"""Tests for the dogfood eval RUNNER (SPEC section 10, checklist item 17).

The runner wires the existing eval harness end-to-end:

    build golden set (git history)  +  clean-PR negative controls
        -> run the review pipeline (orchestrator.review) per sample
        -> grade findings vs golden labels (judge.py)
        -> Scorecard (precision/recall/FP per category + <10% FP budget)

Everything is INJECTED so this runs fully offline with ``FakeProvider`` — no
network, no live credentials, and (per the DATA SAFETY rule) no reading of any
other repo: a synthetic temp git repo is created in-test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pytest

from openrabbit.config import load_config
from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from openrabbit.eval.runner import EvalReport, run_eval
from openrabbit.eval.scorecard import Scorecard
from openrabbit.providers.base import FakeProvider, Provider

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


# --------------------------------------------------------------------------- #
# synthetic temp git repo (created in-test; never reads another repo)          #
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "HOME": str(repo),
    }
    out = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def _commit(repo: Path, filename: str, content: str, message: str) -> str:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _auth_module(body: str) -> str:
    """A multi-line auth module so changed-line counts clear the trivial-diff gate.

    The pipeline's gate skips diffs under ``DEFAULT_MIN_CHANGED_LINES`` (3), so a
    realistic golden sample must touch several lines. Each variant swaps in a
    multi-line ``body`` (the query construction + surrounding statements).
    """
    return (
        "import db\n"
        "\n"
        "\n"
        "def login(user, password):\n"
        "    if not user:\n"
        "        raise ValueError('user required')\n"
        f"{body}"
        "    return row is not None\n"
    )


# Three multi-line bodies, each differing from the others by >= 3 lines so every
# transition clears the trivial-diff gate.
_BODY_BASELINE = (
    "    sql = 'SELECT * FROM u WHERE n=' + user\n"
    "    rows = db.query(sql)\n"
    "    row = rows.first()\n"
)
_BODY_BUGGY = (
    "    sql = 'SELECT * FROM u WHERE n=' + user + ' OR 1=1'\n"
    "    rows = db.query(sql + ';')\n"
    "    row = rows.first_or_none()\n"
)
_BODY_FIXED = (
    "    sql = 'SELECT * FROM u WHERE n=?'\n"
    "    rows = db.query(sql, [user])\n"
    "    row = rows.first()\n"
)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A tiny synthetic repo: a couple of bug-fix + revert commits + a clean one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "auth.py", _auth_module(_BODY_BASELINE), "Initial commit")
    # a buggy commit followed by a revert (known bug)
    _commit(
        repo,
        "auth.py",
        _auth_module(_BODY_BUGGY),
        "introduce bad query that always matches",
    )
    _commit(
        repo,
        "auth.py",
        _auth_module(_BODY_BASELINE),
        'Revert "introduce bad query that always matches"',
    )
    # a fix commit (known bug, security category)
    _commit(
        repo,
        "auth.py",
        _auth_module(_BODY_FIXED),
        "fix: SQL injection in login query construction",
    )
    # a plain refactor (clean, not a bug)
    _commit(
        repo,
        "helpers.py",
        "def helper_one():\n    return 1\n\n\ndef helper_two():\n    return 2\n",
        "refactor: extract helpers module",
    )
    return repo


# --------------------------------------------------------------------------- #
# tool-aware deterministic provider                                            #
# --------------------------------------------------------------------------- #
# A scripted positional FakeProvider is brittle here: the finder is called once
# per (file, lens) and the verifier/judge once each, all sharing whatever script
# order the spine happens to use. Instead we route by the OFFERED tool, which is
# stable regardless of call count/order: emit_findings -> findings,
# verify_findings -> keep verdicts, emit_verdict -> a judge verdict.
_SECURITY_FINDING = {
    "file": "auth.py",
    "startLine": 7,
    "endLine": 9,
    "side": "RIGHT",
    "severity": "high",
    "category": "security",
    "confidence": 95,
    "title": "SQL injection via string concatenation",
    "body": "User input is concatenated into a SQL string.",
    "ruleId": "openrabbit/security/sqli",
}


class _ToolAwareProvider(Provider):
    """A deterministic, network-free provider that answers by offered tool.

    ``emit`` controls whether the finder pass emits the security finding; the
    verifier always keeps everything it is asked about, and the judge always
    returns ``judge_verdict``. This sidesteps positional-script fragility.
    """

    def __init__(
        self, *, emit: bool, judge_verdict: str = "match", name: str = "tool-aware"
    ) -> None:
        self._emit = emit
        self._judge_verdict = judge_verdict
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return "tool-aware-0"

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolSpec]],
        max_tokens: int,
        cache_prefix: Optional[str],
        **opts,
    ) -> CompletionResult:
        names = {t.name for t in (tools or [])}
        usage = Usage(input_tokens=10, output_tokens=5)
        if "emit_findings" in names:
            findings = [dict(_SECURITY_FINDING)] if self._emit else []
            return CompletionResult(
                text="",
                tool_calls=[
                    ToolCall(id="f", name="emit_findings", args={"findings": findings})
                ],
                finish_reason=FinishReason.TOOL_USE,
                usage=usage,
            )
        if "verify_findings" in names:
            # Keep every finding referenced in the batch (ids 0..n-1). We don't
            # know n statically, so emit a generous keep list; unmatched ids are
            # ignored by the verifier.
            verdicts = [
                {"id": i, "keep": True, "confidence": 0.95, "rationale": "ok"}
                for i in range(64)
            ]
            return CompletionResult(
                text="",
                tool_calls=[
                    ToolCall(
                        id="v", name="verify_findings", args={"verdicts": verdicts}
                    )
                ],
                finish_reason=FinishReason.TOOL_USE,
                usage=usage,
            )
        if "emit_verdict" in names:
            return CompletionResult(
                text="",
                tool_calls=[
                    ToolCall(
                        id="j",
                        name="emit_verdict",
                        args={
                            "verdict": self._judge_verdict,
                            "confidence": 0.9,
                            "rationale": "r",
                        },
                    )
                ],
                finish_reason=FinishReason.TOOL_USE,
                usage=usage,
            )
        return CompletionResult(
            text="", tool_calls=[], finish_reason=FinishReason.STOP, usage=usage
        )


def _flagging_provider_factory() -> Callable[[], Provider]:
    """A factory minting a reviewer that flags the security bug and judges match."""
    return lambda: _ToolAwareProvider(emit=True, name="flagging")


def _silent_provider_factory() -> Callable[[], Provider]:
    """A factory minting a clean reviewer that emits no findings."""
    return lambda: _ToolAwareProvider(emit=False, name="silent")


@pytest.fixture()
def config():
    return load_config({"version": 1})


# --------------------------------------------------------------------------- #
# run_eval — end to end                                                        #
# --------------------------------------------------------------------------- #
def test_run_eval_returns_eval_report(git_repo: Path, config):
    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        config=config,
    )
    assert isinstance(report, EvalReport)
    assert isinstance(report.scorecard, Scorecard)
    # The synthetic repo carries bug-labeled commits, so the golden set is built.
    assert report.golden_count >= 1
    # Negative controls were generated and exercised.
    assert report.control_count >= 1


def test_run_eval_flagging_reviewer_scores_true_positives(git_repo: Path, config):
    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        config=config,
    )
    sc = report.scorecard
    # A reviewer that correctly flags the known security bug yields true positives
    # and non-zero recall/precision in the security category.
    assert sc.overall.tp >= 1
    assert 0.0 <= sc.overall.precision <= 1.0
    assert 0.0 <= sc.overall.recall <= 1.0
    sec = next((c for c in sc.categories if c.category == "security"), None)
    assert sec is not None
    assert sec.tp >= 1


def test_run_eval_clean_reviewer_passes_fp_budget(git_repo: Path, config):
    # A reviewer that emits nothing produces no false positives on controls, so
    # the FP budget passes (FP rate 0). It misses the known bugs (recall low).
    report = run_eval(
        git_repo,
        provider=_silent_provider_factory(),
        config=config,
    )
    sc = report.scorecard
    assert sc.overall.fp == 0
    assert sc.overall.false_positive_rate == 0.0
    assert sc.passed is True
    # known bugs went unflagged -> misses (false negatives)
    assert sc.overall.fn >= 1


def test_run_eval_negative_controls_make_false_positives(git_repo: Path, config):
    # A reviewer that flags EVERYTHING (incl. the no-op controls) blows the FP
    # budget: controls are clean by construction, so any finding on them is a FP.
    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        config=config,
    )
    sc = report.scorecard
    # The flagging reviewer fires on every clean control -> false positives, and
    # none stay silent, so the true-negative count is 0 (FP + TN == controls).
    assert sc.overall.fp >= 1
    assert sc.overall.tn == report.control_count - sc.overall.fp
    # with FP on clean controls the budget is exceeded
    assert sc.passed is False


def test_run_eval_respects_limit(git_repo: Path, config):
    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        config=config,
        limit=1,
    )
    assert report.golden_count == 1


def test_run_eval_separate_judge_provider(git_repo: Path, config):
    # judge_provider is injectable independently of the review provider.
    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        judge_provider=_flagging_provider_factory(),
        config=config,
    )
    assert isinstance(report, EvalReport)
    assert report.scorecard.overall.tp >= 1


class _CountingProvider(_ToolAwareProvider):
    """A tool-aware provider that records how many times it was called."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def complete(self, *args, **kwargs) -> CompletionResult:
        self.calls += 1
        return super().complete(*args, **kwargs)


def test_run_eval_uses_distinct_verifier_provider_in_pipeline(git_repo: Path, config):
    # The in-pipeline stage-2 verifier must be a DISTINCT provider object from the
    # finder when verifier_provider is supplied (cross-family routing: Nova finds,
    # GPT-5.5 verifies). Assert the verifier is a separate object AND gets called.
    finder = _CountingProvider(emit=True, name="finder")
    verifier = _CountingProvider(emit=True, name="verifier")
    report = run_eval(
        git_repo,
        provider=finder,
        verifier_provider=verifier,
        config=config,
    )
    assert isinstance(report, EvalReport)
    # Distinct objects: the verifier passed in is not the finder.
    assert verifier is not finder
    # The pipeline's verify stage exercised the injected verifier (it answered
    # verify_findings), proving the cross-family verifier is actually wired.
    assert verifier.calls >= 1
    # A flagged known bug yields a true positive once the verifier keeps it.
    assert report.scorecard.overall.tp >= 1


def test_run_eval_verifier_provider_defaults_to_review_provider(git_repo: Path, config):
    # When verifier_provider is omitted, behavior is unchanged: the finder also
    # self-verifies (back-compat).
    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        config=config,
    )
    assert isinstance(report, EvalReport)
    assert report.scorecard.overall.tp >= 1


def test_eval_report_to_dict_is_json_serializable(git_repo: Path, config):
    import json

    report = run_eval(
        git_repo,
        provider=_flagging_provider_factory(),
        config=config,
    )
    d = report.to_dict()
    json.dumps(d)
    assert d["goldenCount"] == report.golden_count
    assert d["controlCount"] == report.control_count
    assert "scorecard" in d
    assert d["scorecard"]["passed"] == report.scorecard.passed
    # The retained per-finding transcript (graded) is surfaced as a compact,
    # JSON-friendly list so `--json` consumers can read transcripts, not just
    # aggregates (SPEC 10).
    assert "gradedFindings" in d
    assert len(d["gradedFindings"]) == len(report.graded)
    for entry, gf in zip(d["gradedFindings"], report.graded):
        assert entry == {"category": gf.category, "verdict": gf.verdict}


def test_run_eval_accepts_provider_instance(git_repo: Path, config):
    # A bare Provider instance (not a factory) is also accepted: the runner
    # reuses it across samples. A tool-aware provider is stateless, so reuse is
    # safe regardless of how many calls the spine makes.
    provider = _ToolAwareProvider(emit=False, name="instance")
    report = run_eval(git_repo, provider=provider, config=config)
    assert isinstance(report, EvalReport)
    # silent instance: no false positives
    assert report.scorecard.overall.fp == 0


def test_run_eval_with_scripted_fake_provider_instance(git_repo: Path, config):
    # The standard FakeProvider also works as a bare instance when scripted with
    # enough empty-emit results to cover every finder call (no verify needed when
    # nothing is emitted). This documents the FakeProvider compatibility path.
    empty = CompletionResult(
        text="",
        tool_calls=[ToolCall(id="f", name="emit_findings", args={"findings": []})],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )
    provider = FakeProvider([empty for _ in range(200)], name="scripted")
    report = run_eval(git_repo, provider=provider, config=config)
    assert isinstance(report, EvalReport)
    assert report.scorecard.overall.fp == 0


def test_run_eval_raises_on_non_git_dir(tmp_path: Path, config):
    with pytest.raises(Exception):
        run_eval(
            tmp_path / "nope",
            provider=_flagging_provider_factory(),
            config=config,
        )


# --------------------------------------------------------------------------- #
# runner internals — control construction edge cases                           #
# --------------------------------------------------------------------------- #
def test_build_controls_falls_back_when_no_golden_files():
    from openrabbit.eval.runner import _build_controls

    # No golden samples -> a single synthetic fallback control is still produced
    # so the FP-rate denominator is never empty.
    controls = _build_controls([], count=1)
    assert len(controls) == 1
    assert controls[0].known_bug is False


def test_build_controls_respects_count_cap():
    from openrabbit.eval.golden_set import GoldenSample
    from openrabbit.eval.runner import _build_controls

    diff_a = "diff --git a/a.py b/a.py\n+++ b/a.py\n@@\n+x = 1\n"
    diff_b = "diff --git a/b.py b/b.py\n+++ b/b.py\n@@\n+y = 2\n"
    golden = [
        GoldenSample("r@a", "r", "a", diff_a, True, "correctness", "fix", "fix a"),
        GoldenSample("r@b", "r", "b", diff_b, True, "correctness", "fix", "fix b"),
    ]
    # Two golden files exist, but the cap limits controls to 1.
    controls = _build_controls(golden, count=1)
    assert len(controls) == 1


def test_multiline_noop_control_is_a_real_noop():
    from openrabbit.eval.controls import is_noop_diff
    from openrabbit.eval.runner import _multiline_noop_control

    control = _multiline_noop_control("auth.py", "a = 1\nb = 2\nc = 3\nd = 4\n")
    # The synthesized control diff is semantically empty (whitespace-only).
    assert is_noop_diff(control.diff)
    # ... and carries a diff --git header so the router enumerates the file.
    assert "diff --git a/auth.py b/auth.py" in control.diff


# --------------------------------------------------------------------------- #
# CLI `openrabbit eval` subcommand — offline smoke                             #
# --------------------------------------------------------------------------- #
def test_cli_eval_offline_smoke(git_repo: Path, capsys):
    from openrabbit.cli import main

    rc = main(["eval", "--repo", str(git_repo)])
    assert rc == 0
    out = capsys.readouterr().out
    # The offline scripted-fixture run prints the human scorecard table.
    assert "openrabbit eval scorecard" in out
    assert "OVERALL" in out


def test_cli_eval_json_smoke(git_repo: Path, capsys):
    import json

    from openrabbit.cli import main

    rc = main(["eval", "--repo", str(git_repo), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "scorecard" in payload
    assert "goldenCount" in payload
    assert "controlCount" in payload


def test_cli_eval_respects_limit(git_repo: Path, capsys):
    import json

    from openrabbit.cli import main

    rc = main(["eval", "--repo", str(git_repo), "--limit", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["goldenCount"] == 1


def test_cli_eval_online_flag_is_gated_on_creds(git_repo: Path, capsys, monkeypatch):
    # --online requires real Bedrock creds (item 20). Offline-by-default; with
    # --online and no creds the CLI exits non-zero with a clear message rather
    # than attempting a network call.
    from openrabbit.cli import main

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    rc = main(["eval", "--repo", str(git_repo), "--online"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "online" in err.lower()
    assert "cred" in err.lower() or "bedrock" in err.lower()


def test_cli_eval_online_builds_providers_when_creds_present(
    git_repo: Path, tmp_path: Path, capsys, monkeypatch
):
    # With creds present and a finder role configured, --online builds providers
    # via the factory and runs the harness. We stub creds + the model factory so
    # NO network/SDK is touched (the invariant): the factory returns a
    # network-free tool-aware fake.
    import openrabbit.cli as cli

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    # Mint a fresh tool-aware fake per role so the finder and verifier providers
    # are DISTINCT objects (mirrors the real cross-family Nova→GPT-5.5 routing).
    built: dict[str, _ToolAwareProvider] = {}

    def _fake_factory(role):
        prov = _ToolAwareProvider(emit=True, name=str(role.model))
        built[role.model] = prov
        return prov

    monkeypatch.setattr(cli.orch, "model_factory", _fake_factory)

    cfg = tmp_path / ".openrabbit.yaml"
    cfg.write_text(
        "version: 1\n"
        "model_roles:\n"
        "  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n"
        "  verifier: {model: openai.gpt-5.5, region: us-east-2}\n",
        encoding="utf-8",
    )
    rc = cli.main(
        ["eval", "--repo", str(git_repo), "--config", str(cfg), "--online", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "scorecard" in payload
    # The CLI built a DISTINCT verifier provider (GPT-5.5) separate from the
    # Nova finder, wiring the real cross-family routing into the pipeline.
    assert "amazon.nova-pro-v1:0" in built
    assert "openai.gpt-5.5" in built
    assert built["openai.gpt-5.5"] is not built["amazon.nova-pro-v1:0"]


def test_cli_eval_online_requires_finder_role(
    git_repo: Path, tmp_path: Path, capsys, monkeypatch
):
    import openrabbit.cli as cli

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    cfg = tmp_path / ".openrabbit.yaml"
    cfg.write_text("version: 1\n", encoding="utf-8")  # no model_roles
    rc = cli.main(["eval", "--repo", str(git_repo), "--config", str(cfg), "--online"])
    assert rc == 2
    assert "finder" in capsys.readouterr().err.lower()


def test_cli_eval_require_pass_exits_nonzero_on_budget_fail(git_repo: Path, capsys):
    # The default offline fixtures flag a clean control -> FP budget fails. With
    # --require-pass that becomes a non-zero exit (a CI gate).
    from openrabbit.cli import main

    rc = main(["eval", "--repo", str(git_repo), "--require-pass"])
    assert rc == 1
