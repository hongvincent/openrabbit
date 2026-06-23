"""Eval-honesty tests for the dogfood RUNNER (adversarial findings 2, 3, 5 + contract).

Finding 2 (CRITICAL): the FP denominator was EXCLUSIVELY synthetic trailing-
whitespace no-ops, so the FP<10% gate was satisfiable purely on whitespace. The
runner must support REAL-clean controls (diverse no-op kinds) and report
synthetic-vs-real-clean FP SEPARATELY.

Finding 3 (HIGH): run_eval must load a COMMITTED versioned golden JSONL when one
is present (reproducible), treating live mining as an explicit build step.

Finding 5 (MEDIUM): calibrate_agreement must be wired into the run (a
CalibrationReport attached to the report; untrusted when agreement < threshold),
and a distinct judge model is allowed.

Contract (coordinated with the cli agent): EvalReport has ``mode`` and
``call_count``; run_eval populates ``call_count`` from real provider calls and
refuses to emit a numeric FP rate when ``mode != 'live'``.
"""

from __future__ import annotations

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
from openrabbit.providers.base import Provider

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


# --------------------------------------------------------------------------- #
# synthetic temp git repo (created in-test)                                    #
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
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "auth.py", _auth_module(_BODY_BASELINE), "Initial commit")
    _commit(repo, "auth.py", _auth_module(_BODY_BUGGY), "introduce bad query")
    _commit(repo, "auth.py", _auth_module(_BODY_BASELINE), 'Revert "introduce bad query"')
    _commit(
        repo,
        "auth.py",
        _auth_module(_BODY_FIXED),
        "fix: SQL injection in login query construction",
    )
    # A genuinely clean follow-up change (a new feature module) — a real-clean
    # negative control: it is NOT followed by any fix/revert.
    _commit(
        repo,
        "feature.py",
        "def feature_one():\n    return 1\n\n\ndef feature_two():\n    return 2\n",
        "feat: add feature module",
    )
    return repo


# --------------------------------------------------------------------------- #
# deterministic, network-free provider routed by offered tool                  #
# --------------------------------------------------------------------------- #
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
    def __init__(
        self, *, emit: bool, judge_verdict: str = "match", name: str = "tool-aware"
    ) -> None:
        self._emit = emit
        self._judge_verdict = judge_verdict
        self._name = name
        self.calls = 0

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
        self.calls += 1
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
            verdicts = [
                {"id": i, "keep": True, "confidence": 0.95, "rationale": "ok"}
                for i in range(64)
            ]
            return CompletionResult(
                text="",
                tool_calls=[
                    ToolCall(id="v", name="verify_findings", args={"verdicts": verdicts})
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


def _flagging_factory() -> Callable[[], Provider]:
    return lambda: _ToolAwareProvider(emit=True, name="flagging")


def _silent_factory() -> Callable[[], Provider]:
    return lambda: _ToolAwareProvider(emit=False, name="silent")


@pytest.fixture()
def config():
    return load_config({"version": 1})


# --------------------------------------------------------------------------- #
# Contract: mode + call_count + numeric-FP gating                              #
# --------------------------------------------------------------------------- #
def test_eval_report_defaults_to_live_mode(git_repo: Path, config):
    report = run_eval(git_repo, provider=_silent_factory(), config=config)
    assert isinstance(report, EvalReport)
    assert report.mode == "live"


def test_call_count_reflects_real_provider_calls(git_repo: Path, config):
    # A flagging finder + verifier + judge make multiple real provider calls.
    report = run_eval(git_repo, provider=_flagging_factory(), config=config)
    assert report.call_count > 0


def test_silent_run_still_makes_finder_calls(git_repo: Path, config):
    report = run_eval(git_repo, provider=_silent_factory(), config=config)
    # Even with nothing emitted, the finder ran on every sample + control.
    assert report.call_count >= report.golden_count + report.control_count


def test_fixture_mode_refuses_numeric_fp_rate(git_repo: Path, config):
    # When the run is NOT live (fixture/offline), the FP rate is not a real
    # measurement and must be reported as None / N/A, never a number.
    report = run_eval(
        git_repo, provider=_flagging_factory(), config=config, mode="fixture"
    )
    assert report.mode == "fixture"
    assert report.fp_rate() is None
    d = report.to_dict()
    assert d["scorecard"]["falsePositiveRate"] in (None, "N/A")


def test_live_mode_emits_numeric_fp_rate(git_repo: Path, config):
    report = run_eval(git_repo, provider=_flagging_factory(), config=config)
    assert report.fp_rate() is not None
    assert isinstance(report.fp_rate(), float)


# --------------------------------------------------------------------------- #
# Finding 2: real-clean controls + synthetic-vs-real-clean separation          #
# --------------------------------------------------------------------------- #
def test_fp_denominator_can_include_real_clean_controls(git_repo: Path, config):
    # When real-clean controls are mined from the repo, they augment the FP
    # denominator (true negatives) rather than being purely synthetic whitespace.
    report = run_eval(
        git_repo,
        provider=_silent_factory(),
        config=config,
        include_real_clean_controls=True,
    )
    assert report.real_clean_control_count >= 1
    # A silent reviewer turns every clean control into a true negative.
    assert report.scorecard.overall.tn >= report.real_clean_control_count


def test_synthetic_and_real_clean_fp_reported_separately(git_repo: Path, config):
    # A flagging reviewer fires on BOTH synthetic and real-clean controls; the
    # report must break the FP counts out by control kind so the gate cannot be
    # satisfied purely on whitespace.
    report = run_eval(
        git_repo,
        provider=_flagging_factory(),
        config=config,
        include_real_clean_controls=True,
    )
    breakdown = report.fp_breakdown()
    assert "synthetic" in breakdown
    assert "real_clean" in breakdown
    # Distinct counters, both populated for an everything-flagging reviewer.
    assert breakdown["synthetic"] >= 0
    assert breakdown["real_clean"] >= 1
    # The breakdown is reflected in the serialized report.
    assert report.to_dict()["fpBreakdown"]["real_clean"] == breakdown["real_clean"]


def test_real_clean_controls_default_off_keeps_backcompat(git_repo: Path, config):
    # Without the flag, no real-clean controls are mined (back-compat with the
    # existing synthetic-only behavior), so the count is zero.
    report = run_eval(git_repo, provider=_silent_factory(), config=config)
    assert report.real_clean_control_count == 0


# --------------------------------------------------------------------------- #
# Finding 5: calibration wired in; distinct judge model honored                #
# --------------------------------------------------------------------------- #
def test_calibration_report_attached_when_human_labels_supplied(git_repo: Path, config):
    # Supplying held-out human labels wires calibrate_agreement into the run and
    # attaches a CalibrationReport; low agreement marks the run untrusted.
    report = run_eval(
        git_repo,
        provider=_flagging_factory(),
        config=config,
        human_verdicts=["miss", "miss", "miss", "miss"],  # disagree with the judge
    )
    assert report.calibration is not None
    assert report.calibration.n >= 1
    # Forced disagreement -> below threshold -> not calibrated -> untrusted.
    assert report.calibration.calibrated is False
    assert report.trusted is False


def test_no_calibration_when_no_human_labels(git_repo: Path, config):
    report = run_eval(git_repo, provider=_flagging_factory(), config=config)
    assert report.calibration is None
    # Absent calibration evidence, the run is not asserted as calibrated-trusted,
    # but it is not marked untrusted either (nothing to contradict it).
    assert report.trusted is True


def test_distinct_judge_provider_is_used(git_repo: Path, config):
    # The judge model can be distinct from finder/verifier; it must actually be
    # the object that answers emit_verdict.
    judge = _ToolAwareProvider(emit=False, name="judge")
    run_eval(
        git_repo,
        provider=_flagging_factory(),
        judge_provider=judge,
        config=config,
    )
    # The judge object answered at least one emit_verdict call.
    assert judge.calls >= 1


# --------------------------------------------------------------------------- #
# Finding 3: load a committed versioned golden JSONL when present              #
# --------------------------------------------------------------------------- #
def test_run_eval_loads_committed_corpus_when_present(git_repo: Path, config, tmp_path):
    from openrabbit.eval.golden_set import GoldenSample, write_jsonl

    # A committed corpus with a single, deliberately-distinct sample id. If the
    # runner mines live instead of loading it, this id would not appear.
    committed = GoldenSample(
        sample_id="COMMITTED@deadbeef",
        repo="repo",
        commit="deadbeefcafebabe",
        diff=(
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n"
            "-return a - b\n+return a + b\n"
        ),
        known_bug=True,
        bug_category="correctness",
        source="fix",
        message="fix: corrected operator",
        defect_location="x.py:1",
    )
    corpus = tmp_path / "golden.jsonl"
    write_jsonl([committed], corpus)

    report = run_eval(
        git_repo,
        provider=_silent_factory(),
        config=config,
        corpus_path=corpus,
    )
    assert report.golden_count == 1
    # A miss is recorded against the committed sample's category (silent reviewer).
    assert any(g.verdict == "miss" for g in report.graded)
