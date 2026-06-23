"""Live-CLI wiring e2e tests (cli-wiring bucket).

The final adversarial review found several CORRECT, well-tested components that
are NEVER reached by the production CLI/live paths — "wiring gaps, not logic
gaps". Offline-green masked them because the isolated units were tested directly
while the real call sites bypassed them.

Each test here exercises the REAL CLI call site (not the isolated unit) so a
regression that re-detaches the component from the live path is caught:

1. Diff-anchor guard reaches ``emit_github`` (``valid_positions`` /
   ``changed_files``) so an out-of-diff finding is dropped before the batched
   ``createReview`` (one bad position 422s the whole batch otherwise).
2. ``_load_base_config`` retries ``origin/<base>`` / ``refs/remotes/origin/...``
   on a detached-HEAD CI checkout, so the trusted base policy is loaded instead
   of silently trusting the PR-head config.
4. ``openrabbit eval --online`` threads ``include_real_clean_controls=True`` into
   ``run_eval`` so the FP denominator is not synthetic-whitespace-only, plus the
   degenerate-corpus / distinct-judge / committed-corpus guards.

No network, no live AWS/GitHub credentials: GitHub uses an in-process fake
client, providers are monkeypatched FakeProviders, and ``run_eval`` is recorded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from openrabbit import cli
from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    ToolCall,
    Usage,
)
from openrabbit.providers.base import FakeProvider

# A diff with a single hunk on src/api/auth.py (enough changed lines to clear
# the trivial-diff gate). Valid RIGHT anchors run over the hunk's new-side line
# range (11..16 added/context). The point is that a finding far outside (line
# 999) is out-of-diff and must be dropped before the batched createReview POST.
WIRING_DIFF = """\
diff --git a/src/api/auth.py b/src/api/auth.py
index 1111111..2222222 100644
--- a/src/api/auth.py
+++ b/src/api/auth.py
@@ -10,3 +10,7 @@ def login(request):
     user = lookup(request.user)
+    token = request.GET["token"]
+    query = "SELECT * FROM users WHERE token = '" + token + "'"
+    db.execute(query)
+    log("queried")
     return ok()
"""


def _emit_findings_result(findings: list[dict]) -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(id="c1", name="emit_findings", args={"findings": findings})
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(input_tokens=100, output_tokens=50),
    )


def _verify_batch_result(
    verdicts: list[tuple[int, bool, float]],
) -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v1",
                name="verify_findings",
                args={
                    "verdicts": [
                        {"id": vid, "keep": keep, "confidence": conf, "rationale": "ok"}
                        for vid, keep, conf in verdicts
                    ]
                },
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(input_tokens=80, output_tokens=20),
    )


def _out_of_diff_finding() -> dict:
    """A finder finding anchored on a line that is NOT part of the diff (line 999).

    GitHub 422s the whole batched createReview on a single out-of-diff position,
    so this MUST be dropped/clamped before the POST.
    """
    return {
        "file": "src/api/auth.py",
        "startLine": 999,
        "endLine": 999,
        "side": "RIGHT",
        "severity": "high",
        "category": "security",
        "confidence": 95,
        "title": "out-of-diff hallucinated finding",
        "body": "Rationale.",
        "ruleId": "openrabbit/security/sqli",
    }


# --------------------------------------------------------------------------- #
# Finding 1: diff-anchor guard must be live in the CLI post path                #
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _RecordingGitHubClient:
    """In-process fake httpx-like client that records every review POST body."""

    def __init__(self, diff_text: str):
        self._diff = diff_text
        self.posts: list[tuple[str, Any]] = []
        self.review_comment_counts: list[int] = []

    def get(self, url, headers=None):
        if url.endswith("/comments"):
            return _Resp(200, [])
        return _Resp(200, {}, text=self._diff)

    def post(self, url, headers=None, json=None):
        self.posts.append((url, json))
        if "graphql" in url:
            return _Resp(
                200,
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "nodes": [],
                                    "pageInfo": {"hasNextPage": False},
                                }
                            }
                        }
                    }
                },
            )
        if url.endswith("/reviews"):
            self.review_comment_counts.append(len((json or {}).get("comments", [])))
            return _Resp(200, {"id": 1})
        return _Resp(200, {"id": 2})

    def patch(self, url, headers=None, json=None):
        return _Resp(200, {"id": 3})

    def close(self):
        pass


def _install_fake_adapter(monkeypatch, fake_client) -> None:
    # Import the adapter class LAZILY (at call time): another test
    # (test_github.py::test_module_imports_without_httpx) reloads the
    # openrabbit.adapters.github module, rebinding GitHubAdapter to a fresh class
    # object. A module-level import would patch the STALE class while the CLI's
    # own fresh import inside _cmd_review_online resolves the reloaded one — so we
    # must patch whatever class the module currently exposes.
    from openrabbit.adapters.github import GitHubAdapter

    real_init = GitHubAdapter.__init__

    def patched_init(self, repo, pr_number, token, **kwargs):
        kwargs["client"] = fake_client
        real_init(self, repo, pr_number, token, **kwargs)

    monkeypatch.setattr(GitHubAdapter, "__init__", patched_init)


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / ".openrabbit.yaml"
    cfg_path.write_text(
        "version: 1\nreview:\n  lenses: [security]\n"
        "model_roles:\n  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n"
        "  verifier: {model: openai.gpt-5.5, region: us-east-2}\n",
        encoding="utf-8",
    )
    return cfg_path


class TestDiffAnchorGuardLiveInCli:
    def test_out_of_diff_finding_dropped_before_post(
        self, monkeypatch, capsys, tmp_path
    ):
        """A CLI ``review --post`` run where the only finding is out-of-diff must
        post ZERO inline comments (dropped by the diff-anchor guard), not a
        422-prone batch carrying the hallucinated position.

        RED before the wiring: ``_cmd_review_online`` calls ``emit_github``
        WITHOUT ``valid_positions=``/``changed_files=``, so the guard never runs
        and the out-of-diff comment reaches the createReview payload.
        """
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        cfg_path = _write_config(tmp_path)
        fake_client = _RecordingGitHubClient(WIRING_DIFF)
        _install_fake_adapter(monkeypatch, fake_client)

        finder = FakeProvider([_emit_findings_result([_out_of_diff_finding()])])
        verifier = FakeProvider([_verify_batch_result([(0, True, 0.95)])])
        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.build_providers",
            lambda cfg: {"finder": finder, "verifier": verifier},
        )

        rc = cli.main(
            [
                "review",
                "--repo",
                "acme/repo",
                "--pr",
                "7",
                "--commit",
                "headsha",
                "--config",
                str(cfg_path),
                "--post",
            ]
        )
        assert rc == 0
        # If a createReview POST happened at all, it must carry ZERO inline
        # comments — the single out-of-diff finding was dropped by the guard.
        # (When the guard drops every comment the adapter skips the POST entirely,
        # which is also acceptable: review_comment_counts stays empty.)
        assert all(n == 0 for n in fake_client.review_comment_counts), (
            "out-of-diff finding reached the createReview payload — diff-anchor "
            "guard is not wired into the CLI post path"
        )

    def test_in_diff_finding_still_posts(self, monkeypatch, capsys, tmp_path):
        """Regression guard: a legitimately in-diff finding still posts an inline
        comment (the guard must not over-drop everything)."""
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        cfg_path = _write_config(tmp_path)
        fake_client = _RecordingGitHubClient(WIRING_DIFF)
        _install_fake_adapter(monkeypatch, fake_client)

        in_diff = {
            "file": "src/api/auth.py",
            "startLine": 11,
            "endLine": 11,
            "side": "RIGHT",
            "severity": "high",
            "category": "security",
            "confidence": 95,
            "title": "in-diff finding",
            "body": "Rationale.",
            "ruleId": "openrabbit/security/sqli",
        }
        finder = FakeProvider([_emit_findings_result([in_diff])])
        verifier = FakeProvider([_verify_batch_result([(0, True, 0.95)])])
        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.build_providers",
            lambda cfg: {"finder": finder, "verifier": verifier},
        )

        rc = cli.main(
            [
                "review",
                "--repo",
                "acme/repo",
                "--pr",
                "7",
                "--commit",
                "headsha",
                "--config",
                str(cfg_path),
                "--post",
            ]
        )
        assert rc == 0
        assert fake_client.review_comment_counts, "no createReview POST happened"
        assert any(n >= 1 for n in fake_client.review_comment_counts), (
            "in-diff finding was incorrectly dropped"
        )


# --------------------------------------------------------------------------- #
# Finding 2: base-config loader retries origin/<base> on detached HEAD          #
# --------------------------------------------------------------------------- #
class TestBaseConfigDetachedHead:
    def _make_repo(self, tmp_path: Path) -> tuple[Path, str]:
        """Build a git repo whose base config lives only under origin/<base>.

        Simulates the CI detached-HEAD checkout: the local branch name does not
        resolve (we check out a detached commit) but ``origin/<base>`` does.
        Returns ``(checkout_dir, base_ref)``.
        """
        origin = tmp_path / "origin"
        origin.mkdir()

        def run(cwd: Path, *args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
            ).stdout

        run(origin, "init", "-b", "main")
        run(origin, "config", "user.email", "t@t.t")
        run(origin, "config", "user.name", "t")
        # The trusted BASE config: strict gate, security lens kept.
        (origin / ".openrabbit.yaml").write_text(
            "version: 1\nreview:\n  confidence_gate: 0.80\n"
            "  lenses: [correctness, security]\n",
            encoding="utf-8",
        )
        (origin / "app.py").write_text("x = 1\n", encoding="utf-8")
        run(origin, "add", ".")
        run(origin, "commit", "-m", "base")

        # Clone, then detach HEAD so the local 'main' ref is NOT what we resolve.
        checkout = tmp_path / "checkout"
        run(tmp_path, "clone", str(origin), str(checkout))
        head_sha = run(checkout, "rev-parse", "HEAD").strip()
        run(checkout, "checkout", "--detach", head_sha)
        # Delete the local 'main' branch so a bare 'main' git-show cannot resolve;
        # only origin/main remains.
        run(checkout, "branch", "-D", "main")
        return checkout, "main"

    def test_bare_base_ref_fails_but_origin_resolves(self, tmp_path):
        """On a detached-HEAD checkout where only ``origin/<base>`` resolves, the
        base policy MUST be loaded (not silently fall back to head=None).

        RED before the fix: ``_load_base_config`` runs ``git show main:...`` with a
        BARE branch name, which fails on the detached checkout, returns None, and
        the run silently trusts the PR-head config.
        """
        checkout, base_ref = self._make_repo(tmp_path)

        # Sanity: the bare ref does NOT resolve (proves the gap is real).
        bare = subprocess.run(
            ["git", "show", f"{base_ref}:.openrabbit.yaml"],
            cwd=checkout,
            capture_output=True,
            text=True,
        )
        assert bare.returncode != 0, "bare base ref unexpectedly resolved"

        base_config = cli._load_base_config(base_ref, checkout)
        assert base_config is not None, (
            "base config not loaded via origin/<base> fallback — detached-HEAD CI "
            "would silently trust the PR-head config"
        )
        assert base_config.review.confidence_gate == pytest.approx(0.80)
        assert "security" in base_config.review.lenses

    def test_origin_prefixed_ref_also_resolves(self, tmp_path):
        """An explicitly origin-prefixed base ref still resolves (no regression)."""
        checkout, base_ref = self._make_repo(tmp_path)
        base_config = cli._load_base_config(f"origin/{base_ref}", checkout)
        assert base_config is not None
        assert base_config.review.confidence_gate == pytest.approx(0.80)


# --------------------------------------------------------------------------- #
# Finding 4: --online eval threads include_real_clean_controls into run_eval    #
# --------------------------------------------------------------------------- #
class _RecordingRunEval:
    """Records the kwargs the CLI passes to ``run_eval`` (no real eval runs)."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repo_root, **kwargs):
        self.calls.append({"repo_root": repo_root, **kwargs})
        # Return a minimal passing EvalReport-like object. We only need the CLI to
        # finish; assertions are on the recorded kwargs.
        from openrabbit.eval.runner import LIVE_MODE, EvalReport
        from openrabbit.eval.scorecard import CategoryScore, Scorecard

        overall = CategoryScore(category="overall", tp=1, fp=0, fn=0, tn=5)
        sc = Scorecard(
            categories=[overall],
            overall=overall,
            fp_budget=0.10,
            addressed_rate=1.0,
            total_findings=1,
        )
        return EvalReport(
            scorecard=sc,
            golden_count=3,
            control_count=5,
            repo="r",
            graded=[],
            mode=kwargs.get("mode", LIVE_MODE),
            call_count=10,
            real_clean_control_count=5,
            fp_by_kind={"synthetic": 0, "real_clean": 0},
        )


@pytest.fixture()
def online_creds(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer")
    yield


def _online_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / ".openrabbit.yaml"
    cfg_path.write_text(
        "version: 1\nreview:\n  lenses: [correctness, security]\n"
        "model_roles:\n  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n"
        "  verifier: {model: openai.gpt-5.5, region: us-east-2}\n",
        encoding="utf-8",
    )
    return cfg_path


class TestEvalOnlineRealCleanControls:
    def test_online_threads_real_clean_controls(
        self, monkeypatch, online_creds, tmp_path
    ):
        """``openrabbit eval --online`` MUST pass ``include_real_clean_controls=True``
        into ``run_eval`` so the FP denominator is not synthetic-whitespace-only.

        RED before the fix: ``_cmd_eval``'s online ``run_eval`` call omits the
        flag, so the FP rate collapses to ~0 on whitespace no-ops and always
        passes.
        """
        recorder = _RecordingRunEval()
        monkeypatch.setattr("openrabbit.eval.runner.run_eval", recorder)
        # Avoid building real Bedrock providers.
        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.model_factory",
            lambda role: lambda: FakeProvider([], name="fake"),
        )
        cfg_path = _online_config(tmp_path)

        rc = cli.main(
            ["eval", "--repo", str(tmp_path), "--online", "--config", str(cfg_path)]
        )
        assert rc == 0
        assert recorder.calls, "run_eval was never called"
        call = recorder.calls[-1]
        assert call.get("include_real_clean_controls") is True, (
            "online eval did not thread include_real_clean_controls=True into "
            "run_eval — the FP denominator is synthetic-whitespace-only"
        )

    def test_online_passes_distinct_judge_provider(
        self, monkeypatch, online_creds, tmp_path
    ):
        """The verifier must not self-grade: a distinct ``judge_provider`` (the
        cross-family verifier) is threaded into ``run_eval``."""
        recorder = _RecordingRunEval()
        monkeypatch.setattr("openrabbit.eval.runner.run_eval", recorder)
        made: list[str] = []

        def fake_factory(role):
            made.append(role.model)
            return lambda: FakeProvider([], name=role.model)

        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.model_factory", fake_factory
        )
        cfg_path = _online_config(tmp_path)

        rc = cli.main(
            ["eval", "--repo", str(tmp_path), "--online", "--config", str(cfg_path)]
        )
        assert rc == 0
        call = recorder.calls[-1]
        # judge_provider is supplied (not left to default to the finder).
        assert call.get("judge_provider") is not None


class TestEvalDegenerateCorpusGuard:
    def test_online_require_pass_refuses_empty_denominator(
        self, monkeypatch, online_creds, tmp_path
    ):
        """``--online --require-pass`` must NOT pass on a degenerate corpus (the FP
        denominator fp+tn == 0 makes ``false_positive_rate`` 0.0 -> a vacuous PASS).

        RED before the guard: a run with zero controls (tn=0) and zero findings
        passes ``--require-pass`` even though nothing was actually measured.
        """
        from openrabbit.eval.runner import EvalReport
        from openrabbit.eval.scorecard import CategoryScore, Scorecard

        def degenerate_run_eval(repo_root, **kwargs):
            overall = CategoryScore(category="overall", tp=0, fp=0, fn=0, tn=0)
            sc = Scorecard(
                categories=[overall],
                overall=overall,
                fp_budget=0.10,
                addressed_rate=0.0,
                total_findings=0,
            )
            return EvalReport(
                scorecard=sc,
                golden_count=0,
                control_count=0,
                repo="r",
                graded=[],
                mode="live",
                call_count=0,
                real_clean_control_count=0,
                fp_by_kind={"synthetic": 0, "real_clean": 0},
            )

        monkeypatch.setattr("openrabbit.eval.runner.run_eval", degenerate_run_eval)
        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.model_factory",
            lambda role: lambda: FakeProvider([], name="fake"),
        )
        cfg_path = _online_config(tmp_path)

        rc = cli.main(
            [
                "eval",
                "--repo",
                str(tmp_path),
                "--online",
                "--require-pass",
                "--config",
                str(cfg_path),
            ]
        )
        assert rc != 0, (
            "--online --require-pass passed on a degenerate/empty corpus "
            "(vacuous PASS on an empty FP denominator)"
        )


class TestEvalCorpusFlag:
    def test_corpus_flag_threads_into_run_eval(
        self, monkeypatch, online_creds, tmp_path
    ):
        """``--corpus PATH`` is threaded into ``run_eval`` as ``corpus_path`` so the
        committed reproducible corpus can be loaded instead of live mining."""
        recorder = _RecordingRunEval()
        monkeypatch.setattr("openrabbit.eval.runner.run_eval", recorder)
        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.model_factory",
            lambda role: lambda: FakeProvider([], name="fake"),
        )
        cfg_path = _online_config(tmp_path)
        corpus = tmp_path / "golden.jsonl"
        corpus.write_text("", encoding="utf-8")

        rc = cli.main(
            [
                "eval",
                "--repo",
                str(tmp_path),
                "--online",
                "--corpus",
                str(corpus),
                "--config",
                str(cfg_path),
            ]
        )
        assert rc == 0
        call = recorder.calls[-1]
        assert str(call.get("corpus_path")) == str(corpus), (
            "--corpus PATH not threaded into run_eval as corpus_path"
        )
