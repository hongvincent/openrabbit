"""End-to-end integration tests for the deterministic pipeline spine.

These tests exercise gate -> route -> context -> run_lenses -> verify ->
dedup -> emit entirely OFFLINE: every model call uses ``FakeProvider`` with
scripted results, and the GitHub side uses an in-process fake client. No
network, no live AWS/GitHub credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openrabbit.config import load_config
from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    ToolCall,
    Usage,
)
from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.providers.base import FakeProvider

from openrabbit.pipeline import context as ctx_mod
from openrabbit.pipeline import dedup as dedup_mod
from openrabbit.pipeline import emit as emit_mod
from openrabbit.pipeline import gate as gate_mod
from openrabbit.pipeline import orchestrator as orch_mod
from openrabbit.pipeline import route as route_mod
from openrabbit.pipeline import run_lenses as run_lenses_mod
from openrabbit.pipeline import verify as verify_mod


# --------------------------------------------------------------------------- #
# Fixtures / sample data                                                       #
# --------------------------------------------------------------------------- #
SAMPLE_DIFF = """\
diff --git a/src/api/auth.py b/src/api/auth.py
index 1111111..2222222 100644
--- a/src/api/auth.py
+++ b/src/api/auth.py
@@ -10,6 +10,9 @@ def login(request):
     user = lookup(request.user)
     token = request.GET["token"]
-    if token == user.token:
+    query = "SELECT * FROM users WHERE token = '" + token + "'"
+    db.execute(query)
+    if token == user.token:
         return ok()
     return deny()
diff --git a/README.md b/README.md
index 3333333..4444444 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,3 @@
 # Project
+Some docs line.
 More text.
diff --git a/package-lock.json b/package-lock.json
index 5555555..6666666 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,4 @@
 {
+  "added": true,
 }
"""

LOCKFILE_ONLY_DIFF = """\
diff --git a/package-lock.json b/package-lock.json
index 5555555..6666666 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,4 @@
 {
+  "added": true,
 }
"""


@pytest.fixture
def config():
    return load_config(
        {
            "version": 1,
            "review": {
                "profile": "balanced",
                "confidence_gate": 0.80,
                "incremental": True,
                "lenses": ["correctness", "security"],
            },
            "model_roles": {
                "finder": {"model": "amazon.nova-pro-v1:0", "region": "ap-northeast-2"},
                "verifier": {"model": "openai.gpt-5.5", "region": "us-east-2"},
            },
        }
    )


def _finder_finding(file: str, rule: str, conf_int: int, category: str) -> dict:
    """A raw finder finding dict (confidence as integer 0-100, no fingerprint)."""
    return {
        "file": file,
        "startLine": 12,
        "endLine": 14,
        "side": "RIGHT",
        "severity": "high",
        "category": category,
        "confidence": conf_int,
        "title": f"{category} issue in {file}",
        "body": "Rationale.",
        "ruleId": rule,
    }


def _emit_findings_result(findings: list[dict]) -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(id="c1", name="emit_findings", args={"findings": findings})
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(input_tokens=100, output_tokens=50),
    )


def _verify_result(confidence: float, keep: bool = True) -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v1",
                name="verify_finding",
                args={"keep": keep, "confidence": confidence, "rationale": "ok"},
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(input_tokens=80, output_tokens=20),
    )


# --------------------------------------------------------------------------- #
# gate                                                                         #
# --------------------------------------------------------------------------- #
class TestGate:
    def test_skip_draft(self, config):
        decision = gate_mod.evaluate_gate(
            config, {"draft": True, "state": "open", "head_sha": "abc"}, SAMPLE_DIFF
        )
        assert decision.should_review is False
        assert "draft" in decision.reason.lower()

    def test_skip_closed(self, config):
        decision = gate_mod.evaluate_gate(
            config, {"draft": False, "state": "closed", "head_sha": "abc"}, SAMPLE_DIFF
        )
        assert decision.should_review is False

    def test_skip_lockfile_only(self, config):
        decision = gate_mod.evaluate_gate(
            config,
            {"draft": False, "state": "open", "head_sha": "abc"},
            LOCKFILE_ONLY_DIFF,
        )
        assert decision.should_review is False
        assert "lockfile" in decision.reason.lower() or "generated" in decision.reason.lower()

    def test_skip_trivial(self, config):
        tiny = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        )
        decision = gate_mod.evaluate_gate(
            config,
            {"draft": False, "state": "open", "head_sha": "abc"},
            tiny,
            min_changed_lines=5,
        )
        assert decision.should_review is False
        assert "trivial" in decision.reason.lower()

    def test_skip_already_reviewed(self, config, tmp_path):
        state_path = tmp_path / "state.json"
        store = gate_mod.StateStore(state_path)
        store.record_review("acme/repo", 7, "abc123")
        decision = gate_mod.evaluate_gate(
            config,
            {"draft": False, "state": "open", "head_sha": "abc123", "repo": "acme/repo", "number": 7},
            SAMPLE_DIFF,
            store=store,
        )
        assert decision.should_review is False
        assert "already" in decision.reason.lower()

    def test_review_proceeds(self, config):
        decision = gate_mod.evaluate_gate(
            config, {"draft": False, "state": "open", "head_sha": "abc"}, SAMPLE_DIFF
        )
        assert decision.should_review is True

    def test_state_store_roundtrip(self, tmp_path):
        state_path = tmp_path / "s.json"
        store = gate_mod.StateStore(state_path)
        assert store.last_reviewed_sha("acme/repo", 1) is None
        store.record_review("acme/repo", 1, "deadbeef")
        # New instance reads persisted state.
        store2 = gate_mod.StateStore(state_path)
        assert store2.last_reviewed_sha("acme/repo", 1) == "deadbeef"


# --------------------------------------------------------------------------- #
# route                                                                        #
# --------------------------------------------------------------------------- #
class TestRoute:
    def test_parse_per_file_hunks(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        files = {f.path for f in plan.files}
        assert "src/api/auth.py" in files
        assert "README.md" in files
        assert "package-lock.json" in files

    def test_classify_file_types(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        by_path = {f.path: f for f in plan.files}
        assert by_path["README.md"].file_type == "docs"
        assert by_path["package-lock.json"].file_type in ("lockfile", "generated")
        assert by_path["src/api/auth.py"].file_type == "code"

    def test_security_sensitive_gets_security_lens(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        by_path = {f.path: f for f in plan.files}
        auth = by_path["src/api/auth.py"]
        assert "security" in auth.lenses
        assert auth.risk in ("high", "medium", "low")

    def test_docs_drops_correctness_security_lenses(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        by_path = {f.path: f for f in plan.files}
        # Docs files shouldn't get a heavy correctness/security pass.
        assert by_path["README.md"].lenses == [] or "security" not in by_path["README.md"].lenses

    def test_hunks_have_content(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        auth = next(f for f in plan.files if f.path == "src/api/auth.py")
        assert auth.hunks
        assert any("SELECT" in h.text for h in auth.hunks)


# --------------------------------------------------------------------------- #
# context                                                                      #
# --------------------------------------------------------------------------- #
class TestContext:
    def test_byte_stable_prefix(self, config):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        a = ctx_mod.build_prefix(config, pr_context={"title": "T", "body": "B"})
        b = ctx_mod.build_prefix(config, pr_context={"title": "T", "body": "B"})
        assert a == b  # deterministic / byte-stable

    def test_per_file_message_includes_diff(self, config):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        msg = ctx_mod.build_file_message(pf)
        assert "SELECT" in msg.content if isinstance(msg.content, str) else True

    def test_untrusted_fencing(self, config):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        msg = ctx_mod.build_file_message(pf)
        body = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        assert "UNTRUSTED" in body.upper()


# --------------------------------------------------------------------------- #
# run_lenses                                                                   #
# --------------------------------------------------------------------------- #
class TestRunLenses:
    def test_runs_finder_and_parses_findings(self, config):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/security/sqli", 90, "security")]
                ),
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/correctness/x", 70, "correctness")]
                ),
            ]
        )
        prefix = ctx_mod.build_prefix(config, pr_context={})
        findings = run_lenses_mod.run_lenses(
            finder,
            pf,
            lens_prompts={"security": "SEC", "correctness": "COR"},
            prefix=prefix,
        )
        assert len(findings) >= 1
        assert all(isinstance(f, Finding) for f in findings)
        # confidence rescaled 0..1
        assert all(0.0 <= f.confidence <= 1.0 for f in findings)
        # fingerprint computed by harness
        assert all(len(f.fingerprint) == 64 for f in findings)

    def test_no_lens_for_file_returns_empty(self, config):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        readme = next(f for f in plan.files if f.path == "README.md")
        finder = FakeProvider([])  # should never be called
        findings = run_lenses_mod.run_lenses(
            finder, readme, lens_prompts={}, prefix="P"
        )
        assert findings == []
        assert finder.calls == []

    def test_injected_enclosing_fetcher_reaches_finder_message(self, config, tmp_path):
        # Integration: an injected GitEnclosingFetcher must actually surface the
        # enclosing-context block in the message the finder provider receives.
        import subprocess

        from openrabbit.pipeline.enclosing import GitEnclosingFetcher

        repo = tmp_path / "repo"
        repo.mkdir()

        def _git(*args: str) -> None:
            subprocess.run(
                ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
            )

        _git("init", "-q")
        _git("config", "user.email", "t@example.com")
        _git("config", "user.name", "Test")
        (repo / "svc.py").write_text(
            "class Service:\n"
            "    def charge(self, amount):\n"
            "        if amount <= 0:\n"
            '            raise ValueError("bad")\n'
            "        return amount\n"
        )
        _git("add", "svc.py")
        _git("commit", "-q", "-m", "init")

        diff = (
            "diff --git a/svc.py b/svc.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/svc.py\n"
            "+++ b/svc.py\n"
            "@@ -2,3 +2,3 @@ class Service:\n"
            "     def charge(self, amount):\n"
            "-        if amount <= 0:\n"
            "+        if amount < 0:\n"
        )
        plan = route_mod.route_diff(diff, lenses=["correctness"])
        pf = next(f for f in plan.files if f.path == "svc.py")
        finder = FakeProvider([_emit_findings_result([])])
        fetcher = GitEnclosingFetcher(repo_root=repo)

        run_lenses_mod.run_lenses(
            finder,
            pf,
            lens_prompts={"correctness": "COR"},
            prefix="P",
            enclosing_fetcher=fetcher,
        )

        assert len(finder.calls) == 1
        msg = finder.calls[0].messages[0]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert "enclosing-context" in content
        assert "def charge(self, amount):" in content

    def test_default_run_lenses_has_no_enclosing_block(self, config):
        # Without an injected fetcher the finder message stays diff-only (the
        # no-op default), so offline/unit runs are unchanged.
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        finder = FakeProvider([_emit_findings_result([])])
        run_lenses_mod.run_lenses(
            finder, pf, lens_prompts={"correctness": "COR"}, prefix="P"
        )
        msg = finder.calls[0].messages[0]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert "enclosing-context" not in content


# --------------------------------------------------------------------------- #
# verify                                                                       #
# --------------------------------------------------------------------------- #
class TestVerify:
    def _finding(self, conf: float, rule: str = "openrabbit/security/sqli") -> Finding:
        fp = compute_fingerprint("src/api/auth.py", rule, "ctx")
        return Finding(
            file="src/api/auth.py",
            start_line=12,
            end_line=14,
            side="RIGHT",
            severity="high",
            category="security",
            confidence=0.9,
            title="t",
            body="b",
            rule_id=rule,
            fingerprint=fp,
        )

    def test_drops_below_gate(self, config):
        verifier = FakeProvider([_verify_result(0.5, keep=True)])
        kept = verify_mod.verify_findings(
            verifier, [self._finding(0.9)], gate=0.80
        )
        assert kept == []

    def test_keeps_above_gate(self, config):
        verifier = FakeProvider([_verify_result(0.95, keep=True)])
        kept = verify_mod.verify_findings(
            verifier, [self._finding(0.9)], gate=0.80
        )
        assert len(kept) == 1
        assert kept[0].confidence == pytest.approx(0.95)

    def test_refuted_dropped(self, config):
        verifier = FakeProvider([_verify_result(0.95, keep=False)])
        kept = verify_mod.verify_findings(
            verifier, [self._finding(0.9)], gate=0.80
        )
        assert kept == []

    def test_untrusted_fencing_in_prompt(self, config):
        verifier = FakeProvider([_verify_result(0.95, keep=True)])
        verify_mod.verify_findings(verifier, [self._finding(0.9)], gate=0.80)
        msg = verifier.calls[0].messages[0]
        body = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        assert "UNTRUSTED" in body.upper()


# --------------------------------------------------------------------------- #
# dedup                                                                        #
# --------------------------------------------------------------------------- #
class TestDedup:
    def _finding(self, rule: str, severity: str, conf: float) -> Finding:
        fp = compute_fingerprint("f.py", rule, "ctx")
        return Finding(
            file="f.py",
            start_line=1,
            end_line=1,
            side="RIGHT",
            severity=severity,
            category="correctness",
            confidence=conf,
            title="t",
            body="b",
            rule_id=rule,
            fingerprint=fp,
        )

    def test_dedup_within_batch(self):
        f1 = self._finding("r1", "high", 0.9)
        f2 = self._finding("r1", "high", 0.9)  # same fingerprint
        out = dedup_mod.dedup_and_rank([f1, f2], prior_fingerprints=set())
        assert len(out) == 1

    def test_dedup_against_prior(self):
        f1 = self._finding("r1", "high", 0.9)
        out = dedup_mod.dedup_and_rank([f1], prior_fingerprints={f1.fingerprint})
        assert out == []

    def test_ranking_severity_then_confidence(self):
        low = self._finding("r1", "low", 0.99)
        crit = self._finding("r2", "critical", 0.81)
        out = dedup_mod.dedup_and_rank([low, crit], prior_fingerprints=set())
        assert out[0].severity == "critical"


# --------------------------------------------------------------------------- #
# emit                                                                         #
# --------------------------------------------------------------------------- #
class TestEmit:
    def _finding(self) -> Finding:
        fp = compute_fingerprint("src/api/auth.py", "openrabbit/security/sqli", "ctx")
        return Finding(
            file="src/api/auth.py",
            start_line=12,
            end_line=14,
            side="RIGHT",
            severity="high",
            category="security",
            confidence=0.95,
            title="SQL injection",
            body="Concatenated user input.",
            rule_id="openrabbit/security/sqli",
            fingerprint=fp,
            suggestion="db.execute(query, [token])",
        )

    def test_console_emit_returns_payload(self):
        result = emit_mod.emit_console([self._finding()], summary_markdown="Summary")
        assert "review" in result
        review = result["review"]
        assert review["event"] == "COMMENT"
        assert review["comments"]
        assert any("SQL injection" in c["body"] for c in review["comments"])

    def test_console_emit_empty(self):
        result = emit_mod.emit_console([], summary_markdown="No issues")
        assert result["review"]["comments"] == []

    def test_render_markdown_summary(self):
        md = emit_mod.render_summary_markdown([self._finding()], stats={"files": 1})
        assert "SQL injection" in md or "security" in md.lower()


# --------------------------------------------------------------------------- #
# orchestrator (full spine end to end)                                         #
# --------------------------------------------------------------------------- #
class TestOrchestratorEndToEnd:
    def test_full_offline_review(self, config):
        # Finder will be called once per (file, lens) that routes to a lens.
        # auth.py -> {correctness, security}; README/lockfile route to nothing.
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/correctness/x", 70, "correctness")]
                ),
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/security/sqli", 90, "security")]
                ),
            ]
        )
        # Verifier called once per surviving finding. Keep both, high confidence.
        verifier = FakeProvider([_verify_result(0.95), _verify_result(0.92)])

        result = orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "repo": "acme/repo",
                "number": 7,
                "diff": SAMPLE_DIFF,
                "title": "Add auth",
                "body": "PR body",
            },
            providers={"finder": finder, "verifier": verifier},
        )
        assert result.reviewed is True
        assert len(result.findings) >= 1
        assert all(f.confidence >= 0.80 for f in result.findings)
        # emit payload present
        assert "review" in result.emitted
        # finder called twice (2 lenses on auth.py)
        assert len(finder.calls) == 2

    def test_review_threads_enclosing_fetcher_to_finder(self, config):
        # review() must forward an injected enclosing_fetcher all the way to the
        # finder message (the wiring the production CLI relies on).
        captured: list[str] = []

        def _fetcher(file_plan):
            captured.append(file_plan.path)
            return f"# context for {file_plan.path}\nENCLOSED-MARKER"

        finder = FakeProvider(
            [_emit_findings_result([]), _emit_findings_result([])]
        )
        verifier = FakeProvider([])
        result = orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "diff": SAMPLE_DIFF,
                "title": "t",
                "body": "b",
            },
            providers={"finder": finder, "verifier": verifier},
            enclosing_fetcher=_fetcher,
        )
        assert result.reviewed is True
        # The fetcher ran for the reviewable file...
        assert "src/api/auth.py" in captured
        # ...and its output is embedded in the finder message.
        content = finder.calls[0].messages[0].content
        content = content if isinstance(content, str) else str(content)
        assert "ENCLOSED-MARKER" in content
        assert "enclosing-context" in content

    def test_gate_skips_short_circuits(self, config):
        finder = FakeProvider([])
        verifier = FakeProvider([])
        result = orch_mod.review(
            config,
            pr_context={
                "draft": True,
                "state": "open",
                "head_sha": "abc",
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
        )
        assert result.reviewed is False
        assert result.findings == []
        assert finder.calls == []
        assert verifier.calls == []

    def test_verifier_drops_low_confidence(self, config):
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/correctness/x", 70, "correctness")]
                ),
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/security/sqli", 90, "security")]
                ),
            ]
        )
        verifier = FakeProvider([_verify_result(0.10), _verify_result(0.20)])
        result = orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
        )
        assert result.reviewed is True
        assert result.findings == []

    def test_default_lens_prompts_loaded_from_shipped_skills(self, config):
        """When the caller omits lens_prompts, the spine loads the packaged
        SKILL.md rubric (not the one-line stub) into the finder system prompt."""
        finder = FakeProvider(
            [_emit_findings_result([]), _emit_findings_result([])]
        )
        verifier = FakeProvider([])
        orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
            # lens_prompts intentionally omitted -> load shipped skills.
        )
        # auth.py routes to correctness + security; find the security finder call.
        systems = [c.system for c in finder.calls]
        security_systems = [s for s in systems if "LENS: security" in s]
        assert security_systems, "no security lens finder call captured"
        # Distinctive text from the shipped security SKILL.md body, not the stub.
        assert "You are the security finder." in security_systems[0]
        assert "Apply the security review rubric" not in security_systems[0]

    def test_load_packaged_lens_prompts_has_shipped_lenses(self):
        prompts = orch_mod.load_packaged_lens_prompts()
        assert "security" in prompts
        assert "correctness" in prompts
        assert "You are the security finder." in prompts["security"]

    def test_load_packaged_lens_prompts_missing_dir_returns_empty(self, tmp_path):
        # A non-existent directory must degrade to {} (stubs fill the gap),
        # never crash the review.
        assert orch_mod.load_packaged_lens_prompts(tmp_path / "nope") == {}

    def test_records_state_after_review(self, config, tmp_path):
        state_path = tmp_path / "state.json"
        store = gate_mod.StateStore(state_path)
        finder = FakeProvider(
            [
                _emit_findings_result([]),
                _emit_findings_result([]),
            ]
        )
        verifier = FakeProvider([])
        orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "sha-xyz",
                "repo": "acme/repo",
                "number": 7,
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
            store=store,
        )
        assert store.last_reviewed_sha("acme/repo", 7) == "sha-xyz"


# --------------------------------------------------------------------------- #
# model_factory                                                                #
# --------------------------------------------------------------------------- #
class TestModelFactory:
    def test_openai_prefix_builds_responses_adapter(self):
        from openrabbit.config import ModelRole
        from openrabbit.providers.openai_responses import OpenAIResponsesAdapter

        role = ModelRole(model="openai.gpt-5.5", region="us-east-2")
        provider = orch_mod.model_factory(role)
        assert isinstance(provider, OpenAIResponsesAdapter)

    def test_amazon_prefix_builds_converse_adapter(self):
        from openrabbit.config import ModelRole
        from openrabbit.providers.converse import ConverseAdapter

        role = ModelRole(model="amazon.nova-pro-v1:0", region="ap-northeast-2")
        provider = orch_mod.model_factory(role)
        assert isinstance(provider, ConverseAdapter)

    def test_anthropic_prefix_builds_converse_adapter(self):
        from openrabbit.config import ModelRole
        from openrabbit.providers.converse import ConverseAdapter

        role = ModelRole(model="anthropic.claude-x", region="us-east-1")
        provider = orch_mod.model_factory(role)
        assert isinstance(provider, ConverseAdapter)

    def test_unknown_prefix_raises(self):
        from openrabbit.config import ModelRole

        role = ModelRole(model="mystery.model", region="x")
        with pytest.raises(ValueError):
            orch_mod.model_factory(role)


# --------------------------------------------------------------------------- #
# CLI offline mode                                                             #
# --------------------------------------------------------------------------- #
class TestCliOffline:
    def test_offline_review_from_file(self, tmp_path, capsys, config):
        from openrabbit import cli

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        cfg_path = tmp_path / ".openrabbit.yaml"
        cfg_path.write_text(
            "version: 1\nreview:\n  lenses: [correctness, security]\n"
            "model_roles:\n  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n"
            "  verifier: {model: openai.gpt-5.5, region: us-east-2}\n",
            encoding="utf-8",
        )
        rc = cli.main(
            [
                "review",
                "--offline",
                "--diff",
                str(diff_path),
                "--config",
                str(cfg_path),
                "--fixtures",
                "demo",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "review" in out.lower() or "finding" in out.lower()

    def test_offline_no_creds_needed(self, tmp_path, monkeypatch, config):
        """Offline mode must not touch GITHUB_TOKEN / AWS creds."""
        from openrabbit import cli

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        rc = cli.main(["review", "--offline", "--diff", str(diff_path), "--fixtures", "demo"])
        assert rc == 0

    def test_offline_without_fixtures_emits_no_findings(self, tmp_path, capsys):
        from openrabbit import cli

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        rc = cli.main(["review", "--offline", "--diff", str(diff_path)])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reviewed"] is True
        assert payload["findings"] == []

    def test_offline_reads_stdin(self, monkeypatch, capsys):
        import io

        from openrabbit import cli

        monkeypatch.setattr("sys.stdin", io.StringIO(SAMPLE_DIFF))
        rc = cli.main(["review", "--offline", "--fixtures", "demo"])
        assert rc == 0

    def test_offline_demo_finding_present(self, tmp_path, capsys):
        from openrabbit import cli

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        rc = cli.main(
            ["review", "--offline", "--diff", str(diff_path), "--fixtures", "demo"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reviewed"] is True
        assert len(payload["findings"]) == 1
        assert payload["findings"][0]["ruleId"] == "openrabbit/security/sqli"

    def test_no_diff_raises(self, monkeypatch):
        import io

        from openrabbit import cli

        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        with pytest.raises(SystemExit):
            cli.main(["review", "--offline"])


# --------------------------------------------------------------------------- #
# CLI online mode (fake GitHub client + monkeypatched providers)              #
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _FakeGitHubClient:
    """In-process fake httpx-like client for the GitHub adapter."""

    def __init__(self, diff_text: str):
        self._diff = diff_text
        self.posts: list[tuple] = []

    def get(self, url, headers=None):
        if url.endswith("/comments"):
            return _Resp(200, [])
        # diff fetch
        return _Resp(200, {}, text=self._diff)

    def post(self, url, headers=None, json=None):
        self.posts.append((url, json))
        if "graphql" in url:
            return _Resp(200, {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}}})
        if url.endswith("/reviews"):
            return _Resp(200, {"id": 1})
        return _Resp(200, {"id": 2})

    def patch(self, url, headers=None, json=None):
        return _Resp(200, {"id": 3})

    def close(self):
        pass


class TestCliOnline:
    def test_online_requires_token(self, monkeypatch):
        from openrabbit import cli

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        rc = cli.main(["review", "--repo", "a/b", "--pr", "1"])
        assert rc == 2

    def test_online_requires_repo_and_pr(self, monkeypatch):
        from openrabbit import cli

        monkeypatch.setenv("GITHUB_TOKEN", "t")
        rc = cli.main(["review"])
        assert rc == 2

    def test_online_review_with_fakes(self, monkeypatch, capsys, tmp_path):
        from openrabbit import cli
        from openrabbit.adapters.github import GitHubAdapter, GitHubRepo

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        cfg_path = tmp_path / ".openrabbit.yaml"
        cfg_path.write_text(
            "version: 1\nreview:\n  lenses: [correctness, security]\n"
            "model_roles:\n  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n"
            "  verifier: {model: openai.gpt-5.5, region: us-east-2}\n",
            encoding="utf-8",
        )
        fake_client = _FakeGitHubClient(SAMPLE_DIFF)

        real_init = GitHubAdapter.__init__

        def patched_init(self, repo, pr_number, token, **kwargs):
            kwargs["client"] = fake_client
            real_init(self, repo, pr_number, token, **kwargs)

        monkeypatch.setattr(GitHubAdapter, "__init__", patched_init)

        # Replace real provider construction with FakeProviders so no AWS.
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/correctness/x", 70, "correctness")]
                ),
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/security/sqli", 90, "security")]
                ),
            ]
        )
        verifier = FakeProvider([_verify_result(0.9), _verify_result(0.9)])
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
        # A review POST happened.
        assert any(u.endswith("/reviews") for u, _ in fake_client.posts)

    def test_online_clean_pr_posts_no_review(self, monkeypatch, tmp_path):
        """When every finding is dropped (clean PR), no createReview POST fires —
        only the sticky walkthrough comment is upserted."""
        from openrabbit import cli
        from openrabbit.adapters.github import GitHubAdapter

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        cfg_path = tmp_path / ".openrabbit.yaml"
        cfg_path.write_text(
            "version: 1\nreview:\n  lenses: [correctness, security]\n"
            "model_roles:\n  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n"
            "  verifier: {model: openai.gpt-5.5, region: us-east-2}\n",
            encoding="utf-8",
        )
        fake_client = _FakeGitHubClient(SAMPLE_DIFF)

        real_init = GitHubAdapter.__init__

        def patched_init(self, repo, pr_number, token, **kwargs):
            kwargs["client"] = fake_client
            real_init(self, repo, pr_number, token, **kwargs)

        monkeypatch.setattr(GitHubAdapter, "__init__", patched_init)

        # Finder reports, but the verifier refutes/drops everything -> 0 findings.
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/correctness/x", 70, "correctness")]
                ),
                _emit_findings_result(
                    [_finder_finding("src/api/auth.py", "openrabbit/security/sqli", 90, "security")]
                ),
            ]
        )
        verifier = FakeProvider([_verify_result(0.10), _verify_result(0.20)])
        monkeypatch.setattr(
            "openrabbit.pipeline.orchestrator.build_providers",
            lambda cfg: {"finder": finder, "verifier": verifier},
        )

        rc = cli.main(
            [
                "review", "--repo", "acme/repo", "--pr", "7",
                "--commit", "headsha", "--config", str(cfg_path), "--post",
            ]
        )
        assert rc == 0
        # No createReview POST on a clean PR.
        assert not any(u.endswith("/reviews") for u, _ in fake_client.posts)
        # The sticky walkthrough comment IS still posted/updated (single comment).
        assert any(u.endswith("/comments") for u, _ in fake_client.posts)


# --------------------------------------------------------------------------- #
# run_lenses defensive parsing                                                 #
# --------------------------------------------------------------------------- #
class TestRunLensesParsing:
    def _file_plan(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["security"])
        return next(f for f in plan.files if f.path == "src/api/auth.py")

    def test_skips_malformed_findings(self, config):
        bad = [
            {"file": "x.py"},  # missing ruleId + title
            {"ruleId": "r"},  # missing file
            {"file": "x.py", "ruleId": "r", "title": "", "startLine": "nope"},
            {"file": "x.py", "ruleId": "r", "title": "ok", "startLine": "z"},  # bad int
            "not-a-dict",
            {"file": "x.py", "ruleId": "openrabbit/security/y", "title": "good", "confidence": 90,
             "startLine": 1, "endLine": 2, "side": "BOGUS", "severity": "??", "category": "??"},
        ]
        finder = FakeProvider([_emit_findings_result(bad)])
        prefix = ctx_mod.build_prefix(config, pr_context={})
        out = run_lenses_mod.run_lens(
            finder, self._file_plan(), "security", "SEC", prefix=prefix
        )
        # Only the last valid one survives; bad enums normalize to defaults.
        assert len(out) == 1
        assert out[0].side == "RIGHT"
        assert out[0].severity == "low"
        assert out[0].category == "correctness"

    def test_confidence_already_normalized(self, config):
        raw = [
            {"file": "x.py", "ruleId": "openrabbit/security/y", "title": "t",
             "confidence": 0.42, "startLine": 1, "endLine": 1}
        ]
        finder = FakeProvider([_emit_findings_result(raw)])
        prefix = ctx_mod.build_prefix(config, pr_context={})
        out = run_lenses_mod.run_lens(
            finder, self._file_plan(), "security", "SEC", prefix=prefix
        )
        assert out[0].confidence == pytest.approx(0.42)

    def test_confidence_non_numeric_defaults_zero(self, config):
        raw = [
            {"file": "x.py", "ruleId": "openrabbit/security/y", "title": "t",
             "confidence": "high", "startLine": 1, "endLine": 1}
        ]
        finder = FakeProvider([_emit_findings_result(raw)])
        prefix = ctx_mod.build_prefix(config, pr_context={})
        out = run_lenses_mod.run_lens(
            finder, self._file_plan(), "security", "SEC", prefix=prefix
        )
        assert out[0].confidence == 0.0

    def test_non_emit_tool_call_ignored(self, config):
        result = CompletionResult(
            text="",
            tool_calls=[ToolCall(id="x", name="other_tool", args={})],
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(),
        )
        finder = FakeProvider([result])
        prefix = ctx_mod.build_prefix(config, pr_context={})
        out = run_lenses_mod.run_lens(
            finder, self._file_plan(), "security", "SEC", prefix=prefix
        )
        assert out == []

    def test_lens_without_prompt_skipped(self, config):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        finder = FakeProvider([_emit_findings_result([])])  # only one call expected
        prefix = ctx_mod.build_prefix(config, pr_context={})
        # Only provide the correctness prompt; security is skipped.
        out = run_lenses_mod.run_lenses(
            finder, pf, {"correctness": "C"}, prefix=prefix
        )
        assert out == []
        assert len(finder.calls) == 1


# --------------------------------------------------------------------------- #
# emit_github online wiring (fake adapter)                                     #
# --------------------------------------------------------------------------- #
class _FakeAdapter:
    def __init__(self):
        self.reviews = []
        self.walkthroughs = []
        self.resolved = []
        self.minimized = []

    def post_review(self, findings, summary, commit_sha, *, event="COMMENT"):
        self.reviews.append((findings, summary, commit_sha, event))
        return {"id": 1, "event": event}

    def upsert_sticky_walkthrough(self, markdown):
        self.walkthroughs.append(markdown)
        return {"id": 2}

    def resolve_review_thread(self, thread_id):
        self.resolved.append(thread_id)
        return True

    def minimize_comment(self, comment_id, classifier="OUTDATED"):
        self.minimized.append((comment_id, classifier))
        return True


class TestEmitGithub:
    def _finding(self):
        fp = compute_fingerprint("f.py", "r1", "ctx")
        return Finding(
            file="f.py", start_line=1, end_line=1, side="RIGHT", severity="high",
            category="correctness", confidence=0.9, title="t", body="b",
            rule_id="r1", fingerprint=fp,
        )

    def test_emit_github_posts_review_and_walkthrough(self):
        adapter = _FakeAdapter()
        out = emit_mod.emit_github(
            adapter, [self._finding()], summary_markdown="S", commit_sha="sha"
        )
        assert adapter.reviews and adapter.reviews[0][3] == "COMMENT"
        assert adapter.walkthroughs == ["S"]
        assert out["review"]["id"] == 1

    def test_emit_github_skips_review_when_no_findings(self):
        """A clean PR (no findings) must NOT fire a createReview event; only the
        sticky walkthrough updates (low-noise, SPEC principle 1)."""
        adapter = _FakeAdapter()
        out = emit_mod.emit_github(
            adapter, [], summary_markdown="No issues found.", commit_sha="sha"
        )
        assert adapter.reviews == []  # no createReview POST
        assert adapter.walkthroughs == ["No issues found."]  # walkthrough still updates
        assert out["review"] is None

    def test_emit_github_resolves_stale(self):
        from openrabbit.adapters.github import ReviewThread

        adapter = _FakeAdapter()
        # A prior thread whose fingerprint is NOT among current findings -> stale.
        stale = ReviewThread(
            thread_id="T1", comment_id="C1", fingerprint="deadbeef",
            is_resolved=False, is_outdated=False, path="f.py", body="x",
        )
        emit_mod.emit_github(
            adapter, [self._finding()], summary_markdown="S", commit_sha="sha",
            prior_threads=[stale],
        )
        assert adapter.resolved == ["T1"]
        assert adapter.minimized == [("C1", "OUTDATED")]


# --------------------------------------------------------------------------- #
# emit summary rendering + payload                                             #
# --------------------------------------------------------------------------- #
class TestSummaryRendering:
    def _finding(self, title="t", severity="high"):
        fp = compute_fingerprint("f.py", "r1", title)
        return Finding(
            file="f.py", start_line=3, end_line=3, side="RIGHT", severity=severity,
            category="correctness", confidence=0.9, title=title, body="b",
            rule_id="r1", fingerprint=fp,
        )

    def test_empty_summary(self):
        md = emit_mod.render_summary_markdown([])
        assert "No issues" in md

    def test_empty_summary_with_stats(self):
        md = emit_mod.render_summary_markdown([], stats={"files": 2})
        assert "files: 2" in md

    def test_summary_escapes_pipes(self):
        md = emit_mod.render_summary_markdown([self._finding(title="a | b")], stats={"files": 1})
        assert "a \\| b" in md

    def test_build_review_payload_event(self):
        payload = emit_mod.build_review_payload([self._finding()], "S", commit_sha="c")
        assert payload["review"]["event"] == "COMMENT"
        assert payload["review"]["commit_id"] == "c"


# --------------------------------------------------------------------------- #
# gate helpers + route fallbacks                                               #
# --------------------------------------------------------------------------- #
class TestGateHelpers:
    def test_empty_diff_skipped(self, config):
        decision = gate_mod.evaluate_gate(
            config, {"draft": False, "state": "open", "head_sha": "a"}, ""
        )
        assert decision.should_review is False
        assert "empty" in decision.reason.lower()

    def test_is_ignorable_variants(self):
        assert gate_mod.is_ignorable_file("a/b/yarn.lock")
        assert gate_mod.is_ignorable_file("dist/app.js")
        assert gate_mod.is_ignorable_file("x.min.js")
        assert not gate_mod.is_ignorable_file("src/app.py")

    def test_diff_files_plus_header_fallback(self):
        plain = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n+c\n+d\n"
        assert "x.py" in gate_mod.diff_files(plain)

    def test_count_changed_lines(self):
        diff = "+++ b/x\n--- a/x\n+added\n-removed\n unchanged\n"
        assert gate_mod.count_changed_lines(diff) == 2

    def test_count_reviewable_excludes_lockfile_churn(self):
        # 2 real code lines + lots of lockfile churn.
        code = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1,2 @@\n+x = 1\n+y = 2\n"
        )
        lock = "diff --git a/package-lock.json b/package-lock.json\n" + "\n".join(
            ["--- a/package-lock.json", "+++ b/package-lock.json", "@@ -1 +1,40 @@"]
            + [f"+  line{i}" for i in range(40)]
        )
        diff = code + "\n" + lock + "\n"
        # Whole-diff count would be 42; reviewable-only count is 2.
        assert gate_mod.count_changed_lines(diff) > 40
        assert gate_mod.count_reviewable_changed_lines(diff) == 2

    def test_count_reviewable_falls_back_for_non_git_diff(self):
        plain = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+a\n+b\n+c\n"
        assert gate_mod.count_reviewable_changed_lines(plain) == 3

    def test_trivial_skip_fires_despite_lockfile_churn(self, config):
        code = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1,2 @@\n+x = 1\n+y = 2\n"
        )
        lock = "diff --git a/package-lock.json b/package-lock.json\n" + "\n".join(
            ["--- a/package-lock.json", "+++ b/package-lock.json", "@@ -1 +1,40 @@"]
            + [f"+  line{i}" for i in range(40)]
        )
        decision = gate_mod.evaluate_gate(
            config,
            {"draft": False, "state": "open", "head_sha": "a"},
            code + "\n" + lock + "\n",
            min_changed_lines=10,
        )
        # Only 2 real code lines -> still trivial despite the lockfile padding.
        assert decision.should_review is False
        assert "trivial" in decision.reason.lower()
        assert decision.changed_lines == 2

    def test_incremental_off_does_not_skip(self, tmp_path):
        cfg = load_config({"version": 1, "review": {"incremental": False, "lenses": ["correctness"]}})
        store = gate_mod.StateStore(tmp_path / "s.json")
        store.record_review("acme/repo", 7, "abc")
        decision = gate_mod.evaluate_gate(
            cfg,
            {"draft": False, "state": "open", "head_sha": "abc", "repo": "acme/repo", "number": 7},
            SAMPLE_DIFF,
            store=store,
        )
        assert decision.should_review is True


class TestRouteFallbacks:
    def test_test_file_lenses(self):
        diff = (
            "diff --git a/tests/test_x.py b/tests/test_x.py\n"
            "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
            "@@ -1 +1,3 @@\n+def test_y():\n+    assert add(1,2) == 3\n+    assert True\n"
        )
        plan = route_mod.route_diff(diff, lenses=["correctness", "security", "tests", "maintainability"])
        tf = plan.files[0]
        assert tf.file_type == "test"
        assert "security" not in tf.lenses
        assert "correctness" in tf.lenses

    def test_infra_and_frontend_and_migration(self):
        diff = (
            "diff --git a/Dockerfile b/Dockerfile\n--- a/Dockerfile\n+++ b/Dockerfile\n@@ -1 +1,2 @@\n+FROM x\n+RUN y\n"
            "diff --git a/app.tsx b/app.tsx\n--- a/app.tsx\n+++ b/app.tsx\n@@ -1 +1,2 @@\n+const a = 1;\n+const b = 2;\n"
            "diff --git a/migrations/001.py b/migrations/001.py\n--- a/migrations/001.py\n+++ b/migrations/001.py\n@@ -1 +1,2 @@\n+op.create()\n+op.drop()\n"
        )
        plan = route_mod.route_diff(diff, lenses=["correctness", "security"])
        by = {f.path: f for f in plan.files}
        assert by["Dockerfile"].file_type == "infra"
        assert by["app.tsx"].file_type == "frontend"
        assert by["migrations/001.py"].file_type == "migration"
        assert by["migrations/001.py"].risk == "high"

    def test_reviewable_files_excludes_docs(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness", "security"])
        paths = {f.path for f in plan.reviewable_files}
        assert "README.md" not in paths
        assert "package-lock.json" not in paths
        assert "src/api/auth.py" in paths


# --------------------------------------------------------------------------- #
# context enclosing-context hook                                               #
# --------------------------------------------------------------------------- #
class TestContextEnclosing:
    def test_enclosing_fetcher_injected(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        msg = ctx_mod.build_file_message(
            pf, enclosing_fetcher=lambda fp: "def login(): ..."
        )
        body = msg.content
        assert "enclosing-context" in body
        assert "def login()" in body

    def test_default_fetcher_is_noop(self):
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["correctness"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        assert ctx_mod.gather_enclosing_context(pf) is None


# --------------------------------------------------------------------------- #
# verify edge cases                                                            #
# --------------------------------------------------------------------------- #
class TestVerifyEdges:
    def _finding(self):
        fp = compute_fingerprint("src/api/auth.py", "r", "ctx")
        return Finding(
            file="src/api/auth.py", start_line=1, end_line=1, side="RIGHT",
            severity="high", category="security", confidence=0.9, title="t",
            body="b", rule_id="r", fingerprint=fp,
        )

    def test_missing_tool_call_drops(self):
        result = CompletionResult(
            text="no tool", tool_calls=[], finish_reason=FinishReason.STOP, usage=Usage()
        )
        verifier = FakeProvider([result])
        assert verify_mod.verify_findings(verifier, [self._finding()], gate=0.8) == []

    def test_non_numeric_confidence_drops(self):
        result = CompletionResult(
            text="",
            tool_calls=[ToolCall(id="v", name="verify_finding", args={"keep": True, "confidence": "x"})],
            finish_reason=FinishReason.TOOL_USE, usage=Usage(),
        )
        verifier = FakeProvider([result])
        assert verify_mod.verify_findings(verifier, [self._finding()], gate=0.8) == []

    def test_high_risk_prompt_nudge(self):
        verifier = FakeProvider([_verify_result(0.95)])
        verify_mod.verify_findings(
            verifier, [self._finding()], gate=0.8,
            high_risk_files={"src/api/auth.py"},
        )
        msg = verifier.calls[0].messages[0]
        assert "HIGH-RISK" in msg.content


# --------------------------------------------------------------------------- #
# console-script entrypoint                                                     #
# --------------------------------------------------------------------------- #
class TestConsoleScriptEntrypoint:
    """The bare ``openrabbit`` command resolves to ``openrabbit.cli:main``.

    These tests assert the wiring without spawning a subprocess (so they make
    no network calls and need no creds): they import the callable the entry
    point names and confirm pyproject points at exactly that target.
    """

    def test_main_is_callable(self):
        from openrabbit.cli import main

        assert callable(main)

    def test_pyproject_declares_console_script(self):
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redefine]

        root = Path(__file__).resolve().parent.parent
        with (root / "pyproject.toml").open("rb") as fh:
            data = tomllib.load(fh)

        scripts = data["project"]["scripts"]
        assert scripts["openrabbit"] == "openrabbit.cli:main"

    def test_entrypoint_target_runs_offline(self, tmp_path, capsys):
        """Resolve the exact ``module:attr`` the entry point names and run it."""
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redefine]
        import importlib

        root = Path(__file__).resolve().parent.parent
        with (root / "pyproject.toml").open("rb") as fh:
            target = tomllib.load(fh)["project"]["scripts"]["openrabbit"]
        module_path, _, attr = target.partition(":")
        entry = getattr(importlib.import_module(module_path), attr)

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        rc = entry(["review", "--offline", "--diff", str(diff_path), "--fixtures", "demo"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reviewed"] is True
