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
from openrabbit.pipeline import context as ctx_mod
from openrabbit.pipeline import dedup as dedup_mod
from openrabbit.pipeline import emit as emit_mod
from openrabbit.pipeline import gate as gate_mod
from openrabbit.pipeline import orchestrator as orch_mod
from openrabbit.pipeline import route as route_mod
from openrabbit.pipeline import run_lenses as run_lenses_mod
from openrabbit.pipeline import verify as verify_mod
from openrabbit.providers.base import FakeProvider

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
    """A single-verdict batch result (verifier called once for one finding)."""
    return _verify_batch_result([(0, keep, confidence)])


def _verify_batch_result(
    verdicts: list[tuple[int, bool, float]],
) -> CompletionResult:
    """A batched ``verify_findings`` tool call returning a verdict array.

    Each tuple is ``(id, keep, confidence)`` where ``id`` is the stable index
    into the verifier's batch (so verdicts map back to findings by id).
    """
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v1",
                name="verify_findings",
                args={
                    "verdicts": [
                        {
                            "id": vid,
                            "keep": keep,
                            "confidence": conf,
                            "rationale": "ok",
                        }
                        for vid, keep, conf in verdicts
                    ]
                },
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
        assert (
            "lockfile" in decision.reason.lower()
            or "generated" in decision.reason.lower()
        )

    def test_skip_trivial(self, config):
        tiny = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
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
            {
                "draft": False,
                "state": "open",
                "head_sha": "abc123",
                "repo": "acme/repo",
                "number": 7,
            },
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
        assert (
            by_path["README.md"].lenses == []
            or "security" not in by_path["README.md"].lenses
        )

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
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            70,
                            "correctness",
                        )
                    ]
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
                ["git", *args],
                cwd=str(repo),
                check=True,
                capture_output=True,
                text=True,
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
def _finding(
    conf: float = 0.9,
    *,
    severity: str = "high",
    rule: str = "openrabbit/security/sqli",
    file: str = "src/api/auth.py",
) -> Finding:
    fp = compute_fingerprint(file, rule, f"ctx-{conf}-{severity}")
    return Finding(
        file=file,
        start_line=12,
        end_line=14,
        side="RIGHT",
        severity=severity,
        category="security",
        confidence=conf,
        title="t",
        body="b",
        rule_id=rule,
        fingerprint=fp,
    )


class TestVerify:
    def test_drops_below_gate(self, config):
        verifier = FakeProvider([_verify_result(0.5, keep=True)])
        kept = verify_mod.verify_findings(verifier, [_finding(0.9)], gate=0.80)
        assert kept == []

    def test_keeps_above_gate(self, config):
        verifier = FakeProvider([_verify_result(0.95, keep=True)])
        kept = verify_mod.verify_findings(verifier, [_finding(0.9)], gate=0.80)
        assert len(kept) == 1
        assert kept[0].confidence == pytest.approx(0.95)

    def test_refuted_dropped(self, config):
        verifier = FakeProvider([_verify_result(0.95, keep=False)])
        kept = verify_mod.verify_findings(verifier, [_finding(0.9)], gate=0.80)
        assert kept == []

    def test_untrusted_fencing_in_prompt(self, config):
        verifier = FakeProvider([_verify_result(0.95, keep=True)])
        verify_mod.verify_findings(verifier, [_finding(0.9)], gate=0.80)
        msg = verifier.calls[0].messages[0]
        body = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        assert "UNTRUSTED" in body.upper()


class TestVerifyBatchingAndScoping:
    """Item 8: batching (kill the N+1) + HIGH/CRITICAL severity scoping."""

    def test_batches_n_findings_into_one_call(self, config):
        # Three HIGH findings -> exactly ONE verifier call (not three).
        findings = [
            _finding(0.9, rule="r1"),
            _finding(0.9, rule="r2"),
            _finding(0.9, rule="r3"),
        ]
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.91), (1, True, 0.92), (2, True, 0.93)])]
        )
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
        assert len(verifier.calls) == 1  # N+1 killed: ONE call for N findings
        assert len(kept) == 3

    def test_verdict_to_finding_mapping_by_id(self, config):
        # Verdicts arrive out of order and the calibrated score must land on the
        # right finding (mapped by stable id, not position).
        findings = [
            _finding(0.9, rule="r1"),
            _finding(0.9, rule="r2"),
            _finding(0.9, rule="r3"),
        ]
        verifier = FakeProvider(
            [_verify_batch_result([(2, True, 0.93), (0, True, 0.91), (1, False, 0.99)])]
        )
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
        by_rule = {f.rule_id: f for f in kept}
        assert set(by_rule) == {"r1", "r3"}  # r2 refuted (keep=False) → dropped
        assert by_rule["r1"].confidence == pytest.approx(0.91)
        assert by_rule["r3"].confidence == pytest.approx(0.93)

    def test_gate_still_drops_low_confidence_in_batch(self, config):
        findings = [_finding(0.9, rule="r1"), _finding(0.9, rule="r2")]
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.95), (1, True, 0.40)])]
        )
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
        assert [f.rule_id for f in kept] == ["r1"]  # r2 below gate → dropped

    def test_only_high_critical_verified_by_default(self, config):
        # MEDIUM/LOW/nit findings must NOT hit the expensive verifier; HIGH and
        # CRITICAL do. With one HIGH + one CRITICAL + one MEDIUM + one LOW, the
        # verifier batch carries exactly the two severe ones.
        findings = [
            _finding(0.95, severity="high", rule="r-high"),
            _finding(0.95, severity="critical", rule="r-crit"),
            _finding(0.95, severity="medium", rule="r-med"),
            _finding(0.95, severity="low", rule="r-low"),
        ]
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.90), (1, True, 0.90)])]
        )
        kept = verify_mod.verify_findings(
            verifier, findings, gate=0.80, min_severity="high"
        )
        # Exactly one verifier call carrying the two severe findings.
        assert len(verifier.calls) == 1
        # The cheaper-path (medium/low) findings still survive via the gate
        # using their finder confidence (0.95 >= 0.80), so all 4 are kept.
        assert {f.rule_id for f in kept} == {"r-high", "r-crit", "r-med", "r-low"}

    def test_cheaper_path_findings_apply_gate_without_verifier(self, config):
        # A MEDIUM finding below the gate is dropped WITHOUT a verifier call.
        findings = [_finding(0.50, severity="medium", rule="r-med")]
        verifier = FakeProvider([])  # must never be called
        kept = verify_mod.verify_findings(
            verifier, findings, gate=0.80, min_severity="high"
        )
        assert kept == []
        assert verifier.calls == []

    def test_no_verifier_call_when_nothing_severe(self, config):
        # All findings below the severity threshold -> ZERO verifier calls.
        findings = [
            _finding(0.95, severity="medium", rule="r1"),
            _finding(0.95, severity="low", rule="r2"),
        ]
        verifier = FakeProvider([])
        kept = verify_mod.verify_findings(
            verifier, findings, gate=0.80, min_severity="high"
        )
        assert verifier.calls == []
        assert {f.rule_id for f in kept} == {"r1", "r2"}

    def test_config_can_widen_scope_to_medium(self, config):
        # Widening min_severity=medium routes MEDIUM through the verifier too.
        findings = [
            _finding(0.95, severity="high", rule="r-high"),
            _finding(0.95, severity="medium", rule="r-med"),
            _finding(0.95, severity="low", rule="r-low"),
        ]
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.90), (1, True, 0.90)])]
        )
        kept = verify_mod.verify_findings(
            verifier, findings, gate=0.80, min_severity="medium"
        )
        assert len(verifier.calls) == 1
        # batch carried 2 (high + medium); low took the cheaper path and survived
        assert {f.rule_id for f in kept} == {"r-high", "r-med", "r-low"}

    def test_empty_findings_no_call(self, config):
        verifier = FakeProvider([])
        assert verify_mod.verify_findings(verifier, [], gate=0.80) == []
        assert verifier.calls == []

    def test_missing_verdict_for_id_drops_that_finding(self, config):
        # If the verifier omits a verdict for a finding's id, that finding is
        # dropped (find-broad/filter-strict: no verdict == not surfaced).
        findings = [_finding(0.9, rule="r1"), _finding(0.9, rule="r2")]
        verifier = FakeProvider([_verify_batch_result([(0, True, 0.95)])])
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
        assert [f.rule_id for f in kept] == ["r1"]


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
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            70,
                            "correctness",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
            ]
        )
        # Verifier called ONCE for the whole batch (N+1 killed). Keep both.
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.95), (1, True, 0.92)])]
        )

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

        finder = FakeProvider([_emit_findings_result([]), _emit_findings_result([])])
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
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            70,
                            "correctness",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
            ]
        )
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.10), (1, True, 0.20)])]
        )
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
        finder = FakeProvider([_emit_findings_result([]), _emit_findings_result([])])
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
# context cache key (byte-stable per-PR marker)                                #
# --------------------------------------------------------------------------- #
class TestCacheKey:
    def test_byte_stable_across_files_in_a_pr(self, config):
        # The cache key derived from the same prefix + PR identity must be
        # identical regardless of which file is being reviewed (the whole point
        # of prompt caching: cache the shared prefix once per PR).
        pr = {"repo": "acme/repo", "number": 7, "title": "T", "body": "B"}
        prefix = ctx_mod.build_prefix(config, pr)
        k1 = ctx_mod.build_cache_key(prefix, pr)
        k2 = ctx_mod.build_cache_key(prefix, pr)
        assert k1 == k2
        assert k1.startswith("openrabbit-")

    def test_distinct_across_prs(self, config):
        prefix = ctx_mod.build_prefix(config, {"title": "T"})
        k1 = ctx_mod.build_cache_key(prefix, {"repo": "acme/repo", "number": 7})
        k2 = ctx_mod.build_cache_key(prefix, {"repo": "acme/repo", "number": 8})
        assert k1 != k2

    def test_changes_when_prefix_changes(self, config):
        pr = {"repo": "acme/repo", "number": 7}
        a = ctx_mod.build_cache_key(ctx_mod.build_prefix(config, pr), pr)
        b = ctx_mod.build_cache_key("DIFFERENT-PREFIX", pr)
        assert a != b

    def test_falls_back_to_head_sha_without_repo(self, config):
        prefix = ctx_mod.build_prefix(config, {})
        k = ctx_mod.build_cache_key(prefix, {"head_sha": "deadbeef"})
        assert k.startswith("openrabbit-")
        # Same prefix + same sha -> same key.
        assert k == ctx_mod.build_cache_key(prefix, {"head_sha": "deadbeef"})


# --------------------------------------------------------------------------- #
# prompt-cache plumbing through the spine                                       #
# --------------------------------------------------------------------------- #
class TestCachePlumbing:
    def _two_file_diff(self) -> str:
        return (
            "diff --git a/src/api/auth.py b/src/api/auth.py\n"
            "--- a/src/api/auth.py\n+++ b/src/api/auth.py\n"
            "@@ -1 +1,2 @@\n+x = 1\n+y = 2\n"
            "diff --git a/src/api/billing.py b/src/api/billing.py\n"
            "--- a/src/api/billing.py\n+++ b/src/api/billing.py\n"
            "@@ -1 +1,2 @@\n+a = 1\n+b = 2\n"
        )

    def test_same_cache_prefix_for_two_files_of_same_pr(self, config):
        # Two reviewable files, both routed to correctness+security -> 4 finder
        # calls; the cache_prefix passed to the provider must be IDENTICAL across
        # all of them (one cacheable prefix per PR).
        finder = FakeProvider([_emit_findings_result([]) for _ in range(8)])
        verifier = FakeProvider([])
        orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "repo": "acme/repo",
                "number": 7,
                "diff": self._two_file_diff(),
            },
            providers={"finder": finder, "verifier": verifier},
        )
        keys = {c.cache_prefix for c in finder.calls}
        assert len(finder.calls) >= 2
        assert len(keys) == 1  # one byte-stable key for the whole PR
        assert next(iter(keys)) is not None  # caching actually activated

    def test_cache_prefix_set_when_prefix_given(self, config):
        finder = FakeProvider([_emit_findings_result([]), _emit_findings_result([])])
        verifier = FakeProvider([])
        orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "repo": "acme/repo",
                "number": 7,
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
        )
        assert all(c.cache_prefix for c in finder.calls)
        expected = ctx_mod.build_cache_key(
            ctx_mod.build_prefix(config, {"repo": "acme/repo", "number": 7}),
            {"repo": "acme/repo", "number": 7},
        )
        assert finder.calls[0].cache_prefix == expected


# --------------------------------------------------------------------------- #
# per-PR cost telemetry (Usage aggregation + CostSummary)                       #
# --------------------------------------------------------------------------- #
class TestCostTelemetry:
    def test_usage_summed_across_all_model_calls(self, config):
        from openrabbit.domain import Usage

        # 2 finder calls (100/50 each) + 1 verifier batch call (80/20).
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            90,
                            "correctness",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
            ]
        )
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.95), (1, True, 0.92)])]
        )
        result = orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "repo": "acme/repo",
                "number": 7,
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
        )
        # finder: 2*(100 in, 50 out); verifier: 1*(80 in, 20 out).
        assert result.usage == Usage(input_tokens=280, output_tokens=120)
        cost = result.cost_summary
        assert cost is not None
        assert cost.input_tokens == 280
        assert cost.output_tokens == 120
        # 3 model calls total (2 finder + 1 verifier).
        assert cost.calls == 3

    def test_cost_summary_zero_when_no_model_calls(self, config):
        # A gate skip means zero model calls -> a zeroed cost summary.
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
        assert result.cost_summary.input_tokens == 0
        assert result.cost_summary.calls == 0

    def test_cost_summary_has_dollar_estimate_for_priced_finder(self, config):
        # The finder model (amazon.nova-pro) is priced, so a $ estimate appears.
        finder = FakeProvider([_emit_findings_result([]), _emit_findings_result([])])
        verifier = FakeProvider([])
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
        # Finder is amazon.nova-pro-v1:0 in the config -> priced.
        assert result.cost_summary.usd_estimate is not None

    def test_cost_priced_per_role_at_each_models_own_rate(self, config):
        # Item 1: each role's Usage is priced at ITS OWN model rate, then summed.
        # The verifier (gpt-5.5) is pricier than the finder (nova-pro); pricing
        # the verifier's tokens at the finder rate would understate the total.
        from openrabbit import pricing
        from openrabbit.domain import Usage

        # 2 finder lens calls @ (100 in, 50 out) each = (200, 100).
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            90,
                            "correctness",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
            ]
        )
        # 1 verifier batch call @ (80 in, 20 out).
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.95), (1, True, 0.92)])]
        )
        result = orch_mod.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "repo": "acme/repo",
                "number": 7,
                "diff": SAMPLE_DIFF,
            },
            providers={"finder": finder, "verifier": verifier},
        )

        finder_usage = Usage(input_tokens=200, output_tokens=100)
        verifier_usage = Usage(input_tokens=80, output_tokens=20)
        expected = pricing.estimate_cost_for_model(
            finder_usage, "amazon.nova-pro-v1:0"
        ) + pricing.estimate_cost_for_model(verifier_usage, "openai.gpt-5.5")

        assert result.cost_summary.usd_estimate == pytest.approx(expected)

        # A naive single-rate (everything at the finder rate) would be cheaper —
        # prove the per-role split actually adds the more-expensive verifier rate.
        naive = pricing.estimate_cost_for_model(
            finder_usage + verifier_usage, "amazon.nova-pro-v1:0"
        )
        assert result.cost_summary.usd_estimate > naive


# --------------------------------------------------------------------------- #
# usage-recording provider wrapper                                             #
# --------------------------------------------------------------------------- #
class TestUsageRecordingProvider:
    def test_passes_through_identity_and_accumulates_usage(self):
        from openrabbit.domain import Usage

        inner = FakeProvider(
            [
                _emit_findings_result([]),  # Usage(input=100, output=50)
                _emit_findings_result([]),
            ],
            name="nova",
            model="amazon.nova-pro-v1:0",
        )
        wrapper = orch_mod._UsageRecordingProvider(inner)
        # Identity passes through.
        assert wrapper.name == "nova"
        assert wrapper.model == "amazon.nova-pro-v1:0"
        # Forwards complete() and accumulates Usage + a call count.
        plan = route_mod.route_diff(SAMPLE_DIFF, lenses=["security"])
        pf = next(f for f in plan.files if f.path == "src/api/auth.py")
        run_lenses_mod.run_lens(wrapper, pf, "security", "SEC", prefix="P")
        run_lenses_mod.run_lens(wrapper, pf, "security", "SEC", prefix="P")
        assert wrapper.call_count == 2
        assert wrapper.total_usage == Usage(input_tokens=200, output_tokens=100)


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
        rc = cli.main(
            ["review", "--offline", "--diff", str(diff_path), "--fixtures", "demo"]
        )
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

    def test_offline_surfaces_soft_model_role_warning(self, tmp_path, capsys):
        # Item 2: a config with a SOFT model_roles warning (off-allow-list Nova
        # region) must print a warning to stderr on `review` without failing.
        from openrabbit import cli

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        cfg_path = tmp_path / ".openrabbit.yaml"
        cfg_path.write_text(
            "version: 1\nreview:\n  lenses: [correctness, security]\n"
            "model_roles:\n"
            "  finder: {model: amazon.nova-pro-v1:0, region: eu-west-3}\n",
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
        # Soft warning does NOT fail the command.
        assert rc == 0
        err = capsys.readouterr().err
        assert "warning" in err.lower()
        assert "eu-west-3" in err
        assert "finder" in err

    def test_offline_clean_config_emits_no_warning(self, tmp_path, capsys):
        # A clean config (in-allow-list regions) prints no model-role warning.
        from openrabbit import cli

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        cfg_path = tmp_path / ".openrabbit.yaml"
        cfg_path.write_text(
            "version: 1\nreview:\n  lenses: [correctness, security]\n"
            "model_roles:\n"
            "  finder: {model: amazon.nova-pro-v1:0, region: ap-northeast-2}\n",
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
        err = capsys.readouterr().err
        assert "model_roles" not in err

    def test_offline_output_includes_cost_summary(self, tmp_path, capsys):
        # The per-PR cost telemetry (SPEC 7.3) is surfaced in the JSON output
        # and logged to stderr.
        from openrabbit import cli

        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
        rc = cli.main(
            ["review", "--offline", "--diff", str(diff_path), "--fixtures", "demo"]
        )
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        cost = payload["cost"]
        # The cost block carries the four token totals + a call count.
        for key in ("inputTokens", "outputTokens", "cacheRead", "cacheWrite", "calls"):
            assert key in cost
        assert "usdEstimate" in cost
        # The cost line is logged to stderr (CI visibility).
        assert "openrabbit cost:" in captured.err


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

        # Replace real provider construction with FakeProviders so no AWS.
        finder = FakeProvider(
            [
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            70,
                            "correctness",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
            ]
        )
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.9), (1, True, 0.9)])]
        )
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
        # The enriched walkthrough actually ships to GitHub via the sticky
        # comment: its body must carry the grouped changed-files table (not just
        # the minimal summary), mirroring the offline orchestrator assertion so a
        # regression dropping walkthrough_markdown on the CLI path is caught.
        comment_bodies = [
            (json or {}).get("body", "")
            for u, json in fake_client.posts
            if u.endswith("/comments")
        ]
        assert any("## Walkthrough" in b for b in comment_bodies)
        assert any("### Changed files" in b for b in comment_bodies)
        # Parity with the offline orchestrator: the shipped walkthrough carries
        # the stats footer (and the count is labeled "reviewable files" so it
        # never reads as contradicting the all-files changed-files table).
        assert any("reviewable files:" in b for b in comment_bodies)

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
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/correctness/x",
                            70,
                            "correctness",
                        )
                    ]
                ),
                _emit_findings_result(
                    [
                        _finder_finding(
                            "src/api/auth.py",
                            "openrabbit/security/sqli",
                            90,
                            "security",
                        )
                    ]
                ),
            ]
        )
        verifier = FakeProvider(
            [_verify_batch_result([(0, True, 0.10), (1, True, 0.20)])]
        )
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
            {
                "file": "x.py",
                "ruleId": "openrabbit/security/y",
                "title": "good",
                "confidence": 90,
                "startLine": 1,
                "endLine": 2,
                "side": "BOGUS",
                "severity": "??",
                "category": "??",
            },
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
            {
                "file": "x.py",
                "ruleId": "openrabbit/security/y",
                "title": "t",
                "confidence": 0.42,
                "startLine": 1,
                "endLine": 1,
            }
        ]
        finder = FakeProvider([_emit_findings_result(raw)])
        prefix = ctx_mod.build_prefix(config, pr_context={})
        out = run_lenses_mod.run_lens(
            finder, self._file_plan(), "security", "SEC", prefix=prefix
        )
        assert out[0].confidence == pytest.approx(0.42)

    def test_confidence_non_numeric_defaults_zero(self, config):
        raw = [
            {
                "file": "x.py",
                "ruleId": "openrabbit/security/y",
                "title": "t",
                "confidence": "high",
                "startLine": 1,
                "endLine": 1,
            }
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
        out = run_lenses_mod.run_lenses(finder, pf, {"correctness": "C"}, prefix=prefix)
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
            file="f.py",
            start_line=1,
            end_line=1,
            side="RIGHT",
            severity="high",
            category="correctness",
            confidence=0.9,
            title="t",
            body="b",
            rule_id="r1",
            fingerprint=fp,
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
            thread_id="T1",
            comment_id="C1",
            fingerprint="deadbeef",
            is_resolved=False,
            is_outdated=False,
            path="f.py",
            body="x",
        )
        emit_mod.emit_github(
            adapter,
            [self._finding()],
            summary_markdown="S",
            commit_sha="sha",
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
            file="f.py",
            start_line=3,
            end_line=3,
            side="RIGHT",
            severity=severity,
            category="correctness",
            confidence=0.9,
            title=title,
            body="b",
            rule_id="r1",
            fingerprint=fp,
        )

    def test_empty_summary(self):
        md = emit_mod.render_summary_markdown([])
        assert "No issues" in md

    def test_empty_summary_with_stats(self):
        md = emit_mod.render_summary_markdown([], stats={"files": 2})
        assert "files: 2" in md

    def test_summary_escapes_pipes(self):
        md = emit_mod.render_summary_markdown(
            [self._finding(title="a | b")], stats={"files": 1}
        )
        assert "a \\| b" in md

    def test_summary_escapes_pipe_and_backtick_in_file_cell(self):
        # f.file derives from the UNTRUSTED diff path; a pipe would break the
        # table row and a backtick would close the code span. Both must be
        # neutralized so a hostile path can't corrupt the rendered table.
        f = self._finding()
        f = Finding(
            file="src/a|b`c.py",
            start_line=f.start_line,
            end_line=f.end_line,
            side=f.side,
            severity=f.severity,
            category=f.category,
            confidence=f.confidence,
            title=f.title,
            body=f.body,
            rule_id=f.rule_id,
            fingerprint=f.fingerprint,
        )
        md = emit_mod.render_summary_markdown([f], stats={"files": 1})
        # Raw pipe/backtick must not survive verbatim in the file cell.
        assert "a|b" not in md
        assert "b`c" not in md
        assert "a\\|b" in md

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
        cfg = load_config(
            {"version": 1, "review": {"incremental": False, "lenses": ["correctness"]}}
        )
        store = gate_mod.StateStore(tmp_path / "s.json")
        store.record_review("acme/repo", 7, "abc")
        decision = gate_mod.evaluate_gate(
            cfg,
            {
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "repo": "acme/repo",
                "number": 7,
            },
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
        plan = route_mod.route_diff(
            diff, lenses=["correctness", "security", "tests", "maintainability"]
        )
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
        return _finding(0.9, rule="r")

    def test_missing_tool_call_falls_back_not_silent_drop(self):
        # No verify_findings tool call at all (e.g. a refusal) is NOT a "verified
        # and dropped" signal — silently zeroing every HIGH/CRITICAL candidate on
        # one refusal is a catastrophic recall failure. Fail SAFE: fall back to
        # the finder's own confidence through the gate. The finding here is 0.9 >=
        # gate 0.8, so it must SURVIVE rather than vanish.
        result = CompletionResult(
            text="no tool",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            usage=Usage(),
        )
        verifier = FakeProvider([result])
        kept = verify_mod.verify_findings(verifier, [self._finding()], gate=0.8)
        assert [f.rule_id for f in kept] == ["r"]

    def test_missing_tool_call_fallback_still_applies_gate(self):
        # The fail-safe fallback is to finder confidence THROUGH the gate, not an
        # unconditional keep: a below-gate finder confidence still drops.
        result = CompletionResult(
            text="no tool",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
            usage=Usage(),
        )
        verifier = FakeProvider([result])
        low = _finding(0.50, rule="low")
        assert verify_mod.verify_findings(verifier, [low], gate=0.8) == []

    def test_non_numeric_confidence_drops(self):
        # A verdict whose confidence is non-numeric drops that finding.
        result = CompletionResult(
            text="",
            tool_calls=[
                ToolCall(
                    id="v",
                    name="verify_findings",
                    args={"verdicts": [{"id": 0, "keep": True, "confidence": "x"}]},
                )
            ],
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(),
        )
        verifier = FakeProvider([result])
        assert verify_mod.verify_findings(verifier, [self._finding()], gate=0.8) == []

    def test_high_risk_prompt_nudge(self):
        verifier = FakeProvider([_verify_result(0.95)])
        verify_mod.verify_findings(
            verifier,
            [self._finding()],
            gate=0.8,
            high_risk_files={"src/api/auth.py"},
        )
        msg = verifier.calls[0].messages[0]
        assert "HIGH-RISK" in msg.content

    def _batch_tool(self, verdicts) -> CompletionResult:
        return CompletionResult(
            text="",
            tool_calls=[
                ToolCall(id="v", name="verify_findings", args={"verdicts": verdicts})
            ],
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(),
        )

    def test_verdicts_not_a_list_falls_back_not_silent_drop(self):
        # Malformed model output: 'verdicts' isn't an array -> unparseable, NOT a
        # real "drop these" answer. Same fail-safe as a missing tool call: fall
        # back to finder confidence through the gate rather than zero everything.
        result = CompletionResult(
            text="",
            tool_calls=[
                ToolCall(id="v", name="verify_findings", args={"verdicts": "nope"})
            ],
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(),
        )
        verifier = FakeProvider([result])
        kept = verify_mod.verify_findings(verifier, [self._finding()], gate=0.8)
        assert [f.rule_id for f in kept] == ["r"]

    def test_non_dict_verdict_item_skipped(self):
        verifier = FakeProvider([self._batch_tool(["not-a-dict"])])
        assert verify_mod.verify_findings(verifier, [self._finding()], gate=0.8) == []

    def test_verdict_missing_keep_skipped(self):
        verifier = FakeProvider([self._batch_tool([{"id": 0, "confidence": 0.95}])])
        assert verify_mod.verify_findings(verifier, [self._finding()], gate=0.8) == []

    def test_verdict_bad_id_skipped(self):
        verifier = FakeProvider(
            [self._batch_tool([{"id": "x", "keep": True, "confidence": 0.95}])]
        )
        assert verify_mod.verify_findings(verifier, [self._finding()], gate=0.8) == []

    def test_max_tokens_override_passed_through(self):
        verifier = FakeProvider([_verify_result(0.95)])
        verify_mod.verify_findings(
            verifier, [self._finding()], gate=0.8, max_tokens=999
        )
        assert verifier.calls[0].max_tokens == 999


class TestVerifySchemaStrictness:
    """Item 3: the verifier tool schemas reject extra props, matching the
    findings/judge contracts (additionalProperties: false)."""

    def test_verify_schema_rejects_additional_properties(self):
        assert verify_mod._VERIFY_SCHEMA.get("additionalProperties") is False

    def test_verdict_schema_rejects_additional_properties(self):
        assert verify_mod._VERDICT_SCHEMA.get("additionalProperties") is False

    def test_verify_schema_validates_against_jsonschema(self):
        # A well-formed verdict batch passes; an extra top-level / verdict prop
        # is rejected by a real validator (proves the constraint is enforceable).
        import jsonschema

        validator = jsonschema.Draft202012Validator(verify_mod._VERIFY_SCHEMA)
        # Under OpenAI strict mode every property (incl. the now-nullable
        # `rationale`) is required, so a well-formed verdict carries it.
        good = {
            "verdicts": [
                {"id": 0, "keep": True, "confidence": 0.9, "rationale": "ok"}
            ]
        }
        assert list(validator.iter_errors(good)) == []

        # `rationale` is nullable: an explicit null still validates.
        good_null = {
            "verdicts": [
                {"id": 0, "keep": True, "confidence": 0.9, "rationale": None}
            ]
        }
        assert list(validator.iter_errors(good_null)) == []

        extra_top = {
            "verdicts": [
                {"id": 0, "keep": True, "confidence": 0.9, "rationale": "ok"}
            ],
            "surprise": 1,
        }
        assert list(validator.iter_errors(extra_top))

        extra_verdict = {
            "verdicts": [
                {
                    "id": 0,
                    "keep": True,
                    "confidence": 0.9,
                    "rationale": "ok",
                    "injected": "x",
                }
            ]
        }
        assert list(validator.iter_errors(extra_verdict))


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
        rc = entry(
            ["review", "--offline", "--diff", str(diff_path), "--fixtures", "demo"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reviewed"] is True
