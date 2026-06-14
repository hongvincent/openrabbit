"""Incremental-review fingerprint persistence (SPEC 6 step 1/5/6, 9).

These tests cover the LOCAL persisted-fingerprint dedup source added on top of
the existing GitHub-thread dedup: a finding posted once must be suppressed on a
second run with the same fingerprint, even offline / before threads load.

All offline: ``FakeProvider`` with scripted results + a temp JSON state file.
No network, no live AWS/GitHub credentials. The existing ``last_reviewed_sha``
contract must remain unchanged.
"""

from __future__ import annotations

import json

import pytest

from openrabbit.config import load_config
from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    ToolCall,
    Usage,
)
from openrabbit.pipeline import gate as gate_mod
from openrabbit.pipeline import orchestrator as orch_mod
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


def _one_security_finding():
    """A finder that emits exactly one high-confidence security finding.

    ``src/api/auth.py`` is the only reviewable file in SAMPLE_DIFF and gets the
    correctness + security lenses (2 finder calls): only the security lens
    emits, so exactly one finding survives verify.
    """
    return FakeProvider(
        [
            _emit_findings_result([]),  # correctness lens: nothing
            _emit_findings_result(
                [_finder_finding("src/api/auth.py", "openrabbit/security/sqli", 95, "security")]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# StateStore: posted-fingerprint persistence API                              #
# --------------------------------------------------------------------------- #
class TestStateStoreFingerprints:
    def test_get_posted_fingerprints_empty_by_default(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        assert store.get_posted_fingerprints("acme/repo", 7) == set()

    def test_record_and_get_posted_fingerprints_roundtrip(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_posted_fingerprints("acme/repo", 7, {"fp1", "fp2"})
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1", "fp2"}

    def test_record_posted_fingerprints_unions_across_calls(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_posted_fingerprints("acme/repo", 7, {"fp1"})
        store.record_posted_fingerprints("acme/repo", 7, {"fp2", "fp3"})
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1", "fp2", "fp3"}

    def test_record_empty_set_is_noop(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_posted_fingerprints("acme/repo", 7, set())
        assert store.get_posted_fingerprints("acme/repo", 7) == set()

    def test_fingerprints_keyed_per_repo_pr(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_posted_fingerprints("acme/repo", 7, {"fp1"})
        store.record_posted_fingerprints("acme/repo", 8, {"fp2"})
        store.record_posted_fingerprints("other/repo", 7, {"fp3"})
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1"}
        assert store.get_posted_fingerprints("acme/repo", 8) == {"fp2"}
        assert store.get_posted_fingerprints("other/repo", 7) == {"fp3"}

    def test_fingerprints_persist_across_store_instances(self, tmp_path):
        path = tmp_path / "state.json"
        store = gate_mod.StateStore(path)
        store.record_posted_fingerprints("acme/repo", 7, {"fp1", "fp2"})
        # New instance reading the same file (offline restart).
        store2 = gate_mod.StateStore(path)
        assert store2.get_posted_fingerprints("acme/repo", 7) == {"fp1", "fp2"}

    def test_corrupted_non_iterable_fingerprints_degrade_to_empty(self, tmp_path):
        # A corrupted state file whose posted_fingerprints is a non-iterable
        # (e.g. an int) must degrade to empty rather than raising.
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {"acme/repo#7": {"last_reviewed_sha": "s", "posted_fingerprints": 123}}
            ),
            encoding="utf-8",
        )
        store = gate_mod.StateStore(path)
        assert store.get_posted_fingerprints("acme/repo", 7) == set()
        # Recording onto the corrupted entry still works and yields a clean set.
        store.record_posted_fingerprints("acme/repo", 7, {"fp1"})
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1"}

    def test_corrupted_string_fingerprints_degrade_to_empty(self, tmp_path):
        # A bare string value would otherwise iterate per-character; treat any
        # non-list/tuple/set as empty.
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {"acme/repo#7": {"last_reviewed_sha": "s", "posted_fingerprints": "abc"}}
            ),
            encoding="utf-8",
        )
        store = gate_mod.StateStore(path)
        assert store.get_posted_fingerprints("acme/repo", 7) == set()


# --------------------------------------------------------------------------- #
# last_reviewed_sha contract unchanged + coexists with fingerprints           #
# --------------------------------------------------------------------------- #
class TestLastReviewedShaUnchanged:
    def test_record_review_and_last_reviewed_sha_still_work(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        assert store.last_reviewed_sha("acme/repo", 1) is None
        store.record_review("acme/repo", 1, "deadbeef")
        assert store.last_reviewed_sha("acme/repo", 1) == "deadbeef"

    def test_sha_and_fingerprints_coexist_same_key(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_review("acme/repo", 7, "sha-1")
        store.record_posted_fingerprints("acme/repo", 7, {"fp1"})
        # Each stored without clobbering the other.
        assert store.last_reviewed_sha("acme/repo", 7) == "sha-1"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1"}
        # And recording a new SHA does not wipe persisted fingerprints.
        store.record_review("acme/repo", 7, "sha-2")
        assert store.last_reviewed_sha("acme/repo", 7) == "sha-2"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1"}

    def test_record_review_with_fingerprints_single_call(self, tmp_path):
        # The combined call records SHA and unions fingerprints in one save.
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_review("acme/repo", 7, "sha-1", fingerprints={"fp1", "fp2"})
        assert store.last_reviewed_sha("acme/repo", 7) == "sha-1"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1", "fp2"}
        # A second combined call updates SHA and unions more fingerprints.
        store.record_review("acme/repo", 7, "sha-2", fingerprints={"fp3"})
        assert store.last_reviewed_sha("acme/repo", 7) == "sha-2"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1", "fp2", "fp3"}

    def test_record_review_without_fingerprints_preserves_existing(self, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_posted_fingerprints("acme/repo", 7, {"fp1"})
        store.record_review("acme/repo", 7, "sha-1")  # no fingerprints arg
        assert store.last_reviewed_sha("acme/repo", 7) == "sha-1"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1"}

    def test_legacy_bare_string_value_read_as_sha(self, tmp_path):
        """A pre-existing state file written by the old (bare-string) format
        must still be readable: ``last_reviewed_sha`` returns the string and
        fingerprints default to empty."""
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"acme/repo#7": "legacy-sha"}), encoding="utf-8")
        store = gate_mod.StateStore(path)
        assert store.last_reviewed_sha("acme/repo", 7) == "legacy-sha"
        assert store.get_posted_fingerprints("acme/repo", 7) == set()
        # Adding fingerprints migrates the value in place; SHA preserved.
        store.record_posted_fingerprints("acme/repo", 7, {"fp1"})
        assert store.last_reviewed_sha("acme/repo", 7) == "legacy-sha"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"fp1"}


# --------------------------------------------------------------------------- #
# Orchestrator wiring: persisted fingerprints suppress re-posts               #
# --------------------------------------------------------------------------- #
class TestOrchestratorPersistedDedup:
    PR = {
        "draft": False,
        "state": "open",
        "head_sha": "sha-1",
        "repo": "acme/repo",
        "number": 7,
        "diff": SAMPLE_DIFF,
    }

    def test_finding_recorded_after_first_review(self, config, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        result = orch_mod.review(
            config,
            pr_context=dict(self.PR),
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store,
        )
        assert result.reviewed is True
        assert len(result.findings) == 1
        fp = result.findings[0].fingerprint
        # The kept finding's fingerprint is persisted for next run.
        assert store.get_posted_fingerprints("acme/repo", 7) == {fp}

    def test_same_finding_suppressed_on_second_run(self, config, tmp_path):
        store = gate_mod.StateStore(tmp_path / "state.json")
        # Run 1: posts the finding, persists its fingerprint.
        first = orch_mod.review(
            config,
            pr_context=dict(self.PR),
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store,
        )
        assert len(first.findings) == 1
        fp = first.findings[0].fingerprint

        # Run 2: NEW commit (so the gate doesn't short-circuit on SHA), same
        # finding re-found. Persisted fingerprint must suppress the re-post.
        pr2 = dict(self.PR)
        pr2["head_sha"] = "sha-2"
        second = orch_mod.review(
            config,
            pr_context=pr2,
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store,
        )
        assert second.reviewed is True
        assert second.findings == []
        # Still recorded exactly once (idempotent union).
        assert store.get_posted_fingerprints("acme/repo", 7) == {fp}

    def test_persisted_unioned_with_thread_fingerprints(self, config, tmp_path):
        """A fingerprint coming ONLY from GitHub threads (prior_fingerprints)
        still dedups, AND persisted ones still dedup — both sources union."""
        store = gate_mod.StateStore(tmp_path / "state.json")
        first = orch_mod.review(
            config,
            pr_context=dict(self.PR),
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store,
        )
        fp = first.findings[0].fingerprint
        # Clear the store to simulate "threads loaded but no local persistence":
        # pass the same fp via prior_fingerprints only.
        store2 = gate_mod.StateStore(tmp_path / "fresh.json")
        pr2 = dict(self.PR)
        pr2["head_sha"] = "sha-2"
        second = orch_mod.review(
            config,
            pr_context=pr2,
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store2,
            prior_fingerprints={fp},
        )
        assert second.findings == []

    def test_new_finding_not_suppressed(self, config, tmp_path):
        """A genuinely new fingerprint (different rule) survives even when the
        store has other persisted fingerprints."""
        store = gate_mod.StateStore(tmp_path / "state.json")
        store.record_posted_fingerprints("acme/repo", 7, {"some-old-fp"})
        result = orch_mod.review(
            config,
            pr_context=dict(self.PR),
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store,
        )
        assert len(result.findings) == 1
        new_fp = result.findings[0].fingerprint
        assert new_fp != "some-old-fp"
        assert store.get_posted_fingerprints("acme/repo", 7) == {"some-old-fp", new_fp}

    def test_no_store_does_not_crash(self, config):
        """Without a store, persisted dedup is simply skipped (offline demo)."""
        result = orch_mod.review(
            config,
            pr_context={"draft": False, "state": "open", "diff": SAMPLE_DIFF},
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
        )
        assert result.reviewed is True
        assert len(result.findings) == 1

    def test_findings_persisted_when_no_head_sha(self, config, tmp_path):
        """With a store + repo/number but NO head_sha, kept fingerprints are
        still persisted (so dedup works even when the SHA is unknown)."""
        store = gate_mod.StateStore(tmp_path / "state.json")
        pr = dict(self.PR)
        pr.pop("head_sha")
        result = orch_mod.review(
            config,
            pr_context=pr,
            providers={"finder": _one_security_finding(), "verifier": FakeProvider([_verify_result(0.95)])},
            store=store,
        )
        assert result.reviewed is True
        assert len(result.findings) == 1
        fp = result.findings[0].fingerprint
        assert store.get_posted_fingerprints("acme/repo", 7) == {fp}
        # No SHA was recorded.
        assert store.last_reviewed_sha("acme/repo", 7) is None

    def test_not_reviewed_records_nothing(self, config, tmp_path):
        """A gated-out PR (draft) records no fingerprints."""
        store = gate_mod.StateStore(tmp_path / "state.json")
        pr = dict(self.PR)
        pr["draft"] = True
        result = orch_mod.review(
            config,
            pr_context=pr,
            providers={"finder": FakeProvider([]), "verifier": FakeProvider([])},
            store=store,
        )
        assert result.reviewed is False
        assert store.get_posted_fingerprints("acme/repo", 7) == set()
