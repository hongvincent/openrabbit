"""Learnings / memory store + feedback loop (SPEC section 10, checklist item 7).

The learnings store is the key trust differentiator: it captures team knowledge
(team-authored learnings injected into the cacheable prefix) and a negative
signal (dismissals) that deterministically LOWERS confidence for findings that
look like ones a human already dismissed — so noisy rule/category/file patterns
fall below the gate on a re-review.

All offline: a temp JSON file (Phase-2 local stand-in for DynamoDB, mirroring
``StateStore``) + ``FakeProvider`` where a model is needed. No network, no live
AWS/GitHub credentials.
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
from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.learnings import Learning, LearningsStore
from openrabbit.pipeline import context as ctx
from openrabbit.pipeline import orchestrator as orch_mod
from openrabbit.providers.base import FakeProvider


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _finding(
    *,
    file: str = "src/api/auth.py",
    rule_id: str = "openrabbit/correctness/bounds-check",
    category: str = "correctness",
    confidence: float = 0.90,
    title: str = "Unvalidated index can raise IndexError",
) -> Finding:
    body = "Rationale."
    fp = compute_fingerprint(file, rule_id, f"{title}\n{body}")
    return Finding(
        file=file,
        start_line=42,
        end_line=47,
        side="RIGHT",
        severity="high",
        category=category,
        confidence=confidence,
        title=title,
        body=body,
        rule_id=rule_id,
        fingerprint=fp,
    )


@pytest.fixture
def store(tmp_path):
    return LearningsStore(tmp_path / "learnings.json")


# --------------------------------------------------------------------------- #
# add_learning / get_in_scope_learnings                                        #
# --------------------------------------------------------------------------- #
def test_add_learning_returns_learning_with_metadata(store):
    learning = store.add_learning(
        scope="acme/widgets",
        text="Always use parameterized SQL queries in this repo.",
        provenance={"pr": 42, "file": "src/db.py", "user": "alice"},
        category="security",
    )
    assert isinstance(learning, Learning)
    assert learning.id  # non-empty id assigned
    assert learning.text == "Always use parameterized SQL queries in this repo."
    assert learning.provenance == {"pr": 42, "file": "src/db.py", "user": "alice"}
    assert learning.category == "security"
    assert learning.created_at  # timestamp assigned
    assert learning.usage_count == 0
    assert learning.embedding is None


def test_get_in_scope_returns_repo_scoped_learning(store):
    store.add_learning(
        scope="acme/widgets",
        text="Prefer pathlib over os.path here.",
        provenance={"pr": 1, "file": "x.py", "user": "bob"},
        category="maintainability",
    )
    in_scope = store.get_in_scope_learnings("acme/widgets", ["x.py"])
    assert len(in_scope) == 1
    assert in_scope[0].text == "Prefer pathlib over os.path here."


def test_get_in_scope_empty_when_no_learnings(store):
    assert store.get_in_scope_learnings("acme/widgets", ["a.py"]) == []


# --------------------------------------------------------------------------- #
# learning text length bound (item 4)                                          #
# --------------------------------------------------------------------------- #
def test_add_learning_truncates_overlong_text(store):
    # One oversized learning must not bloat/poison the byte-stable cached prefix:
    # add_learning caps the stored text to MAX_LEARNING_TEXT_CHARS.
    from openrabbit.learnings import MAX_LEARNING_TEXT_CHARS

    huge = "x" * (MAX_LEARNING_TEXT_CHARS + 5000)
    learning = store.add_learning(
        scope="acme/widgets",
        text=huge,
        provenance={"pr": 1, "file": "x.py", "user": "u"},
        category="maintainability",
    )
    assert len(learning.text) == MAX_LEARNING_TEXT_CHARS
    # The cap is persisted, not just on the returned object.
    reloaded = store.get_in_scope_learnings("acme/widgets", ["x.py"])
    assert len(reloaded) == 1
    assert len(reloaded[0].text) == MAX_LEARNING_TEXT_CHARS


def test_add_learning_keeps_short_text_verbatim(store):
    text = "Always parameterize SQL."
    learning = store.add_learning(
        scope="acme/widgets",
        text=text,
        provenance={"pr": 1, "file": "x.py", "user": "u"},
        category="security",
    )
    assert learning.text == text


def test_max_learning_text_chars_is_sane(store):
    # The cap is a sane, bounded value (a couple thousand chars), not unbounded.
    from openrabbit.learnings import MAX_LEARNING_TEXT_CHARS

    assert 0 < MAX_LEARNING_TEXT_CHARS <= 4000


# --------------------------------------------------------------------------- #
# scope filtering: repo vs org                                                 #
# --------------------------------------------------------------------------- #
def test_scope_filtering_repo_vs_org(store):
    # An org-wide learning (scope = the org / owner) applies to every repo in it.
    store.add_learning(
        scope="acme",
        text="Org rule: log no PII.",
        provenance={"pr": 0, "file": "", "user": "secteam"},
        category="security",
    )
    # A repo-specific learning applies only to that repo.
    store.add_learning(
        scope="acme/widgets",
        text="Widgets repo: use the shared retry helper.",
        provenance={"pr": 7, "file": "net.py", "user": "carol"},
        category="maintainability",
    )

    # The widgets repo sees BOTH the org learning and its own.
    widgets = {
        ln.text for ln in store.get_in_scope_learnings("acme/widgets", ["net.py"])
    }
    assert widgets == {
        "Org rule: log no PII.",
        "Widgets repo: use the shared retry helper.",
    }

    # A different repo in the same org sees only the org learning.
    gadgets = {ln.text for ln in store.get_in_scope_learnings("acme/gadgets", ["a.py"])}
    assert gadgets == {"Org rule: log no PII."}

    # A repo in a different org sees neither.
    other = store.get_in_scope_learnings("other/thing", ["a.py"])
    assert other == []


def test_repo_only_learning_not_leaked_to_other_repos(store):
    store.add_learning(
        scope="acme/widgets",
        text="repo-only",
        provenance={"pr": 1, "file": "x.py", "user": "u"},
        category="correctness",
    )
    assert store.get_in_scope_learnings("acme/gadgets", ["x.py"]) == []


# --------------------------------------------------------------------------- #
# persistence round-trip via temp file                                        #
# --------------------------------------------------------------------------- #
def test_persistence_round_trip(tmp_path):
    path = tmp_path / "learnings.json"
    store1 = LearningsStore(path)
    created = store1.add_learning(
        scope="acme/widgets",
        text="Persisted learning.",
        provenance={"pr": 9, "file": "p.py", "user": "dan"},
        category="tests",
    )
    # A fresh store over the same file reads the persisted learning back.
    store2 = LearningsStore(path)
    reloaded = store2.get_in_scope_learnings("acme/widgets", ["p.py"])
    assert len(reloaded) == 1
    assert reloaded[0].id == created.id
    assert reloaded[0].text == "Persisted learning."
    assert reloaded[0].category == "tests"
    # On-disk JSON is well-formed and human-readable.
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)


def test_record_dismissal_round_trip(tmp_path):
    path = tmp_path / "learnings.json"
    store1 = LearningsStore(path)
    store1.record_dismissal("acme/widgets", _finding())
    # A fresh store still down-weights a repeat of the dismissed finding.
    store2 = LearningsStore(path)
    repeat = _finding(confidence=0.90)
    adjusted = store2.adjust_confidence(repeat)
    assert adjusted < 0.90


# --------------------------------------------------------------------------- #
# record_dismissal + adjust_confidence (negative signal)                       #
# --------------------------------------------------------------------------- #
def test_adjust_confidence_no_dismissals_is_identity(store):
    f = _finding(confidence=0.88)
    assert store.adjust_confidence(f) == 0.88


def test_dismissal_lowers_confidence_on_repeat_finding(store):
    f = _finding(confidence=0.90)
    store.record_dismissal("acme/widgets", f)
    repeat = _finding(confidence=0.90)
    adjusted = store.adjust_confidence(repeat)
    assert adjusted < 0.90
    assert 0.0 <= adjusted <= 1.0


def test_repeated_dismissals_compound_the_penalty(store):
    f = _finding(confidence=0.95)
    store.record_dismissal("acme/widgets", f)
    once = store.adjust_confidence(_finding(confidence=0.95))
    store.record_dismissal("acme/widgets", f)
    twice = store.adjust_confidence(_finding(confidence=0.95))
    assert twice < once < 0.95


def test_dismissal_matches_similar_finding_same_rule_and_path(store):
    # Dismiss a finding, then a NEW finding with the same rule_id + category +
    # file but different line/title still gets down-weighted (it's "similar").
    store.record_dismissal("acme/widgets", _finding(title="Original title"))
    similar = _finding(title="Totally different wording", confidence=0.90)
    assert store.adjust_confidence(similar) < 0.90


def test_dismissal_does_not_affect_unrelated_finding(store):
    store.record_dismissal("acme/widgets", _finding(rule_id="openrabbit/correctness/a"))
    unrelated = _finding(
        rule_id="openrabbit/security/b",
        category="security",
        file="src/other.py",
        confidence=0.88,
    )
    assert store.adjust_confidence(unrelated) == 0.88


# --------------------------------------------------------------------------- #
# pipeline wiring: prefix injection                                            #
# --------------------------------------------------------------------------- #
def test_build_prefix_includes_in_scope_learnings():
    config = load_config({"version": 1})
    learnings = [
        "Always validate request bodies.",
        "Never log secrets.",
    ]
    prefix = ctx.build_prefix(config, {"title": "t", "body": "b"}, learnings=learnings)
    assert "Always validate request bodies." in prefix
    assert "Never log secrets." in prefix


def test_build_prefix_without_learnings_is_unchanged():
    config = load_config({"version": 1})
    base = ctx.build_prefix(config, {"title": "t", "body": "b"})
    with_empty = ctx.build_prefix(config, {"title": "t", "body": "b"}, learnings=[])
    assert base == with_empty


def test_build_prefix_learnings_fenced_as_untrusted():
    config = load_config({"version": 1})
    prefix = ctx.build_prefix(config, {}, learnings=["Use the shared logger."])
    # Team learnings are still data, not instructions: fenced like other context.
    assert "Use the shared logger." in prefix
    assert "learnings" in prefix.lower()


# --------------------------------------------------------------------------- #
# pipeline wiring: orchestrator end-to-end (dismissal drops below gate)        #
# --------------------------------------------------------------------------- #
SAMPLE_DIFF = """\
diff --git a/src/api/auth.py b/src/api/auth.py
index 1111111..2222222 100644
--- a/src/api/auth.py
+++ b/src/api/auth.py
@@ -40,6 +40,9 @@ def handler(request):
     items = load()
-    return items[idx]
+    idx = int(request.GET["i"])
+    value = items[idx]
+    return value
"""


def _finder_emit(findings: list[dict]) -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(id="c1", name="emit_findings", args={"findings": findings})
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )


def _verify_keep(confidence: float) -> CompletionResult:
    """A batched verify result keeping a single finding (verdict id 0)."""
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v",
                name="verify_findings",
                args={
                    "verdicts": [
                        {
                            "id": 0,
                            "keep": True,
                            "confidence": confidence,
                            "rationale": "ok",
                        }
                    ]
                },
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )


@pytest.fixture
def single_lens_config():
    return load_config(
        {
            "version": 1,
            "review": {
                "profile": "balanced",
                "confidence_gate": 0.80,
                "incremental": False,
                "lenses": ["correctness"],
            },
        }
    )


def _raw_finding_dict() -> dict:
    return {
        "file": "src/api/auth.py",
        "startLine": 42,
        "endLine": 44,
        "side": "RIGHT",
        "severity": "high",
        "category": "correctness",
        "confidence": 90,
        "title": "Unvalidated index can raise IndexError",
        "body": "User-controlled index used without bounds check.",
        "ruleId": "openrabbit/correctness/bounds-check",
    }


def test_orchestrator_keeps_finding_without_dismissal(single_lens_config, tmp_path):
    finder = FakeProvider([_finder_emit([_raw_finding_dict()])], name="finder")
    verifier = FakeProvider([_verify_keep(0.85)], name="verifier")
    providers = {"finder": finder, "verifier": verifier}
    pr_context = {
        "draft": False,
        "state": "open",
        "head_sha": "abc",
        "repo": "acme/widgets",
        "number": 5,
        "diff": SAMPLE_DIFF,
    }
    learnings_store = LearningsStore(tmp_path / "learnings.json")
    result = orch_mod.review(
        single_lens_config,
        pr_context,
        providers,
        learnings_store=learnings_store,
        emit=False,
    )
    assert result.reviewed is True
    assert len(result.findings) == 1


def test_orchestrator_dismissal_drops_repeat_below_gate(single_lens_config, tmp_path):
    learnings_store = LearningsStore(tmp_path / "learnings.json")

    # The reviewer dismisses this finding (negative signal) before the re-review.
    # adjust_confidence will pull a verified 0.85 below the 0.80 gate.
    dismissed = _finding(
        file="src/api/auth.py",
        rule_id="openrabbit/correctness/bounds-check",
        category="correctness",
        confidence=0.85,
    )
    learnings_store.record_dismissal("acme/widgets", dismissed)

    finder = FakeProvider([_finder_emit([_raw_finding_dict()])], name="finder")
    verifier = FakeProvider([_verify_keep(0.85)], name="verifier")
    providers = {"finder": finder, "verifier": verifier}
    pr_context = {
        "draft": False,
        "state": "open",
        "head_sha": "def",
        "repo": "acme/widgets",
        "number": 5,
        "diff": SAMPLE_DIFF,
    }
    result = orch_mod.review(
        single_lens_config,
        pr_context,
        providers,
        learnings_store=learnings_store,
        emit=False,
    )
    assert result.reviewed is True
    # The dismissal down-weighted the repeat finding below the gate → dropped.
    assert result.findings == []


def test_orchestrator_injects_learnings_into_prefix(single_lens_config, tmp_path):
    learnings_store = LearningsStore(tmp_path / "learnings.json")
    learnings_store.add_learning(
        scope="acme/widgets",
        text="MEMORY-MARKER: always bounds-check indices.",
        provenance={"pr": 1, "file": "src/api/auth.py", "user": "lead"},
        category="correctness",
    )
    finder = FakeProvider([_finder_emit([])], name="finder")
    verifier = FakeProvider([], name="verifier")
    providers = {"finder": finder, "verifier": verifier}
    pr_context = {
        "draft": False,
        "state": "open",
        "head_sha": "ghi",
        "repo": "acme/widgets",
        "number": 5,
        "diff": SAMPLE_DIFF,
    }
    orch_mod.review(
        single_lens_config,
        pr_context,
        providers,
        learnings_store=learnings_store,
        emit=False,
    )
    # The in-scope learning reached the finder's system prompt (cacheable prefix).
    assert finder.calls, "finder should have been called"
    assert any("MEMORY-MARKER" in c.system for c in finder.calls)


def test_orchestrator_downweights_but_keeps_when_still_above_gate(tmp_path):
    # A low gate + high confidence: one dismissal down-weights the finding
    # (0.98 * 0.65 ~= 0.637) but it stays above a 0.50 gate, so it survives with
    # a *lowered* confidence (the down-weight-but-keep path).
    config = load_config(
        {
            "version": 1,
            "review": {
                "profile": "balanced",
                "confidence_gate": 0.50,
                "incremental": False,
                "lenses": ["correctness"],
            },
        }
    )
    learnings_store = LearningsStore(tmp_path / "learnings.json")
    learnings_store.record_dismissal(
        "acme/widgets",
        _finding(
            file="src/api/auth.py",
            rule_id="openrabbit/correctness/bounds-check",
            category="correctness",
        ),
    )
    finder = FakeProvider([_finder_emit([_raw_finding_dict()])], name="finder")
    verifier = FakeProvider([_verify_keep(0.98)], name="verifier")
    providers = {"finder": finder, "verifier": verifier}
    pr_context = {
        "draft": False,
        "state": "open",
        "head_sha": "jkl",
        "repo": "acme/widgets",
        "number": 5,
        "diff": SAMPLE_DIFF,
    }
    result = orch_mod.review(
        config, pr_context, providers, learnings_store=learnings_store, emit=False
    )
    assert len(result.findings) == 1
    # Confidence was lowered by the dismissal penalty but stayed above the gate.
    assert result.findings[0].confidence < 0.98
    assert result.findings[0].confidence >= 0.50


def test_orchestrator_without_learnings_store_unchanged(single_lens_config):
    finder = FakeProvider([_finder_emit([_raw_finding_dict()])], name="finder")
    verifier = FakeProvider([_verify_keep(0.85)], name="verifier")
    providers = {"finder": finder, "verifier": verifier}
    pr_context = {
        "draft": False,
        "state": "open",
        "head_sha": "abc",
        "repo": "acme/widgets",
        "number": 5,
        "diff": SAMPLE_DIFF,
    }
    # No learnings_store passed → behavior identical to before this feature.
    result = orch_mod.review(single_lens_config, pr_context, providers, emit=False)
    assert result.reviewed is True
    assert len(result.findings) == 1


# --------------------------------------------------------------------------- #
# CLI `learn` feedback-capture hook (offline, no creds)                        #
# --------------------------------------------------------------------------- #
def test_cli_learn_records_learning(tmp_path, capsys):
    from openrabbit import cli

    store_path = tmp_path / "learnings.json"
    rc = cli.main(
        [
            "learn",
            "--store",
            str(store_path),
            "--scope",
            "acme/widgets",
            "--text",
            "Always use the shared retry helper.",
            "--category",
            "maintainability",
            "--pr",
            "12",
            "--user",
            "alice",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recorded"] == "learning"
    # The learning is now in scope for the repo.
    learnings = LearningsStore(store_path).get_in_scope_learnings(
        "acme/widgets", ["x.py"]
    )
    assert any("shared retry helper" in ln.text for ln in learnings)


def test_cli_learn_records_dismissal(tmp_path, capsys):
    from openrabbit import cli

    store_path = tmp_path / "learnings.json"
    rc = cli.main(
        [
            "learn",
            "--store",
            str(store_path),
            "--dismiss",
            "--repo",
            "acme/widgets",
            "--rule-id",
            "openrabbit/correctness/bounds-check",
            "--category",
            "correctness",
            "--file",
            "src/api/auth.py",
        ]
    )
    assert rc == 0
    # The dismissal now down-weights a matching finding.
    store = LearningsStore(store_path)
    f = _finding(
        file="src/api/auth.py",
        rule_id="openrabbit/correctness/bounds-check",
        category="correctness",
        confidence=0.90,
    )
    assert store.adjust_confidence(f) < 0.90


def test_cli_learn_dismiss_missing_args_errors(tmp_path):
    from openrabbit import cli

    rc = cli.main(
        ["learn", "--store", str(tmp_path / "l.json"), "--dismiss", "--repo", "acme/w"]
    )
    assert rc == 2


def test_cli_learn_no_action_errors(tmp_path):
    from openrabbit import cli

    rc = cli.main(["learn", "--store", str(tmp_path / "l.json")])
    assert rc == 2


def test_cli_learn_text_without_scope_errors(tmp_path):
    from openrabbit import cli

    rc = cli.main(["learn", "--store", str(tmp_path / "l.json"), "--text", "hi"])
    assert rc == 2


def test_cli_review_offline_with_learnings_store_injects_prefix(tmp_path, capsys):
    """The offline review path threads --learnings-store into the spine."""
    from openrabbit import cli

    store_path = tmp_path / "learnings.json"
    LearningsStore(store_path).add_learning(
        scope="acme/widgets",
        text="CLI-MEMORY: prefer explicit bounds checks.",
        provenance={"pr": 1, "file": "src/api/auth.py", "user": "lead"},
        category="correctness",
    )
    diff_path = tmp_path / "pr.diff"
    diff_path.write_text(SAMPLE_DIFF, encoding="utf-8")
    rc = cli.main(
        [
            "review",
            "--offline",
            "--diff",
            str(diff_path),
            "--repo",
            "acme/widgets",
            "--learnings-store",
            str(store_path),
            "--fixtures",
            "demo",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reviewed"] is True
