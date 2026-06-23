"""Tests for emit_github threading valid positions + logging cleanup failures.

Offline: a tiny stub adapter records calls; no network.

Covers the github-anchor bucket emit findings:

* emit_github forwards parsed valid-position info + changed files to
  post_review so out-of-diff comments are filtered before the single POST.
* the stale-thread cleanup loop LOGS resolve/minimize failures instead of
  silently swallowing them with ``except: continue``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from openrabbit.findings import Finding
from openrabbit.pipeline import emit as emit_mod


def _finding(fp: str = "fp-1") -> Finding:
    return Finding(
        file="src/api/auth.py",
        start_line=11,
        end_line=11,
        side="RIGHT",
        severity="high",
        category="security",
        confidence=0.95,
        title="t",
        body="b",
        rule_id="openrabbit/security/x",
        fingerprint=fp,
    )


class StubAdapter:
    """Records post_review kwargs and scripts cleanup outcomes."""

    def __init__(self, *, resolve_exc: Optional[Exception] = None) -> None:
        self.post_review_kwargs: Optional[dict[str, Any]] = None
        self.resolve_exc = resolve_exc
        self.upserted = False

    def post_review(self, findings, summary, commit_sha, *, event="COMMENT", **kw):
        self.post_review_kwargs = {
            "findings": findings,
            "valid_positions": kw.get("valid_positions"),
            "changed_files": kw.get("changed_files"),
        }
        return {"id": 1}

    def upsert_sticky_walkthrough(self, markdown):
        self.upserted = True
        return {"id": 2}

    def resolve_review_thread(self, thread_id):
        if self.resolve_exc is not None:
            raise self.resolve_exc
        return True

    def minimize_comment(self, subject_id, classifier="OUTDATED"):
        return True


class _Thread:
    def __init__(self, thread_id: str, fingerprint: str) -> None:
        self.thread_id = thread_id
        self.comment_id = f"{thread_id}-c"
        self.fingerprint = fingerprint
        self.is_resolved = False
        self.is_outdated = False


def test_emit_github_forwards_valid_positions_to_post_review():
    adapter = StubAdapter()
    valid = {"src/api/auth.py": {("RIGHT", 11)}}
    emit_mod.emit_github(
        adapter,
        [_finding()],
        summary_markdown="s",
        commit_sha="sha",
        valid_positions=valid,
        changed_files={"src/api/auth.py"},
    )
    kw = adapter.post_review_kwargs
    assert kw is not None
    assert kw["valid_positions"] == valid
    assert kw["changed_files"] == {"src/api/auth.py"}


def test_emit_github_logs_cleanup_failure_instead_of_swallowing(caplog):
    """A failing resolve_review_thread must be LOGGED (not silently continued)."""
    adapter = StubAdapter(resolve_exc=RuntimeError("graphql boom"))
    stale = [_Thread("T-gone", "fp-gone")]
    # current findings do not include fp-gone -> it is stale -> cleanup attempted.
    with caplog.at_level(logging.WARNING):
        out = emit_mod.emit_github(
            adapter,
            [_finding(fp="fp-current")],
            summary_markdown="s",
            commit_sha="sha",
            prior_threads=stale,
        )
    # the failure surfaced in logs, not swallowed silently
    assert any("graphql boom" in r.getMessage() or "T-gone" in r.getMessage()
               for r in caplog.records)
    # cleanup failure does not crash emit; thread not counted as resolved
    assert "T-gone" not in out["resolved_threads"]
