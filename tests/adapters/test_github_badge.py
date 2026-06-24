"""Tests for the enriched inline review-comment badge.

The inline badge now carries the same at-a-glance visual encoding as the
findings table: an emoji severity, an emoji (gitmoji) category, and a calibrated
confidence token — borrowing the legacy GPT-5.5 reviewer's per-finding numeric
confidence idea. Deterministic & offline.
"""

from __future__ import annotations

from openrabbit.adapters.github import build_review_comment
from openrabbit.findings import Finding, compute_fingerprint


def _finding(*, severity="critical", category="security", confidence=0.96) -> Finding:
    fp = compute_fingerprint("src/auth.py", "r", "t")
    return Finding(
        file="src/auth.py",
        start_line=42,
        end_line=42,
        side="RIGHT",
        severity=severity,
        category=category,
        confidence=confidence,
        title="Hardcoded AWS secret key",
        body="Rationale.",
        rule_id="openrabbit/security/secret",
        fingerprint=fp,
    )


class TestInlineBadge:
    def test_badge_has_emoji_severity_and_category(self):
        comment = build_review_comment(_finding())
        first = comment["body"].splitlines()[0]
        # Bold-bracketed emoji severity + emoji category, slash-separated.
        assert "🔴 CRITICAL" in first
        assert "🔒 security" in first
        assert first.startswith("**[")

    def test_badge_carries_confidence_token(self):
        comment = build_review_comment(_finding(confidence=0.96))
        first = comment["body"].splitlines()[0]
        assert "🟢 96%" in first

    def test_near_gate_confidence_is_yellow(self):
        comment = build_review_comment(_finding(confidence=0.83))
        first = comment["body"].splitlines()[0]
        assert "🟡 83%" in first

    def test_badge_still_contains_title_and_body(self):
        comment = build_review_comment(_finding())
        body = comment["body"]
        assert "Hardcoded AWS secret key" in body
        assert "Rationale." in body

    def test_low_severity_uses_its_own_emoji(self):
        comment = build_review_comment(_finding(severity="nit", category="tests"))
        first = comment["body"].splitlines()[0]
        assert "⚪ NIT" in first
        assert "✅ tests" in first
