"""Tests for the enriched findings summary table (emoji severity/category +
calibrated Confidence column + 정량판단 severity histogram).

The Confidence column and the severity/category emoji are UNIVERSAL triage aids
(rendered regardless of ``persona``); the ASCII severity histogram is a rich
visual flourish gated behind ``persona``. All deterministic & offline.
"""

from __future__ import annotations

from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import emit as emit_mod


def _finding(
    *, file="src/api/auth.py", severity="high", category="security", confidence=0.95
) -> Finding:
    fp = compute_fingerprint(file, "r", "t")
    return Finding(
        file=file,
        start_line=12,
        end_line=14,
        side="RIGHT",
        severity=severity,
        category=category,
        confidence=confidence,
        title="SQL injection",
        body="b",
        rule_id="openrabbit/security/sqli",
        fingerprint=fp,
    )


class TestConfidenceColumn:
    def test_header_carries_confidence_column(self):
        md = emit_mod.render_summary_markdown([_finding()])
        # 6-column header now includes Confidence between Category and File.
        assert "| Severity | Category | Confidence | File | Line | Finding |" in md
        # Separator row grew to six columns.
        assert "| --- | --- | --- | --- | --- | --- |" in md

    def test_ko_header_localizes_confidence(self):
        md = emit_mod.render_summary_markdown([_finding()], response_language="ko")
        assert "확신도" in md  # Confidence
        assert "심각도" in md and "카테고리" in md

    def test_row_has_severity_category_emoji_and_confidence_token(self):
        md = emit_mod.render_summary_markdown([_finding(confidence=0.96)])
        row = next(
            ln for ln in md.splitlines() if ln.startswith("|") and "SQL injection" in ln
        )
        assert "🟠 **High**" in row  # severity emoji + bold
        assert "🔒 Security" in row  # category gitmoji
        assert "🟢 96%" in row  # calibrated confidence token

    def test_near_gate_confidence_is_yellow(self):
        md = emit_mod.render_summary_markdown([_finding(confidence=0.84)])
        row = next(ln for ln in md.splitlines() if "SQL injection" in ln)
        assert "🟡 84%" in row

    def test_confidence_column_present_regardless_of_persona(self):
        # The Confidence column is a triage aid, not branding — it shows even
        # when persona is OFF.
        md = emit_mod.render_summary_markdown([_finding()], persona=False)
        assert "Confidence" in md
        assert "🟢" in md or "🟡" in md

    def test_file_cell_still_escapes_pipe_and_backtick(self):
        # Regression: the UNTRUSTED file path must still be neutralized.
        f = _finding(file="src/a|b`c.py")
        md = emit_mod.render_summary_markdown([f])
        assert "a|b" not in md
        assert "b`c" not in md


class TestSeverityHistogram:
    def test_histogram_present_when_persona_on_and_findings(self):
        md = emit_mod.render_summary_markdown(
            [_finding(severity="critical"), _finding(severity="high")],
            persona=True,
        )
        assert "```" in md
        # The histogram lists the severities present with their colored dots.
        assert "🔴" in md and "🟠" in md

    def test_histogram_absent_when_persona_off(self):
        md = emit_mod.render_summary_markdown(
            [_finding(severity="critical")], persona=False
        )
        # persona OFF stays lean: no fenced histogram block.
        assert "```" not in md

    def test_no_histogram_when_no_findings(self):
        md = emit_mod.render_summary_markdown([], persona=True)
        assert "```" not in md
