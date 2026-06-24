"""Tests for the shared visual-render helpers (severity/category emoji,
confidence token + bars, severity histogram).

These primitives are the single source of truth for the at-a-glance visual
encoding used by both the findings summary table (``emit``) and the inline
review badge (``adapters.github``), plus the walkthrough's quantitative
``정량판단`` ASCII histogram. Every primitive must render IDENTICALLY on
github.com (color via emoji only; ASCII bars only inside fenced code blocks).

Fully offline & deterministic: no network, no model calls.
"""

from __future__ import annotations

from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import render


def _finding(*, severity="high", category="security", confidence=0.95) -> Finding:
    fp = compute_fingerprint("f.py", "r", "t")
    return Finding(
        file="f.py",
        start_line=1,
        end_line=1,
        side="RIGHT",
        severity=severity,
        category=category,
        confidence=confidence,
        title="t",
        body="b",
        rule_id="r",
        fingerprint=fp,
    )


# --------------------------------------------------------------------------- #
# severity display (table) — emoji + bold for the top-3 severities             #
# --------------------------------------------------------------------------- #
class TestSeverityDisplay:
    def test_each_severity_has_distinct_emoji(self):
        out = {s: render.severity_display(s) for s in render.SEVERITIES}
        emojis = [v.split()[0] for v in out.values()]
        # All five severities map to a DISTINCT colored marker (at-a-glance).
        assert len(set(emojis)) == len(render.SEVERITIES)

    def test_top_three_are_bold_rest_are_not(self):
        assert render.severity_display("critical") == "🔴 **Critical**"
        assert render.severity_display("high") == "🟠 **High**"
        assert render.severity_display("medium") == "🟡 **Medium**"
        # low / nit carry the emoji but NOT bold (weight reserved for the top-3).
        assert render.severity_display("low") == "🟢 Low"
        assert render.severity_display("nit") == "⚪ Nit"

    def test_unknown_severity_degrades_safely(self):
        # An out-of-vocabulary value must never crash; it passes through plain.
        assert render.severity_display("weird") == "weird"


# --------------------------------------------------------------------------- #
# severity inline (badge) — emoji + UPPER word, no bold (bold is added by the   #
# badge's own ``**[...]**`` wrapper)                                            #
# --------------------------------------------------------------------------- #
class TestSeverityInline:
    def test_inline_is_emoji_plus_upper(self):
        assert render.severity_inline("critical") == "🔴 CRITICAL"
        assert render.severity_inline("nit") == "⚪ NIT"

    def test_unknown_inline_degrades_safely(self):
        assert render.severity_inline("weird") == "WEIRD"


# --------------------------------------------------------------------------- #
# category display — canonical gitmoji                                          #
# --------------------------------------------------------------------------- #
class TestCategoryDisplay:
    def test_canonical_gitmoji_mapping(self):
        # Canonical gitmoji: lock (NOT shield), zap (NOT horse), bug, ✅, ♻️.
        assert render.category_display("correctness") == "🐛 Correctness"
        assert render.category_display("security") == "🔒 Security"
        assert render.category_display("performance") == "⚡ Performance"
        assert render.category_display("tests") == "✅ Tests"
        assert render.category_display("maintainability") == "♻️ Maintainability"

    def test_inline_category_keeps_lowercase(self):
        # The inline badge historically renders category lowercase
        # (``[CRITICAL/correctness]``); the emoji is prepended, case preserved.
        assert render.category_inline("security") == "🔒 security"

    def test_unknown_category_degrades_safely(self):
        assert render.category_display("weird") == "weird"
        assert render.category_inline("weird") == "weird"


# --------------------------------------------------------------------------- #
# confidence token — colored dot + literal percent                             #
# --------------------------------------------------------------------------- #
class TestConfidenceToken:
    def test_high_bucket_is_green(self):
        assert render.confidence_token(0.96) == "🟢 96%"
        assert render.confidence_token(0.90) == "🟢 90%"
        assert render.confidence_token(1.0) == "🟢 100%"

    def test_near_gate_bucket_is_yellow(self):
        # 0.80 <= conf < 0.90 → yellow (just above the 0.80 gate; scrutinize).
        assert render.confidence_token(0.89) == "🟡 89%"
        assert render.confidence_token(0.80) == "🟡 80%"

    def test_always_prints_the_literal_number(self):
        # The number is the calibration signal users came for — never hide it.
        assert "84%" in render.confidence_token(0.84)


# --------------------------------------------------------------------------- #
# bars (monospace-only; for the fenced ASCII scorecard/histogram)              #
# --------------------------------------------------------------------------- #
class TestBars:
    def test_confidence_bar_is_proportional_and_fixed_width(self):
        assert render.confidence_bar(1.0) == "██████████"
        assert render.confidence_bar(0.0) == "░░░░░░░░░░"
        # Every bar is the same width (alignment in a code block).
        assert len(render.confidence_bar(0.84)) == 10
        # Partial fill is monotonic.
        assert render.confidence_bar(0.9).count("█") > render.confidence_bar(0.8).count(
            "█"
        )

    def test_severity_bar_maps_ordinal_scale(self):
        # critical is the full bar, nit the smallest; monotonic decreasing.
        levels = [render.severity_bar(s).count("█") for s in render.SEVERITIES]
        assert levels == sorted(levels, reverse=True)
        assert render.severity_bar("critical").count("█") == 10
        assert render.severity_bar("nit").count("█") < render.severity_bar(
            "critical"
        ).count("█")


# --------------------------------------------------------------------------- #
# severity histogram — the 정량판단 ASCII table (fenced code block)              #
# --------------------------------------------------------------------------- #
class TestSeverityHistogram:
    def test_histogram_is_fenced_codeblock_with_counts(self):
        findings = [
            _finding(severity="critical"),
            _finding(severity="high"),
            _finding(severity="high"),
            _finding(severity="nit"),
        ]
        block = render.severity_histogram(findings)
        # Fenced so the monospace bars actually align on github.com.
        assert block.startswith("```")
        assert block.rstrip().endswith("```")
        # Each present severity shows its emoji + count.
        assert "🔴" in block and "🟠" in block and "⚪" in block
        # The high bucket has 2; the count is rendered.
        high_line = next(ln for ln in block.splitlines() if "🟠" in ln)
        assert "2" in high_line

    def test_histogram_omits_zero_severities(self):
        # Only severities that actually occur are listed (compact / low-noise).
        block = render.severity_histogram([_finding(severity="critical")])
        assert "🔴" in block
        # No medium/low/nit rows when none occur.
        assert "🟡" not in block
        assert "⚪" not in block

    def test_histogram_localizes_title_ko(self):
        block_en = render.severity_histogram([_finding()], response_language="en")
        block_ko = render.severity_histogram([_finding()], response_language="ko")
        assert block_en != block_ko
        # Korean variant carries Hangul in its title line.
        assert any(any("가" <= ch <= "힣" for ch in ln) for ln in block_ko.splitlines())

    def test_empty_findings_histogram_is_empty_string(self):
        # No findings → no histogram (caller omits the section entirely).
        assert render.severity_histogram([]) == ""
