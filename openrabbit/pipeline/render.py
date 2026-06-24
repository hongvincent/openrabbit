"""Shared visual-render helpers — the single source of truth for openrabbit's
at-a-glance output encoding (severity/category emoji, calibrated-confidence
token, and the fenced ASCII ``정량판단`` severity histogram).

Design constraints (GitHub-Flavored Markdown, verified against the renderer):

* **Color only via emoji.** GitHub sanitizes ``style=``/``color=``/``<font>``;
  there is no way to color text in a comment/table cell. Colored emoji circles
  (🔴🟠🟡🟢⚪) are the only robust, accessible color primitive — and they are
  always paired with a word/number so the meaning survives if color is removed
  (WCAG 1.4.1 — never color alone).
* **ASCII/block bars only inside a fenced code block.** In a proportional table
  cell ``█``/``░`` render ragged and misrepresent the value, so the bars here are
  used ONLY by the fenced histogram, never in a table cell. Table cells get the
  emoji-dot + literal percent instead.

Severity emoji map 1:1 onto :data:`openrabbit.findings.SEVERITIES`; category
emoji are the CANONICAL gitmoji (lock not shield, zap not horse). Confidence is
kept on the green/yellow axis only — red is owned by severity, so it is never
reused across both axes — with two honest buckets because the default
``confidence_gate=0.80`` truncates the bottom of the visible range.

Pure + deterministic: no network, no model calls, no imports beyond the
findings contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from openrabbit.findings import SEVERITIES, Finding

# Severity → colored dot (ordered critical→nit, matching findings.SEVERITIES).
_SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
    "nit": "⚪",
}
# Only the top-3 severities carry **bold** in the table — weight without shouting.
_SEVERITY_BOLD = {"critical", "high", "medium"}
# Ordinal weight (critical=5 … nit=1) for the proportional severity bar.
_SEVERITY_LEVEL = {"critical": 5, "high": 4, "medium": 3, "low": 2, "nit": 1}

# Category → canonical gitmoji (gitmoji.dev): bug / lock / zap / ✅ / ♻️.
_CATEGORY_EMOJI = {
    "correctness": "🐛",
    "security": "🔒",
    "performance": "⚡",
    "tests": "✅",
    "maintainability": "♻️",
}

#: Confidence high-bucket cutoff. ``>= 0.90`` is 🟢 (verified/very likely real);
#: ``0.80 <= conf < 0.90`` is 🟡 (just above the gate — scrutinize). Anything
#: below the gate is dropped before render and never reaches these helpers.
_CONFIDENCE_HIGH = 0.90

_BAR_WIDTH = 10

#: Localized title for the fenced severity histogram (the 정량판단 ASCII table).
_HISTOGRAM_TITLE = {
    "en": "Findings by severity",
    "ko": "심각도별 분포",
}


def severity_display(severity: str) -> str:
    """Table-cell severity: ``🔴 **Critical**`` (top-3 bold), ``🟢 Low`` else.

    An out-of-vocabulary severity degrades to the plain word (no crash).
    """
    emoji = _SEVERITY_EMOJI.get(severity)
    if emoji is None:
        return severity
    word = severity.capitalize()
    if severity in _SEVERITY_BOLD:
        return f"{emoji} **{word}**"
    return f"{emoji} {word}"


def severity_inline(severity: str) -> str:
    """Inline-badge severity: ``🔴 CRITICAL`` (the badge adds its own bold)."""
    emoji = _SEVERITY_EMOJI.get(severity)
    if emoji is None:
        return severity.upper()
    return f"{emoji} {severity.upper()}"


def category_display(category: str) -> str:
    """Table-cell category: ``🔒 Security`` (canonical gitmoji + Title word)."""
    emoji = _CATEGORY_EMOJI.get(category)
    if emoji is None:
        return category
    return f"{emoji} {category.capitalize()}"


def category_inline(category: str) -> str:
    """Inline-badge category: ``🔒 security`` (lowercase preserved, gitmoji added)."""
    emoji = _CATEGORY_EMOJI.get(category)
    if emoji is None:
        return category
    return f"{emoji} {category}"


def confidence_token(confidence: float) -> str:
    """Calibrated confidence as ``🟢 96%`` / ``🟡 84%`` (dot bucket + literal %).

    The number is always printed — surfacing 96% vs 84% *within* a bucket is
    exactly the calibration signal the product exists to provide.
    """
    pct = round(confidence * 100)
    dot = "🟢" if confidence >= _CONFIDENCE_HIGH else "🟡"
    return f"{dot} {pct}%"


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def confidence_bar(confidence: float, width: int = _BAR_WIDTH) -> str:
    """Fixed-width proportional confidence bar (monospace/fenced use only)."""
    return _bar(confidence, width)


def severity_bar(severity: str, width: int = _BAR_WIDTH) -> str:
    """Fixed-width proportional severity bar from the ordinal scale (fenced use)."""
    level = _SEVERITY_LEVEL.get(severity, 0)
    return _bar(level / 5, width)


def severity_histogram(
    findings: Iterable[Finding], *, response_language: str = "en"
) -> str:
    """Render the 정량판단 severity histogram as a fenced ASCII code block.

    One row per severity that actually occurs (zero-count severities are
    omitted — compact/low-noise), ordered critical→nit, each row = emoji + bar
    + count. Returns ``""`` when there are no findings so the caller can omit
    the whole section. Fenced so the monospace bars align on github.com.
    """
    counts = Counter(f.severity for f in findings)
    if not counts:
        return ""
    present = [s for s in SEVERITIES if counts.get(s)]
    most = max(counts[s] for s in present)
    title = _HISTOGRAM_TITLE.get(response_language, _HISTOGRAM_TITLE["en"])
    lines = ["```", title]
    for sev in present:
        n = counts[sev]
        bar = _bar(n / most)
        lines.append(f"{_SEVERITY_EMOJI[sev]} {sev:<9} {bar} {n}")
    lines.append("```")
    return "\n".join(lines)
