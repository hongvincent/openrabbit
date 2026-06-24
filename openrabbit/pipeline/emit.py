"""Stage 7 — emit (SPEC section 6, step 6).

Two emit paths:

* :func:`emit_console` — offline: builds the would-be GitHub review payload
  (the exact ``comments[]`` the adapter would POST) plus a summary, returning a
  JSON-serializable dict. No network, no credentials.
* :func:`emit_github` — online: delegates to
  :class:`~openrabbit.adapters.github.GitHubAdapter` (one batched createReview +
  sticky walkthrough). The adapter is injected so this module never imports
  ``httpx`` and stays offline-importable.

:func:`render_summary_markdown` builds the sticky walkthrough body: a summary
line + a grouped, per-file findings table.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from openrabbit.adapters.github import build_review_comment
from openrabbit.findings import Finding

DEFAULT_EVENT = "COMMENT"

_LOG = logging.getLogger("openrabbit.pipeline.emit")


# --------------------------------------------------------------------------- #
# localized summary strings (Feature 1 — response_language)                     #
# --------------------------------------------------------------------------- #
#: Per-language strings for the findings summary table. ``en`` is the SSOT
#: default and must stay pixel-identical to today (existing tests pin these).
#: An unknown code falls back to ``en``. ``{n}`` is the kept-finding count;
#: ``stat_keys`` localizes the per-PR stats-line keys (reviewable files/raw/kept).
_SUMMARY_STRINGS: dict[str, dict[str, Any]] = {
    "en": {
        # Standalone heading word (the inline review body). Localized + branded
        # (🐰) only when persona is ON; see ``render_summary_markdown``.
        "heading": "openrabbit review",
        "no_issues": "No issues found above the confidence gate. ✅",
        "found": "Found **{n}** issue(s) above the confidence gate.",
        "table_header": "| Severity | Category | File | Line | Finding |",
        "stat_keys": {
            "reviewable files": "reviewable files",
            "raw": "raw",
            "kept": "kept",
        },
    },
    "ko": {
        # Natural Korean ("리뷰" = the standard loanword for "review"), never the
        # literal English "openrabbit review".
        "heading": "openrabbit 리뷰",
        "no_issues": "신뢰도 기준을 넘는 문제가 발견되지 않았습니다. ✅",
        "found": "신뢰도 기준을 넘는 문제 **{n}**건을 찾았어요.",
        "table_header": "| 심각도 | 카테고리 | 파일 | 라인 | 발견 사항 |",
        "stat_keys": {
            "reviewable files": "검토 대상 파일",
            "raw": "원본",
            "kept": "유지",
        },
    },
}

#: Branding marker (kept in sync with ``walkthrough._RABBIT``). Only rendered on
#: the standalone heading when ``persona`` is ON.
_RABBIT = "🐰"


def _summary_strings(response_language: str) -> dict[str, Any]:
    """Return the summary-string map for a language, falling back to English."""
    return _SUMMARY_STRINGS.get(response_language, _SUMMARY_STRINGS["en"])


def render_summary_markdown(
    findings: list[Finding],
    *,
    stats: Optional[Mapping[str, Any]] = None,
    response_language: str = "en",
    heading: bool = True,
    persona: bool = True,
) -> str:
    """Render the findings summary markdown: a heading + count + grouped table.

    ``response_language`` (default ``"en"``) localizes the heading word, the
    count line, the "no issues" line, the table header, and the stats-line keys.

    ``heading`` (default ``True``) controls the standalone ``## …`` heading. It
    is the natural single heading for the inline-review BODY (this function's
    standalone use). When embedded inside
    :func:`openrabbit.pipeline.walkthrough.build_walkthrough` — which already
    emits its own localized "Findings"/"발견 사항" heading right above — the caller
    passes ``heading=False`` so we don't DOUBLE-head; only the localized count
    line (and table) is rendered.

    ``persona`` (default ``True``) gates the 🐰 marker on the standalone heading
    (branding appears only when persona is ON); ignored when ``heading`` is
    ``False``. With ``persona=False`` the heading is the plain localized word.
    """
    stats = stats or {}
    strings = _summary_strings(response_language)
    head_line = None
    if heading:
        word = strings["heading"]
        head_line = f"## {_RABBIT} {word}" if persona else f"## {word}"

    if not findings:
        parts = [head_line] if head_line is not None else []
        parts.append(strings["no_issues"])
        body = "\n\n".join(parts)
        if stats:
            body += "\n\n" + _stats_line(stats, strings["stat_keys"])
        return body

    lines: list[str] = []
    if head_line is not None:
        lines += [head_line, ""]
    lines.append(strings["found"].format(n=len(findings)))
    if stats:
        lines += ["", _stats_line(stats, strings["stat_keys"])]
    lines += [
        "",
        strings["table_header"],
        "| --- | --- | --- | --- | --- |",
    ]
    for f in findings:
        title = f.title.replace("|", "\\|")
        # f.file derives from the UNTRUSTED diff path: a pipe would break the
        # table row and a backtick would close the code span, so neutralize both
        # before placing it in the `\`...\`` file cell.
        file_cell = f.file.replace("|", "\\|").replace("`", "\\`")
        lines.append(
            f"| {f.severity} | {f.category} | `{file_cell}` | {f.start_line} | {title} |"
        )
    return "\n".join(lines)


def _stats_line(
    stats: Mapping[str, Any], stat_keys: Optional[Mapping[str, str]] = None
) -> str:
    """Render the italic per-PR stats line, optionally localizing the keys.

    ``stat_keys`` maps an English stats key (``reviewable files``/``raw``/
    ``kept``) to its localized label; an unmapped key passes through unchanged so
    a caller adding a new stat never crashes.
    """
    stat_keys = stat_keys or {}
    parts = [f"{stat_keys.get(k, k)}: {v}" for k, v in stats.items()]
    return "_" + ", ".join(parts) + "_"


def build_review_payload(
    findings: list[Finding],
    summary_markdown: str,
    *,
    commit_sha: Optional[str] = None,
    event: str = DEFAULT_EVENT,
) -> dict[str, Any]:
    """Build the exact review payload the GitHub adapter would POST."""
    return {
        "review": {
            "commit_id": commit_sha,
            "event": event,
            "body": summary_markdown,
            "comments": [build_review_comment(f) for f in findings],
        },
        "sticky_walkthrough": summary_markdown,
    }


def emit_console(
    findings: list[Finding],
    *,
    summary_markdown: str,
    commit_sha: Optional[str] = None,
    stats: Optional[Mapping[str, Any]] = None,
    walkthrough_markdown: Optional[str] = None,
) -> dict[str, Any]:
    """Offline emit: return the would-be GitHub payload (no network).

    ``walkthrough_markdown`` (the enriched
    :func:`openrabbit.pipeline.walkthrough.build_walkthrough` body) overrides the
    payload's ``sticky_walkthrough`` field when supplied; otherwise the minimal
    ``summary_markdown`` is used (preserving the prior behavior).
    """
    payload = build_review_payload(findings, summary_markdown, commit_sha=commit_sha)
    if walkthrough_markdown is not None:
        payload["sticky_walkthrough"] = walkthrough_markdown
    return payload


def emit_github(
    adapter: Any,
    findings: list[Finding],
    *,
    summary_markdown: str,
    commit_sha: str,
    walkthrough_markdown: Optional[str] = None,
    prior_threads: Optional[list[Any]] = None,
    resolve_stale: bool = True,
    valid_positions: Optional[dict[str, set[tuple[str, int]]]] = None,
    changed_files: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Online emit via an injected :class:`GitHubAdapter`.

    Posts ONE advisory (event=COMMENT) review with all inline comments, upserts
    the sticky walkthrough comment, and (optionally) resolves+minimizes stale
    bot threads whose finding no longer appears.

    **Diff-anchor validation:** ``valid_positions`` (from
    :func:`openrabbit.pipeline.route.valid_positions_by_file`) and
    ``changed_files`` are forwarded to :meth:`GitHubAdapter.post_review`, which
    drops any finding whose path/line falls outside the real diff — so one
    hallucinated position can't 422 the entire batched review.

    **Low-noise guard (SPEC 1.3 / 3 / principle 1):** when ``findings`` is empty
    (clean PR, or an incremental re-run where dedup suppressed everything), NO
    ``createReview`` event is fired — an empty review on every push is exactly
    the noise this product avoids. The single sticky walkthrough is still
    upserted (it just reflects "no issues"), and stale prior threads are still
    resolved.

    The adapter is injected (never constructed here) so this module imports with
    zero external deps; the orchestrator wires the real adapter in online mode.
    """
    review: Optional[dict[str, Any]] = None
    if findings:
        # Forward the diff-anchor validation maps only when supplied so adapters
        # (and test fakes) with the legacy post_review signature keep working;
        # the real GitHubAdapter accepts the optional kwargs.
        extra: dict[str, Any] = {}
        if valid_positions is not None:
            extra["valid_positions"] = valid_positions
        if changed_files is not None:
            extra["changed_files"] = changed_files
        review = adapter.post_review(
            findings,
            summary_markdown,
            commit_sha,
            event=DEFAULT_EVENT,
            **extra,
        )
    walkthrough = adapter.upsert_sticky_walkthrough(
        walkthrough_markdown or summary_markdown
    )

    resolved: list[str] = []
    if resolve_stale and prior_threads is not None:
        from openrabbit.adapters.github import GitHubAdapter

        for thread in GitHubAdapter.stale_threads(findings, prior_threads):
            try:
                adapter.resolve_review_thread(thread.thread_id)
                if thread.comment_id:
                    adapter.minimize_comment(thread.comment_id, "OUTDATED")
                resolved.append(thread.thread_id)
            except Exception as exc:
                # Best-effort cleanup: a failed resolve/minimize must not abort
                # the review, but it MUST be logged (not silently swallowed) so a
                # persistently failing GraphQL call is diagnosable in CI logs.
                _LOG.warning(
                    "emit_github: failed to resolve/minimize stale thread %s: %s",
                    thread.thread_id,
                    exc,
                )
                continue

    return {
        "review": review,
        "sticky_walkthrough": walkthrough,
        "resolved_threads": resolved,
    }
