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


def render_summary_markdown(
    findings: list[Finding], *, stats: Optional[Mapping[str, Any]] = None
) -> str:
    """Render the sticky walkthrough markdown: summary + grouped findings table."""
    stats = stats or {}
    if not findings:
        header = "## openrabbit review\n\nNo issues found above the confidence gate. ✅"
        if stats:
            header += "\n\n" + _stats_line(stats)
        return header

    lines = ["## openrabbit review", ""]
    lines.append(f"Found **{len(findings)}** issue(s) above the confidence gate.")
    if stats:
        lines += ["", _stats_line(stats)]
    lines += [
        "",
        "| Severity | Category | File | Line | Finding |",
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


def _stats_line(stats: Mapping[str, Any]) -> str:
    parts = [f"{k}: {v}" for k, v in stats.items()]
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
