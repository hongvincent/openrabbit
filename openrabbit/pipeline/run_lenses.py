"""Stage 4 — parallel lenses (bounded LLM, report-all) (SPEC section 6, step 4).

For each lens assigned to a file, call the FINDER provider with the byte-stable
prefix as the system prompt + lens prompt, plus the per-file diff message,
forcing the single ``emit_findings`` tool. Parse raw finding dicts into
:class:`~openrabbit.findings.Finding` objects:

* confidence is rescaled from the finder's integer 0-100 to the contract's
  0..1 (values already in 0..1 are passed through),
* the harness (not the model) computes the fingerprint,
* malformed/partial findings are skipped defensively rather than crashing the
  whole pass.

This module never imports a cloud SDK; it talks only to the neutral
:class:`~openrabbit.providers.base.Provider` interface, so unit tests drive it
with ``FakeProvider``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from openrabbit.domain import Message
from openrabbit.findings import (
    CATEGORIES,
    SEVERITIES,
    SIDES,
    Finding,
    compute_fingerprint,
)
from openrabbit.pipeline.context import (
    EnclosingFetcher,
    build_file_message,
    gather_enclosing_context,
)
from openrabbit.pipeline.route import FilePlan
from openrabbit.providers.base import Provider

EMIT_FINDINGS_TOOL = "emit_findings"
DEFAULT_FINDER_MAX_TOKENS = 4096

# JSON Schema for the forced emit_findings tool input (matches the finding
# wire contract minus the harness-computed fingerprint).
_FINDING_PROPS: dict[str, Any] = {
    "file": {"type": "string"},
    "startLine": {"type": "integer"},
    "endLine": {"type": "integer"},
    "side": {"type": "string", "enum": list(SIDES)},
    "severity": {"type": "string", "enum": list(SEVERITIES)},
    "category": {"type": "string", "enum": list(CATEGORIES)},
    "confidence": {"type": "number"},
    "title": {"type": "string"},
    "body": {"type": "string"},
    "suggestion": {"type": ["string", "null"]},
    "ruleId": {"type": "string"},
}

_EMIT_FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {"type": "object", "properties": _FINDING_PROPS},
        }
    },
    "required": ["findings"],
}


def emit_findings_tool_spec() -> Any:
    """Return the ``emit_findings`` :class:`~openrabbit.domain.ToolSpec`."""
    from openrabbit.domain import ToolSpec

    return ToolSpec(
        name=EMIT_FINDINGS_TOOL,
        description=(
            "Emit ALL findings for this lens as a structured array. Report every "
            "issue; do not self-filter."
        ),
        json_schema=_EMIT_FINDINGS_SCHEMA,
    )


def _rescale_confidence(value: Any) -> float:
    """Normalize a finder confidence to 0..1.

    Finder skills emit an integer 0-100; values already in [0, 1] are passed
    through. Out-of-range values are clamped.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num > 1.0:
        num = num / 100.0
    return max(0.0, min(1.0, num))


def _normalized_context(raw: Mapping[str, Any]) -> str:
    """Stable context string for fingerprinting (title + body of the finding)."""
    return f"{raw.get('title', '')}\n{raw.get('body', '')}"


def _parse_finding(raw: Mapping[str, Any]) -> Optional[Finding]:
    """Build a :class:`Finding` from a raw finder dict; return None if invalid."""
    file = raw.get("file")
    rule_id = raw.get("ruleId") or raw.get("rule_id")
    if not file or not rule_id:
        return None
    try:
        start_line = int(raw.get("startLine", raw.get("start_line", 1)))
        end_line = int(raw.get("endLine", raw.get("end_line", start_line)))
    except (TypeError, ValueError):
        return None
    side = raw.get("side", "RIGHT")
    if side not in SIDES:
        side = "RIGHT"
    severity = raw.get("severity", "low")
    if severity not in SEVERITIES:
        severity = "low"
    category = raw.get("category", "correctness")
    if category not in CATEGORIES:
        category = "correctness"
    title = str(raw.get("title", "")).strip()
    if not title:
        return None
    body = str(raw.get("body", ""))
    suggestion = raw.get("suggestion")
    if suggestion is not None:
        suggestion = str(suggestion)

    fingerprint = compute_fingerprint(str(file), str(rule_id), _normalized_context(raw))
    return Finding(
        file=str(file),
        start_line=start_line,
        end_line=end_line,
        side=side,
        severity=severity,
        category=category,
        confidence=_rescale_confidence(raw.get("confidence", 0)),
        title=title,
        body=body,
        rule_id=str(rule_id),
        fingerprint=fingerprint,
        suggestion=suggestion,
    )


def _parse_emit_findings(result: Any) -> list[Finding]:
    calls = getattr(result, "tool_calls", None) or []
    findings: list[Finding] = []
    for call in calls:
        if getattr(call, "name", None) != EMIT_FINDINGS_TOOL:
            continue
        raw_list = (call.args or {}).get("findings", [])
        if not isinstance(raw_list, list):
            continue
        for raw in raw_list:
            if not isinstance(raw, Mapping):
                continue
            parsed = _parse_finding(raw)
            if parsed is not None:
                findings.append(parsed)
    return findings


def run_lens(
    finder: Provider,
    file_plan: FilePlan,
    lens_name: str,
    lens_prompt: str,
    *,
    prefix: str,
    file_message: Optional[Message] = None,
    enclosing_fetcher: EnclosingFetcher = gather_enclosing_context,
    max_tokens: int = DEFAULT_FINDER_MAX_TOKENS,
    cache_prefix: Optional[str] = None,
) -> list[Finding]:
    """Run ONE lens over one file via the finder provider; return findings.

    ``enclosing_fetcher`` is forwarded to :func:`build_file_message` only when a
    pre-built ``file_message`` is not supplied. The default is the offline-safe
    no-op so unit tests never shell out; production passes a
    :class:`~openrabbit.pipeline.enclosing.GitEnclosingFetcher`.
    """
    if file_message is None:
        file_message = build_file_message(
            file_plan, enclosing_fetcher=enclosing_fetcher
        )
    system = f"{prefix}\n\n--- LENS: {lens_name} ---\n{lens_prompt}"
    result = finder.complete(
        system,
        [file_message],
        [emit_findings_tool_spec()],
        max_tokens,
        cache_prefix,
        tool_choice=EMIT_FINDINGS_TOOL,
    )
    return _parse_emit_findings(result)


def run_lenses(
    finder: Provider,
    file_plan: FilePlan,
    lens_prompts: Mapping[str, str],
    *,
    prefix: str,
    enclosing_fetcher: EnclosingFetcher = gather_enclosing_context,
    max_tokens: int = DEFAULT_FINDER_MAX_TOKENS,
    cache_prefix: Optional[str] = None,
) -> list[Finding]:
    """Run every assigned lens over one file; aggregate findings.

    A file with no assigned lenses (docs/lockfile/generated) returns ``[]``
    without ever calling the provider. Lenses with no available prompt are
    skipped.

    ``enclosing_fetcher`` (default: offline-safe no-op) is used once to build the
    shared per-file message that every lens reuses, so the enclosing-context
    block is fetched at most once per file. Production injects a
    :class:`~openrabbit.pipeline.enclosing.GitEnclosingFetcher`.
    """
    if not file_plan.lenses:
        return []
    file_message = build_file_message(file_plan, enclosing_fetcher=enclosing_fetcher)
    findings: list[Finding] = []
    for lens_name in file_plan.lenses:
        prompt = lens_prompts.get(lens_name)
        if not prompt:
            continue
        findings.extend(
            run_lens(
                finder,
                file_plan,
                lens_name,
                prompt,
                prefix=prefix,
                file_message=file_message,
                max_tokens=max_tokens,
                cache_prefix=cache_prefix,
            )
        )
    return findings
