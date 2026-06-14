"""Check-Run output adapter (SPEC section 6 emit + section 9).

Turns structured :class:`~openrabbit.findings.Finding` objects into a GitHub
Checks API payload (the optional merge-gating tier, SPEC 1.2 / 9). The check run
carries inline ``annotations[]`` and a ``conclusion`` that can gate merges.

Posture (SPEC 12 — advisory-only by default): with no ``gate`` configured the
conclusion is ``neutral`` (or ``success`` when there are no findings) — it never
fails a build. A blocking ``gate`` severity is **opt-in**: only then does a
finding at-or-above the gate flip the conclusion to ``failure``.

``post_check_run`` reuses the existing :class:`~openrabbit.adapters.github.GitHubAdapter`
HTTP layer (``httpx`` imported lazily *inside* the adapter), so importing this
module and every unit test needs ZERO external deps and makes no network calls.
"""

from __future__ import annotations

from typing import Any, Optional

from openrabbit.adapters.github import GitHubAdapter
from openrabbit.findings import SEVERITIES, Finding

CHECK_NAME = "openrabbit"

# GitHub caps a single Check-Run create/update at 50 annotations; extras are
# delivered via follow-up PATCH calls in batches of this size.
_MAX_ANNOTATIONS_PER_REQUEST = 50

# annotation_level per severity (GitHub: notice | warning | failure).
_ANNOTATION_LEVEL_BY_SEVERITY = {
    "critical": "failure",
    "high": "failure",
    "medium": "warning",
    "low": "notice",
    "nit": "notice",
}

# Severity ordering (most severe first) for gate comparison. Mirrors
# ``findings.SEVERITIES`` so the two never drift.
_SEVERITY_RANK = {sev: rank for rank, sev in enumerate(SEVERITIES)}


def annotation_level_for_severity(severity: str) -> str:
    """Map an openrabbit severity to a Checks API ``annotation_level``.

    ``failure`` for critical/high, ``warning`` for medium, ``notice`` for
    low/nit. Unknown severities fall back to ``warning``.
    """
    return _ANNOTATION_LEVEL_BY_SEVERITY.get(severity, "warning")


def _meets_gate(severity: str, gate: str) -> bool:
    """True when ``severity`` is at least as severe as ``gate``.

    Lower rank == more severe (critical=0). Unknown severities never meet a
    gate (treated as least severe).
    """
    sev_rank = _SEVERITY_RANK.get(severity, len(SEVERITIES))
    return sev_rank <= _SEVERITY_RANK[gate]


def build_annotation(finding: Finding) -> dict[str, Any]:
    """Build one Checks API ``annotations[]`` entry for a finding.

    Shape: ``path`` / ``start_line`` / ``end_line`` / ``annotation_level`` /
    ``message`` (+ a short ``title``). The finding body is untrusted model
    output and is embedded as plain text, never interpreted. The ``message``
    leads with the finding title so the annotation is self-describing.
    """
    body = (finding.body or "").strip()
    message = f"{finding.title}\n\n{body}" if body else finding.title
    return {
        "path": finding.file,
        "start_line": finding.start_line,
        "end_line": finding.end_line,
        "annotation_level": annotation_level_for_severity(finding.severity),
        "title": finding.title,
        "message": message,
    }


def _conclusion(findings: list[Finding], gate: Optional[str]) -> str:
    """Decide the check-run conclusion.

    * no findings -> ``success``
    * ``gate`` is ``None`` (advisory-only default) -> ``neutral``
    * any finding meets the ``gate`` severity -> ``failure``
    * otherwise (findings exist, none meet the gate) -> ``neutral``
    """
    if not findings:
        return "success"
    if gate is None:
        return "neutral"
    if any(_meets_gate(f.severity, gate) for f in findings):
        return "failure"
    return "neutral"


def _summary(findings: list[Finding], conclusion: str) -> str:
    """Compose the check-run ``output.summary`` markdown."""
    if not findings:
        return "openrabbit found no issues."
    by_sev: dict[str, int] = {}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    breakdown = ", ".join(f"{by_sev[sev]} {sev}" for sev in SEVERITIES if sev in by_sev)
    return (
        f"openrabbit found {len(findings)} finding(s) ({breakdown}). "
        f"Conclusion: {conclusion}."
    )


def build_check_run(
    findings: list[Finding],
    *,
    head_sha: str,
    gate: Optional[str] = None,
) -> dict[str, Any]:
    """Build a GitHub Checks API ``POST /check-runs`` payload.

    ``name`` is ``openrabbit``, ``head_sha`` anchors the run, ``status`` is
    ``completed`` with a ``conclusion`` decided by :func:`_conclusion`.
    ``output.annotations[]`` carries one annotation per finding.

    ``gate`` is the optional blocking severity (one of
    :data:`~openrabbit.findings.SEVERITIES`); when ``None`` the check is
    advisory-only and never fails. An invalid ``gate`` raises ``ValueError``.
    """
    if gate is not None and gate not in _SEVERITY_RANK:
        raise ValueError(
            f"invalid gate severity {gate!r}; expected one of {SEVERITIES}"
        )
    conclusion = _conclusion(findings, gate)
    annotations = [build_annotation(f) for f in findings]
    return {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": f"openrabbit: {len(findings)} finding(s)",
            "summary": _summary(findings, conclusion),
            "annotations": annotations,
        },
    }


def post_check_run(
    adapter: GitHubAdapter,
    findings: list[Finding],
    *,
    head_sha: str,
    gate: Optional[str] = None,
) -> dict[str, Any]:
    """Create the check run via the GitHub adapter's HTTP layer.

    Reuses ``adapter``'s lazy ``httpx`` client + auth headers + error handling.
    GitHub caps a Check-Run request at 50 annotations, so the first 50 ship in
    the create call and any remainder are appended via follow-up PATCH calls in
    batches of 50. Returns the created check-run JSON.
    """
    payload = build_check_run(findings, head_sha=head_sha, gate=gate)
    all_annotations = payload["output"]["annotations"]
    payload["output"]["annotations"] = all_annotations[:_MAX_ANNOTATIONS_PER_REQUEST]

    url = adapter._rest_url(f"/repos/{adapter.repo.slug}/check-runs")
    resp = adapter._client().post(url, headers=adapter._headers(), json=payload)
    adapter._raise_for(resp, "post_check_run")
    created = resp.json()

    remaining = all_annotations[_MAX_ANNOTATIONS_PER_REQUEST:]
    if remaining:
        _append_annotations(adapter, created["id"], payload["output"], remaining)
    return created


def _append_annotations(
    adapter: GitHubAdapter,
    check_run_id: Any,
    output: dict[str, Any],
    remaining: list[dict[str, Any]],
) -> None:
    """PATCH the remaining annotations onto an existing check run (batches of 50)."""
    url = adapter._rest_url(f"/repos/{adapter.repo.slug}/check-runs/{check_run_id}")
    for start in range(0, len(remaining), _MAX_ANNOTATIONS_PER_REQUEST):
        batch = remaining[start : start + _MAX_ANNOTATIONS_PER_REQUEST]
        body = {
            "output": {
                "title": output["title"],
                "summary": output["summary"],
                "annotations": batch,
            }
        }
        resp = adapter._client().patch(url, headers=adapter._headers(), json=body)
        adapter._raise_for(resp, "post_check_run (append annotations)")
