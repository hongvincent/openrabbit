"""SARIF 2.1.0 output adapter (SPEC section 6 emit + section 9).

Turns structured :class:`~openrabbit.findings.Finding` objects into a valid
SARIF 2.1.0 document (the GitHub code-scanning / Security-tab interchange
format). This is the optional GHAS tier (SPEC 1.2 / 9): SARIF is **data only**
here — uploading the produced file is left to ``github/codeql-action/upload-sarif``
in the workflow, and surfacing it on the Security tab requires **GitHub Advanced
Security (GHAS)**.

Pure stdlib (``json`` + ``pathlib``) — no network, no cloud SDK, no heavy
parser. Importing this module and every unit test needs ZERO external deps.

severity -> SARIF mapping
-------------------------
* ``level`` (SARIF result level): ``error`` (critical/high) / ``warning``
  (medium) / ``note`` (low/nit).
* ``security-severity`` (GHAS numeric property, emitted as a *string* per the
  GHAS contract) by band, per the design doc:
  Critical > 9 / High 7-8.9 / Medium 4-6.9 / Low 0.1-3.9.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

from openrabbit.findings import Finding

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
    "Schemata/sarif-schema-2.1.0.json"
)

TOOL_NAME = "openrabbit"
TOOL_INFO_URI = "https://openrabbit.dev"

# SARIF result ``level`` per severity. Unknown -> ``warning`` (safe middle).
_LEVEL_BY_SEVERITY = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "nit": "note",
}

# Representative numeric ``security-severity`` (GHAS) per severity band.
# Bands (design doc): Critical > 9 / High 7-8.9 / Medium 4-6.9 / Low 0.1-3.9.
# A single representative point inside each band keeps the mapping stable.
_SECURITY_SEVERITY_BY_SEVERITY = {
    "critical": 9.5,
    "high": 8.0,
    "medium": 5.5,
    "low": 2.0,
    "nit": 0.5,
}


def level_for_severity(severity: str) -> str:
    """Map an openrabbit severity to a SARIF result ``level``.

    ``error`` for critical/high, ``warning`` for medium, ``note`` for low/nit.
    Unknown severities fall back to ``warning`` (a safe, visible middle).
    """
    return _LEVEL_BY_SEVERITY.get(severity, "warning")


def security_severity_for_severity(severity: str) -> float:
    """Map an openrabbit severity to a GHAS ``security-severity`` number.

    Returns a float inside the doc band (Critical > 9 / High 7-8.9 /
    Medium 4-6.9 / Low 0.1-3.9). Unknown severities fall back to the Medium
    representative.
    """
    return _SECURITY_SEVERITY_BY_SEVERITY.get(severity, 5.5)


def _security_severity_str(severity: str) -> str:
    """GHAS expects ``security-severity`` as a string property."""
    return f"{security_severity_for_severity(severity):.1f}"


def _relativize(file: str, repo_root: str) -> str:
    """Return a forward-slash URI for ``file`` relative to ``repo_root``.

    Absolute paths under ``repo_root`` are made repo-relative (SARIF URIs are
    conventionally relative to the repo). Paths that are already relative — or
    that lie outside the root — are returned unchanged (with back-slashes
    normalized to forward-slashes so Windows-authored paths stay valid URIs).
    """
    normalized = file.replace("\\", "/")
    if not os.path.isabs(file):
        return normalized
    try:
        rel = os.path.relpath(file, repo_root)
    except ValueError:  # pragma: no cover - different drive on Windows
        return normalized
    if rel.startswith(".."):
        return normalized
    return rel.replace("\\", "/")


def _rule_for(finding: Finding) -> dict[str, Any]:
    """Build a ``tool.driver.rules[]`` entry for a finding's ruleId."""
    return {
        "id": finding.rule_id,
        "name": finding.rule_id,
        "shortDescription": {"text": finding.title},
        "defaultConfiguration": {"level": level_for_severity(finding.severity)},
        "properties": {
            "category": finding.category,
            "security-severity": _security_severity_str(finding.severity),
        },
    }


def _result_for(finding: Finding, rule_index: int, repo_root: str) -> dict[str, Any]:
    """Build a single ``runs[].results[]`` entry."""
    return {
        "ruleId": finding.rule_id,
        "ruleIndex": rule_index,
        "level": level_for_severity(finding.severity),
        "message": {"text": _result_message(finding)},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": _relativize(finding.file, repo_root)},
                    "region": {
                        "startLine": finding.start_line,
                        "endLine": finding.end_line,
                    },
                }
            }
        ],
        "partialFingerprints": {"primaryLocationLineHash": finding.fingerprint},
        "properties": {
            "category": finding.category,
            "confidence": finding.confidence,
            "security-severity": _security_severity_str(finding.severity),
        },
    }


def _result_message(finding: Finding) -> str:
    """Compose the SARIF result message (title + rationale).

    The finding body is untrusted model output; it is embedded as plain text
    (SARIF ``message.text``), never interpreted, so there is no injection
    surface here.
    """
    body = (finding.body or "").strip()
    if body:
        return f"{finding.title}\n\n{body}"
    return finding.title


def findings_to_sarif(
    findings: list[Finding],
    *,
    tool_version: str,
    repo_root: str,
) -> dict[str, Any]:
    """Build a valid SARIF 2.1.0 document from ``findings``.

    The document has ``runs[0].tool.driver`` (one ``rules[]`` entry per distinct
    ``ruleId``, carrying its ``security-severity``) and ``results[]`` referencing
    those rules by index. File URIs are made relative to ``repo_root``.

    SARIF is data-only: uploading requires the ``github/codeql-action`` and the
    Security tab requires GitHub Advanced Security (GHAS).
    """
    rules: list[dict[str, Any]] = []
    rule_index: dict[str, int] = {}
    results: list[dict[str, Any]] = []

    for finding in findings:
        if finding.rule_id not in rule_index:
            rule_index[finding.rule_id] = len(rules)
            rules.append(_rule_for(finding))
        results.append(_result_for(finding, rule_index[finding.rule_id], repo_root))

    return {
        "$schema": SARIF_SCHEMA_URI,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": TOOL_NAME,
                        "informationUri": TOOL_INFO_URI,
                        "version": tool_version,
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def write_sarif(
    path: Union[str, os.PathLike[str]],
    findings: list[Finding],
    *,
    tool_version: str,
    repo_root: str,
) -> dict[str, Any]:
    """Write a SARIF 2.1.0 document for ``findings`` to ``path``.

    Parent directories are created as needed. Returns the document dict that was
    written (handy for logging/telemetry). The file is the artifact a workflow
    later feeds to ``github/codeql-action/upload-sarif`` (GHAS-gated surface).
    """
    document = findings_to_sarif(
        findings, tool_version=tool_version, repo_root=repo_root
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document
