"""Stage 6 — dedup & rank (SPEC section 6, step 5/6).

Pure, deterministic code. Removes:

* intra-batch duplicates (same fingerprint emitted by two lenses),
* findings already posted in a prior review (``prior_fingerprints``),

then ranks the survivors by severity (critical first) and confidence so the
most important findings surface first.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from openrabbit.findings import SEVERITIES, Finding

# Higher rank == more severe; used to sort descending.
_SEVERITY_RANK = {sev: len(SEVERITIES) - i for i, sev in enumerate(SEVERITIES)}


def _sort_key(finding: Finding) -> tuple[int, float, str]:
    # Sort by severity (desc), then confidence (desc), then file for stability.
    return (
        -_SEVERITY_RANK.get(finding.severity, 0),
        -finding.confidence,
        finding.file,
    )


def dedup_and_rank(
    findings: Iterable[Finding],
    *,
    prior_fingerprints: Optional[set[str]] = None,
) -> list[Finding]:
    """Dedup by fingerprint (intra-batch + vs prior) and rank the survivors.

    When two findings share a fingerprint within the batch, the one with the
    higher confidence is kept. Findings whose fingerprint appears in
    ``prior_fingerprints`` are dropped (already reported / dismissed).
    """
    prior = prior_fingerprints or set()
    best: dict[str, Finding] = {}
    for finding in findings:
        if finding.fingerprint in prior:
            continue
        existing = best.get(finding.fingerprint)
        if existing is None or finding.confidence > existing.confidence:
            best[finding.fingerprint] = finding
    return sorted(best.values(), key=_sort_key)
