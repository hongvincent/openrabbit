"""Learnings / memory store + feedback loop (SPEC section 10, principle 1).

The learnings store is openrabbit's **trust differentiator**: it captures team
knowledge and a negative feedback signal, then folds both back into the review
loop so the bot gets quieter where humans said it was noisy.

Two kinds of memory live here, keyed by **scope**:

* **Learnings** — team-authored conventions (``"always parameterize SQL here"``)
  injected into the byte-stable, cacheable system prefix (SPEC 6.3) so every
  finder lens sees them. A learning is::

      {id, text, provenance{pr, file, user}, category, created_at,
       usage_count, embedding(None for now)}

* **Dismissals** — a negative signal. When a human dismisses a finding,
  :func:`LearningsStore.record_dismissal` records its
  ``(rule_id, category, file)`` shape. :func:`LearningsStore.adjust_confidence`
  then **lowers** the confidence of future findings with that same shape so
  drifting/noisy rule+file patterns fall below the gate on a re-review
  (deterministic Phase-2 heuristic — no real embeddings yet).

**Scope** is either a repo (``"owner/name"``) or an org (``"owner"``). A repo is
in scope for its own learnings *and* its org's learnings (org applies fleet-wide;
repo is repo-specific). See :func:`LearningsStore.get_in_scope_learnings`.

Phase 2 persists everything to a small local JSON file (a stand-in for the
DynamoDB single-table in SPEC 9), mirroring the :class:`StateStore` pattern in
:mod:`openrabbit.pipeline.gate`. The store hides its storage behind a tiny
interface so DynamoDB can be swapped in later without touching the spine. No
network, no cloud SDKs — import-cheap and offline-safe.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from openrabbit.findings import Finding

PathLike = Union[str, Path]

# Each recorded dismissal of a matching (rule_id, category, file) finding shape
# multiplies the surviving confidence by this factor, so repeated dismissals
# compound (0.85 -> 0.5525 -> 0.359...). One dismissal of a verified 0.85
# finding lands at ~0.5525, comfortably below the default 0.80 gate. Bounded so
# a single noisy pattern never silences everything outright.
DEFAULT_DISMISSAL_PENALTY = 0.65
# Cap how far confidence can be driven down so a finding is suppressed (below
# any reasonable gate) but the signal is never zeroed entirely.
MIN_ADJUSTED_CONFIDENCE = 0.05


@dataclass(frozen=True)
class Learning:
    """A single team-authored learning (SPEC 9 ``LEARNING#`` entity, 10).

    ``provenance`` carries ``{pr, file, user}``. ``embedding`` is reserved for a
    later semantic-similarity phase and is ``None`` in Phase 2 (the dismissal
    heuristic is deterministic, not vector-based).
    """

    id: str
    text: str
    provenance: dict[str, Any]
    category: str
    created_at: str
    usage_count: int = 0
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-able dict (the on-disk / wire shape)."""
        return {
            "id": self.id,
            "text": self.text,
            "provenance": dict(self.provenance),
            "category": self.category,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Learning":
        """Build a :class:`Learning` from a stored dict (tolerant of missing keys)."""
        embedding = data.get("embedding")
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            provenance=dict(data.get("provenance") or {}),
            category=str(data.get("category", "")),
            created_at=str(data.get("created_at", "")),
            usage_count=int(data.get("usage_count", 0) or 0),
            embedding=list(embedding) if isinstance(embedding, list) else None,
        )


def _dismissal_signature(rule_id: str, category: str, file: str) -> str:
    """Stable key for a dismissed-finding shape (the negative signal).

    Phase-2 similarity is the exact ``(rule_id, category, file)`` triple — a
    deterministic heuristic standing in for embedding similarity. NUL-joined so
    no concatenation collision is possible.
    """
    return "\x00".join((rule_id, category, file))


class LearningsStore:
    """Local-JSON learnings + dismissal store (Phase-2 stand-in for DynamoDB).

    The on-disk document is::

        {
          "learnings": {"<scope>": [<learning dict>, ...], ...},
          "dismissals": {"<signature>": {rule_id, category, file, count}, ...}
        }

    The file is created lazily on the first write. Storage is hidden behind this
    interface so a DynamoDB-backed implementation can replace it without the
    spine knowing (mirrors :class:`openrabbit.pipeline.gate.StateStore`).
    Concurrent writers are not a concern for the local CLI path.
    """

    _LEARNINGS_KEY = "learnings"
    _DISMISSALS_KEY = "dismissals"

    def __init__(self, path: PathLike) -> None:
        self._path = Path(path)

    # -- persistence ------------------------------------------------------- #
    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _scopes_for(repo: str) -> list[str]:
        """Scopes that apply to ``repo``: the org (owner) then the repo itself.

        ``"acme/widgets"`` -> ``["acme", "acme/widgets"]`` so org-wide learnings
        and repo-specific learnings both surface. A bare scope with no ``"/"``
        (already an org) yields just itself.
        """
        owner = repo.split("/", 1)[0] if "/" in repo else repo
        scopes = [owner]
        if repo != owner:
            scopes.append(repo)
        return scopes

    # -- learnings --------------------------------------------------------- #
    def add_learning(
        self,
        scope: str,
        text: str,
        provenance: Mapping[str, Any],
        category: str,
    ) -> Learning:
        """Add a learning under ``scope`` (a repo ``"owner/name"`` or org ``"owner"``).

        Returns the created :class:`Learning` (with a fresh id + UTC timestamp).
        """
        learning = Learning(
            id=uuid.uuid4().hex,
            text=text,
            provenance=dict(provenance),
            category=category,
            created_at=datetime.now(timezone.utc).isoformat(),
            usage_count=0,
            embedding=None,
        )
        data = self._load()
        bucket = data.setdefault(self._LEARNINGS_KEY, {})
        if not isinstance(bucket, dict):  # pragma: no cover - defensive
            bucket = {}
            data[self._LEARNINGS_KEY] = bucket
        scope_list = bucket.setdefault(scope, [])
        if not isinstance(scope_list, list):  # pragma: no cover - defensive
            scope_list = []
            bucket[scope] = scope_list
        scope_list.append(learning.to_dict())
        self._save(data)
        return learning

    def get_in_scope_learnings(
        self, repo: str, paths: Optional[list[str]] = None
    ) -> list[Learning]:
        """Return learnings in scope for ``repo`` (its org's + its own).

        ``paths`` (the PR's changed files) is accepted for forward-compatibility
        with future path-scoped learnings; Phase 2 returns all in-scope
        learnings regardless of path (every learning applies repo/org-wide).
        """
        data = self._load()
        bucket = data.get(self._LEARNINGS_KEY, {})
        if not isinstance(bucket, dict):  # pragma: no cover - defensive
            return []
        out: list[Learning] = []
        for scope in self._scopes_for(repo):
            entries = bucket.get(scope, [])
            if not isinstance(entries, list):  # pragma: no cover - defensive
                continue
            for raw in entries:
                if isinstance(raw, Mapping):
                    out.append(Learning.from_dict(raw))
        return out

    # -- dismissals (negative signal) -------------------------------------- #
    def record_dismissal(self, repo: str, finding: Finding) -> None:
        """Record a human dismissal of ``finding`` as a negative signal.

        Stores (and increments) the ``(rule_id, category, file)`` shape so a
        future similar finding is down-weighted by :func:`adjust_confidence`.
        ``repo`` is recorded for provenance/observability; matching is by shape
        (Phase-2 heuristic), not scope.
        """
        sig = _dismissal_signature(finding.rule_id, finding.category, finding.file)
        data = self._load()
        bucket = data.setdefault(self._DISMISSALS_KEY, {})
        if not isinstance(bucket, dict):  # pragma: no cover - defensive
            bucket = {}
            data[self._DISMISSALS_KEY] = bucket
        entry = bucket.get(sig)
        count = (entry.get("count", 0) if isinstance(entry, Mapping) else 0) + 1
        bucket[sig] = {
            "rule_id": finding.rule_id,
            "category": finding.category,
            "file": finding.file,
            "count": count,
            "repo": repo,
        }
        self._save(data)

    def _dismissal_count(self, finding: Finding) -> int:
        """How many times a finding of this exact shape has been dismissed."""
        sig = _dismissal_signature(finding.rule_id, finding.category, finding.file)
        bucket = self._load().get(self._DISMISSALS_KEY, {})
        if not isinstance(bucket, dict):  # pragma: no cover - defensive
            return 0
        entry = bucket.get(sig)
        if not isinstance(entry, Mapping):
            return 0
        try:
            return max(0, int(entry.get("count", 0)))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    def adjust_confidence(
        self, finding: Finding, *, penalty: float = DEFAULT_DISMISSAL_PENALTY
    ) -> float:
        """Return a (possibly lowered) confidence for ``finding``.

        Deterministic Phase-2 heuristic: if a finding sharing this one's
        ``(rule_id, category, file)`` shape was dismissed ``n`` times, scale
        confidence by ``penalty ** n`` (floored at :data:`MIN_ADJUSTED_CONFIDENCE`)
        so dismissed-style findings drop below the gate. With no matching
        dismissal the confidence is returned unchanged (identity).
        """
        count = self._dismissal_count(finding)
        if count <= 0:
            return finding.confidence
        adjusted = finding.confidence * (penalty ** count)
        return max(MIN_ADJUSTED_CONFIDENCE, adjusted)
