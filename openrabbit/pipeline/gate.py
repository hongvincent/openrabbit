"""Stage 1 — trigger & gate (SPEC section 6, step 1).

Pure, deterministic code (no model calls). Decides whether a PR should be
reviewed at all, skipping:

* drafts / closed PRs,
* trivial diffs (fewer than ``min_changed_lines`` changed lines),
* lockfile / generated-only diffs,
* an already-reviewed head SHA (incremental state).

Incremental state for Phase 0 is a small local JSON file (:class:`StateStore`).
Production swaps this for DynamoDB behind the same tiny interface, but the
spine never depends on that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from openrabbit.config import Config

PathLike = Union[str, Path]

# Files that carry no reviewable intent: dependency lockfiles, vendored or
# generated output. Matched by filename suffix/segment, case-insensitively.
_LOCKFILE_NAMES = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
    "go.sum",
    "composer.lock",
    "gemfile.lock",
    "uv.lock",
)
_GENERATED_SEGMENTS = ("/dist/", "/build/", "/generated/", "/__generated__/", "/vendor/")
_GENERATED_SUFFIXES = (".min.js", ".min.css", ".lock", ".map")

# Match the "+++ b/<path>" header lines of a unified diff to enumerate files.
_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
# Match git's "diff --git a/<x> b/<y>" header (covers renames/deletes too).
_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$", re.MULTILINE)

DEFAULT_MIN_CHANGED_LINES = 3


@dataclass(frozen=True)
class GateDecision:
    """The outcome of :func:`evaluate_gate`."""

    should_review: bool
    reason: str
    changed_files: int = 0
    changed_lines: int = 0


class StateStore:
    """Tiny local JSON state for incremental review (Phase 0 stand-in for DynamoDB).

    Keyed by ``"<repo>#<pr_number>"``. Each entry holds two dedup-relevant
    facts:

    * ``last_reviewed_sha`` — the head SHA of the most recent review (drives the
      gate's incremental skip / synchronize-only-new-commits behavior), and
    * ``posted_fingerprints`` — a set of finding fingerprints already posted, so
      re-reviews suppress the same findings even **offline / before GitHub
      review threads load** (a second, local dedup source alongside threads).

    The on-disk value is a small dict ``{"last_reviewed_sha": str,
    "posted_fingerprints": [str, ...]}``. For backward compatibility a legacy
    bare-string value (the original Phase-0 format that stored only the SHA) is
    read transparently and migrated in place on the next write. The file is
    created lazily; concurrent writers are not a concern for the local CLI.
    """

    _SHA_KEY = "last_reviewed_sha"
    _FP_KEY = "posted_fingerprints"

    def __init__(self, path: PathLike) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _key(repo: str, pr_number: int) -> str:
        return f"{repo}#{pr_number}"

    @classmethod
    def _normalize_entry(cls, raw: Any) -> dict[str, Any]:
        """Coerce a stored entry (legacy bare SHA string or dict) to a dict."""
        if isinstance(raw, str):
            # Legacy Phase-0 format: the value was the bare head SHA.
            return {cls._SHA_KEY: raw, cls._FP_KEY: []}
        if isinstance(raw, dict):
            return raw
        return {}

    def last_reviewed_sha(self, repo: str, pr_number: int) -> Optional[str]:
        entry = self._normalize_entry(self._load().get(self._key(repo, pr_number)))
        sha = entry.get(self._SHA_KEY)
        return sha if isinstance(sha, str) else None

    def record_review(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        *,
        fingerprints: Optional[set[str]] = None,
    ) -> None:
        """Record the reviewed head SHA and (optionally) union posted fingerprints.

        Passing ``fingerprints`` folds the SHA update and the fingerprint union
        into a single load/save, avoiding a second read-modify-write cycle and
        the brief window where the SHA is recorded but fingerprints are not.
        """
        data = self._load()
        key = self._key(repo, pr_number)
        entry = self._normalize_entry(data.get(key))
        entry[self._SHA_KEY] = head_sha
        existing = {str(fp) for fp in self._coerce_fp_list(entry.get(self._FP_KEY))}
        if fingerprints:
            existing |= {str(fp) for fp in fingerprints}
        entry[self._FP_KEY] = sorted(existing)
        data[key] = entry
        self._save(data)

    @staticmethod
    def _coerce_fp_list(value: Any) -> list[Any]:
        """Treat only list/tuple/set as a fingerprint collection.

        A corrupted state file whose ``posted_fingerprints`` is a non-iterable
        (an int) or a bare string (which would iterate per character) degrades
        to empty rather than raising — mirroring the defensive coercion used for
        legacy entry shapes.
        """
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return []

    def get_posted_fingerprints(self, repo: str, pr_number: int) -> set[str]:
        """Return the set of finding fingerprints already posted for this PR."""
        entry = self._normalize_entry(self._load().get(self._key(repo, pr_number)))
        fps = self._coerce_fp_list(entry.get(self._FP_KEY))
        return {str(fp) for fp in fps}

    def record_posted_fingerprints(
        self, repo: str, pr_number: int, fingerprints: set[str]
    ) -> None:
        """Union ``fingerprints`` into the PR's persisted posted set.

        Idempotent and additive: recording the same fingerprint twice (e.g. a
        re-review that re-finds an already-posted issue) is a no-op. Recording
        an empty set does not create or rewrite state.
        """
        if not fingerprints:
            return
        data = self._load()
        key = self._key(repo, pr_number)
        entry = self._normalize_entry(data.get(key))
        existing = {str(fp) for fp in self._coerce_fp_list(entry.get(self._FP_KEY))}
        existing |= {str(fp) for fp in fingerprints}
        entry[self._FP_KEY] = sorted(existing)
        data[key] = entry
        self._save(data)


# --------------------------------------------------------------------------- #
# diff inspection helpers (pure)                                               #
# --------------------------------------------------------------------------- #
def diff_files(diff: str) -> list[str]:
    """Return the set of changed file paths referenced by a unified diff."""
    files: list[str] = []
    seen: set[str] = set()
    for match in _DIFF_GIT_RE.finditer(diff):
        path = match.group("b")
        if path not in seen:
            seen.add(path)
            files.append(path)
    if files:
        return files
    # Fall back to "+++ b/<path>" headers for non-git unified diffs.
    for match in _DIFF_FILE_RE.finditer(diff):
        path = match.group(1)
        if path != "/dev/null" and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def count_changed_lines(diff: str) -> int:
    """Count added/removed content lines (excludes diff/hunk headers)."""
    total = 0
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            total += 1
    return total


def _split_file_sections(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into ``(path, section_text)`` per file.

    Sections are delimited by ``diff --git`` headers (the new-side ``b/`` path
    names the file). A leading segment before the first header is ignored.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_DIFF_GIT_RE.finditer(diff))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(diff)
        sections.append((m.group("b"), diff[start:end]))
    return sections


def count_reviewable_changed_lines(diff: str) -> int:
    """Count changed lines over reviewable (non-ignorable) files only.

    Lockfile/generated/vendored churn (e.g. a 40-line ``package-lock.json``
    update) is excluded so it cannot lift a tiny real code change over the
    trivial-diff threshold. Falls back to whole-diff counting only when the diff
    has no parseable ``diff --git`` sections (non-git unified diffs).
    """
    sections = _split_file_sections(diff)
    if not sections:
        return count_changed_lines(diff)
    return sum(
        count_changed_lines(text)
        for path, text in sections
        if not is_ignorable_file(path)
    )


def is_ignorable_file(path: str) -> bool:
    """True if a path is a lockfile / generated / vendored artifact."""
    lower = path.lower()
    base = lower.rsplit("/", 1)[-1]
    if base in _LOCKFILE_NAMES:
        return True
    if any(seg in f"/{lower}" for seg in _GENERATED_SEGMENTS):
        return True
    if any(lower.endswith(suf) for suf in _GENERATED_SUFFIXES):
        return True
    return False


# --------------------------------------------------------------------------- #
# main entry                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_gate(
    config: Config,
    pr_context: Mapping[str, Any],
    diff: str,
    *,
    store: Optional[StateStore] = None,
    min_changed_lines: int = DEFAULT_MIN_CHANGED_LINES,
) -> GateDecision:
    """Decide whether to review this PR. Pure & deterministic (no model calls).

    ``pr_context`` keys honored: ``draft`` (bool), ``state`` (``"open"`` /
    ``"closed"``), ``head_sha`` (str), ``repo`` (``"owner/name"``),
    ``number`` (int). ``store`` enables already-reviewed skip when the config's
    incremental flag is on.
    """
    if pr_context.get("draft"):
        return GateDecision(False, "PR is a draft")

    state = pr_context.get("state")
    if state and state != "open":
        return GateDecision(False, f"PR state is {state!r}, not open")

    files = diff_files(diff)
    if not files:
        return GateDecision(False, "empty diff: no changed files")

    reviewable = [f for f in files if not is_ignorable_file(f)]
    if not reviewable:
        return GateDecision(
            False,
            "lockfile/generated-only diff (no reviewable files)",
            changed_files=len(files),
        )

    # Count only reviewable (non-ignorable) lines so lockfile/generated churn
    # co-changing with a tiny real edit cannot bypass the trivial-diff skip.
    changed_lines = count_reviewable_changed_lines(diff)
    if changed_lines < min_changed_lines:
        return GateDecision(
            False,
            f"trivial diff ({changed_lines} changed lines < {min_changed_lines})",
            changed_files=len(files),
            changed_lines=changed_lines,
        )

    # Incremental: skip an already-reviewed head SHA.
    if config.review.incremental and store is not None:
        repo = pr_context.get("repo")
        number = pr_context.get("number")
        head_sha = pr_context.get("head_sha")
        if repo is not None and number is not None and head_sha is not None:
            last = store.last_reviewed_sha(str(repo), int(number))
            if last == head_sha:
                return GateDecision(
                    False,
                    f"head SHA {head_sha} already reviewed",
                    changed_files=len(files),
                    changed_lines=changed_lines,
                )

    return GateDecision(
        True,
        "review proceeds",
        changed_files=len(reviewable),
        changed_lines=changed_lines,
    )
