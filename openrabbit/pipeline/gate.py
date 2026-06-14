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

    Keyed by ``"<repo>#<pr_number>"`` -> last reviewed head SHA. The file is
    created lazily; concurrent writers are not a concern for the local CLI.
    """

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

    def last_reviewed_sha(self, repo: str, pr_number: int) -> Optional[str]:
        return self._load().get(self._key(repo, pr_number))

    def record_review(self, repo: str, pr_number: int, head_sha: str) -> None:
        data = self._load()
        data[self._key(repo, pr_number)] = head_sha
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
