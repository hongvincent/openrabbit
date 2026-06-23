"""Golden-set builder: a labeled corpus from a local git repo's history.

SPEC section 10: the offline golden set is built from the team's *own* merged
PRs plus revert / hotfix / incident-linked commits (representative, not
synthetic-injected). This module mines a local git repository's log via
``subprocess`` and extracts ``(diff, known_bug?)`` samples, then serializes them
to JSONL.

Design constraints (per task rules):
- ``subprocess`` / git are invoked LAZILY inside functions, never at import time,
  so importing this module needs zero external tooling.
- Heuristics operate purely on commit messages + ``git log``/``git show``; no
  network access. Tests drive a tiny real local git repo fixture.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

# Vocabulary kept consistent with openrabbit.findings.CATEGORIES so labels can be
# scored per-category by the scorecard.
_CATEGORIES = ("correctness", "security", "performance", "tests", "maintainability")

# Commit-message heuristics. Ordered: the first matching kind wins.
_REVERT_RE = re.compile(r"\brevert(s|ed|ing)?\b", re.IGNORECASE)
_HOTFIX_RE = re.compile(r"\bhot[\s-]?fix(es|ed)?\b", re.IGNORECASE)
_FIX_RE = re.compile(r"\b(fix(es|ed|ing)?|bug ?fix|patch|repair)\b", re.IGNORECASE)

# Trivial-fix exclusions (finding 3): a "fix" whose subject is purely doc /
# format / test / cosmetic carries NO real runtime defect, so grading recall
# against it pollutes the bug class. Such commits are demoted to clean. Checked
# against the subject line only (a body may legitimately mention these words).
_TRIVIAL_FIX_RE = re.compile(
    r"\b(typo|lint|format(ting|ter)?|reformat|whitespace|docs?|"
    r"comment|readme|rename(d|s)?|spelling|style|prettier|black|isort|"
    r"gofmt|test(s|ing)?|spec(s)?|changelog|version bump|bump version)\b",
    re.IGNORECASE,
)

# Bug-category keyword hints, checked in order (security before correctness so a
# "fix: SQL injection" maps to security rather than generic correctness).
_CATEGORY_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "security",
        re.compile(
            r"\b(security|injection|xss|csrf|auth(entication|orization)?|"
            r"secret|credential|vuln|cve|sanitiz|escap)\w*",
            re.IGNORECASE,
        ),
    ),
    (
        "performance",
        re.compile(
            r"\b(perf(ormance)?|slow|latency|memory leak|n\+1|throughput|"
            r"timeout|deadlock)\w*",
            re.IGNORECASE,
        ),
    ),
    (
        "tests",
        re.compile(r"\b(test|flaky|coverage|assertion)\w*", re.IGNORECASE),
    ),
    (
        "maintainability",
        re.compile(
            r"\b(refactor|cleanup|lint|typo|rename|dead code)\w*", re.IGNORECASE
        ),
    ),
)


@dataclass(frozen=True)
class CommitLabel:
    """A heuristic label derived from a single commit message."""

    known_bug: bool
    source: str  # "revert" | "hotfix" | "fix" | "clean"
    bug_category: str  # one of _CATEGORIES (defaults to "correctness")


@dataclass
class GoldenSample:
    """One labeled corpus sample: a diff plus whether it is a known bug.

    ``source`` records the heuristic that labeled it (revert/hotfix/fix/clean).
    ``bug_category`` is the best-guess lens category for per-category scoring.
    """

    sample_id: str
    repo: str
    commit: str
    diff: str
    known_bug: bool
    bug_category: str
    source: str
    message: str
    #: Provenance: where the real defect lives (e.g. ``"auth.py"`` or
    #: ``"auth.py:42"``). Derived from the files the fix touched. Used to judge a
    #: finding BLIND to ``known_bug`` (finding 4): a "match" must hit this
    #: location rather than rely on a leaked yes/no label. Empty when unknown.
    defect_location: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase, JSON-serializable dict."""
        return {
            "sampleId": self.sample_id,
            "repo": self.repo,
            "commit": self.commit,
            "diff": self.diff,
            "knownBug": self.known_bug,
            "bugCategory": self.bug_category,
            "source": self.source,
            "message": self.message,
            "defectLocation": self.defect_location,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenSample:
        """Build a :class:`GoldenSample` from a camelCase dict (ignores extras)."""
        return cls(
            sample_id=data["sampleId"],
            repo=data["repo"],
            commit=data["commit"],
            diff=data["diff"],
            known_bug=bool(data["knownBug"]),
            bug_category=data.get("bugCategory", "correctness"),
            source=data["source"],
            message=data["message"],
            defect_location=data.get("defectLocation", ""),
        )


def classify_commit(message: str) -> CommitLabel:
    """Heuristically label a commit from its message.

    Revert / hotfix / fix commits are treated as evidence that the *preceding*
    state contained a real bug (``known_bug=True``). Everything else is treated
    as a clean change.
    """
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    category = _guess_category(message)

    if _REVERT_RE.search(message):
        return CommitLabel(known_bug=True, source="revert", bug_category=category)
    if _HOTFIX_RE.search(message):
        return CommitLabel(known_bug=True, source="hotfix", bug_category=category)
    if _FIX_RE.search(first_line) or _FIX_RE.search(message):
        # Finding 3: a doc/format/test/cosmetic "fix" is not a real runtime bug.
        # Demote it to clean so it does not pollute the recall denominator. The
        # subject line is the signal (a body may mention 'tests' incidentally).
        if _TRIVIAL_FIX_RE.search(first_line):
            return CommitLabel(known_bug=False, source="clean", bug_category=category)
        return CommitLabel(known_bug=True, source="fix", bug_category=category)
    return CommitLabel(known_bug=False, source="clean", bug_category=category)


def _guess_category(message: str) -> str:
    for category, pattern in _CATEGORY_HINTS:
        if pattern.search(message):
            return category
    return "correctness"


def build_golden_set(
    repo: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    include_clean: bool = False,
    max_commits: int = 1000,
) -> list[GoldenSample]:
    """Mine a local git repo's history into labeled :class:`GoldenSample` objects.

    By default only bug-labeled commits (revert/hotfix/fix) are returned, since
    those carry a ground-truth ``known_bug=True``. Pass ``include_clean=True`` to
    also emit clean commits as negative samples.

    Raises if ``repo`` is not a git repository (``git`` errors propagate).
    """
    repo_path = Path(repo)
    repo_name = repo_path.name or str(repo_path)

    samples: list[GoldenSample] = []
    for commit_sha, message in _iter_commits(repo_path, max_commits):
        label = classify_commit(message)
        if not label.known_bug and not include_clean:
            continue
        # Finding 1 (CRITICAL): a fix/revert/hotfix commit's forward diff is the
        # ALREADY-CORRECTED code; reviewing it would grade patched code. For a
        # known bug we present the PRE-FIX (buggy) state via the REVERSE diff
        # (``git show -R``) so the sample actually contains the defect. Clean
        # negatives keep their forward diff (there is no bug to un-fix).
        reverse = label.known_bug
        diff = _commit_diff(repo_path, commit_sha, reverse=reverse)
        if not diff.strip():
            continue
        samples.append(
            GoldenSample(
                sample_id=f"{repo_name}@{commit_sha[:12]}",
                repo=repo_name,
                commit=commit_sha,
                diff=diff,
                known_bug=label.known_bug,
                bug_category=label.bug_category,
                source=label.source,
                message=message.strip().splitlines()[0] if message.strip() else "",
                defect_location=(_defect_location(diff) if label.known_bug else ""),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def write_jsonl(samples: Iterable[GoldenSample], path: Union[str, Path]) -> int:
    """Serialize samples to a JSONL file (one JSON object per line). Returns count."""
    import json  # stdlib; local keeps the module import surface tiny

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def iter_jsonl(path: Union[str, Path]) -> Iterator[GoldenSample]:
    """Stream :class:`GoldenSample` objects from a JSONL file."""
    import json

    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield GoldenSample.from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# git plumbing (subprocess imported lazily)                                    #
# --------------------------------------------------------------------------- #
_LOG_SEP = "\x1e"  # ASCII record separator between sha and message


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stdout (lazy subprocess import)."""
    import subprocess  # lazy: keeps module import free of process spawning

    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _has_parent(repo: Path, commit_sha: str) -> bool:
    """Return True if ``commit_sha`` has a first parent (i.e. is not a root commit)."""
    import subprocess  # lazy: matches _git's import discipline

    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{commit_sha}^"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _iter_commits(repo: Path, max_commits: int) -> list[tuple[str, str]]:
    """Return ``(sha, full_message)`` pairs newest-first, bounded by ``max_commits``."""
    # %H sha, then RS, then raw body+subject; commits separated by NUL.
    fmt = f"%H{_LOG_SEP}%B"
    raw = _git(
        repo,
        "log",
        f"--max-count={max_commits}",
        "-z",
        f"--format={fmt}",
    )
    commits: list[tuple[str, str]] = []
    for record in raw.split("\x00"):
        if not record.strip():
            continue
        sha, _, message = record.partition(_LOG_SEP)
        commits.append((sha.strip(), message))
    return commits


def _commit_diff(repo: Path, commit_sha: str, *, reverse: bool = False) -> str:
    """Return the unified diff a commit introduced (vs its first parent).

    When ``reverse`` is True, return the REVERSE diff that would UNDO the commit
    via ``git diff <sha> <sha>^`` (commit → first parent). For a bug-fix commit
    this presents the PRE-FIX (buggy) code as the side under review (finding 1):
    the buggy lines appear as additions (``+++ b/<path>``) and the corrected
    lines as removals. Crucially this keeps the standard forward ``a/`` (old) →
    ``b/`` (new) prefixes so the diff router still enumerates the file, unlike
    ``git show -R`` which swaps the prefixes.
    """
    if reverse:
        if not _has_parent(repo, commit_sha):
            # A root commit has no pre-fix state to review; skip it (the caller
            # drops empty diffs). Reversing the whole initial import is not a
            # meaningful "bug under review".
            return ""
        return _git(
            repo,
            "diff",
            "--no-color",
            commit_sha,
            f"{commit_sha}^",
        )
    return _git(
        repo,
        "show",
        "--no-color",
        "--format=",  # suppress the commit header; we already have the message
        commit_sha,
    )


def _defect_location(diff: str) -> str:
    """Best-effort defect location from a unified diff's first changed file.

    Returns the first ``+++ b/<path>`` target (the file under review). Pure
    string parsing — no git, no network. Empty when no file header is present.
    """
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            return raw[len("+++ b/") :].strip()
        if raw.startswith("+++ ") and not raw.startswith("+++ /dev/null"):
            return raw[len("+++ ") :].strip()
    return ""
