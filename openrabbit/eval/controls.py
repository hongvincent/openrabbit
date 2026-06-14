"""Negative controls: clean / no-op PR samples (SPEC section 10).

Negative controls measure the worst failure mode of an LLM reviewer: inventing
problems on unchanged code. A control is, by construction, a diff that contains
no real defect (``known_bug=False``); therefore *any* finding the reviewer emits
on a control counts as a false positive in the scorecard.

Controls reuse :class:`~openrabbit.eval.golden_set.GoldenSample` so they flow
through the exact same judge + scorecard path as real golden samples. This module
is pure stdlib — no git, no network.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openrabbit.eval.golden_set import GoldenSample

# A diff hunk line is added ('+'), removed ('-'), or context (' '). We compare the
# stripped payload of each +/- pair to decide whether a change is a real edit or a
# cosmetic no-op (whitespace, blank line, comment reflow).
_HUNK_HEADER_RE = re.compile(r"^@@ .* @@")
_FILE_META_RE = re.compile(r"^(diff --git|index |--- |\+\+\+ |new file|deleted file)")


@dataclass
class ControlSample(GoldenSample):
    """A clean / no-op sample. ``known_bug`` is always ``False``.

    Adds ``control_kind`` (e.g. ``"whitespace"``) on top of the base
    :class:`GoldenSample` fields so controls can be filtered/reported distinctly.
    """

    control_kind: str = "noop"

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["controlKind"] = self.control_kind
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlSample:
        return cls(
            sample_id=data["sampleId"],
            repo=data["repo"],
            commit=data["commit"],
            diff=data["diff"],
            known_bug=False,
            bug_category=data.get("bugCategory", "correctness"),
            source=data.get("source", "control"),
            message=data.get("message", ""),
            control_kind=data.get("controlKind", "noop"),
        )


def make_whitespace_control(path: str, source: str) -> ControlSample:
    """Build a whitespace-only no-op control diff for ``source`` (file ``path``).

    The synthesized diff re-emits the first content line with trailing whitespace
    appended — a semantically-empty change. A reviewer that flags it is inventing
    a problem.
    """
    lines = source.splitlines()
    first = lines[0] if lines else ""
    diff = f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,1 @@\n-{first}\n+{first}  \n"
    return ControlSample(
        sample_id=f"control:{path}",
        repo="control",
        commit="0" * 40,
        diff=diff,
        known_bug=False,
        bug_category="correctness",
        source="control",
        message=f"no-op whitespace change to {path}",
        control_kind="whitespace",
    )


def generate_noop_controls(files: Mapping[str, str], count: int) -> list[ControlSample]:
    """Generate up to ``count`` whitespace no-op controls from ``files``.

    ``files`` maps path -> file contents. One control is produced per file, in
    sorted-path order for determinism, capped at the number of files available.
    """
    if count <= 0 or not files:
        return []
    controls: list[ControlSample] = []
    for path in sorted(files):
        if len(controls) >= count:
            break
        controls.append(make_whitespace_control(path, files[path]))
    return controls


def is_noop_diff(diff: str) -> bool:
    """Return ``True`` if ``diff`` makes no semantic change.

    A diff is a no-op when, ignoring metadata/hunk headers, every added line has
    a matching removed line with identical whitespace-stripped content (and vice
    versa), and any unmatched added lines are blank. This catches trailing-space
    edits, blank-line insertions, and comment reflows.
    """
    added: list[str] = []
    removed: list[str] = []
    for raw in diff.splitlines():
        if not raw or _HUNK_HEADER_RE.match(raw) or _FILE_META_RE.match(raw):
            continue
        marker, payload = raw[0], raw[1:]
        if marker == "+":
            added.append(payload)
        elif marker == "-":
            removed.append(payload)
        # context lines (' ') are unchanged; ignore.

    norm_added = [a.strip() for a in added]
    norm_removed = [r.strip() for r in removed]

    # Remove matched (whitespace-insensitive) pairs.
    remaining_removed = list(norm_removed)
    leftover_added: list[str] = []
    for a in norm_added:
        if a in remaining_removed:
            remaining_removed.remove(a)
        else:
            leftover_added.append(a)

    # Any leftover removed line that wasn't whitespace-matched => real change.
    if any(r for r in remaining_removed):
        return False
    # No-op iff there is also no leftover added line with actual content.
    return not any(a for a in leftover_added)
