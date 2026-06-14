"""Real enclosing-context fetcher (SPEC section 6, step 3).

:class:`GitEnclosingFetcher` implements the :data:`EnclosingFetcher` protocol
that :mod:`openrabbit.pipeline.context` exposes. For each changed hunk it pulls,
**lazily and best-effort**, a bounded slice of surrounding code so the finder
lenses see the change in its structural context:

* a window of ``N`` lines around the hunk (from a git ``ref`` or the worktree),
* the enclosing function/class (nearest preceding ``def``/``class`` at a lower
  indent — Python-aware, with a language-agnostic brace/indent fallback),
* optionally a short ``git log`` history block for the file.

Everything is exception-safe: any failure (missing file, ``git`` absent,
unparseable header) degrades to ``None`` rather than raising, so the pipeline
stays robust offline. Output is capped (``max_lines``) to protect the prompt
budget. ``subprocess`` is imported at module scope but only *invoked* inside
methods; unit tests drive it against a temporary local git repo (no network).

The diff/PR text the fetcher operates over is UNTRUSTED data; this module only
*reads* — it never executes anything from the diff and holds no write access.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from openrabbit.pipeline.route import FilePlan, Hunk

# Hunk header: "@@ -oldStart,oldLen +newStart,newLen @@ optional-section".
# The new-side range (``+newStart,newLen``) indexes the post-change file, which
# is what the working tree / target ref contains.
_HUNK_RANGE_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<len>\d+))? @@")

# A line that opens a Python (or similar) scope we care about.
_DEF_RE = re.compile(r"^(?P<indent>\s*)(?:async\s+)?(?:def|class)\b")

# Defensive caps so a pathological repo can never blow the prompt budget.
_DEFAULT_WINDOW = 12
_DEFAULT_MAX_LINES = 80
_MAX_GIT_TIMEOUT_S = 5
_HISTORY_LINES = 5
# Hard cap on how many bytes we ever read for a single file before windowing,
# so an attacker-crafted diff referencing a huge tracked blob cannot force the
# whole file into memory (memory-pressure DoS on the CI runner).
_MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MiB


class GitEnclosingFetcher:
    """Best-effort enclosing-context fetcher backed by git + file reads.

    Instances are callables matching ``EnclosingFetcher`` so they can be passed
    straight to :func:`openrabbit.pipeline.context.build_file_message` /
    wired in via :func:`gather_enclosing_context`.

    Parameters
    ----------
    repo_root:
        Root of the git working tree (defaults to the current directory).
    ref:
        If set (e.g. ``"HEAD"`` or a SHA), read file content from that git ref
        via ``git show`` instead of the working tree.
    window:
        Lines of context to include above/below a hunk for non-structured files
        or when no enclosing scope is found.
    max_lines:
        Hard cap on the total number of output lines (prompt-budget guard).
    include_history:
        When true, append a short ``git log`` block for the file.
    """

    def __init__(
        self,
        repo_root: Optional[Path | str] = None,
        *,
        ref: Optional[str] = None,
        window: int = _DEFAULT_WINDOW,
        max_lines: int = _DEFAULT_MAX_LINES,
        include_history: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.ref = ref
        self.window = max(1, int(window))
        self.max_lines = max(1, int(max_lines))
        self.include_history = include_history

    # ------------------------------------------------------------------ #
    # public protocol entry point                                         #
    # ------------------------------------------------------------------ #
    def __call__(self, file_plan: FilePlan) -> Optional[str]:
        """Return bounded enclosing context for ``file_plan`` or ``None``."""
        try:
            return self._gather(file_plan)
        except Exception:
            # Best-effort contract: never propagate failures into the pipeline.
            return None

    # ------------------------------------------------------------------ #
    # core (kept private; the public surface is __call__)                  #
    # ------------------------------------------------------------------ #
    def _gather(self, file_plan: FilePlan) -> Optional[str]:
        if not file_plan.hunks:
            return None
        if not self._path_is_safe(file_plan.path):
            return None

        lines = self._read_file_lines(file_plan.path)
        if not lines:
            return None

        blocks: list[str] = []
        seen: set[tuple[int, int]] = set()
        for hunk in file_plan.hunks:
            span = self._extract_block(lines, hunk)
            if span is None:
                continue
            start, end = span
            if (start, end) in seen:
                continue
            seen.add((start, end))
            blocks.append(self._render_block(file_plan.path, lines, start, end))

        if not blocks:
            return None

        if self.include_history:
            history = self._git_history(file_plan.path)
            if history:
                blocks.append(history)

        return self._cap("\n".join(blocks))

    # ------------------------------------------------------------------ #
    # block extraction (pure given file lines)                            #
    # ------------------------------------------------------------------ #
    def _extract_block(
        self, lines: list[str], hunk: Hunk
    ) -> Optional[tuple[int, int]]:
        """Compute a (start, end) 0-based inclusive line span for one hunk."""
        rng = _HUNK_RANGE_RE.match(hunk.header)
        if rng is None:
            return None
        start_1 = int(rng.group("start"))
        length = int(rng.group("len")) if rng.group("len") else 1
        # Convert to 0-based, clamp into range. A length-0 hunk (pure deletion)
        # still anchors a useful window at the deletion point.
        hunk_start = max(0, start_1 - 1)
        hunk_end = min(len(lines) - 1, max(hunk_start, start_1 - 1 + max(length, 1) - 1))

        enclosing = self._find_enclosing_scope(lines, hunk_start)
        if enclosing is not None:
            scope_start, scope_end = enclosing
            # Expand to also cover the hunk window if the hunk extends past it.
            start = min(scope_start, max(0, hunk_start - 1))
            end = max(scope_end, hunk_end)
            return start, end

        # No structural scope: fall back to a plain window around the hunk.
        start = max(0, hunk_start - self.window)
        end = min(len(lines) - 1, hunk_end + self.window)
        return start, end

    def _find_enclosing_scope(
        self, lines: list[str], anchor: int
    ) -> Optional[tuple[int, int]]:
        """Find the nearest preceding ``def``/``class`` at a lower indent.

        Returns the (start, end) inclusive span of that scope's body, where the
        body ends at the last line indented deeper than the opening line.
        """
        if anchor >= len(lines):
            anchor = len(lines) - 1
        # A blank or whitespace-only anchor (e.g. the gap between two methods)
        # has indent 0, which would widen the match to the whole enclosing class.
        # Advance to the nearest following non-blank line so both the anchor
        # indent and the backward def-search reflect the actual code being
        # changed, resolving the tighter method scope rather than the class.
        if not lines[anchor].strip():
            probe = anchor
            while probe < len(lines) and not lines[probe].strip():
                probe += 1
            if probe < len(lines):
                anchor = probe
        anchor_indent = _indent_of(lines[anchor])

        best: Optional[tuple[int, int]] = None
        for i in range(anchor, -1, -1):
            m = _DEF_RE.match(lines[i])
            if not m:
                continue
            def_indent = len(m.group("indent").expandtabs())
            # Only accept a header whose indent is shallower than the anchor
            # line (or equal when the anchor itself is the def line).
            if i != anchor and def_indent > anchor_indent:
                continue
            end = self._scope_end(lines, i, def_indent)
            if end < anchor:
                # The scope closed before the change — not actually enclosing.
                continue
            best = (i, end)
            break
        return best

    @staticmethod
    def _scope_end(lines: list[str], def_line: int, def_indent: int) -> int:
        """Last line index belonging to the scope opened at ``def_line``."""
        end = def_line
        for j in range(def_line + 1, len(lines)):
            stripped = lines[j].strip()
            if not stripped:
                # Blank lines belong to the scope but don't extend it on their own.
                continue
            if _indent_of(lines[j]) <= def_indent:
                break
            end = j
        return end

    def _render_block(
        self, path: str, lines: list[str], start: int, end: int
    ) -> str:
        header = f"# {path}:{start + 1}-{end + 1} (enclosing context)"
        body = lines[start : end + 1]
        return "\n".join([header, *body])

    # ------------------------------------------------------------------ #
    # file/content access (lazy git/IO)                                   #
    # ------------------------------------------------------------------ #
    def _read_file_lines(self, path: str) -> Optional[list[str]]:
        """Read file content from the configured ref, else the working tree."""
        if self.ref:
            content = self._git_show(path)
            if content is not None:
                return content.splitlines()
        return self._read_worktree(path)

    def _read_worktree(self, path: str) -> Optional[list[str]]:
        try:
            # Resolve symlinks and assert the target stays inside repo_root. The
            # path comes verbatim from the UNTRUSTED diff header, so an in-repo
            # symlink pointing outside the repo must not leak out-of-tree files.
            root = self.repo_root.resolve()
            target = (self.repo_root / path).resolve()
            if target != root and root not in target.parents:
                return None
            if not target.is_file():
                return None
            # Bound the read: never slurp an arbitrarily large tracked blob.
            if target.stat().st_size > _MAX_READ_BYTES:
                return None
            return target.read_text(errors="replace").splitlines()
        except (OSError, ValueError):
            return None

    def _git_show(self, path: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "show", f"{self.ref}:{path}"],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
                timeout=_MAX_GIT_TIMEOUT_S,
            )
            out = proc.stdout
            # Bound the buffered output too (git show is timeout- but not
            # size-bounded); a huge committed blob must not blow memory.
            if len(out) > _MAX_READ_BYTES:
                return None
            return out
        except (OSError, subprocess.SubprocessError):
            return None

    def _git_history(self, path: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                [
                    "git",
                    "log",
                    f"-{_HISTORY_LINES}",
                    "--pretty=format:%h %s",
                    "--",
                    path,
                ],
                cwd=str(self.repo_root),
                check=True,
                capture_output=True,
                text=True,
                timeout=_MAX_GIT_TIMEOUT_S,
            )
            out = proc.stdout.strip()
            if not out:
                return None
            return "# recent history:\n" + out
        except (OSError, subprocess.SubprocessError):
            return None

    # ------------------------------------------------------------------ #
    # safety / bounds                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _path_is_safe(path: str) -> bool:
        """Reject absolute paths and parent-dir traversal (defense in depth)."""
        if not path:
            return False
        p = Path(path)
        if p.is_absolute():
            return False
        return ".." not in p.parts

    def _cap(self, text: str) -> str:
        out_lines = text.splitlines()
        if len(out_lines) <= self.max_lines:
            return text
        kept = out_lines[: self.max_lines - 1]
        kept.append(f"# ... ({len(out_lines) - len(kept)} more lines truncated)")
        return "\n".join(kept)


def _indent_of(line: str) -> int:
    """Leading-whitespace width (tabs expanded) of ``line``."""
    return len(line) - len(line.expandtabs().lstrip(" "))
