"""Stage 2 — parse & route (SPEC section 6, step 2).

Pure, deterministic code. Parses a unified diff into per-file hunks, classifies
each file by type and risk, and assigns the lenses + model role that should run
on it. No model calls.

File-type taxonomy (SPEC 6.2): ``docs``, ``test``, ``migration``, ``frontend``,
``infra``, ``security``-sensitive, ``lockfile``/``generated``, otherwise plain
``code``. Security-sensitive files always keep the ``security`` lens and are
marked high-risk so the verifier can do recall-recovery on them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from openrabbit.pipeline.gate import is_ignorable_file

# git diff section header: "diff --git a/<a> b/<b>".
_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$")
# Hunk header: "@@ -l,s +l,s @@ optional".
_HUNK_RE = re.compile(r"^@@ .* @@")

# Path-classification signals.
_DOC_SUFFIXES = (".md", ".mdx", ".rst", ".txt", ".adoc")
_TEST_SIGNALS = ("/tests/", "/test/", "test_", "_test.", ".spec.", ".test.")
_MIGRATION_SIGNALS = ("/migrations/", "/migrate/", "alembic/")
_FRONTEND_SUFFIXES = (".tsx", ".jsx", ".vue", ".svelte", ".css", ".scss", ".html")
_INFRA_SIGNALS = (
    "dockerfile",
    "docker-compose",
    ".tf",
    ".yaml",
    ".yml",
    "/.github/",
    "/k8s/",
    "/helm/",
    "terraform",
)
# Security-sensitive path/name signals → always run the security lens.
_SECURITY_SIGNALS = (
    "/auth",
    "/api/",
    "/security",
    "login",
    "password",
    "token",
    "crypto",
    "secret",
    "/admin",
    "session",
    "permission",
)

DEFAULT_MODEL_ROLE = "finder"


@dataclass(frozen=True)
class Hunk:
    """One ``@@ ... @@`` hunk of a file's diff."""

    header: str
    text: str  # the hunk header + body lines


@dataclass(frozen=True)
class FilePlan:
    """The routing plan for a single changed file."""

    path: str
    file_type: str  # docs|test|migration|frontend|infra|lockfile|generated|code
    risk: str  # high|medium|low
    lenses: list[str]
    model_role: str
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def diff_text(self) -> str:
        return "\n".join(h.text for h in self.hunks)


@dataclass(frozen=True)
class RoutePlan:
    """The full routing plan for a diff."""

    files: list[FilePlan]

    @property
    def reviewable_files(self) -> list[FilePlan]:
        return [f for f in self.files if f.lenses]


# --------------------------------------------------------------------------- #
# classification (pure)                                                         #
# --------------------------------------------------------------------------- #
def classify_file_type(path: str) -> str:
    """Classify a path into the SPEC 6.2 file-type taxonomy."""
    lower = path.lower()
    base = lower.rsplit("/", 1)[-1]

    if is_ignorable_file(path):
        # Lockfiles vs other generated artifacts (gate already skips both).
        if base.endswith(".lock") or base in (
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "go.sum",
            "uv.lock",
        ):
            return "lockfile"
        return "generated"
    if any(lower.endswith(suf) for suf in _DOC_SUFFIXES):
        return "docs"
    if any(sig in f"/{lower}" or sig in base for sig in _TEST_SIGNALS):
        return "test"
    if any(sig in f"/{lower}" for sig in _MIGRATION_SIGNALS):
        return "migration"
    if any(lower.endswith(suf) for suf in _FRONTEND_SUFFIXES):
        return "frontend"
    if any(sig in lower for sig in _INFRA_SIGNALS):
        return "infra"
    return "code"


def is_security_sensitive(path: str) -> bool:
    """True if a path looks security-sensitive (auth/api/secrets/...)."""
    lower = f"/{path.lower()}"
    return any(sig in lower for sig in _SECURITY_SIGNALS)


def _assess_risk(file_type: str, security_sensitive: bool) -> str:
    if security_sensitive or file_type in ("migration", "infra"):
        return "high"
    if file_type in ("code", "frontend"):
        return "medium"
    return "low"


def _assign_lenses(file_type: str, security_sensitive: bool, available: list[str]) -> list[str]:
    """Pick which configured lenses run on this file.

    Docs/lockfile/generated files get nothing. Tests get correctness +
    maintainability only. Everything else gets all configured lenses, with the
    security lens force-kept for security-sensitive paths.
    """
    if file_type in ("docs", "lockfile", "generated"):
        return []

    avail = list(available)
    if file_type == "test":
        chosen = [l for l in avail if l in ("correctness", "tests", "maintainability")]
    elif file_type in ("frontend", "infra", "migration"):
        chosen = list(avail)
    else:  # code
        chosen = list(avail)

    if security_sensitive and "security" in avail and "security" not in chosen:
        chosen.append("security")
    # Preserve the configured order.
    order = {l: i for i, l in enumerate(avail)}
    return sorted(set(chosen), key=lambda l: order.get(l, len(order)))


def _model_role_for(risk: str) -> str:
    # Phase 0: a single finder role does the broad pass; the verifier role is
    # applied later in the spine. High-risk files could be promoted here.
    return DEFAULT_MODEL_ROLE


# --------------------------------------------------------------------------- #
# diff parsing (pure)                                                          #
# --------------------------------------------------------------------------- #
def _parse_file_sections(diff: str) -> list[tuple[str, list[str]]]:
    """Split a unified diff into ``(path, [lines])`` sections, one per file."""
    sections: list[tuple[str, list[str]]] = []
    current_path: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_path is not None:
            sections.append((current_path, current_lines))

    for line in diff.splitlines():
        git = _DIFF_GIT_RE.match(line)
        if git:
            flush()
            current_path = git.group("b")
            current_lines = []
            continue
        if current_path is not None:
            current_lines.append(line)
    flush()
    return sections


def _parse_hunks(lines: list[str]) -> list[Hunk]:
    hunks: list[Hunk] = []
    header: Optional[str] = None
    body: list[str] = []

    def flush() -> None:
        if header is not None:
            text = "\n".join([header, *body])
            hunks.append(Hunk(header=header, text=text))

    for line in lines:
        if _HUNK_RE.match(line):
            flush()
            header = line
            body = []
        elif header is not None:
            body.append(line)
    flush()
    return hunks


def route_diff(diff: str, *, lenses: list[str]) -> RoutePlan:
    """Parse + classify + route a unified diff into a :class:`RoutePlan`.

    ``lenses`` is the configured lens list (``config.review.lenses``).
    """
    files: list[FilePlan] = []
    for path, lines in _parse_file_sections(diff):
        file_type = classify_file_type(path)
        sec = is_security_sensitive(path)
        risk = _assess_risk(file_type, sec)
        assigned = _assign_lenses(file_type, sec, lenses)
        files.append(
            FilePlan(
                path=path,
                file_type=file_type,
                risk=risk,
                lenses=assigned,
                model_role=_model_role_for(risk),
                hunks=_parse_hunks(lines),
            )
        )
    return RoutePlan(files=files)
