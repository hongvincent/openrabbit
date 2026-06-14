"""Loader for portable review *lenses* (SPEC 3 + 8.3).

A lens is a single source of review intelligence stored as an agentskills.io
``SKILL.md`` file: a YAML frontmatter block (``name``, ``description``,
``allowed-tools``) followed by a markdown body. The body is the **system
prompt** the harness hands to any Bedrock model to run that lens's report-all
finder pass; Claude Code / Codex / Gemini can also run the same file directly
for local review. One ``SKILL.md`` => one prompt => every runtime.

This module is a thin parser/loader only — it never calls a model and never
touches the network. ``pyyaml`` is imported lazily inside the parser so the
module imports with zero third-party deps.

Public API:

* :class:`Lens` — ``(name, description, system_prompt, allowed_tools)``.
* :func:`parse_skill` — parse one ``SKILL.md`` file into a :class:`Lens`.
* :func:`load_lenses` — discover every ``<dir>/<lens>/SKILL.md`` under a
  directory and return ``{name: Lens}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]

# A leading YAML frontmatter block delimited by ``---`` fences. Tolerates an
# optional leading BOM / blank line and CRLF endings.
_FRONTMATTER_RE = re.compile(
    r"\A﻿?\s*---[ \t]*\r?\n(?P<frontmatter>.*?)\r?\n---[ \t]*\r?\n?(?P<body>.*)\Z",
    re.DOTALL,
)


class LensError(ValueError):
    """Raised when a lens ``SKILL.md`` is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class Lens:
    """A parsed review lens.

    Attributes
    ----------
    name:
        Stable lens identifier (e.g. ``"correctness"``). Used as the dict key
        in :func:`load_lenses`.
    description:
        Terse third-person trigger description from the frontmatter.
    system_prompt:
        The markdown body, used verbatim as the model system prompt.
    allowed_tools:
        Least-privilege tool allowlist declared in the frontmatter (may be
        empty).
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)


def parse_skill(path: PathLike) -> Lens:
    """Parse a single ``SKILL.md`` into a :class:`Lens`.

    The file must start with a YAML frontmatter block (``---`` fenced). The
    remaining text becomes :attr:`Lens.system_prompt` (stripped). Missing
    optional fields degrade gracefully: ``description`` -> ``""``,
    ``allowed-tools`` -> ``[]``, and a missing ``name`` falls back to the
    containing directory's name.

    Raises :class:`LensError` for a missing file, absent/invalid frontmatter,
    or malformed YAML.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise LensError(f"cannot read lens skill at {p}: {exc}") from exc

    match = _FRONTMATTER_RE.match(raw)
    if match is None:
        raise LensError(
            f"{p} is missing a YAML frontmatter block "
            "(expected a leading '---' fenced section)"
        )

    meta = _parse_frontmatter(match.group("frontmatter"), p)
    body = match.group("body").strip()

    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        # Fall back to the lens directory name (``.../correctness/SKILL.md``).
        name = p.parent.name
    else:
        name = name.strip()

    description = meta.get("description")
    description = description.strip() if isinstance(description, str) else ""

    allowed_tools = _coerce_allowed_tools(
        meta.get("allowed-tools", meta.get("allowed_tools"))
    )

    return Lens(
        name=name,
        description=description,
        system_prompt=body,
        allowed_tools=allowed_tools,
    )


def load_lenses(skills_dir: PathLike) -> dict[str, Lens]:
    """Discover and load every lens under ``skills_dir``.

    Each immediate subdirectory containing a ``SKILL.md`` is treated as a lens.
    Returns a mapping keyed by each lens's frontmatter ``name`` (not the
    directory name). Subdirectories without a ``SKILL.md`` are ignored.

    Raises :class:`LensError` if ``skills_dir`` does not exist or is not a
    directory.
    """
    root = Path(skills_dir)
    if not root.is_dir():
        raise LensError(f"lens skills directory not found: {root}")

    lenses: dict[str, Lens] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill = child / "SKILL.md"
        if not skill.is_file():
            continue
        lens = parse_skill(skill)
        lenses[lens.name] = lens
    return lenses


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #
def _parse_frontmatter(text: str, path: Path) -> dict[str, Any]:
    import yaml  # local import: pure-python parser, no network

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LensError(f"malformed YAML frontmatter in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise LensError(
            f"frontmatter in {path} must be a mapping, got {type(data).__name__}"
        )
    return data


def _coerce_allowed_tools(value: Any) -> list[str]:
    """Normalize ``allowed-tools`` to a list of stripped, non-empty strings.

    Accepts a YAML list (``[Read, Grep]``) or a comma-separated string
    (``"Read, Grep"``); anything else (or missing) yields ``[]``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [tok.strip() for tok in value.split(",") if tok.strip()]
    if isinstance(value, (list, tuple)):
        return [str(tok).strip() for tok in value if str(tok).strip()]
    return []
