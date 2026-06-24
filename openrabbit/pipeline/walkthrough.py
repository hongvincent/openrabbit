"""Walkthrough enrichment (SPEC section 6, step 6 — parity target).

:func:`build_walkthrough` renders the body of the sticky walkthrough comment:

1. a 2-3 sentence **high-level summary** of the change,
2. a **grouped changed-files table** — related files (by directory/feature) are
   collapsed into one row, each with a plain-language description + a change
   purpose inferred from the path/diff via deterministic heuristics (no LLM in
   Phase 1),
3. a **Mermaid diagram** — emitted ONLY when the change touches component
   interactions / API / event / async flows (detected heuristically from the
   changed files); omitted otherwise,
4. the **findings summary table** that :mod:`openrabbit.pipeline.emit` already
   produces today (reused, not duplicated).

The output is fully deterministic and bounded: it is a pure function of the
inputs, makes no model/network calls, and caps the changed-files table at
:data:`MAX_TABLE_ROWS` rows with a "+N more" note. PR title/body text is treated
as UNTRUSTED data — markdown control characters are neutralized so it can never
break out of a table cell or inject markup.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from itertools import pairwise
from typing import Any, Optional

from openrabbit.findings import Finding
from openrabbit.pipeline.emit import render_summary_markdown
from openrabbit.pipeline.route import FilePlan

# Boundedness caps. The grouped table never grows past this many rows; overflow
# collapses into a "+N more" note so a huge PR cannot produce an unbounded body.
MAX_TABLE_ROWS = 30
#: Per-group, only the first few filenames are listed inline (the rest become a
#: "+N more" suffix) so a directory with hundreds of files stays readable.
MAX_FILES_PER_GROUP = 6
#: Hard cap on the high-level summary length (defensive, untrusted PR title).
_MAX_TITLE_CHARS = 160


# --------------------------------------------------------------------------- #
# localized static labels (Feature 1 — response_language)                       #
# --------------------------------------------------------------------------- #
#: Static walkthrough labels, keyed by canonical language code. ``en`` is the
#: SSOT default and must stay pixel-identical to today (existing tests pin these
#: English strings). A non-``en`` code renders the localized labels; an unknown
#: code falls back to ``en`` so a bad value degrades to English rather than
#: crashing. Keys map 1:1 to the section headings + table columns rendered below.
_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "walkthrough": "Walkthrough",
        "changed_files": "Changed files",
        "group": "Group",
        "files": "Files",
        "change_summary": "Change summary",
        "interaction_flow": "Interaction flow",
        "findings": "Findings",
        # Rabbit sign-off (Feature 2). Casual + friendly, not stiff/spammy.
        "sign_off": "openrabbit hopped through your changes — ping me anytime!",
    },
    "ko": {
        # "둘러보기" is the natural Korean for a walkthrough; the prior
        # transliteration "워크스루" read awkwardly.
        "walkthrough": "둘러보기",
        "changed_files": "변경된 파일",
        "group": "그룹",
        "files": "파일",
        "change_summary": "변경 요약",
        "interaction_flow": "상호작용 흐름",
        "findings": "발견 사항",
        # Casual + friendly (the prior "검토했습니다 — 들러 주세요" was too stiff).
        "sign_off": "openrabbit가 쓱 둘러봤어요. 또 불러줘요!",
    },
}

# --------------------------------------------------------------------------- #
# rabbit persona / branding (Feature 2)                                         #
# --------------------------------------------------------------------------- #
#: The single branding marker. Kept in one place so the header marker, the
#: optional findings marker, and the sign-off all stay consistent and tasteful.
_RABBIT = "🐰"

#: Small ASCII-art rabbit shown at the very TOP of the walkthrough (persona ON
#: only). Wrapped in a fenced code block so GitHub renders it monospace/aligned
#: (proportional fonts would otherwise skew the art). Same art for every
#: language. Kept tiny and tasteful (SPEC principle 1: low-noise).
_ASCII_RABBIT = "```\n(\\__/)\n(='.'=)\n(\")_(\")\n```"

#: Attribution line shown right after the sign-off (persona ON only). Italic so
#: it renders small/secondary. Same for every language.
_ATTRIBUTION = "_made by Subin Hong_"


def _labels(response_language: str) -> dict[str, str]:
    """Return the static-label map for a language, falling back to English."""
    return _LABELS.get(response_language, _LABELS["en"])


# --------------------------------------------------------------------------- #
# untrusted-text hardening                                                      #
# --------------------------------------------------------------------------- #
def _sanitize(text: str, *, limit: Optional[int] = None) -> str:
    """Neutralize markdown/HTML control chars in UNTRUSTED text (SPEC 12).

    Pipes (table-cell delimiter) are escaped; angle brackets (HTML/comment
    injection) and newlines (row injection) are stripped. Optionally truncated.
    """
    cleaned = (
        text.replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r", " ")
        .replace("\n", " ")
    ).strip()
    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "…"
    return cleaned


# --------------------------------------------------------------------------- #
# file grouping (deterministic)                                                 #
# --------------------------------------------------------------------------- #
def _group_key(path: str) -> str:
    """Group key for a path: its parent directory (or ``(root)`` for top-level).

    Grouping by directory collapses related files (same feature/module) into one
    table row, the CodeRabbit-style walkthrough grouping.
    """
    head, sep, _ = path.rpartition("/")
    return head if sep else "(root)"


_TYPE_DESCRIPTIONS = {
    "docs": "Documentation updates",
    "test": "Test changes",
    "migration": "Database migration changes",
    "frontend": "Frontend/UI changes",
    "infra": "Infrastructure / CI configuration changes",
    "lockfile": "Dependency lockfile updates",
    "generated": "Generated artifact updates",
    "code": "Code changes",
}


def _describe_group(plans: list[FilePlan]) -> str:
    """Plain-language description of a group, inferred from its files' types.

    Deterministic: when a group mixes types the dominant (most common, then
    alphabetical for ties) type wins; security-sensitive paths get a hint.
    """
    counts: OrderedDict[str, int] = OrderedDict()
    for p in plans:
        counts[p.file_type] = counts.get(p.file_type, 0) + 1
    dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    desc = _TYPE_DESCRIPTIONS.get(dominant, "Code changes")
    if any(p.risk == "high" for p in plans):
        desc += " (security/risk-sensitive)"
    return desc


def _purpose(plans: list[FilePlan]) -> str:
    """Infer the change purpose for a group from path/diff heuristics.

    Heuristic, bounded, no LLM: looks at whether the hunks predominantly add or
    remove lines and whether interaction signals are present.
    """
    added = removed = 0
    for p in plans:
        for line in p.diff_text.splitlines():
            # The +++/--- guards are belt-and-suspenders: FilePlan.diff_text is the
            # join of HUNK bodies only, so the `--- a/.. / +++ b/..` file headers
            # (consumed by route._parse_file_sections before the first @@) are
            # never present here. They remain to stay robust if that ever changes.
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
    if added and not removed:
        kind = "Adds"
    elif removed and not added:
        kind = "Removes"
    elif added or removed:
        kind = "Updates"
    else:
        kind = "Modifies"
    # Strip the parenthetical risk hint for the purpose phrasing; keep it terse.
    noun = _describe_group(plans).split(" (")[0].lower()
    return f"{kind} {noun}"


def _filenames(plans: list[FilePlan]) -> str:
    """Render the (bounded) filename list for a group cell."""
    names = [p.path.rsplit("/", 1)[-1] for p in plans]
    shown = names[:MAX_FILES_PER_GROUP]
    cell = ", ".join(f"`{_sanitize(n)}`" for n in shown)
    extra = len(names) - len(shown)
    if extra > 0:
        cell += f", +{extra} more"
    return cell


def _grouped_table(file_plans: list[FilePlan], labels: dict[str, str]) -> str:
    """Build the grouped changed-files table (bounded)."""
    groups: OrderedDict[str, list[FilePlan]] = OrderedDict()
    for plan in file_plans:
        groups.setdefault(_group_key(plan.path), []).append(plan)

    # Deterministic ordering: group key alphabetical.
    ordered = sorted(groups.items(), key=lambda kv: kv[0])

    lines = [
        f"### {labels['changed_files']}",
        "",
        f"| {labels['group']} | {labels['files']} | {labels['change_summary']} |",
        "| --- | --- | --- |",
    ]
    shown = ordered[:MAX_TABLE_ROWS]
    for key, plans in shown:
        group_label = f"`{_sanitize(key)}`"
        files_cell = _filenames(plans)
        summary = f"{_describe_group(plans)} — {_purpose(plans)}"
        lines.append(f"| {group_label} | {files_cell} | {summary} |")
    hidden = len(ordered) - len(shown)
    if hidden > 0:
        lines.append(f"| … | … | +{hidden} more group(s) |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# interaction detection + Mermaid                                              #
# --------------------------------------------------------------------------- #
# Path signals that suggest component interaction / API / event / async flows.
_INTERACTION_PATH_SIGNALS = (
    "/api/",
    "/api.",
    "/service",
    "/services/",
    "/handler",
    "/handlers/",
    "/controller",
    "/route",
    "/routes/",
    "/worker",
    "/workers/",
    "/queue",
    "/consumer",
    "/producer",
    "/event",
    "/client",
    "/rpc",
    "/grpc",
    "/webhook",
)
# Diff-content signals that suggest a genuine cross-component flow change. These
# are deliberately concrete network/IPC/event verbs: bare `await `/`async def`
# are NOT included because pure-compute async helpers use them too, which would
# manufacture noisy diagrams (SPEC principle 1: low-noise). An interaction-y
# PATH (``_INTERACTION_PATH_SIGNALS``) is the other way a file qualifies.
_INTERACTION_DIFF_SIGNALS = (
    "requests.",
    "httpx.",
    "fetch(",
    ".publish(",
    ".send(",
    ".emit(",
    "queue",
    "broker",
    "rpc",
    "grpc",
    "@app.route",
    "@router.",
    "webhook",
)


def _is_interaction_file(plan: FilePlan) -> bool:
    lower_path = f"/{plan.path.lower()}"
    if any(sig in lower_path for sig in _INTERACTION_PATH_SIGNALS):
        return True
    diff_lower = plan.diff_text.lower()
    return any(sig in diff_lower for sig in _INTERACTION_DIFF_SIGNALS)


def _interaction_files(file_plans: list[FilePlan]) -> list[FilePlan]:
    """Reviewable, non-docs files that look interaction-flavored."""
    return [
        p
        for p in file_plans
        if p.file_type not in ("docs", "lockfile", "generated")
        and _is_interaction_file(p)
    ]


def _should_render_mermaid(file_plans: list[FilePlan]) -> bool:
    """True only when ≥2 interaction-flavored files participate.

    A lone tweak (even an interaction-y one) does not justify a diagram; a
    docs-only change never does. Requiring two collaborating components keeps the
    diagram meaningful and avoids noise (SPEC principle 1: low-noise).
    """
    return len(_interaction_files(file_plans)) >= 2


def _node_label(plan: FilePlan) -> str:
    """Short, sanitized node label for a file in the Mermaid graph."""
    name = plan.path.rsplit("/", 1)[-1]
    name = name.rsplit(".", 1)[0]
    # Mermaid identifiers / labels: keep it alnum + underscore.
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def _render_mermaid(file_plans: list[FilePlan], labels_map: dict[str, str]) -> str:
    """Render a bounded Mermaid flowchart of the interacting components.

    Deterministic: nodes are the (sorted, capped) interaction files; edges chain
    them in order to convey "these components now collaborate". This is an
    advisory sketch, not a verified call graph.
    """
    files = sorted(_interaction_files(file_plans), key=lambda p: p.path)
    files = files[:MAX_FILES_PER_GROUP]
    labels = []
    seen: set[str] = set()
    for p in files:
        node = _node_label(p)
        # Disambiguate label collisions deterministically.
        base = node
        i = 2
        while node in seen:
            node = f"{base}_{i}"
            i += 1
        seen.add(node)
        labels.append((node, p.path.rsplit("/", 1)[-1]))

    lines = [
        f"### {labels_map['interaction_flow']}",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for node, fname in labels:
        # The label sits inside a `["..."]` Mermaid string; a bare double-quote
        # in an UNTRUSTED filename would close it early and corrupt the diagram,
        # so neutralize it to the Mermaid HTML entity (`_sanitize` handles the
        # other markdown/HTML control chars).
        label = _sanitize(fname).replace('"', "&quot;")
        lines.append(f'    {node}["{label}"]')
    for (a, _), (b, _) in pairwise(labels):
        lines.append(f"    {a} --> {b}")
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# high-level summary                                                            #
# --------------------------------------------------------------------------- #
def _summary(pr_context: Mapping[str, Any], file_plans: list[FilePlan]) -> str:
    """2-3 sentence deterministic high-level summary of the change."""
    n_files = len(file_plans)
    n_groups = len({_group_key(p.path) for p in file_plans})
    types = sorted({p.file_type for p in file_plans})

    title = str(pr_context.get("title", "") or "")
    sentences: list[str] = []
    if title:
        sentences.append(f"**{_sanitize(title, limit=_MAX_TITLE_CHARS)}**")

    if n_files == 0:
        sentences.append("No reviewable file changes were detected.")
    else:
        file_word = "file" if n_files == 1 else "files"
        group_word = "area" if n_groups == 1 else "areas"
        sentences.append(
            f"This change touches {n_files} {file_word} across {n_groups} {group_word}."
        )
        if types:
            sentences.append("Affected categories: " + ", ".join(types) + ".")

    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# public API                                                                    #
# --------------------------------------------------------------------------- #
def build_walkthrough(
    pr_context: Mapping[str, Any],
    file_plans: list[FilePlan],
    findings: list[Finding],
    *,
    stats: Optional[Mapping[str, Any]] = None,
    response_language: str = "en",
    persona: bool = True,
) -> str:
    """Build the enriched sticky-walkthrough markdown.

    Sections, in order: a ``## Walkthrough`` heading + high-level summary, the
    grouped changed-files table, an optional Mermaid interaction diagram, and the
    findings summary table (reused from :mod:`openrabbit.pipeline.emit`).

    The result is deterministic and bounded. ``pr_context`` text is treated as
    UNTRUSTED data.

    ``response_language`` (default ``"en"``) localizes the STATIC labels (section
    headings + the changed-files table columns) and the findings summary table
    header/count/stats line via :func:`render_summary_markdown`. ``"en"`` is
    pixel-identical to the pre-feature output; an unknown code degrades to ``en``.

    ``persona`` (default ``True``) controls the rabbit branding (Feature 2). When
    ON the body opens with a small fenced ASCII-art rabbit, the walkthrough +
    findings headings carry a tasteful 🐰 marker, and a casual one-line sign-off
    (localized via ``response_language``) plus a small italic "made by Subin Hong"
    attribution close the body; when OFF the output is plain/neutral (no art, no
    emoji, no sign-off, no attribution) so a team can opt out. ``persona=False``
    is byte-identical to the pre-Feature-2 rendering.
    """
    labels = _labels(response_language)
    plans = list(file_plans)
    # Branding is purely additive: a 🐰 prefix on the heading when persona is ON.
    walkthrough_heading = (
        f"## {_RABBIT} {labels['walkthrough']}"
        if persona
        else f"## {labels['walkthrough']}"
    )
    findings_heading = (
        f"### {_RABBIT} {labels['findings']}"
        if persona
        else f"### {labels['findings']}"
    )
    sections: list[str] = []
    # Tasteful ASCII-art rabbit at the very top (persona ON only). Fenced so it
    # renders monospace/aligned on GitHub. Omitted when persona is OFF so the
    # plain rendering stays byte-identical.
    if persona:
        sections += [_ASCII_RABBIT, ""]
    sections += [
        walkthrough_heading,
        "",
        _summary(pr_context, plans),
    ]

    if plans:
        sections += ["", _grouped_table(plans, labels)]

    if _should_render_mermaid(plans):
        sections += ["", _render_mermaid(plans, labels)]

    # Reuse today's findings summary table (requirement 4); pass the language so
    # its count/stats line localize too. We already emit our own localized
    # `findings_heading` right above, so pass ``heading=False`` to suppress the
    # renderer's own "## openrabbit review" heading — otherwise the body would
    # DOUBLE-head (our "Findings"/"발견 사항" + its heading). The emit renderer
    # already escapes its own cells and handles the empty case.
    sections += [
        "",
        findings_heading,
        "",
        render_summary_markdown(
            findings,
            stats=stats,
            response_language=response_language,
            heading=False,
        ),
    ]

    # Casual one-line rabbit sign-off + a small italic attribution, localized;
    # omitted when persona is OFF so the plain rendering is byte-identical to the
    # pre-Feature-2 output.
    if persona:
        sections += [
            "",
            f"{_RABBIT} _{labels['sign_off']}_",
            "",
            _ATTRIBUTION,
        ]

    return "\n".join(sections)
