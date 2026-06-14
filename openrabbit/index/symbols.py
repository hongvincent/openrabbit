"""Offline codebase SymbolIndex (SPEC 1.2 / Phase 4, item 16).

A Greptile-style codebase index, but **offline-safe and bounded for v1**: it
builds a graph of *file* and *symbol* nodes plus *defines* / *imports* / *calls*
edges WITHOUT heavy native dependencies. Python files are parsed with the
standard-library :mod:`ast`; js/ts/go files use lightweight regex heuristics.

Key safety property: the target repository's code is **never imported or
executed**. ``.py`` files are read as text and parsed via :func:`ast.parse`
only — so a malicious module in the indexed repo cannot run on the CI runner.
The diff/repo content is UNTRUSTED data; this module only reads and parses it.

tree-sitter is reserved as an OPTIONAL, lazily-imported richer backend selected
via ``backend="tree-sitter"``, but it is **never required**. The richer parse
path is not yet implemented, so for v1 every backend extracts via the stdlib
``ast``/regex path; requesting ``"tree-sitter"`` when it is unavailable still
falls back cleanly (never raises), and unit tests (plus the default production
path) need no native deps. The lazy availability probe is kept so the eventual
backend can be wired in without changing the public surface.

Public surface:
    build_index(repo_root, paths, *, backend="stdlib") -> SymbolIndex
    SymbolIndex.callers_of(symbol) -> list[str]      # files calling the symbol
    SymbolIndex.defined_in(symbol) -> list[str]      # files defining the symbol
    SymbolIndex.impacted_by(changed_files) -> list[str]  # cross-file impact
"""

from __future__ import annotations

import ast
import enum
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Hard cap on file size we will read+parse, so an attacker-crafted giant blob in
# the indexed repo cannot exhaust memory on the runner.
_MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MiB

_PY_SUFFIXES = (".py", ".pyi")
_JS_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_GO_SUFFIXES = (".go",)


class NodeKind(enum.Enum):
    """A node is either a source file or a defined symbol."""

    FILE = "file"
    SYMBOL = "symbol"


class EdgeKind(enum.Enum):
    """Best-effort relationship between nodes."""

    DEFINES = "defines"  # file -> symbol
    IMPORTS = "imports"  # file -> imported module/name
    CALLS = "calls"  # file -> symbol name (best-effort)


@dataclass(frozen=True)
class Node:
    """A graph node (a file or a symbol).

    ``name`` is the file path (FILE) or the bare symbol name (SYMBOL).
    ``file`` is the defining file for a SYMBOL (and equals ``name`` for a FILE).
    """

    kind: NodeKind
    name: str
    file: str


@dataclass(frozen=True)
class Edge:
    """A directed best-effort edge between a file and a target.

    For ``DEFINES`` the ``dst`` is a symbol name; for ``IMPORTS`` it is the
    imported module/name; for ``CALLS`` it is the called symbol name. ``src`` is
    always the absolute file path the edge originates from.
    """

    kind: EdgeKind
    src: str  # source file (absolute)
    dst: str  # target symbol / module name


@dataclass
class _FileFacts:
    """Raw facts extracted from a single file before graph assembly."""

    path: str
    defines: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    calls: set[str] = field(default_factory=set)


class SymbolIndex:
    """An in-memory graph of files and symbols with cross-file queries.

    Build via :func:`build_index`. All query methods return **file paths**
    (absolute, as supplied at build time) and are pure/read-only.
    """

    def __init__(self, repo_root: str) -> None:
        self._repo_root = str(Path(repo_root).resolve())
        self._nodes: list[Node] = []
        self._edges: list[Edge] = []
        # symbol name -> set of files defining it
        self._def_files: dict[str, set[str]] = {}
        # symbol name -> set of files that call it
        self._call_files: dict[str, set[str]] = {}
        # file -> set of symbol names it references (calls + imported names)
        self._file_refs: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ #
    # construction (used by build_index)                                  #
    # ------------------------------------------------------------------ #
    def _ingest(self, facts: _FileFacts) -> None:
        path = facts.path
        self._nodes.append(Node(kind=NodeKind.FILE, name=path, file=path))
        for sym in sorted(facts.defines):
            self._nodes.append(Node(kind=NodeKind.SYMBOL, name=sym, file=path))
            self._edges.append(Edge(kind=EdgeKind.DEFINES, src=path, dst=sym))
            self._def_files.setdefault(sym, set()).add(path)
        for mod in sorted(facts.imports):
            self._edges.append(Edge(kind=EdgeKind.IMPORTS, src=path, dst=mod))
        for sym in sorted(facts.calls):
            self._edges.append(Edge(kind=EdgeKind.CALLS, src=path, dst=sym))
            self._call_files.setdefault(sym, set()).add(path)
        refs = set(facts.calls)
        # The trailing component of an imported dotted path is a referenced name
        # (e.g. ``from pkg.util import add`` references ``add``; ``import x.y``
        # references ``y``). This lets impacted_by catch import-only references.
        for mod in facts.imports:
            tail = mod.rsplit(".", 1)[-1]
            if tail:
                refs.add(tail)
        self._file_refs[path] = refs

    # ------------------------------------------------------------------ #
    # public queries                                                      #
    # ------------------------------------------------------------------ #
    def nodes(self) -> list[Node]:
        """All graph nodes (files + symbols), in insertion order."""
        return list(self._nodes)

    def edges(self) -> list[Edge]:
        """All graph edges, in insertion order."""
        return list(self._edges)

    def defined_in(self, symbol: str) -> list[str]:
        """Files that define ``symbol`` (empty if unknown)."""
        return sorted(self._def_files.get(symbol, set()))

    def callers_of(self, symbol: str) -> list[str]:
        """Files that call ``symbol`` (best-effort, empty if none)."""
        return sorted(self._call_files.get(symbol, set()))

    def impacted_by(self, changed_files: Iterable[str]) -> list[str]:
        """Files that reference symbols defined in any of ``changed_files``.

        Cross-file impact (SPEC 1.2): given a set of changed files, return the
        OTHER indexed files that reference (call/import) a symbol defined in one
        of them. The changed files themselves are excluded from the result.

        Accepts absolute or repo-relative paths; unknown files are ignored.

        Best-effort approximation (no import resolution): matching is by bare
        symbol *name*, so if the same name is defined in two modules this can
        over-report files that actually reference the OTHER definition. That is
        an intentional false-positive bias for v1 (safe for an advisory impact
        hint); disambiguation via the dotted ``IMPORTS`` edges is future work.
        """
        changed_abs: set[str] = {
            normalized
            for normalized in (self._normalize(p) for p in changed_files)
            if normalized is not None
        }
        # All symbols defined in the changed files.
        changed_symbols: set[str] = set()
        for sym, files in self._def_files.items():
            if files & changed_abs:
                changed_symbols.add(sym)
        if not changed_symbols:
            return []
        impacted: set[str] = set()
        for path, refs in self._file_refs.items():
            if path in changed_abs:
                continue
            if refs & changed_symbols:
                impacted.add(path)
        return sorted(impacted)

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #
    def _normalize(self, path: str) -> Optional[str]:
        """Resolve a supplied path to the same absolute form used internally."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._repo_root) / p
        resolved = str(p.resolve())
        # Only return paths we actually indexed (a file node exists).
        if resolved in self._file_refs:
            return resolved
        return None


# --------------------------------------------------------------------------- #
# builder                                                                       #
# --------------------------------------------------------------------------- #
def build_index(
    repo_root: str,
    paths: Iterable[str],
    *,
    backend: str = "stdlib",
) -> SymbolIndex:
    """Build a :class:`SymbolIndex` from ``paths`` rooted at ``repo_root``.

    Parameters
    ----------
    repo_root:
        Repository root; used to resolve relative paths in queries.
    paths:
        Iterable of file paths to index (absolute or relative to ``repo_root``).
    backend:
        ``"stdlib"`` (default) uses :mod:`ast` + regex with no native deps.
        ``"tree-sitter"`` is reserved for the optional richer backend; it is not
        yet implemented, so it currently resolves to the same stdlib extraction
        and **falls back cleanly** when tree-sitter is absent (never raises).
    """
    # Probe (lazy, never raises) so a future tree-sitter backend can branch here
    # without changing the public surface. The richer parse path is not wired up
    # yet, so the stdlib extraction below is always used regardless.
    _ = backend == "tree-sitter" and _tree_sitter_available()
    index = SymbolIndex(repo_root)
    for path in paths:
        facts = _extract_file(repo_root, path)
        if facts is not None:
            index._ingest(facts)
    return index


def _tree_sitter_available() -> bool:
    """Best-effort, LAZY check for the optional tree-sitter backend.

    Never raises; returns ``False`` (→ stdlib fallback) when tree-sitter is not
    installed, which is the case in unit tests and the default path.
    """
    try:  # pragma: no cover - exercised only when tree-sitter is installed
        import importlib.util

        return importlib.util.find_spec("tree_sitter") is not None
    except Exception:  # pragma: no cover - defensive
        return False


def _extract_file(repo_root: str, path: str) -> Optional[_FileFacts]:
    """Read + parse one file into raw facts, or ``None`` if unindexable."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(repo_root) / p
    try:
        resolved = p.resolve()
        if not resolved.is_file():
            return None
        if resolved.stat().st_size > _MAX_READ_BYTES:
            return None
        source = resolved.read_text(errors="replace")
    except (OSError, ValueError):
        return None

    abs_path = str(resolved)
    suffix = resolved.suffix.lower()
    if suffix in _PY_SUFFIXES:
        return _extract_python(abs_path, source)
    if suffix in _JS_SUFFIXES:
        return _extract_regex_js(abs_path, source)
    if suffix in _GO_SUFFIXES:
        return _extract_regex_go(abs_path, source)
    return None


# --------------------------------------------------------------------------- #
# python backend (ast — never imports/execs the target)                         #
# --------------------------------------------------------------------------- #
class _PyVisitor(ast.NodeVisitor):
    """Collect top-level/nested defs, imports, and call names from an AST."""

    def __init__(self) -> None:
        self.defines: set[str] = set()
        self.imports: set[str] = set()
        self.calls: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defines.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.defines.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defines.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            # Record the fully-qualified imported name so the trailing component
            # (the bound name) is recoverable for reference tracking.
            full = f"{module}.{alias.name}" if module else alias.name
            self.imports.add(full)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            self.calls.add(name)
        self.generic_visit(node)


def _call_name(func: ast.expr) -> Optional[str]:
    """Best-effort name of a called function (``f()`` or ``obj.f()`` → ``f``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_python(path: str, source: str) -> Optional[_FileFacts]:
    """Parse Python source with ast ONLY — never import/exec the module."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Unparseable file: skip gracefully (still index the rest).
        return None
    visitor = _PyVisitor()
    visitor.visit(tree)
    return _FileFacts(
        path=path,
        defines=visitor.defines,
        imports=visitor.imports,
        calls=visitor.calls,
    )


# --------------------------------------------------------------------------- #
# regex heuristics backend (js/ts/go)                                           #
# --------------------------------------------------------------------------- #
# function foo(...)  |  export function foo(...)  |  export default function foo
_JS_FUNC_RE = re.compile(r"\bfunction\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(", re.MULTILINE)
# const foo = (...) =>   |   let foo = function   |   var foo = (...) =>
_JS_ASSIGN_FN_RE = re.compile(
    r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
    re.MULTILINE,
)
# class Foo  |  class Foo extends Bar
_JS_CLASS_RE = re.compile(r"\bclass\s+(?P<name>[A-Za-z_$][\w$]*)", re.MULTILINE)
# import { a, b } from '...'   |  import x from '...'
_JS_IMPORT_RE = re.compile(
    r"\bimport\s+(?P<spec>[^;]+?)\s+from\s+['\"](?P<mod>[^'\"]+)['\"]",
    re.MULTILINE,
)
_JS_REQUIRE_RE = re.compile(r"\brequire\(\s*['\"](?P<mod>[^'\"]+)['\"]\s*\)")
# bare call: foo(   (filtered against keywords/known defs)
_JS_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_$][\w$]*)\s*\(", re.MULTILINE)

# go: func Name(...)   |   func (r *T) Name(...)
_GO_FUNC_RE = re.compile(
    r"\bfunc\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_][\w]*)\s*\(", re.MULTILINE
)
_GO_TYPE_RE = re.compile(
    r"\btype\s+(?P<name>[A-Za-z_][\w]*)\s+(?:struct|interface)\b", re.MULTILINE
)
_GO_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_][\w]*)\s*\(", re.MULTILINE)

# Reserved words that the bare-call regex would otherwise pick up.
_CALL_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "function",
        "typeof",
        "await",
        "new",
        "super",
        "func",
        "go",
        "defer",
        "select",
        "case",
        "println",
        "print",
        "make",
        "len",
        "cap",
        "append",
    }
)


def _strip_comments(source: str) -> str:
    """Remove ``//`` line comments and ``/* */`` block comments (best-effort)."""
    no_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", no_block)


def _extract_regex_js(path: str, source: str) -> _FileFacts:
    text = _strip_comments(source)
    defines: set[str] = set()
    for rx in (_JS_FUNC_RE, _JS_ASSIGN_FN_RE, _JS_CLASS_RE):
        defines.update(m.group("name") for m in rx.finditer(text))

    imports: set[str] = set()
    for m in _JS_IMPORT_RE.finditer(text):
        imports.add(m.group("mod"))
    imports.update(m.group("mod") for m in _JS_REQUIRE_RE.finditer(text))

    calls = {
        m.group("name")
        for m in _JS_CALL_RE.finditer(text)
        if m.group("name") not in _CALL_KEYWORDS
    }
    calls -= defines  # a definition site is not a call site
    return _FileFacts(path=path, defines=defines, imports=imports, calls=calls)


def _extract_regex_go(path: str, source: str) -> _FileFacts:
    text = _strip_comments(source)
    defines: set[str] = set()
    defines.update(m.group("name") for m in _GO_FUNC_RE.finditer(text))
    defines.update(m.group("name") for m in _GO_TYPE_RE.finditer(text))

    imports: set[str] = set()
    # import "x"  or  import ( "x"\n "y" )
    for m in re.finditer(r'import\s+(?:\(\s*([^)]*)\)|"([^"]+)")', text, re.DOTALL):
        block, single = m.group(1), m.group(2)
        if single:
            imports.add(single)
        if block:
            imports.update(re.findall(r'"([^"]+)"', block))

    calls = {
        m.group("name")
        for m in _GO_CALL_RE.finditer(text)
        if m.group("name") not in _CALL_KEYWORDS
    }
    calls -= defines
    return _FileFacts(path=path, defines=defines, imports=imports, calls=calls)
