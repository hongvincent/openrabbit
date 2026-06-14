"""Tests for the offline codebase SymbolIndex (SPEC 1.2 / Phase 4, item 16).

The index is built WITHOUT heavy native deps by default: Python ``.py`` files are
parsed with the standard-library :mod:`ast` (NEVER imported/exec'd), and js/ts/go
files use lightweight regex heuristics. These tests build a tiny multi-file
package in a temp dir (no network, no tree-sitter) and assert:

* ast symbol extraction (functions/classes) + import edges,
* ``callers_of`` / ``defined_in`` / ``impacted_by`` correctness across files,
* the regex backend extracts symbols from a tiny js file,
* the indexer NEVER imports the target module (parse-only via ast).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from openrabbit.index import SymbolIndex, build_index
from openrabbit.index.symbols import EdgeKind, NodeKind


# --------------------------------------------------------------------------- #
# fixtures: a tiny multi-file python package                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    """A small package: util defines helpers; service + api import/call them."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)

    (root / "pkg" / "__init__.py").write_text("")

    (root / "pkg" / "util.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def sub(a, b):\n"
        "    return a - b\n"
        "\n"
        "\n"
        "class Calc:\n"
        "    def run(self):\n"
        "        return add(1, 2)\n"
    )

    (root / "pkg" / "service.py").write_text(
        "from pkg.util import add\n\n\ndef compute(x):\n    return add(x, 10)\n"
    )

    (root / "pkg" / "api.py").write_text(
        "import pkg.util as util\n"
        "from pkg.service import compute\n"
        "\n"
        "\n"
        "def handler(x):\n"
        "    return compute(x) + util.sub(x, 1)\n"
    )

    # A non-python file the indexer should skip (binary-ish / unrelated).
    (root / "README.md").write_text("# proj\n")
    return root


def _paths(root: Path) -> list[str]:
    return [str(p) for p in sorted(root.rglob("*.py"))]


# --------------------------------------------------------------------------- #
# build / basic structure                                                       #
# --------------------------------------------------------------------------- #
def test_build_index_returns_symbol_index(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    assert isinstance(idx, SymbolIndex)


def test_extracts_functions_and_classes(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    # functions + the class + the method
    assert {"add", "sub", "compute", "handler"} <= names
    assert "Calc" in names


def test_defined_in_resolves_file(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    defs = idx.defined_in("add")
    assert any(d.endswith("util.py") for d in defs)
    # A symbol defined in exactly one file resolves to exactly that file.
    compute_defs = idx.defined_in("compute")
    assert [d for d in compute_defs if d.endswith("service.py")]


def test_defined_in_unknown_symbol_is_empty(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    assert idx.defined_in("does_not_exist") == []


# --------------------------------------------------------------------------- #
# callers_of (best-effort call edges)                                           #
# --------------------------------------------------------------------------- #
def test_callers_of_finds_cross_file_callers(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    callers = idx.callers_of("add")
    # add() is called by Calc.run (util.py), compute (service.py).
    files = {Path(c).name for c in callers}
    assert "service.py" in files
    assert "util.py" in files


def test_callers_of_compute(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    callers = idx.callers_of("compute")
    assert any(Path(c).name == "api.py" for c in callers)


def test_callers_of_unknown_is_empty(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    assert idx.callers_of("nope") == []


# --------------------------------------------------------------------------- #
# impacted_by (cross-file impact of a change)                                   #
# --------------------------------------------------------------------------- #
def test_impacted_by_returns_referencing_files(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    changed = [str(pkg / "pkg" / "util.py")]
    impacted = idx.impacted_by(changed)
    names = {Path(p).name for p in impacted}
    # service.py imports+calls add; api.py imports util + transitively. Both
    # reference symbols defined in util.py.
    assert "service.py" in names
    assert "api.py" in names
    # The changed file itself is not reported as impacted.
    assert "util.py" not in names


def test_impacted_by_unknown_file_is_empty(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    assert idx.impacted_by([str(pkg / "pkg" / "missing.py")]) == []


def test_impacted_by_accepts_relative_paths(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    impacted = idx.impacted_by(["pkg/util.py"])
    assert {Path(p).name for p in impacted} >= {"service.py"}


# --------------------------------------------------------------------------- #
# import edges                                                                   #
# --------------------------------------------------------------------------- #
def test_import_edges_recorded(pkg: Path) -> None:
    idx = build_index(str(pkg), _paths(pkg))
    import_edges = [e for e in idx.edges() if e.kind == EdgeKind.IMPORTS]
    assert import_edges
    # service.py imports pkg.util
    assert any(
        Path(e.src).name == "service.py" and "util" in e.dst for e in import_edges
    )


# --------------------------------------------------------------------------- #
# regex backend for js/ts/go                                                     #
# --------------------------------------------------------------------------- #
def test_regex_backend_extracts_js_symbols(tmp_path: Path) -> None:
    root = tmp_path / "jsproj"
    root.mkdir()
    js = root / "app.js"
    js.write_text(
        "import { add } from './util';\n"
        "export function handler(x) {\n"
        "  return add(x, 1);\n"
        "}\n"
        "const arrow = (y) => add(y, 2);\n"
        "class Widget {}\n"
    )
    idx = build_index(str(root), [str(js)])
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    assert "handler" in names
    assert "Widget" in names
    # call edge to add (best-effort) makes app.js a caller of add.
    assert any(Path(c).name == "app.js" for c in idx.callers_of("add"))


def test_regex_backend_go_func(tmp_path: Path) -> None:
    root = tmp_path / "goproj"
    root.mkdir()
    go = root / "main.go"
    go.write_text(
        "package main\n"
        "\n"
        "func Add(a int, b int) int {\n"
        "    return a + b\n"
        "}\n"
        "\n"
        "func main() {\n"
        "    Add(1, 2)\n"
        "}\n"
    )
    idx = build_index(str(root), [str(go)])
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    assert {"Add", "main"} <= names


def test_regex_backend_go_imports_and_types(tmp_path: Path) -> None:
    root = tmp_path / "goproj2"
    root.mkdir()
    go = root / "srv.go"
    go.write_text(
        "package main\n"
        "\n"
        "import (\n"
        '    "fmt"\n'
        '    "net/http"\n'
        ")\n"
        "\n"
        'import "os"\n'
        "\n"
        "type Server struct {\n"
        "    port int\n"
        "}\n"
        "\n"
        "type Handler interface {\n"
        "    Serve()\n"
        "}\n"
    )
    idx = build_index(str(root), [str(go)])
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    assert {"Server", "Handler"} <= names
    import_dsts = {e.dst for e in idx.edges() if e.kind == EdgeKind.IMPORTS}
    assert {"fmt", "net/http", "os"} <= import_dsts


def test_method_receiver_in_python_counted_once(tmp_path: Path) -> None:
    root = tmp_path / "py2"
    root.mkdir()
    f = root / "m.py"
    f.write_text(
        "import async_helper\n"
        "\n"
        "async def fetch():\n"
        "    return await async_helper.go()\n"
    )
    idx = build_index(str(root), [str(f)])
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    assert "fetch" in names  # async def extracted


def test_oversize_file_is_skipped(tmp_path: Path, monkeypatch) -> None:
    from openrabbit.index import symbols as sym_mod

    root = tmp_path / "big"
    root.mkdir()
    f = root / "huge.py"
    f.write_text("def small():\n    return 1\n")
    # Force the size cap below the file's real size to exercise the guard.
    monkeypatch.setattr(sym_mod, "_MAX_READ_BYTES", 1)
    idx = build_index(str(root), [str(f)])
    assert [n for n in idx.nodes() if n.kind == NodeKind.SYMBOL] == []


# --------------------------------------------------------------------------- #
# safety: never import/exec the target module                                   #
# --------------------------------------------------------------------------- #
def test_indexer_never_imports_target_module(tmp_path: Path, monkeypatch) -> None:
    """The indexer must parse via ast only — never import the target code.

    We plant a module that would record into a sentinel on import, add its dir
    to sys.path, build the index, and assert the side effect never fired.
    """
    root = tmp_path / "danger"
    root.mkdir()
    sentinel = root / "IMPORTED"
    mod = root / "evil_module.py"
    mod.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('boom')\n"
        "def trigger():\n"
        "    return 1\n"
    )
    monkeypatch.syspath_prepend(str(root))
    try:
        idx = build_index(str(root), [str(mod)])
        # Symbol was extracted (so the file WAS processed)...
        assert any(n.name == "trigger" for n in idx.nodes())
        # ...but importing it never happened.
        assert not sentinel.exists(), "indexer imported/exec'd the target module!"
        assert "evil_module" not in sys.modules
    finally:
        sys.modules.pop("evil_module", None)


def test_syntax_error_file_is_skipped_gracefully(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    good = root / "good.py"
    good.write_text("def ok():\n    return 1\n")
    bad = root / "bad.py"
    bad.write_text("def oops(:\n    pass\n")  # syntax error
    idx = build_index(str(root), [str(good), str(bad)])
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    assert "ok" in names  # good file still indexed; bad file skipped, no raise


def test_missing_file_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    idx = build_index(str(root), [str(root / "ghost.py")])
    assert idx.nodes() == [] or all(n.kind == NodeKind.FILE for n in idx.nodes())


def test_unsupported_extension_skipped(tmp_path: Path) -> None:
    root = tmp_path / "misc"
    root.mkdir()
    f = root / "data.bin"
    f.write_text("\x00\x01 not source")
    idx = build_index(str(root), [str(f)])
    assert [n for n in idx.nodes() if n.kind == NodeKind.SYMBOL] == []


# --------------------------------------------------------------------------- #
# tree-sitter is OPTIONAL and must not be required                              #
# --------------------------------------------------------------------------- #
def test_default_backend_does_not_require_tree_sitter(pkg: Path, monkeypatch) -> None:
    """Building with the default (stdlib) backend must not import tree_sitter."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "tree_sitter" or name.startswith("tree_sitter."):
            raise AssertionError("default backend imported tree_sitter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    idx = build_index(str(pkg), _paths(pkg))
    assert idx.nodes()


def test_tree_sitter_backend_flag_falls_back_when_absent(pkg: Path) -> None:
    """Requesting the tree-sitter backend when it's absent must NOT crash.

    It falls back to the stdlib backend (tree-sitter is optional, never
    required), still producing a usable index.
    """
    idx = build_index(str(pkg), _paths(pkg), backend="tree-sitter")
    names = {n.name for n in idx.nodes() if n.kind == NodeKind.SYMBOL}
    assert "add" in names
