"""Tests for the real enclosing-context fetcher (SPEC section 6, step 3).

The fetcher is best-effort and exception-safe: it shells out to ``git`` and
reads files from the working tree to pull a bounded window of context around
each changed hunk, plus the enclosing function/class. These tests build a
**real local git repo** in a temp dir (no network, no live creds) and assert:

* it returns the enclosing function/class body for a Python hunk,
* it caps output to protect the prompt budget,
* it degrades gracefully when git/the file is missing,
* it wires into :mod:`openrabbit.pipeline.context` via the existing
  ``EnclosingFetcher`` hook while the default stays a safe no-op.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from openrabbit.pipeline import context as ctx_mod
from openrabbit.pipeline import enclosing as enc_mod
from openrabbit.pipeline.route import FilePlan, Hunk


# --------------------------------------------------------------------------- #
# helpers / fixtures                                                            #
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )


def _file_plan(path: str, *hunk_headers: str, file_type: str = "code") -> FilePlan:
    hunks = [Hunk(header=h, text=h) for h in hunk_headers]
    return FilePlan(
        path=path,
        file_type=file_type,
        risk="medium",
        lenses=["correctness"],
        model_role="finder",
        hunks=hunks,
    )


_SAMPLE_PY = '''\
import os


def helper(value):
    """A helper that is far from the change."""
    return value + 1


class Account:
    def __init__(self, owner):
        self.owner = owner
        self.balance = 0

    def deposit(self, amount):
        # the changed line lives in here
        if amount <= 0:
            raise ValueError("bad amount")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        self.balance -= amount
        return self.balance
'''


@pytest.fixture
def py_repo(tmp_path: Path) -> Path:
    """A real local git repo containing one committed Python file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "account.py").write_text(_SAMPLE_PY)
    _git(repo, "add", "account.py")
    _git(repo, "commit", "-q", "-m", "add account")
    return repo


# --------------------------------------------------------------------------- #
# GitEnclosingFetcher — happy path                                              #
# --------------------------------------------------------------------------- #
class TestEnclosingHappyPath:
    def test_returns_enclosing_function_body(self, py_repo: Path):
        # The change touches `deposit` (the `if amount <= 0:` line ~ line 16).
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        out = fetcher(plan)
        assert out is not None
        # Enclosing function body must be present.
        assert "def deposit(self, amount):" in out
        assert 'raise ValueError("bad amount")' in out
        # A distant function should NOT be pulled in.
        assert "def helper(value):" not in out

    def test_includes_window_around_hunk(self, tmp_path: Path):
        # An unstructured (no def/class) file so the *window* logic is the only
        # thing that can pull surrounding lines. Anchor the hunk far from both
        # ends and assert lines reachable ONLY via the window appear.
        repo = tmp_path / "win"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        src = "\n".join(f"line-{i}" for i in range(60)) + "\n"
        (repo / "data.txt").write_text(src)
        _git(repo, "add", "data.txt")
        _git(repo, "commit", "-q", "-m", "data")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo, window=4, max_lines=200)
        # New-side line 30 (1-based) -> 0-based index 29 == "line-29".
        plan = _file_plan("data.txt", "@@ -30,1 +30,1 @@", file_type="docs")
        out = fetcher(plan)
        assert out is not None
        # window=4 pulls 4 lines above/below; these are ONLY reachable via the
        # window (there is no enclosing scope in a plain text file).
        assert "line-26" in out
        assert "line-33" in out
        # ...but not lines outside the window.
        assert "line-20" not in out
        assert "line-40" not in out

    def test_reports_path_and_line_marker(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        out = fetcher(plan)
        assert out is not None
        # Some reference to the file path is included for the model.
        assert "account.py" in out

    def test_finds_enclosing_class_when_no_method(self, tmp_path: Path):
        repo = tmp_path / "r"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        src = "class Widget:\n    x = 1\n    y = 2\n    z = 3\n"
        (repo / "w.py").write_text(src)
        _git(repo, "add", "w.py")
        _git(repo, "commit", "-q", "-m", "init")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo)
        plan = _file_plan("w.py", "@@ -2,1 +3,1 @@")
        out = fetcher(plan)
        assert out is not None
        assert "class Widget:" in out

    def test_blank_anchor_resolves_to_tighter_method_scope(self, tmp_path: Path):
        # When the hunk anchor lands on the blank line BETWEEN two methods, the
        # fetcher should still resolve the nearest method (not widen to the whole
        # class) by advancing past the blank anchor to the next non-blank line.
        repo = tmp_path / "blank"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        src = (
            "class C:\n"
            "    def first(self):\n"
            "        return 1\n"
            "\n"  # line 4 (1-based): the blank line between methods
            "    def second(self):\n"
            "        return 2\n"
        )
        (repo / "c.py").write_text(src)
        _git(repo, "add", "c.py")
        _git(repo, "commit", "-q", "-m", "init")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo)
        # New-side line 4 == the blank line between `first` and `second`.
        plan = _file_plan("c.py", "@@ -4,1 +4,1 @@")
        out = fetcher(plan)
        assert out is not None
        # Should resolve the tighter `second` method, not the whole class body.
        assert "def second(self):" in out
        assert "def first(self):" not in out

    def test_multiple_hunks_aggregated(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan(
            "account.py",
            "@@ -15,3 +16,3 @@ def deposit(self, amount):",
            "@@ -21,2 +22,2 @@ def withdraw(self, amount):",
        )
        out = fetcher(plan)
        assert out is not None
        assert "def deposit(self, amount):" in out
        assert "def withdraw(self, amount):" in out


# --------------------------------------------------------------------------- #
# bounded output                                                                #
# --------------------------------------------------------------------------- #
class TestBoundedOutput:
    def test_caps_total_lines(self, tmp_path: Path):
        repo = tmp_path / "big"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        # One giant function body — far more lines than the cap.
        body = "\n".join(f"    a{i} = {i}" for i in range(500))
        src = f"def big():\n{body}\n"
        (repo / "big.py").write_text(src)
        _git(repo, "add", "big.py")
        _git(repo, "commit", "-q", "-m", "big")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo, max_lines=40)
        plan = _file_plan("big.py", "@@ -10,1 +10,1 @@ def big():")
        out = fetcher(plan)
        assert out is not None
        assert len(out.splitlines()) <= 40

    def test_window_param_bounds_unstructured_file(self, tmp_path: Path):
        repo = tmp_path / "txt"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        src = "\n".join(f"line {i}" for i in range(200)) + "\n"
        (repo / "data.txt").write_text(src)
        _git(repo, "add", "data.txt")
        _git(repo, "commit", "-q", "-m", "data")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo, window=5, max_lines=200)
        plan = _file_plan("data.txt", "@@ -100,1 +100,1 @@", file_type="docs")
        out = fetcher(plan)
        assert out is not None
        # Only a small window around line 100, not the whole 200-line file.
        assert len(out.splitlines()) < 50


# --------------------------------------------------------------------------- #
# graceful degradation                                                          #
# --------------------------------------------------------------------------- #
class TestGracefulDegradation:
    def test_missing_file_returns_none(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("does_not_exist.py", "@@ -1,1 +1,1 @@")
        assert fetcher(plan) is None

    def test_no_hunks_returns_none(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py")  # no hunks
        assert fetcher(plan) is None

    def test_unparseable_hunk_header_returns_none(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py", "not a hunk header at all")
        assert fetcher(plan) is None

    def test_nonexistent_repo_root_returns_none(self, tmp_path: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=tmp_path / "nope")
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@")
        assert fetcher(plan) is None

    def test_git_show_failure_falls_back_to_worktree(self, py_repo: Path, monkeypatch):
        # With a ref set, _git_show is actually invoked; when it raises
        # (git absent / FileNotFoundError) the fetcher must degrade to the
        # working-tree read and still produce enclosing context.
        def _boom(*a, **k):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(enc_mod.subprocess, "run", _boom)
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo, ref="HEAD")
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        out = fetcher(plan)
        # The git-show path failed, so the worktree fallback must supply context.
        assert out is not None
        assert "deposit" in out

    def test_never_raises_on_garbage(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        # Absolute path traversal / weird path must not raise.
        plan = _file_plan("../../etc/passwd", "@@ -1,1 +1,1 @@")
        # Best-effort: returns None rather than leaking or raising.
        assert fetcher(plan) is None

    def test_absolute_path_rejected(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("/etc/passwd", "@@ -1,1 +1,1 @@")
        assert fetcher(plan) is None

    def test_empty_path_rejected(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("", "@@ -1,1 +1,1 @@")
        assert fetcher(plan) is None

    def test_internal_failure_swallowed(self, py_repo: Path, monkeypatch):
        # Force the core to blow up; __call__ must still return None, not raise.
        def _boom(_self, _plan):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(enc_mod.GitEnclosingFetcher, "_gather", _boom)
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@")
        assert fetcher(plan) is None

    def test_duplicate_hunks_deduped(self, py_repo: Path):
        # Two identical hunks resolve to the same span and aren't rendered twice.
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        header = "@@ -15,3 +16,3 @@ def deposit(self, amount):"
        plan = _file_plan("account.py", header, header)
        out = fetcher(plan)
        assert out is not None
        assert out.count("(enclosing context)") == 1

    def test_history_empty_for_unknown_path(self, py_repo: Path):
        # include_history on an untracked (but present) file yields no log block.
        (py_repo / "untracked.py").write_text("def f():\n    return 1\n")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo, include_history=True)
        plan = _file_plan("untracked.py", "@@ -1,2 +1,2 @@ def f():")
        out = fetcher(plan)
        assert out is not None
        assert "recent history" not in out


# --------------------------------------------------------------------------- #
# security: containment (no out-of-repo reads via symlink)                      #
# --------------------------------------------------------------------------- #
class TestPathContainment:
    def test_in_repo_symlink_pointing_outside_yields_none(self, tmp_path: Path):
        # A symlink that lives inside the repo but targets a file OUTSIDE the
        # repo root must NOT have its target content read in worktree mode.
        # FilePlan.path comes verbatim from the UNTRUSTED diff header, so this
        # is attacker-controlled; resolving it must stay within repo_root.
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        secret = tmp_path / "outside.txt"
        secret.write_text("TOPSECRET\n" * 5)
        link = repo / "linkout.txt"
        link.symlink_to(secret)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add link")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo)
        plan = _file_plan("linkout.txt", "@@ -1,1 +1,1 @@")
        out = fetcher(plan)
        # No content from outside the repo may leak into the prompt.
        assert out is None

    def test_symlinked_subdir_escape_yields_none(self, tmp_path: Path):
        # A symlinked directory inside the repo that points outside must not let
        # a path through it escape the repo root.
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        outside_dir = tmp_path / "elsewhere"
        outside_dir.mkdir()
        (outside_dir / "leak.py").write_text("SECRET = 1\n")
        (repo / "sub").symlink_to(outside_dir, target_is_directory=True)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add symdir")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo)
        plan = _file_plan("sub/leak.py", "@@ -1,1 +1,1 @@")
        out = fetcher(plan)
        assert out is None

    def test_in_repo_symlink_to_in_repo_file_is_allowed(self, tmp_path: Path):
        # A symlink that stays inside the repo is fine; content may be returned.
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        (repo / "real.py").write_text("def f():\n    return 42\n")
        (repo / "alias.py").symlink_to(repo / "real.py")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add alias")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo)
        plan = _file_plan("alias.py", "@@ -1,2 +1,2 @@ def f():")
        out = fetcher(plan)
        assert out is not None
        assert "def f():" in out


# --------------------------------------------------------------------------- #
# DoS guard: huge tracked file is not slurped whole                             #
# --------------------------------------------------------------------------- #
class TestReadSizeCap:
    def test_oversized_worktree_file_yields_none(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "Test")
        big = repo / "huge.py"
        # Write a file larger than the read cap so the fetcher must skip it
        # instead of slurping it whole into memory.
        big.write_text("x = 0\n" * (enc_mod._MAX_READ_BYTES // 5 + 1000))
        _git(repo, "add", "huge.py")
        _git(repo, "commit", "-q", "-m", "huge")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=repo)
        plan = _file_plan("huge.py", "@@ -1,1 +1,1 @@")
        out = fetcher(plan)
        assert out is None


# --------------------------------------------------------------------------- #
# ref support (read from a git ref, not just the worktree)                      #
# --------------------------------------------------------------------------- #
class TestRefSupport:
    def test_reads_from_ref(self, py_repo: Path):
        # Modify the worktree, but ask for the committed HEAD content.
        (py_repo / "account.py").write_text("# clobbered\n")
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo, ref="HEAD")
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        out = fetcher(plan)
        assert out is not None
        assert "def deposit(self, amount):" in out


# --------------------------------------------------------------------------- #
# git log / blame enrichment (best-effort)                                      #
# --------------------------------------------------------------------------- #
class TestHistoryEnrichment:
    def test_optional_history_block_when_enabled(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo, include_history=True)
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        out = fetcher(plan)
        assert out is not None
        # The single commit subject should surface somewhere.
        assert "add account" in out

    def test_history_disabled_by_default(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        out = fetcher(plan)
        assert out is not None
        assert "add account" not in out


# --------------------------------------------------------------------------- #
# wiring into context.build_file_message                                        #
# --------------------------------------------------------------------------- #
class TestContextWiring:
    def test_default_fetcher_is_noop(self):
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@")
        assert ctx_mod.gather_enclosing_context(plan) is None

    def test_injected_git_fetcher_adds_block(self, py_repo: Path):
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("account.py", "@@ -15,3 +16,3 @@ def deposit(self, amount):")
        msg = ctx_mod.build_file_message(plan, enclosing_fetcher=fetcher)
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert "enclosing-context" in body
        assert "def deposit(self, amount):" in body

    def test_injected_fetcher_failure_does_not_break_message(self, py_repo: Path):
        # A fetcher that returns None must still yield a valid (diff-only) message.
        fetcher = enc_mod.GitEnclosingFetcher(repo_root=py_repo)
        plan = _file_plan("missing.py", "@@ -1,1 +1,1 @@")
        msg = ctx_mod.build_file_message(plan, enclosing_fetcher=fetcher)
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert "UNTRUSTED" in body.upper()
        assert "enclosing-context" not in body

    def test_raising_fetcher_degrades_to_diff_only(self):
        # A custom fetcher that *raises* must not break message assembly.
        def _boom(_fp):
            raise RuntimeError("boom")

        plan = _file_plan("account.py", "@@ -1,1 +1,1 @@")
        msg = ctx_mod.build_file_message(plan, enclosing_fetcher=_boom)
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert "UNTRUSTED" in body.upper()
        assert "enclosing-context" not in body
