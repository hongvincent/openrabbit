"""Eval-honesty tests for the golden-set builder (adversarial findings 1 & 3).

These guard two correctness defects that made the headline FP<10% metric measure
the wrong thing:

Finding 1 (CRITICAL): a 'fix' commit's diff is the ALREADY-CORRECTED code, so
grading recall against it grades patched code. A bug sample must contain the
PRE-FIX (buggy) state — the reverse diff ``git show -R <fix_sha>``.

Finding 3 (HIGH): commit-message regex alone mislabels doc/format/test-only
'fix' commits ('fix typo', 'fix lint', 'fix tests') as real bugs, and the corpus
is regenerated live per run (irreproducible). A committed versioned JSONL must be
loadable, and trivial 'fix's must NOT be classified as known bugs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from openrabbit.eval.golden_set import (
    build_golden_set,
    classify_commit,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "HOME": str(repo),
    }
    out = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def _commit(repo: Path, filename: str, content: str, message: str) -> str:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    # Buggy baseline, then a fix commit that REMOVES the buggy line.
    _commit(
        repo,
        "calc.py",
        "def add(a, b):\n    return a - b  # BUG: subtraction\n",
        "Initial commit",
    )
    _commit(
        repo,
        "calc.py",
        "def add(a, b):\n    return a + b  # fixed\n",
        "fix: correct addition operator in add()",
    )
    return repo


# --------------------------------------------------------------------------- #
# Finding 1: bug samples must show the PRE-FIX (buggy) state, not the patch.   #
# --------------------------------------------------------------------------- #
def test_bug_sample_contains_the_buggy_code_not_the_fix(git_repo: Path):
    samples = build_golden_set(git_repo)
    assert samples, "expected the fix commit to be mined as a bug sample"
    fix_sample = next(s for s in samples if s.source == "fix")
    diff = fix_sample.diff

    # The reviewer must SEE the defect: the buggy ``return a - b`` line is what
    # gets ADDED ('+') in a reverse diff (the state to review), and the corrected
    # ``return a + b`` line is what is REMOVED ('-').
    added = [
        ln[1:]
        for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    removed = [
        ln[1:]
        for ln in diff.splitlines()
        if ln.startswith("-") and not ln.startswith("---")
    ]
    assert any("a - b" in ln for ln in added), (
        "bug sample must present the PRE-FIX buggy line as the code under review; "
        f"added lines were {added!r}"
    )
    assert any("a + b" in ln for ln in removed), (
        "the corrected line must be the removed side of the reverse diff; "
        f"removed lines were {removed!r}"
    )


def test_bug_sample_records_a_defect_location(git_repo: Path):
    # Provenance for blind judging (finding 4): the sample must expose WHERE the
    # real defect lives (the file the fix touched) so 'match' can be checked
    # against the true location instead of a leaked label.
    samples = build_golden_set(git_repo)
    fix_sample = next(s for s in samples if s.source == "fix")
    assert "calc.py" in (fix_sample.defect_location or "")


# --------------------------------------------------------------------------- #
# Finding 3: trivial doc/format/test 'fix' commits are NOT real bugs.          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        "fix typo in README",
        "fix lint",
        "fix: fix tests",
        "fix: reformat with black",
        "fix: update docs",
        "fix: rename variable for clarity",
    ],
)
def test_trivial_fix_messages_are_not_known_bugs(message: str):
    label = classify_commit(message)
    assert label.known_bug is False, (
        f"{message!r} is a doc/format/test-only change and must NOT be graded as "
        "a real defect (it would pollute the recall denominator)"
    )


def test_real_fix_message_is_still_a_known_bug():
    # A genuine defect fix must remain a known bug.
    assert classify_commit("fix: correct off-by-one in pagination").known_bug is True
    assert classify_commit("fix: SQL injection in login handler").known_bug is True
