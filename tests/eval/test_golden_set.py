"""Tests for the golden-set builder (SPEC section 10).

Builds a labeled corpus from a *local* git repo's history. These tests create a
tiny real git repo in a temp dir (local-only; not a network call) and assert the
builder finds revert/hotfix/fix commits and serializes samples to JSONL.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from openrabbit.eval.golden_set import (
    GoldenSample,
    build_golden_set,
    classify_commit,
    iter_jsonl,
    write_jsonl,
)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not available"
)


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
    _commit(repo, "a.py", "def f():\n    return 1\n", "Initial commit")
    _commit(repo, "a.py", "def f():\n    return 2\n", "feat: add feature")
    # a buggy commit followed by a revert
    _commit(repo, "a.py", "def f():\n    return 999\n", "introduce bad value")
    _commit(repo, "a.py", "def f():\n    return 2\n", 'Revert "introduce bad value"')
    # a hotfix / fix commit
    _commit(repo, "a.py", "def f():\n    return 3\n", "fix: correct off-by-one in f")
    # a plain refactor (not a bug)
    _commit(repo, "a.py", "def f():\n    return 3  # ok\n", "refactor: tidy f")
    return repo


# --------------------------------------------------------------------------- #
# classify_commit heuristics                                                   #
# --------------------------------------------------------------------------- #
def test_classify_revert():
    label = classify_commit('Revert "introduce bad value"')
    assert label.known_bug is True
    assert label.source == "revert"


def test_classify_fix():
    label = classify_commit("fix: correct off-by-one in f")
    assert label.known_bug is True
    assert label.source in ("fix", "hotfix")


def test_classify_hotfix():
    label = classify_commit("hotfix: emergency patch for prod outage")
    assert label.known_bug is True
    assert label.source == "hotfix"


def test_classify_feature_is_not_a_bug():
    label = classify_commit("feat: add a brand new feature")
    assert label.known_bug is False


def test_classify_is_case_insensitive():
    assert classify_commit("FIX: thing").known_bug is True
    assert classify_commit("Reverts the previous commit").known_bug is True


def test_classify_assigns_a_category():
    label = classify_commit("fix: SQL injection in login handler")
    assert label.bug_category in (
        "correctness",
        "security",
        "performance",
        "tests",
        "maintainability",
    )
    # security keywords should bias toward security
    assert label.bug_category == "security"


# --------------------------------------------------------------------------- #
# build_golden_set against the temp repo                                       #
# --------------------------------------------------------------------------- #
def test_build_finds_bug_samples(git_repo: Path):
    samples = build_golden_set(git_repo)
    assert all(isinstance(s, GoldenSample) for s in samples)
    messages = [s.message for s in samples]
    assert any("Revert" in m for m in messages)
    assert any(m.startswith("fix:") for m in messages)
    # every sample carries a diff and a commit sha
    for s in samples:
        assert s.diff.strip()
        assert len(s.commit) >= 7
        assert s.repo


def test_build_only_includes_bug_labeled_by_default(git_repo: Path):
    samples = build_golden_set(git_repo)
    assert samples, "expected at least the revert and fix commits"
    assert all(s.known_bug for s in samples)
    # the refactor and feat commits are excluded
    assert not any("refactor" in s.message for s in samples)


def test_build_include_clean_yields_negatives(git_repo: Path):
    samples = build_golden_set(git_repo, include_clean=True)
    assert any(s.known_bug for s in samples)
    assert any(not s.known_bug for s in samples)


def test_build_respects_max_samples(git_repo: Path):
    samples = build_golden_set(git_repo, max_samples=1)
    assert len(samples) == 1


def test_build_raises_on_non_git_dir(tmp_path: Path):
    with pytest.raises(Exception):
        build_golden_set(tmp_path / "not-a-repo")


# --------------------------------------------------------------------------- #
# JSONL serialization                                                          #
# --------------------------------------------------------------------------- #
def test_write_and_read_jsonl_roundtrip(git_repo: Path, tmp_path: Path):
    samples = build_golden_set(git_repo)
    out = tmp_path / "golden.jsonl"
    n = write_jsonl(samples, out)
    assert n == len(samples)
    assert out.exists()
    # each line is valid JSON
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(samples)
    for line in lines:
        json.loads(line)
    loaded = list(iter_jsonl(out))
    assert [s.sample_id for s in loaded] == [s.sample_id for s in samples]
    assert [s.known_bug for s in loaded] == [s.known_bug for s in samples]


def test_sample_to_dict_is_json_serializable():
    s = GoldenSample(
        sample_id="x",
        repo="r",
        commit="deadbeef",
        diff="@@",
        known_bug=True,
        bug_category="correctness",
        source="fix",
        message="fix: thing",
    )
    d = s.to_dict()
    json.dumps(d)
    again = GoldenSample.from_dict(d)
    assert again == s
