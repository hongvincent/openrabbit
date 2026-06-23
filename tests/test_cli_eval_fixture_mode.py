"""CLI eval fixture-mode honesty (cli-security-demo finding 1).

The default/offline ``openrabbit eval`` runs a scripted-fixture provider + an
always-``match`` judge. That run is NOT a real scorecard, so it must:

* stamp ``mode == "fixture"`` on the human output AND the ``--json`` payload,
* never emit a numeric false-positive rate as if it were measured (the offline
  scorecard prints ``N/A — fixture`` instead), and
* hard-error (never pass / never exit 0) under ``--require-pass`` so a fixture
  run can NEVER be cited as a passing CI gate. The real scorecard is reserved
  strictly for ``--online``.

These are pure-CLI assertions: no network, no credentials, no AWS/GitHub SDK.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from openrabbit.cli import main


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one 'fix' commit so the golden set is non-empty."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    run("init")
    run("config", "user.email", "t@t.t")
    run("config", "user.name", "t")
    (repo / "app.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    run("add", ".")
    run("commit", "-m", "initial")
    (repo / "app.py").write_text("x = 1\ny = 22\nz = 3\n", encoding="utf-8")
    run("add", ".")
    run("commit", "-m", "fix: correct a bug in app")
    return repo


def test_offline_eval_json_stamps_fixture_mode(git_repo: Path, capsys):
    rc = main(["eval", "--repo", str(git_repo), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # The offline run is fixture-driven, not a measured scorecard.
    assert payload.get("mode") == "fixture", payload


def test_offline_eval_json_does_not_emit_numeric_fp_rate(git_repo: Path, capsys):
    # A fixture run's "FP rate" is meaningless (always-match judge), so the JSON
    # must NOT advertise a numeric falsePositiveRate as if it were measured.
    rc = main(["eval", "--repo", str(git_repo), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    fp = payload.get("falsePositiveRate")
    assert fp == "N/A — fixture", payload


def test_offline_eval_pretty_marks_fixture_and_hides_fp_number(git_repo: Path, capsys):
    rc = main(["eval", "--repo", str(git_repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fixture" in out.lower()
    # The human output must not present a bare measured FP percentage as a result.
    assert "N/A" in out


def test_offline_require_pass_hard_errors(git_repo: Path, capsys):
    # A fixture run can NEVER be cited as a passing gate: --require-pass offline
    # is a hard error (non-zero), regardless of the meaningless fixture verdict.
    rc = main(["eval", "--repo", str(git_repo), "--require-pass"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "require-pass" in err or "require_pass" in err or "online" in err


def test_offline_require_pass_can_never_exit_zero(git_repo: Path, capsys):
    # Even with a limit (which changes corpus sizes), offline --require-pass must
    # never be a passing (rc == 0) gate.
    rc = main(["eval", "--repo", str(git_repo), "--require-pass", "--limit", "1"])
    assert rc != 0
