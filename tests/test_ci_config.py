"""Structural tests for openrabbit's *self* CI (PRD §12, checklist item 19).

These parse ``.github/workflows/ci.yml`` and ``.github/dependabot.yml`` (plus the
``[tool.ruff]`` config in ``pyproject.toml``) and assert the CI-hardening
invariants the design spec requires. They are **pure structural** tests — no
network calls and no live credentials. SHA pins are verified by shape (40-hex),
not by re-resolving them online.

Invariants asserted:

* the CI workflow triggers on ``pull_request`` and on push to the default branch;
* top-level ``permissions`` is least-privilege ``contents: read``;
* every ``uses:`` is pinned to a 40-hex commit SHA (no floating tags/branches),
  each carrying a trailing version comment;
* CI runs ``pytest`` with a coverage gate (``--cov-fail-under``) and runs
  ``ruff check`` (lint) + ``ruff format --check``;
* the test matrix covers Python 3.11 and 3.12;
* ``dependabot.yml`` covers both the ``github-actions`` and the Python
  (``pip``) ecosystems on a weekly cadence;
* ``[tool.ruff]`` exists in ``pyproject.toml`` and ``ruff`` is a dev dependency;
* ``ruff check .`` actually exits 0 on the current tree (run as a subprocess).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS_DIR / "ci.yml"
DEPENDABOT = REPO_ROOT / ".github" / "dependabot.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.exists(), f"missing config file: {path}"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _on_block(workflow: dict[str, Any]) -> Any:
    """Return the ``on:`` block.

    PyYAML parses the bare key ``on`` as the boolean ``True`` (the YAML 1.1
    "norway problem"), so accept either spelling.
    """
    if "on" in workflow:
        return workflow["on"]
    return workflow.get(True)


def _iter_uses(node: Any):
    """Yield every ``uses:`` string anywhere in a parsed YAML tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                yield value
            else:
                yield from _iter_uses(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_uses(item)


def _iter_run(node: Any):
    """Yield every ``run:`` string anywhere in a parsed YAML tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                yield value
            else:
                yield from _iter_run(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_run(item)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ci() -> dict[str, Any]:
    return _load_yaml(CI_WORKFLOW)


@pytest.fixture(scope="module")
def dependabot() -> dict[str, Any]:
    return _load_yaml(DEPENDABOT)


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return tomllib.loads(_text(PYPROJECT))


# --------------------------------------------------------------------------- #
# files exist                                                                 #
# --------------------------------------------------------------------------- #
def test_ci_config_files_exist() -> None:
    for path in (CI_WORKFLOW, DEPENDABOT, PYPROJECT):
        assert path.is_file(), f"missing CI config: {path}"


# --------------------------------------------------------------------------- #
# CI workflow: triggers                                                       #
# --------------------------------------------------------------------------- #
def test_ci_triggers_on_pull_request_and_push(ci: dict[str, Any]) -> None:
    on = _on_block(ci)
    assert isinstance(on, dict), "ci `on:` must be a mapping"
    assert "pull_request" in on, "CI must trigger on pull_request"
    assert "push" in on, "CI must trigger on push (to the default branch)"
    # The push trigger targets the default branch (main), not every branch.
    push = on["push"]
    assert isinstance(push, dict) and "branches" in push, (
        "push trigger must scope to specific branches (the default branch)"
    )
    assert "main" in push["branches"], "push must target the default branch `main`"


# --------------------------------------------------------------------------- #
# CI workflow: permissions (least privilege)                                  #
# --------------------------------------------------------------------------- #
def test_ci_top_level_permissions_contents_read(ci: dict[str, Any]) -> None:
    perms = ci.get("permissions")
    assert isinstance(perms, dict), "top-level permissions must be a mapping"
    assert perms.get("contents") == "read", "top-level must be contents: read"
    # Least privilege: a lint/test workflow needs no write scopes at all.
    for scope, level in perms.items():
        assert level != "write", (
            f"CI must not grant any write scope at the top level (got {scope}: write)"
        )


# --------------------------------------------------------------------------- #
# CI workflow: SHA-pinned actions                                             #
# --------------------------------------------------------------------------- #
def test_ci_every_uses_is_sha_pinned(ci: dict[str, Any]) -> None:
    uses_values = list(_iter_uses(ci))
    assert uses_values, "CI declares no `uses:` steps"
    for ref in uses_values:
        if ref.startswith("./") or ref.startswith("../"):
            continue
        owner_repo, sep, ref_spec = ref.partition("@")
        assert sep == "@", f"`uses: {ref}` is not pinned (@<sha>)"
        assert SHA_RE.match(ref_spec), f"`uses: {ref}` is not pinned to a 40-hex SHA"


def test_ci_pins_carry_version_comment() -> None:
    text = _text(CI_WORKFLOW)
    pin_lines = [
        line
        for line in text.splitlines()
        if "uses:" in line and "@" in line and not re.search(r"uses:\s*\.", line)
    ]
    assert pin_lines, "CI has no third-party pinned uses lines"
    for line in pin_lines:
        assert "#" in line.split("@", 1)[1], (
            f"pinned `uses` line lacks a trailing version comment: {line.strip()}"
        )


def test_ci_no_floating_tag_or_branch_refs(ci: dict[str, Any]) -> None:
    for ref in _iter_uses(ci):
        if ref.startswith("./") or ref.startswith("../"):
            continue
        _, _, ref_spec = ref.partition("@")
        assert not ref_spec.startswith("v"), f"floating tag ref: {ref}"
        assert ref_spec not in {"main", "master", "HEAD"}, f"branch ref: {ref}"


# --------------------------------------------------------------------------- #
# CI workflow: steps (pytest + coverage gate, ruff lint + format)             #
# --------------------------------------------------------------------------- #
def test_ci_sets_up_uv(ci: dict[str, Any]) -> None:
    refs = " ".join(_iter_uses(ci))
    assert "astral-sh/setup-uv@" in refs, "CI must set up uv (astral-sh/setup-uv)"


def test_ci_runs_uv_sync_dev_extra(ci: dict[str, Any]) -> None:
    runs = "\n".join(_iter_run(ci))
    assert "uv sync" in runs, "CI must install deps via `uv sync`"
    assert "--extra dev" in runs, "CI must install the dev extra (`--extra dev`)"


def test_ci_runs_pytest_with_coverage_gate(ci: dict[str, Any]) -> None:
    runs = "\n".join(_iter_run(ci))
    assert "pytest" in runs, "CI must run pytest"
    # A coverage gate that FAILS the build below a threshold.
    m = re.search(r"--cov-fail-under=(\d+)", runs)
    assert m, "CI must enforce a coverage gate (--cov-fail-under=N)"
    threshold = int(m.group(1))
    assert threshold >= 95, (
        f"coverage gate must be >= 95% (found --cov-fail-under={threshold})"
    )


def test_ci_runs_ruff_lint_and_format_check(ci: dict[str, Any]) -> None:
    runs = "\n".join(_iter_run(ci))
    assert re.search(r"ruff check", runs), "CI must run `ruff check` (lint)"
    assert re.search(r"ruff format --check", runs), (
        "CI must run `ruff format --check` (format gate)"
    )


# --------------------------------------------------------------------------- #
# CI workflow: Python matrix                                                  #
# --------------------------------------------------------------------------- #
def test_ci_matrix_covers_311_and_312(ci: dict[str, Any]) -> None:
    jobs = ci.get("jobs", {})
    assert jobs, "CI declares no jobs"
    matrices: list[Any] = []
    for job in jobs.values():
        if isinstance(job, dict):
            strategy = job.get("strategy")
            if isinstance(strategy, dict) and isinstance(strategy.get("matrix"), dict):
                matrices.append(strategy["matrix"])
    assert matrices, "CI must declare a build matrix"
    # Collect every python-version value across all matrices.
    versions: set[str] = set()
    for matrix in matrices:
        for key, value in matrix.items():
            if "python" in key.lower() and isinstance(value, list):
                versions.update(str(v) for v in value)
    assert "3.11" in versions, f"matrix must include Python 3.11 (got {versions})"
    assert "3.12" in versions, f"matrix must include Python 3.12 (got {versions})"


# --------------------------------------------------------------------------- #
# dependabot.yml                                                              #
# --------------------------------------------------------------------------- #
def test_dependabot_version_is_2(dependabot: dict[str, Any]) -> None:
    assert dependabot.get("version") == 2, "dependabot config must be version 2"


def test_dependabot_covers_github_actions_and_python(
    dependabot: dict[str, Any],
) -> None:
    updates = dependabot.get("updates")
    assert isinstance(updates, list) and updates, "dependabot has no updates"
    ecosystems = {u.get("package-ecosystem") for u in updates if isinstance(u, dict)}
    assert "github-actions" in ecosystems, (
        "dependabot must cover github-actions (to bump pinned SHAs)"
    )
    # uv projects read pyproject.toml; the "pip" ecosystem is how Dependabot
    # tracks PEP 621 Python deps.
    assert ecosystems & {"pip", "uv"}, (
        "dependabot must cover Python deps (pip/uv ecosystem)"
    )


def test_dependabot_schedules_are_weekly(dependabot: dict[str, Any]) -> None:
    for update in dependabot["updates"]:
        if not isinstance(update, dict):
            continue
        schedule = update.get("schedule", {})
        assert isinstance(schedule, dict), "each update needs a schedule"
        assert schedule.get("interval") == "weekly", (
            f"update {update.get('package-ecosystem')!r} must be weekly"
        )


# --------------------------------------------------------------------------- #
# pyproject: [tool.ruff] config + ruff dev dependency                          #
# --------------------------------------------------------------------------- #
def test_pyproject_has_tool_ruff(pyproject: dict[str, Any]) -> None:
    tool = pyproject.get("tool", {})
    assert "ruff" in tool, "pyproject must define [tool.ruff]"
    ruff_cfg = tool["ruff"]
    assert isinstance(ruff_cfg, dict) and ruff_cfg, "[tool.ruff] must be non-empty"
    # A lint section selecting rules is what makes `ruff check` meaningful.
    lint = ruff_cfg.get("lint", {})
    assert isinstance(lint.get("select"), list) and lint["select"], (
        "[tool.ruff.lint] must select a rule set"
    )


def test_ruff_is_a_dev_dependency(pyproject: dict[str, Any]) -> None:
    optional = pyproject["project"].get("optional-dependencies", {})
    dev = optional.get("dev", [])
    assert any(req.lower().startswith("ruff") for req in dev), (
        "ruff must be declared in the dev optional-dependencies"
    )


# --------------------------------------------------------------------------- #
# `ruff check .` actually passes on the current tree (no network)             #
# --------------------------------------------------------------------------- #
def test_ruff_check_exits_zero_on_current_tree() -> None:
    """`ruff check .` must exit 0 on the committed codebase.

    This guards the invariant that the lint gate in CI is green: a future change
    that introduces a lint error fails here locally too. Skipped only if the
    ruff binary is unavailable in the environment (it is a dev dependency).
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        # Fall back to `python -m ruff` if the console script is not on PATH.
        probe = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if probe.returncode != 0:
            pytest.skip("ruff not installed in this environment")
        cmd = [sys.executable, "-m", "ruff", "check", "."]
    else:
        cmd = [ruff, "check", "."]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"`ruff check .` failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
