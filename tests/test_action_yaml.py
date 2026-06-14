"""Structural tests for the reusable GitHub Action (PRD §11, §12, item 4).

These are *pure structural* tests: they parse ``actions/reusable-workflow.yml``
and ``actions/action.yml`` with pyyaml and assert the CI-hardening invariants the
design spec requires. There are **no network calls** and no live credentials —
the SHA pins are verified by shape (40-hex), not by re-resolving them online.

Invariants asserted (SPEC §12 "CI hardening"):

* every ``uses:`` references a 40-hex commit SHA (no floating tags/branches);
* top-level ``permissions`` is least-privilege ``contents: read``;
* ``id-token: write`` is present (OIDC -> STS, keyless Bedrock auth);
* ``pull-requests: write`` is scoped to the review job (never top-level);
* the ``openrabbit review`` command is actually invoked;
* the reusable workflow declares the ``workflow_call`` trigger.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIONS_DIR = REPO_ROOT / "actions"
REUSABLE_WORKFLOW = ACTIONS_DIR / "reusable-workflow.yml"
COMPOSITE_ACTION = ACTIONS_DIR / "action.yml"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.exists(), f"missing action file: {path}"
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


def _all_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reusable() -> dict[str, Any]:
    return _load_yaml(REUSABLE_WORKFLOW)


@pytest.fixture(scope="module")
def composite() -> dict[str, Any]:
    return _load_yaml(COMPOSITE_ACTION)


# --------------------------------------------------------------------------- #
# SHA pinning (both files)                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_every_uses_is_sha_pinned(path: Path) -> None:
    tree = _load_yaml(path)
    uses_values = list(_iter_uses(tree))
    assert uses_values, f"{path} declares no `uses:` steps"
    for ref in uses_values:
        # Local/relative references (composite reuse of repo actions) are fine.
        if ref.startswith("./") or ref.startswith("../"):
            continue
        owner_repo, sep, ref_spec = ref.partition("@")
        assert sep == "@", f"`uses: {ref}` in {path.name} is not pinned (@<sha>)"
        assert SHA_RE.match(ref_spec), (
            f"`uses: {ref}` in {path.name} is not pinned to a 40-hex SHA"
        )


@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_pins_carry_version_comment(path: Path) -> None:
    """Each SHA-pinned `uses:` line carries a trailing `# vX` version comment."""
    text = _all_text(path)
    pin_lines = [
        line
        for line in text.splitlines()
        if "uses:" in line and "@" in line and not re.search(r"uses:\s*\.", line)
    ]
    assert pin_lines, f"{path.name} has no third-party pinned uses lines"
    for line in pin_lines:
        assert "#" in line.split("@", 1)[1], (
            f"pinned `uses` line lacks a trailing version comment: {line.strip()}"
        )


@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_no_floating_tag_or_branch_refs(path: Path) -> None:
    """Defensive: no `@vN`, `@main`, `@master` style floating refs."""
    for ref in _iter_uses(_load_yaml(path)):
        if ref.startswith("./") or ref.startswith("../"):
            continue
        _, _, ref_spec = ref.partition("@")
        assert not ref_spec.startswith("v"), f"floating tag ref: {ref}"
        assert ref_spec not in {"main", "master", "HEAD"}, f"branch ref: {ref}"


def test_third_party_actions_present(reusable: dict[str, Any]) -> None:
    """Checkout, setup-uv and configure-aws-credentials are all wired in."""
    refs = " ".join(_iter_uses(reusable))
    assert "actions/checkout@" in refs
    assert "astral-sh/setup-uv@" in refs
    assert "aws-actions/configure-aws-credentials@" in refs


# --------------------------------------------------------------------------- #
# reusable workflow: triggers + permissions                                   #
# --------------------------------------------------------------------------- #
def test_workflow_call_trigger_present(reusable: dict[str, Any]) -> None:
    on = _on_block(reusable)
    assert isinstance(on, dict), "reusable workflow `on:` must be a mapping"
    assert "workflow_call" in on, "reusable workflow must declare `workflow_call`"


def test_top_level_permissions_contents_read(reusable: dict[str, Any]) -> None:
    perms = reusable.get("permissions")
    assert isinstance(perms, dict), "top-level permissions must be a mapping"
    assert perms.get("contents") == "read", "top-level must be contents: read"
    # Least privilege: do not grant write at the top level.
    assert perms.get("pull-requests") in (None, "read"), (
        "pull-requests: write must NOT be granted at the top level"
    )


def test_id_token_write_present(reusable: dict[str, Any]) -> None:
    """OIDC needs id-token: write somewhere (top level or the review job)."""
    perms_blocks: list[dict[str, Any]] = []
    if isinstance(reusable.get("permissions"), dict):
        perms_blocks.append(reusable["permissions"])
    for job in reusable.get("jobs", {}).values():
        if isinstance(job, dict) and isinstance(job.get("permissions"), dict):
            perms_blocks.append(job["permissions"])
    assert any(b.get("id-token") == "write" for b in perms_blocks), (
        "id-token: write is required for OIDC -> STS keyless auth"
    )


def test_pull_requests_write_scoped_to_review_job(reusable: dict[str, Any]) -> None:
    jobs = reusable.get("jobs", {})
    assert jobs, "reusable workflow declares no jobs"
    review_jobs = [
        j
        for name, j in jobs.items()
        if isinstance(j, dict)
        and isinstance(j.get("permissions"), dict)
        and j["permissions"].get("pull-requests") == "write"
    ]
    assert review_jobs, "no job grants pull-requests: write (needed to post review)"
    # The review job that posts comments must also hold id-token: write (OIDC).
    assert any(j["permissions"].get("id-token") == "write" for j in review_jobs), (
        "the review job needs id-token: write for OIDC Bedrock auth"
    )


# --------------------------------------------------------------------------- #
# the review command is actually invoked (both files)                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_openrabbit_review_invoked(path: Path) -> None:
    text = _all_text(path)
    assert "openrabbit review" in text, f"{path.name} never runs `openrabbit review`"
    # The review must target a specific PR + commit and post results.
    assert "--repo" in text and "--pr" in text and "--commit" in text, (
        f"{path.name} review step is missing --repo/--pr/--commit"
    )
    assert "--post" in text, f"{path.name} review step is missing --post"
    assert "--config" in text, f"{path.name} review step is missing --config"


def test_reusable_runs_on_review_job_only_with_pr_write(
    reusable: dict[str, Any],
) -> None:
    """The job that invokes `openrabbit review` is the one holding pr:write."""
    jobs = reusable.get("jobs", {})
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps") or []
        runs_review = any(
            isinstance(s, dict)
            and isinstance(s.get("run"), str)
            and "openrabbit review" in s["run"]
            for s in steps
        )
        if runs_review:
            perms = job.get("permissions") or {}
            assert perms.get("pull-requests") == "write"
            assert perms.get("id-token") == "write"
            return
    pytest.fail("no job runs `openrabbit review`")


# --------------------------------------------------------------------------- #
# composite action shape                                                       #
# --------------------------------------------------------------------------- #
def test_composite_is_a_composite_action(composite: dict[str, Any]) -> None:
    runs = composite.get("runs")
    assert isinstance(runs, dict), "composite action needs a `runs:` mapping"
    assert runs.get("using") == "composite", "action.yml must be a composite action"
    assert isinstance(runs.get("steps"), list) and runs["steps"], (
        "composite action needs steps"
    )


def test_composite_declares_inputs(composite: dict[str, Any]) -> None:
    inputs = composite.get("inputs")
    assert isinstance(inputs, dict) and inputs, "composite action needs inputs"
    # The review needs to know which PR/commit to look at + where the config is.
    for required in ("pr", "commit", "config"):
        assert required in inputs, f"composite missing input: {required}"


def test_reusable_declares_workflow_call_inputs_and_secrets(
    reusable: dict[str, Any],
) -> None:
    on = _on_block(reusable)
    wc = on["workflow_call"]
    assert isinstance(wc, dict)
    inputs = wc.get("inputs") or {}
    secrets = wc.get("secrets") or {}
    # region + config path are tunable inputs; the AWS role ARN is a secret.
    assert "config" in inputs, "workflow_call must expose a `config` input"
    assert any("region" in k for k in inputs), "workflow_call must expose a region input"
    assert secrets, "workflow_call must declare secrets (AWS role ARN)"
