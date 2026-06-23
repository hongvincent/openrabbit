"""Live-wiring guards for the config-as-policy BASE REF (SPEC §12 trust boundary).

These are *pure structural*, fully-offline tests (no network, no live creds).
They parse ``.github/workflows/reusable-workflow.yml``, ``actions/action.yml`` and
the org starter template with pyyaml + raw text and assert the one onboarding
invariant the trust boundary actually depends on at run time:

The review POLICY (gate / lenses / path_filters) is anchored to the *trusted*
base ref, not the attacker-controlled PR head. ``cli.py`` resolves that base ref
via ``--base-ref`` / ``$GITHUB_BASE_REF`` and reads it with
``git show <base_ref>:.openrabbit.yaml`` (see ``openrabbit/cli._load_base_config``).

The CI checkouts are *detached-HEAD on the head commit* (``ref: <commit sha>``),
so unless the workflow **fetches the base branch** into the local repo AND
**passes a resolvable base ref** to ``openrabbit review``, the ``git show`` lands
on an unknown ref, ``_load_base_config`` returns ``None``, and the trust boundary
silently degrades to the untrusted head policy. Offline unit tests never caught
this because they exercise ``_load_base_config`` against a hand-built local repo,
never the real CI call site.

Guards (per workflow + composite action + starter template):

(1) A resolvable base ref reaches ``openrabbit review`` via ``--base-ref`` and
    that ref is the explicit ``origin/<base>`` form (a remote-tracking ref that
    exists only after a fetch) — the most-robust value for a detached-HEAD
    checkout where a bare ``main`` has no local branch.
(2) The base branch is actually fetched before the review step (an explicit
    ``git fetch origin <base>`` so ``origin/<base>`` exists locally).
(3) The base ref value derives from the GitHub PR base (``github.base_ref`` /
    ``github.event.pull_request.base.ref`` / a piped ``inputs.base_ref``), not a
    hard-coded branch name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
ACTIONS_DIR = REPO_ROOT / "actions"
TEMPLATES_DIR = REPO_ROOT / "org" / ".github" / "workflow-templates"
REUSABLE_WORKFLOW = WORKFLOWS_DIR / "reusable-workflow.yml"
COMPOSITE_ACTION = ACTIONS_DIR / "action.yml"
STARTER_TEMPLATE = TEMPLATES_DIR / "openrabbit.yml"

# Files that actually invoke `openrabbit review` (so must wire the base ref end to
# end: fetch the base + pass --base-ref).
REVIEW_RUNNERS = [REUSABLE_WORKFLOW, COMPOSITE_ACTION]


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.exists(), f"missing CI file: {path}"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _step_index_running(steps: list[dict[str, Any]], needle: str) -> int:
    for i, step in enumerate(steps):
        run = step.get("run")
        if isinstance(run, str) and needle in run:
            return i
    return -1


def _job_steps_running(tree: dict[str, Any], needle: str) -> list[dict[str, Any]]:
    """Return the linear step list of the job/action that runs ``needle``."""
    for job in tree.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if isinstance(steps, list) and _step_index_running(steps, needle) >= 0:
            return [s for s in steps if isinstance(s, dict)]
    runs = tree.get("runs")
    if isinstance(runs, dict):
        steps = runs.get("steps")
        if isinstance(steps, list) and _step_index_running(steps, needle) >= 0:
            return [s for s in steps if isinstance(s, dict)]
    return []


# --------------------------------------------------------------------------- #
# (1) a resolvable base ref reaches `openrabbit review`                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", REVIEW_RUNNERS)
def test_review_passes_resolvable_base_ref(path: Path) -> None:
    """The review invocation passes ``--base-ref origin/<base>``.

    ``cli.py`` re-anchors the policy from the base ref via
    ``git show <base_ref>:.openrabbit.yaml``. The checkout is detached-HEAD on the
    head commit where a bare ``<base>`` has no local branch, so the workflow
    passes the explicit remote-tracking ``origin/<base>`` form — the
    most-robust value that resolves directly (cli.py also retries an ``origin/``
    fallback as belt-and-suspenders, but shipping the explicit form keeps the
    boundary independent of that fallback). Either way the ref only resolves once
    the base branch is fetched (guarded separately below).
    """
    tree = _load_yaml(path)
    steps = _job_steps_running(tree, "openrabbit review")
    assert steps, f"{path.name}: no step runs `openrabbit review`"
    review_idx = _step_index_running(steps, "openrabbit review")
    review_run = steps[review_idx]["run"]

    assert "--base-ref" in review_run, (
        f"{path.name}: review step must pass --base-ref so cli.py can anchor the "
        "review policy to the trusted base ref (config-as-policy trust boundary)"
    )
    # The value passed must be the explicit origin/<base> remote-tracking form
    # (resolvable directly after a fetch), not a bare detached branch name.
    assert re.search(r"--base-ref\s+\S*origin/", review_run), (
        f"{path.name}: --base-ref must reference an explicit origin/<base> "
        "remote-tracking ref (resolvable directly on a detached-HEAD checkout "
        "after the fetch), not a bare branch name"
    )


# --------------------------------------------------------------------------- #
# (2) the base branch is actually fetched before the review step               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", REVIEW_RUNNERS)
def test_base_branch_is_fetched_before_review(path: Path) -> None:
    """Some step must ``git fetch origin <base>`` before the review step.

    The checkout pins the head commit (detached HEAD); ``origin/<base>`` does not
    exist locally until fetched. Without the fetch, ``--base-ref origin/<base>``
    is an unknown ref and ``_load_base_config`` silently returns ``None`` (policy
    degrades to the untrusted head).
    """
    tree = _load_yaml(path)
    steps = _job_steps_running(tree, "openrabbit review")
    assert steps, f"{path.name}: no step runs `openrabbit review`"
    review_idx = _step_index_running(steps, "openrabbit review")

    fetch_idx = -1
    for i, step in enumerate(steps):
        run = step.get("run")
        if isinstance(run, str) and re.search(r"git\s+fetch\s+(?:--\S+\s+)*origin", run):
            fetch_idx = i
            break
    assert fetch_idx >= 0, (
        f"{path.name}: no step runs `git fetch origin <base>` — origin/<base> "
        "never exists locally so the base ref is unresolvable"
    )
    assert fetch_idx < review_idx, (
        f"{path.name}: base branch must be fetched BEFORE the review step "
        f"(fetch at step {fetch_idx}, review at step {review_idx})"
    )


# --------------------------------------------------------------------------- #
# (3) the base ref derives from the GitHub PR base, not a hard-coded branch     #
# --------------------------------------------------------------------------- #
def test_reusable_workflow_accepts_base_ref_input() -> None:
    """The reusable workflow must accept a ``base_ref`` workflow_call input.

    It has no ``pull_request`` event context of its own, so the caller (org
    template) must pipe the PR base branch in. Without the input there is no way
    for the reusable workflow to know which base to fetch / anchor to.
    """
    tree = _load_yaml(REUSABLE_WORKFLOW)
    on = tree.get(True, tree.get("on", {}))  # PyYAML parses bare `on:` as True
    wc = on.get("workflow_call", {}) if isinstance(on, dict) else {}
    inputs = wc.get("inputs", {}) if isinstance(wc, dict) else {}
    assert "base_ref" in inputs, (
        "reusable-workflow.yml: must declare a `base_ref` workflow_call input "
        "(the caller pipes github.base_ref in; the reusable workflow has no PR "
        "event context of its own)"
    )


def test_starter_template_pipes_pr_base_ref() -> None:
    """The org starter template must pass the PR base branch to the reusable wf.

    GitHub exposes the PR target branch as ``github.base_ref`` /
    ``github.event.pull_request.base.ref`` in a ``pull_request`` workflow. The
    starter must wire one of those into the reusable workflow's ``base_ref``
    input so the policy anchors to the real target branch, not a hard-coded name.
    """
    text = _text(STARTER_TEMPLATE)
    assert re.search(r"base_ref:\s*\$\{\{\s*github\.base_ref\s*\}\}", text) or re.search(
        r"base_ref:\s*\$\{\{\s*github\.event\.pull_request\.base\.ref\s*\}\}", text
    ), (
        "openrabbit.yml (starter): must pipe github.base_ref (or "
        "github.event.pull_request.base.ref) into the reusable workflow's "
        "base_ref input"
    )


def test_composite_action_derives_base_from_github_context() -> None:
    """The composite action must derive the base ref from the GitHub PR context.

    A composite action runs inside a caller's ``pull_request`` job, so
    ``github.base_ref`` is directly available. The fetch + --base-ref must use it
    (not a literal branch), so the boundary tracks the real target branch.
    """
    text = _text(COMPOSITE_ACTION)
    assert "github.base_ref" in text or "github.event.pull_request.base.ref" in text, (
        "action.yml: must derive the base ref from github.base_ref "
        "(or github.event.pull_request.base.ref), not a hard-coded branch"
    )
