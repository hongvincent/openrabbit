"""Onboarding/auth wiring guards for the review action + reusable workflow.

These are *pure structural*, fully-offline tests (no network, no live creds).
They parse ``.github/workflows/reusable-workflow.yml`` and ``actions/action.yml``
with pyyaml and assert the two onboarding invariants an in-house user actually
depends on to get a working review run:

(a) GPT-5.5 verifier bearer-token auth is wired. The verifier provider reads
    ``AWS_BEARER_TOKEN_BEDROCK`` (see ``openrabbit/providers/openai_responses.py``);
    the OIDC -> STS step only yields raw AWS creds, so a dedicated step MUST mint
    a short-lived Bedrock bearer token and export it to ``$GITHUB_ENV`` *before*
    the review runs — otherwise the verifier (default ``model_roles.verifier``
    ships ``openai.gpt-5.5``) hard-fails with "no Bedrock bearer token found".

(b) The review job installs openrabbit ITSELF. The reviewed repo can be Node/Go/
    anything with no Python ``pyproject.toml``; a bare ``uv sync`` either errors
    (no project to sync) or syncs the *wrong* project, so ``openrabbit`` is never
    on PATH. The step must install openrabbit independent of the caller repo
    (``uvx --from git+...`` / ``uv run --with git+...`` / a pinned ``--project``
    checkout).

The default ``.openrabbit.yaml`` rendered by ``gh openrabbit init`` configures an
``openai.*`` verifier (asserted below), so the bearer-token step is required for
the shipped defaults. The token is minted unconditionally (documented in the
workflow) which is a strict superset of "guarded on an openai.* verifier".
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
REUSABLE_WORKFLOW = WORKFLOWS_DIR / "reusable-workflow.yml"
COMPOSITE_ACTION = ACTIONS_DIR / "action.yml"

# The env var the verifier provider reads for the Bedrock bearer token.
BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"


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


def _iter_steps(node: Any):
    """Yield every step mapping (dict with a ``run`` or ``uses``) in a tree.

    Walks both the reusable workflow (``jobs.*.steps``) and the composite action
    (``runs.steps``) without caring about the exact nesting.
    """
    if isinstance(node, dict):
        steps = node.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    yield step
        for value in node.values():
            yield from _iter_steps(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_steps(item)


def _step_index_running(steps: list[dict[str, Any]], needle: str) -> int:
    """Index of the first step whose ``run:`` contains ``needle`` (-1 if none)."""
    for i, step in enumerate(steps):
        run = step.get("run")
        if isinstance(run, str) and needle in run:
            return i
    return -1


def _job_steps_running(tree: dict[str, Any], needle: str) -> list[dict[str, Any]]:
    """Return the linear step list of the job/action that runs ``needle``."""
    # Reusable workflow: jobs.<name>.steps
    for job in tree.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if isinstance(steps, list) and _step_index_running(steps, needle) >= 0:
            return [s for s in steps if isinstance(s, dict)]
    # Composite action: runs.steps
    runs = tree.get("runs")
    if isinstance(runs, dict):
        steps = runs.get("steps")
        if isinstance(steps, list) and _step_index_running(steps, needle) >= 0:
            return [s for s in steps if isinstance(s, dict)]
    return []


# --------------------------------------------------------------------------- #
# default config ships an openai.* verifier (so bearer auth IS required)       #
# --------------------------------------------------------------------------- #
def test_default_scaffold_config_has_openai_verifier() -> None:
    """The init-generated ``.openrabbit.yaml`` configures an ``openai.*`` model
    role, so the shipped defaults genuinely require a Bedrock bearer token."""
    from openrabbit.init import DetectedStack, _render_config_yaml

    rendered = _render_config_yaml(DetectedStack())
    config = yaml.safe_load(rendered)
    roles = config.get("model_roles", {})
    models = [r.get("model", "") for r in roles.values() if isinstance(r, dict)]
    assert any(m.startswith("openai.") for m in models), (
        "default .openrabbit.yaml must configure an openai.* verifier "
        f"(got models {models}) — otherwise the bearer-auth guard is vacuous"
    )


# --------------------------------------------------------------------------- #
# (a) the verifier credential (Bedrock bearer token) is provisioned            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_action_yaml_provides_verifier_credential(path: Path) -> None:
    """A step must mint + export ``AWS_BEARER_TOKEN_BEDROCK`` to ``$GITHUB_ENV``.

    The OIDC->STS step only yields raw AWS creds; the GPT-5.5 verifier needs the
    Bedrock bearer token. Without this the default (openai.* verifier) config
    hard-fails at runtime.
    """
    tree = _load_yaml(path)
    steps = _job_steps_running(tree, "openrabbit review")
    assert steps, f"{path.name}: no step runs `openrabbit review`"

    # Some step's run script must write AWS_BEARER_TOKEN_BEDROCK into $GITHUB_ENV.
    export_idx = -1
    for i, step in enumerate(steps):
        run = step.get("run")
        if not isinstance(run, str):
            continue
        if (
            BEARER_ENV in run
            and "GITHUB_ENV" in run
            and re.search(rf"{BEARER_ENV}=", run)
        ):
            export_idx = i
            break
    assert export_idx >= 0, (
        f"{path.name}: no step exports {BEARER_ENV} to $GITHUB_ENV "
        "(GPT-5.5 verifier bearer-token auth is never wired)"
    )

    # It must use the proven token generator, not a hand-rolled signer.
    minting_run = steps[export_idx]["run"]
    assert "aws-bedrock-token-generator" in minting_run, (
        f"{path.name}: bearer token must be minted via aws-bedrock-token-generator"
    )
    assert "provide_token" in minting_run, (
        f"{path.name}: bearer step must call provide_token(...)"
    )

    # The token must be masked in logs.
    assert "add-mask" in minting_run, (
        f"{path.name}: minted bearer token must be masked (::add-mask::)"
    )

    # It must be exported BEFORE the review step (env propagates via $GITHUB_ENV
    # only to *subsequent* steps).
    review_idx = _step_index_running(steps, "openrabbit review")
    assert export_idx < review_idx, (
        f"{path.name}: {BEARER_ENV} must be exported BEFORE the review step "
        f"(export at step {export_idx}, review at step {review_idx})"
    )


# --------------------------------------------------------------------------- #
# (b) the review job installs openrabbit ITSELF (not the caller repo)          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_review_job_installs_openrabbit(path: Path) -> None:
    """openrabbit must be installed independent of the reviewed repo.

    The reviewed repo may be Node/Go/etc. with no Python project, so a bare
    ``uv sync`` against it cannot put the ``openrabbit`` console script on PATH.
    The install must reference openrabbit itself: ``uvx --from git+...`` /
    ``uv run --with git+...`` / a pinned ``--project <dir>`` checkout.
    """
    text = _text(path)

    # The reviewed-repo bare `uv sync` (no project/with/from args) is the bug.
    bare_sync = re.search(r"(?m)^\s*run:\s*uv sync\s*$", text) or re.search(
        r"(?m)^\s*uv sync\s*$", text
    )
    assert not bare_sync, (
        f"{path.name}: bare `uv sync` syncs the REVIEWED repo, not openrabbit — "
        "openrabbit is never installed / not on PATH"
    )

    installs_self = (
        re.search(r"uvx\s+(--\S+\s+)*--from\s+git\+", text)
        or re.search(r"uv run\s+(--\S+\s+)*--with\s+git\+", text)
        or re.search(r"--with\s+git\+https://\S*openrabbit", text)
        or re.search(r"uv run\s+(--\S+\s+)*--project\s+\S+", text)
        or re.search(r"pip install\s+(--\S+\s+)*git\+https://\S*openrabbit", text)
    )
    assert installs_self, (
        f"{path.name}: must install openrabbit ITSELF "
        "(uvx --from git+ / uv run --with git+ / --project <checkout>), "
        "not `uv sync` against the caller repo"
    )

    # And the openrabbit source must be SHA-pinned, not a floating ref. The repo
    # convention uses an explicit <PINNED_SHA> placeholder the owner replaces.
    m = re.search(r"git\+https://\S*openrabbit(?:\.git)?@(\S+)", text)
    assert m, (
        f"{path.name}: openrabbit install must pin a git ref "
        "(git+https://.../openrabbit@<sha>)"
    )
    ref = m.group(1).rstrip("\"'")
    assert not ref.startswith("v"), (
        f"{path.name}: openrabbit pin is a floating tag: {ref}"
    )
    assert ref not in {"main", "master", "HEAD"}, (
        f"{path.name}: openrabbit pin is a branch ref: {ref}"
    )


@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_review_run_is_isolated_from_consumer_project(path: Path) -> None:
    """The openrabbit review ``uv run`` must ignore the consumer repo's project.

    The review runs from the checked-out consumer repo's working dir, so a bare
    ``uv run --with git+...openrabbit`` makes uv adopt the CONSUMER's pyproject
    and its ``requires-python`` — which can conflict with the workflow's pinned
    Python (observed live: consumer ``requires-python >=3.13`` vs runner 3.12 ->
    ``uv`` aborts before openrabbit ever starts). openrabbit is a TOOL, not a
    consumer dependency, so the invocation must pass ``--no-project`` (or
    ``--isolated``) to run in an ephemeral env decoupled from the consumer's
    project metadata.
    """
    text = path.read_text(encoding="utf-8")
    assert re.search(r"--with\s+git\+https://\S*openrabbit", text), (
        f"{path.name}: no `uv run --with git+...openrabbit` invocation"
    )
    assert re.search(r"--no-project|--isolated", text), (
        f"{path.name}: the openrabbit review `uv run` must use --no-project (or "
        "--isolated) so the consumer repo's pyproject/requires-python cannot "
        "constrain the tool's ephemeral env (live: consumer >=3.13 vs runner 3.12)"
    )


# --------------------------------------------------------------------------- #
# (c) MEDIUM: the review command engages the bot-login author filter           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_review_passes_bot_login(path: Path) -> None:
    """The review invocation passes ``--bot-login`` so the auto-resolve author
    dedup filter engages (otherwise prior bot comments are not recognized)."""
    text = _text(path)
    assert "--bot-login" in text, (
        f"{path.name}: review step must pass --bot-login (author dedup filter)"
    )
