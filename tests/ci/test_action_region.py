"""Region-correctness guard for the GPT-5.5 verifier bearer-token mint step.

Pure structural, fully-offline tests (no network, no live creds). They parse
``.github/workflows/reusable-workflow.yml`` and ``actions/action.yml`` with pyyaml
and assert the bearer-token mint step is region-scoped to the VERIFIER's region,
not the general ``aws_region`` (which is typically the Nova model region).

WHY THIS MATTERS (the bug this guards against)
----------------------------------------------
The GPT-5.5 verifier runs ONLY on the ``bedrock-mantle`` us-east-1/us-east-2
endpoint (default us-east-2). But ``aws_region`` is the primary OIDC/STS + Nova
model region, which in-house users routinely set to a Nova inference-profile
region such as ``ap-northeast-2`` (Seoul). A Bedrock bearer token minted FOR
Seoul and then presented to the us-east-2 mantle endpoint is rejected (401):
``provide_token(region=...)`` bakes the region into the SigV4 scope of the token.

So the verifier bearer token MUST be minted for a dedicated verifier region
(``verifier_region``, default ``us-east-2``) that is independent of the Nova
``aws_region``. These tests fail if the mint step regresses to keying the bearer
token off ``aws_region``.
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

BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"

# The us-east-{1,2} mantle endpoints are the only place GPT-5.5 lives; us-east-2
# is the shipped default verifier region.
_VERIFIER_REGIONS = {"us-east-1", "us-east-2"}


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.exists(), f"missing CI file: {path}"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _all_steps(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every step mapping from a reusable workflow or composite action."""
    steps: list[dict[str, Any]] = []
    for job in tree.get("jobs", {}).values():
        if isinstance(job, dict) and isinstance(job.get("steps"), list):
            steps.extend(s for s in job["steps"] if isinstance(s, dict))
    runs = tree.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
        steps.extend(s for s in runs["steps"] if isinstance(s, dict))
    return steps


def _mint_step(tree: dict[str, Any]) -> dict[str, Any]:
    """Return the step that mints + exports the Bedrock bearer token."""
    for step in _all_steps(tree):
        run = step.get("run")
        if (
            isinstance(run, str)
            and BEARER_ENV in run
            and "GITHUB_ENV" in run
            and "provide_token" in run
        ):
            return step
    raise AssertionError("no bearer-token mint step found")


def _verifier_region_input(tree: dict[str, Any]) -> Any:
    """Return the declared default of the dedicated ``verifier_region`` input."""
    # Reusable workflow: on.workflow_call.inputs.verifier_region
    on = tree.get("on", tree.get(True, {}))
    if isinstance(on, dict):
        wc = on.get("workflow_call")
        if isinstance(wc, dict):
            inputs = wc.get("inputs") or {}
            spec = inputs.get("verifier_region")
            if isinstance(spec, dict):
                return spec.get("default")
    # Composite action: inputs.verifier_region
    inputs = tree.get("inputs") or {}
    spec = inputs.get("verifier_region")
    if isinstance(spec, dict):
        return spec.get("default")
    return None


# --------------------------------------------------------------------------- #
# the bearer token is minted for the VERIFIER region, not the Nova aws_region   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_bearer_mint_uses_verifier_region_not_aws_region(path: Path) -> None:
    """The mint step must NOT key the bearer token off ``aws_region``.

    GPT-5.5 lives only on the us-east-{1,2} mantle endpoint. Minting the bearer
    token with the Nova ``aws_region`` (e.g. ap-northeast-2) yields a token the
    mantle endpoint rejects (401). The mint step must use a dedicated verifier
    region input/value instead.
    """
    tree = _load_yaml(path)
    step = _mint_step(tree)
    env = step.get("env") or {}

    # The step's env must NOT carry the general aws_region into the mint.
    env_values = " ".join(str(v) for v in env.values())
    assert "inputs.aws_region" not in env_values, (
        f"{path.name}: bearer mint step still references inputs.aws_region "
        "(token would be minted for the Nova region, not the GPT-5.5 mantle "
        "endpoint — 401 at runtime)"
    )

    # And the run body must not provide_token() from an aws_region-derived var.
    run = step["run"]
    assert "aws_region" not in run, (
        f"{path.name}: mint run script must not reference aws_region"
    )


@pytest.mark.parametrize("path", [REUSABLE_WORKFLOW, COMPOSITE_ACTION])
def test_bearer_mint_region_is_us_east(path: Path) -> None:
    """The verifier region the mint step uses resolves to us-east-{1,2}.

    Either a dedicated ``verifier_region`` input defaulting to us-east-1/2, or a
    literal us-east region hardcoded in the mint step env, is acceptable. A token
    minted anywhere else cannot authenticate to the GPT-5.5 mantle endpoint.
    """
    tree = _load_yaml(path)
    step = _mint_step(tree)
    env = step.get("env") or {}
    env_values = " ".join(str(v) for v in env.values())

    declared_default = _verifier_region_input(tree)
    uses_verifier_input = "inputs.verifier_region" in env_values
    has_us_east_default = declared_default in _VERIFIER_REGIONS
    has_literal_region = any(r in env_values for r in _VERIFIER_REGIONS)

    assert (uses_verifier_input and has_us_east_default) or has_literal_region, (
        f"{path.name}: bearer mint must use a verifier region in {_VERIFIER_REGIONS} "
        f"(verifier_region input default={declared_default!r}, "
        f"mint env={env!r})"
    )


def test_verifier_region_input_declared_with_us_east_default_reusable() -> None:
    """The reusable workflow exposes a ``verifier_region`` input, default us-east-2."""
    tree = _load_yaml(REUSABLE_WORKFLOW)
    on = tree.get("on", tree.get(True, {}))
    wc = on["workflow_call"]
    inputs = wc.get("inputs") or {}
    assert "verifier_region" in inputs, (
        "reusable workflow must declare a `verifier_region` input distinct from "
        "aws_region (GPT-5.5 mantle endpoint region)"
    )
    assert inputs["verifier_region"].get("default") == "us-east-2"


def test_verifier_region_input_declared_with_us_east_default_composite() -> None:
    """The composite action exposes a ``verifier_region`` input, default us-east-2."""
    tree = _load_yaml(COMPOSITE_ACTION)
    inputs = tree.get("inputs") or {}
    assert "verifier_region" in inputs, (
        "composite action must declare a `verifier_region` input distinct from "
        "aws_region (GPT-5.5 mantle endpoint region)"
    )
    assert inputs["verifier_region"].get("default") == "us-east-2"


def test_mint_log_line_does_not_echo_aws_region() -> None:
    """Defensive: the mint step's human log line must not claim the Nova region.

    (Catches a half-fix where the env is renamed but the echo still prints
    aws_region, which would mislead an operator debugging a 401.)
    """
    for path in (REUSABLE_WORKFLOW, COMPOSITE_ACTION):
        run = _mint_step(_load_yaml(path))["run"]
        # No bare `aws_region` token anywhere in the mint run script.
        assert not re.search(r"\baws_region\b", run), (
            f"{path.name}: mint run script references aws_region"
        )
