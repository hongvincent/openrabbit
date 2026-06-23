"""Wording guard: CI artifacts reflect the GPT-5.4 DEFAULT verifier.

The live-verified default verifier is ``openai.gpt-5.4`` (see
``openrabbit/init.py`` scaffold + ``openrabbit/bedrock_models.py``). GPT-5.5 is
still *supported* (a strict-region mantle model), but the CI comments that name
the verifier in the *default* path used to say only "GPT-5.5", which is stale and
misleads an onboarding reader about which model their default config runs.

These offline tests assert the owned CI artifacts mention the GPT-5.4 default
verifier, and that the file is internally consistent (where it talks about the
default verifier it does not name only 5.5). They do NOT forbid the string
"5.5" outright — the region-allow-list note is historically accurate for both
5.4 and 5.5 (both live only on the us-east-1/2 mantle endpoint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REUSABLE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "reusable-workflow.yml"
COMPOSITE_ACTION = REPO_ROOT / "actions" / "action.yml"

CI_ARTIFACTS = [REUSABLE_WORKFLOW, COMPOSITE_ACTION]


@pytest.mark.parametrize("path", CI_ARTIFACTS)
def test_ci_artifact_names_gpt54_default(path: Path) -> None:
    """The CI artifact references the GPT-5.4 default verifier (not only 5.5)."""
    text = path.read_text(encoding="utf-8")
    assert "5.4" in text, (
        f"{path.name}: must reference the GPT-5.4 default verifier "
        "(the live-verified default is openai.gpt-5.4, not 5.5)"
    )


@pytest.mark.parametrize("path", CI_ARTIFACTS)
def test_ci_artifact_keeps_both_supported_note(path: Path) -> None:
    """Where the artifact names the verifier model it acknowledges 5.4 AND 5.5.

    Both are supported; the region constraint is shared. The artifact must not
    regress to a 5.5-only description of the default verifier.
    """
    text = path.read_text(encoding="utf-8")
    assert "5.4/5.5" in text or ("5.4" in text and "5.5" in text), (
        f"{path.name}: verifier wording must reflect that 5.4 (default) and 5.5 "
        "are both supported"
    )
