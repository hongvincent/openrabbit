"""Live smoke: the eval harness ONLINE against openrabbit's OWN git history.

Runs the real review pipeline end-to-end with REAL providers — Nova 2 Lite
finder + GPT-5.4 cross-family verifier + GPT-5.4 judge — over a tiny slice of
this repo's own commits. This is the integration that 1000+ unit tests cannot
cover: it
exercises ``orchestrator.review`` -> ``run_lenses`` -> ``verify_findings`` ->
``judge`` against live Bedrock, the same wiring the CLI's ``eval --online`` uses.

Asserts the run COMPLETES (no ProviderError bubbling up), is stamped ``live``,
made real calls (``call_count > 0``), and produced a real ``fp_rate`` in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openrabbit.config import Config, ModelRole, ReviewConfig
from openrabbit.eval.runner import LIVE_MODE, run_eval
from openrabbit.pipeline import orchestrator as orch
from openrabbit.providers.base import ProviderError

from .conftest import GPT_MODEL, GPT_REGION, NOVA_MODEL, NOVA_REGION

#: This repository — the eval mines its own git history for the golden set.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Tiny cap: a real (paid) run, so keep the corpus small per the live-call budget.
EVAL_LIMIT = 2


def _live_config() -> Config:
    """A Config with the verified live model roles (global Nova 2 Lite + GPT-5.4).

    Mirrors the CLI's ``eval --online`` wiring: a Nova 2 Lite finder in Seoul and
    a GPT-5.4 cross-family verifier/judge in us-east-2. ``reasoning_effort=low``
    on the verifier keeps the (paid) judge/verify calls cheap.
    """
    return Config(
        review=ReviewConfig(),
        model_roles={
            "finder": ModelRole(model=NOVA_MODEL, region=NOVA_REGION),
            "verifier": ModelRole(
                model=GPT_MODEL,
                region=GPT_REGION,
                options={"reasoning_effort": "low"},
            ),
        },
    )


@pytest.mark.live
def test_live_eval_online_smoke(aws_profile_env: str, bearer_token: str) -> None:
    config = _live_config()

    # Build the real providers exactly as the CLI does: Nova 2 Lite finder +
    # GPT-5.4 verifier driving BOTH the in-pipeline verify pass and the judge.
    finder = orch.model_factory(config.model_roles["finder"])
    verifier = orch.model_factory(config.model_roles["verifier"])

    try:
        report = run_eval(
            REPO_ROOT,
            provider=finder,
            verifier_provider=verifier,
            judge_provider=verifier,
            config=config,
            limit=EVAL_LIMIT,
            mode=LIVE_MODE,
        )
    except ProviderError as exc:  # a real-API failure must surface, not be hidden
        pytest.fail(f"live eval raised a ProviderError (real-API bug): {exc}")

    # The run is stamped live -> fp_rate() returns a real measurement, not None.
    assert report.mode == LIVE_MODE

    # call_count > 0 proves REAL provider calls happened (not fixtures / empty
    # corpus). If the golden set mined empty this would be 0 and the FP rate
    # would be meaningless.
    assert report.call_count > 0, (
        "expected real provider calls — call_count==0 means nothing actually ran "
        "(empty golden corpus or short-circuited pipeline)"
    )

    fp_rate = report.fp_rate()
    assert fp_rate is not None, "live run must yield a real (non-None) fp_rate"
    assert 0.0 <= fp_rate <= 1.0, f"fp_rate out of range: {fp_rate}"

    # Surface the real number so the run is visible in `-s` / -rA output.
    print(
        f"\n[live eval] repo={report.repo} mode={report.mode} "
        f"calls={report.call_count} golden={report.golden_count} "
        f"controls={report.control_count} fp_rate={fp_rate:.4f}"
    )
