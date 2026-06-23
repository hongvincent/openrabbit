"""Verifier reasoning-token budget starvation must be observable + headroomed.

Adversarial scenario (item 1, the bug): the shipped verifier role runs with
``reasoning_effort: medium``. ``verify._batch_max_tokens`` floored the budget at
``MIN_VERIFY_MAX_TOKENS == 512`` and passed it as the GPT verifier's
``max_output_tokens``. Under medium reasoning the (billed) reasoning tokens eat
that tiny budget, so the turn comes back ``status=incomplete /
max_output_tokens`` (FinishReason.MAX_TOKENS) with ZERO verdicts. The soft-skip
fallback then posts UN-verified HIGH/CRITICAL findings, logged ONLY as the
generic "unparseable" path — indistinguishable from a real content-filter
refusal, so reasoning-starvation was invisible to the operator.

Two fixes, both pinned here, OFFLINE (FakeProvider / canned results):

1. Observability: a verifier result with ``finish_reason == MAX_TOKENS`` and no
   ``verify_findings`` tool call must log a DISTINCT "reasoning-starved/budget"
   warning, NOT the generic refusal/unparseable warning.
2. Headroom: the per-batch budget must give a reasoning verifier real room — the
   floor is raised substantially, and when the caller declares the verifier's
   reasoning effort the budget scales above that floor. The budget that actually
   reaches the provider's ``complete()`` is asserted (the live call site).
"""

from __future__ import annotations

import logging

from openrabbit.domain import CompletionResult, FinishReason, ToolCall, Usage
from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import verify as verify_mod
from openrabbit.providers.base import FakeProvider


def _finding(conf: float, *, rule: str, severity: str = "high") -> Finding:
    fp = compute_fingerprint("src/api/auth.py", rule, f"ctx-{rule}")
    return Finding(
        file="src/api/auth.py",
        start_line=12,
        end_line=14,
        side="RIGHT",
        severity=severity,
        category="security",
        confidence=conf,
        title="t",
        body="b",
        rule_id=rule,
        fingerprint=fp,
    )


def _max_tokens_no_verdicts() -> CompletionResult:
    """A verifier turn that ran out of budget mid-reasoning: truncated, no tool call."""
    return CompletionResult(
        text="",
        tool_calls=[],
        finish_reason=FinishReason.MAX_TOKENS,
        usage=Usage(input_tokens=200, output_tokens=512),
    )


def _refusal_no_verdicts() -> CompletionResult:
    """A verifier turn that REFUSED (content filter): terminal STOP, refusal text."""
    return CompletionResult(
        text="I'm sorry, I can't comply with that request.",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=50, output_tokens=5),
    )


# --------------------------------------------------------------------------- #
# Fix 1: MAX_TOKENS-with-no-verdicts logs a DISTINCT budget/starvation warning #
# --------------------------------------------------------------------------- #
def test_max_tokens_no_verdicts_logs_distinct_starvation_warning(caplog):
    findings = [_finding(0.95, rule="r1")]
    verifier = FakeProvider([_max_tokens_no_verdicts()])

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)

    # Fail-safe still keeps the genuine candidate (find-broad).
    assert [f.rule_id for f in kept] == ["r1"]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a budget-truncated verifier turn must emit a WARNING"
    msg = "\n".join(r.getMessage() for r in warnings).lower()
    # Must read as a BUDGET / reasoning-starvation problem, distinct vocabulary.
    assert "budget" in msg or "reasoning" in msg or "max_tokens" in msg, (
        "a MAX_TOKENS-with-no-verdicts turn must be logged as reasoning/budget "
        "starvation, not the generic refusal/unparseable path"
    )
    # And must NOT be mislabeled as a content-filter refusal.
    assert "refusal" not in msg and "content-filter" not in msg, (
        "reasoning-starvation must be distinguishable from a content-filter refusal"
    )


def test_refusal_still_logs_generic_path_not_starvation(caplog):
    """A genuine STOP refusal must KEEP its own (non-budget) warning vocabulary."""
    findings = [_finding(0.95, rule="r1")]
    verifier = FakeProvider([_refusal_no_verdicts()])

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        verify_mod.verify_findings(verifier, findings, gate=0.80)

    msg = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ).lower()
    # The refusal path must NOT claim a token-budget starvation.
    assert "max_tokens" not in msg, (
        "a content-filter refusal (STOP) must not be mislabeled as a "
        "MAX_TOKENS/budget starvation"
    )


# --------------------------------------------------------------------------- #
# Fix 2: budget headroom — raised floor + scales with verifier reasoning effort#
# --------------------------------------------------------------------------- #
def test_min_verify_max_tokens_floor_raised_for_reasoning_headroom():
    """The hard floor must give a reasoning verifier real headroom (>= 2048)."""
    assert verify_mod.MIN_VERIFY_MAX_TOKENS >= 2048


def test_default_budget_at_least_floor(caplog):
    """Even with NO declared reasoning effort the budget never dips below the floor."""
    verifier = FakeProvider([_verify_one_verdict()])
    verify_mod.verify_findings(verifier, [_finding(0.95, rule="r1")], gate=0.80)
    assert verifier.calls[0].max_tokens >= verify_mod.MIN_VERIFY_MAX_TOKENS


def test_reasoning_verifier_budget_scales_above_floor():
    """When the verifier's reasoning effort is declared, the budget gives headroom.

    Live-wiring: the orchestrator runs the verifier with ``reasoning_effort:
    medium`` (config), but ``verify_findings`` floored the budget blind to that.
    Passing the declared effort must lift the budget reaching ``complete()`` to at
    least the reasoning floor so reasoning tokens don't starve the verdict.
    """
    verifier = FakeProvider([_verify_one_verdict()])
    verify_mod.verify_findings(
        verifier,
        [_finding(0.95, rule="r1")],
        gate=0.80,
        verifier_reasoning_effort="medium",
    )
    budget = verifier.calls[0].max_tokens
    assert budget >= verify_mod.MIN_REASONING_VERIFY_MAX_TOKENS
    assert budget >= verify_mod.MIN_VERIFY_MAX_TOKENS


def test_non_reasoning_verifier_keeps_lower_budget():
    """A verifier with reasoning OFF must NOT pay for the larger reasoning floor."""
    verifier = FakeProvider([_verify_one_verdict()])
    verify_mod.verify_findings(
        verifier,
        [_finding(0.95, rule="r1")],
        gate=0.80,
        verifier_reasoning_effort=None,
    )
    budget = verifier.calls[0].max_tokens
    # Below the reasoning floor (no reasoning to fund) but still >= the base floor.
    assert budget < verify_mod.MIN_REASONING_VERIFY_MAX_TOKENS
    assert budget >= verify_mod.MIN_VERIFY_MAX_TOKENS


def _verify_one_verdict() -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v1",
                name="verify_findings",
                args={
                    "verdicts": [
                        {"id": 0, "keep": True, "confidence": 0.95, "rationale": "ok"}
                    ]
                },
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )
