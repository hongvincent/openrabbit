"""Verifier refusal / empty result must NOT silently zero every candidate.

Adversarial scenario (the bug): GPT-5.5 emits a ``{type:"refusal"}`` block (or
the turn comes back ``incomplete`` with ``reason == "content_filter"``). The old
adapter dropped the refusal text on the floor, so the verifier turn looked like
a clean empty completion: no tool_calls, empty text, FinishReason.STOP. The
verify stage then parsed an empty verdict map and SILENTLY DROPPED every
candidate finding (every HIGH/CRITICAL issue vanished with no signal).

Two layers are exercised here, both OFFLINE (FakeProvider / canned payloads):

1. ``OpenAIResponsesAdapter._extract_message_text`` must SURFACE refusal text
   (not skip it) so a refusal is observable on ``CompletionResult.text``.
2. ``verify_findings`` must DISTINGUISH "the verifier returned a verdict array
   that happens to drop finding X" (filter-strict, fine) from "the verifier
   gave us NO usable verdicts at all" (refusal/empty) — in the latter case it
   must NOT zero all candidates; it falls back to the finder's own confidence
   through the gate.
"""

from __future__ import annotations

from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Usage,
)
from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import verify as verify_mod
from openrabbit.providers.base import FakeProvider, Provider
from openrabbit.providers.openai_responses import OpenAIResponsesAdapter


# --------------------------------------------------------------------------- #
# Layer 1: the adapter must surface a refusal block as text.                  #
# --------------------------------------------------------------------------- #
def test_refusal_block_is_surfaced_as_text():
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
    }
    text = OpenAIResponsesAdapter._extract_message_text(item)
    assert "I cannot help with that." in text, (
        "a refusal block must be surfaced, not silently dropped (which would "
        "make a refused turn look like a clean empty completion)"
    )


def test_content_filter_incomplete_is_not_mislabeled_length():
    """incomplete + content_filter must be distinct from the LENGTH truncation."""
    payload = {
        "id": "r",
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [],
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }
    finish = OpenAIResponsesAdapter._normalize_finish(payload, [])
    assert finish is not FinishReason.LENGTH, (
        "content_filter must not be mislabeled as a LENGTH truncation"
    )


# --------------------------------------------------------------------------- #
# Layer 2: verify_findings must not silently zero all candidates on refusal.  #
# --------------------------------------------------------------------------- #
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


def _refusal_result() -> CompletionResult:
    """A verifier turn that refused: no tool calls, refusal surfaced as text."""
    return CompletionResult(
        text="I'm sorry, I can't comply with that request.",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=50, output_tokens=5),
    )


def test_verifier_refusal_does_not_silently_drop_findings():
    """A refused verifier turn must NOT silently zero all candidate findings.

    Both candidates have high finder confidence (>= gate). Since the verifier
    produced no usable verdicts, the stage falls back to finder confidence and
    keeps them rather than dropping every HIGH/CRITICAL finding.
    """
    findings = [
        _finding(0.95, rule="r1"),
        _finding(0.90, rule="r2"),
    ]
    verifier = FakeProvider([_refusal_result()])
    kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
    assert {f.rule_id for f in kept} == {"r1", "r2"}, (
        "a refusal/empty verifier result must not silently drop every candidate"
    )


def test_verifier_refusal_below_gate_still_applies_gate():
    """Fallback is to finder confidence THROUGH the gate, not unconditional keep."""
    findings = [
        _finding(0.95, rule="keep"),
        _finding(0.50, rule="drop"),  # below gate on finder confidence
    ]
    verifier = FakeProvider([_refusal_result()])
    kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
    assert {f.rule_id for f in kept} == {"keep"}


def test_normal_empty_verdict_array_still_filter_strict():
    """A REAL (non-refused) verdict array that omits an id still drops that id.

    This guards the find-broad/filter-strict contract: only a genuinely missing
    verifier turn triggers the fallback, NOT a verifier that returned a verdict
    array which simply did not vouch for a finding.
    """
    from openrabbit.domain import ToolCall

    # Verifier DID respond with a verdict array, vouching only for r1.
    result = CompletionResult(
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
    findings = [_finding(0.95, rule="r1"), _finding(0.95, rule="r2")]
    verifier = FakeProvider([result])
    kept = verify_mod.verify_findings(verifier, findings, gate=0.80)
    # r2 has NO verdict in a valid array -> dropped (filter-strict preserved).
    assert [f.rule_id for f in kept] == ["r1"]


def test_verifier_refusal_with_empty_text_is_treated_as_no_verdicts():
    """Even a fully-empty turn (no text, no tool calls) is a no-verdicts case."""

    class _EmptyVerifier(Provider):
        name = "fake"
        model = "fake"

        def complete(self, *a, **k):  # type: ignore[override]
            return CompletionResult(
                text="",
                tool_calls=[],
                finish_reason=FinishReason.STOP,
                usage=Usage(),
            )

    findings = [_finding(0.95, rule="r1")]
    kept = verify_mod.verify_findings(_EmptyVerifier(), findings, gate=0.80)
    assert [f.rule_id for f in kept] == ["r1"], (
        "an empty/no-verdict verifier turn must fall back rather than drop all"
    )
