"""Eval-honesty tests for the LLM-as-judge (adversarial finding 4).

The judge prompt previously prepended ``GROUND-TRUTH LABEL (trusted):
knownBug=True``, biasing the model toward 'match'. A blind judge must NOT see the
held-out ``known_bug`` label; instead it is given the true defect LOCATION (from
provenance) so a 'match' must hit the real bug, and its verdict is compared to
the label OUTSIDE the prompt.
"""

from __future__ import annotations

from openrabbit.domain import CompletionResult, FinishReason, ToolCall, Usage
from openrabbit.eval.golden_set import GoldenSample
from openrabbit.eval.judge import JUDGE_TOOL_NAME, Judge
from openrabbit.findings import Finding
from openrabbit.providers.base import FakeProvider


def _finding() -> Finding:
    return Finding(
        file="src/a.py",
        start_line=10,
        end_line=12,
        side="RIGHT",
        severity="high",
        category="correctness",
        confidence=0.9,
        title="Off-by-one",
        body="loop runs one too many times",
        rule_id="openrabbit/correctness/bounds-check",
        fingerprint="f" * 64,
    )


def _sample(known_bug: bool = True) -> GoldenSample:
    return GoldenSample(
        sample_id="s1",
        repo="local",
        commit="abc123",
        diff="@@ -1 +1 @@\n-x\n+y\n",
        known_bug=known_bug,
        bug_category="correctness",
        source="fix",
        message="fix: correct loop bound",
        defect_location="src/a.py:11",
    )


def _tool_result(verdict: str = "match") -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="c1",
                name=JUDGE_TOOL_NAME,
                args={"verdict": verdict, "confidence": 0.9, "rationale": "ok"},
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(input_tokens=10, output_tokens=3),
    )


def _judge_payload(known_bug: bool) -> str:
    fp = FakeProvider([_tool_result("match")])
    Judge(fp).judge(_finding(), _sample(known_bug=known_bug))
    call = fp.calls[0]
    user = "".join(m.content for m in call.messages if isinstance(m.content, str))
    return call.system + "\n" + user


def test_judge_prompt_does_not_leak_known_bug_label():
    # Neither a True nor a False label may appear in any form in the prompt.
    for known in (True, False):
        payload = _judge_payload(known)
        lowered = payload.lower()
        assert "knownbug" not in lowered, "judge must be BLIND to the known_bug label"
        assert "known_bug" not in lowered
        assert "ground-truth label" not in lowered
        # The literal label values must not be stated as the sample's truth.
        assert "knownbug=true" not in lowered
        assert "knownbug=false" not in lowered


def test_judge_prompt_includes_the_true_defect_location():
    # Blind judging still needs to know WHERE the real defect is so a 'match'
    # must hit the real bug (provenance-driven), not be a leaked yes/no.
    payload = _judge_payload(known_bug=True)
    assert "src/a.py:11" in payload


def test_judge_verdict_is_independent_of_the_label_value():
    # Same finding + same diff, only the held-out label differs -> identical
    # prompt (the label is decided OUTSIDE the model), so the judge cannot be
    # nudged by it.
    payload_true = _judge_payload(known_bug=True)
    payload_false = _judge_payload(known_bug=False)
    assert payload_true == payload_false
