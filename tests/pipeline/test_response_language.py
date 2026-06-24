"""response_language: localize USER-FACING text while REASONING stays English.

Feature 1 (give users the choice: Korean response, English reasoning):

* When ``review.response_language != 'en'``, the finder system prompt AND the
  verifier system prompt must carry a language instruction telling the model to
  write the user-facing ``title``/``description`` in that language while doing
  all REASONING in English.
* The default (``'en'``) must leave both prompts byte-for-byte unchanged (no
  appended instruction), so the existing offline suite is unaffected.

Everything here is OFFLINE: ``FakeProvider`` records the exact system prompt of
each ``complete()`` call, so we assert on the captured Converse/Responses
request rather than on any live model behavior.
"""

from __future__ import annotations

from openrabbit.domain import CompletionResult, FinishReason, ToolCall, Usage
from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import run_lenses as run_lenses_mod
from openrabbit.pipeline import verify as verify_mod
from openrabbit.pipeline.route import FilePlan, Hunk
from openrabbit.providers.base import FakeProvider


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _file_plan() -> FilePlan:
    return FilePlan(
        path="src/api/auth.py",
        file_type="code",
        risk="medium",
        lenses=["correctness"],
        model_role="finder",
        hunks=[Hunk(header="@@ -1 +1 @@", text="+changed line")],
    )


def _empty_findings_result() -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[ToolCall(id="t1", name="emit_findings", args={"findings": []})],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )


def _finding(conf: float = 0.95, *, rule: str = "r1") -> Finding:
    fp = compute_fingerprint("src/api/auth.py", rule, f"ctx-{rule}")
    return Finding(
        file="src/api/auth.py",
        start_line=12,
        end_line=14,
        side="RIGHT",
        severity="high",
        category="security",
        confidence=conf,
        title="t",
        body="b",
        rule_id=rule,
        fingerprint=fp,
    )


def _verdict_result() -> CompletionResult:
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


# --------------------------------------------------------------------------- #
# finder (run_lenses) — the language instruction rides on the system prompt   #
# --------------------------------------------------------------------------- #
def test_finder_prompt_carries_korean_instruction_when_ko():
    finder = FakeProvider([_empty_findings_result()])
    run_lenses_mod.run_lenses(
        finder,
        _file_plan(),
        {"correctness": "rubric"},
        prefix="PREFIX",
        response_language="ko",
    )
    assert finder.calls, "finder must be called for an assigned lens"
    system = finder.calls[0].system
    assert "Korean" in system or "한국어" in system, (
        "the finder system prompt must instruct Korean user-facing output"
    )
    # REASONING must stay English so cross-family verification stays robust.
    assert "REASONING" in system and "English" in system


def test_finder_prompt_unchanged_for_default_en():
    finder = FakeProvider([_empty_findings_result()])
    run_lenses_mod.run_lenses(
        finder,
        _file_plan(),
        {"correctness": "rubric"},
        prefix="PREFIX",
        # response_language omitted -> default 'en'
    )
    system = finder.calls[0].system
    assert "Korean" not in system and "한국어" not in system, (
        "the default (en) finder prompt must NOT carry a language instruction"
    )


def test_finder_default_prompt_is_byte_identical_to_no_kwarg():
    """The en path must be byte-for-byte identical to passing nothing."""
    a = FakeProvider([_empty_findings_result()])
    b = FakeProvider([_empty_findings_result()])
    run_lenses_mod.run_lenses(
        a, _file_plan(), {"correctness": "rubric"}, prefix="PREFIX"
    )
    run_lenses_mod.run_lenses(
        b,
        _file_plan(),
        {"correctness": "rubric"},
        prefix="PREFIX",
        response_language="en",
    )
    assert a.calls[0].system == b.calls[0].system


# --------------------------------------------------------------------------- #
# verifier (verify_findings) — same instruction on the verifier prompt        #
# --------------------------------------------------------------------------- #
def test_verifier_prompt_carries_korean_instruction_when_ko():
    verifier = FakeProvider([_verdict_result()])
    verify_mod.verify_findings(
        verifier,
        [_finding()],
        gate=0.80,
        response_language="ko",
    )
    assert verifier.calls, "verifier must be called for a HIGH/security finding"
    system = verifier.calls[0].system
    assert "Korean" in system or "한국어" in system, (
        "the verifier system prompt must instruct Korean user-facing output"
    )
    assert "REASONING" in system and "English" in system


def test_verifier_prompt_unchanged_for_default_en():
    verifier = FakeProvider([_verdict_result()])
    verify_mod.verify_findings(
        verifier,
        [_finding()],
        gate=0.80,
    )
    system = verifier.calls[0].system
    assert "Korean" not in system and "한국어" not in system, (
        "the default (en) verifier prompt must NOT carry a language instruction"
    )
