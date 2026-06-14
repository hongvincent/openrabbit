"""Tests for the neutral provider domain model (SPEC 7.1).

The spine, routing, and aggregation only ever see these types — never raw
Bedrock/OpenAI payloads. No network imports here.
"""

from __future__ import annotations

import dataclasses

import pytest

from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


# --------------------------------------------------------------------------- #
# Message                                                                      #
# --------------------------------------------------------------------------- #
def test_message_holds_role_and_blocks():
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_message_content_can_be_blocks():
    blocks = [{"type": "text", "text": "hi"}]
    m = Message(role="assistant", content=blocks)
    assert m.content == blocks


# --------------------------------------------------------------------------- #
# ToolSpec                                                                     #
# --------------------------------------------------------------------------- #
def test_toolspec_fields():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    t = ToolSpec(name="emit_findings", description="emit", json_schema=schema)
    assert t.name == "emit_findings"
    assert t.description == "emit"
    assert t.json_schema == schema


# --------------------------------------------------------------------------- #
# ToolCall / ToolResult                                                        #
# --------------------------------------------------------------------------- #
def test_toolcall_fields():
    c = ToolCall(id="call_1", name="grep", args={"pattern": "foo"})
    assert c.id == "call_1"
    assert c.name == "grep"
    assert c.args == {"pattern": "foo"}


def test_toolresult_fields_default_not_error():
    r = ToolResult(id="call_1", content="matches: 3")
    assert r.id == "call_1"
    assert r.content == "matches: 3"
    assert r.is_error is False


def test_toolresult_is_error_flag():
    r = ToolResult(id="call_1", content="boom", is_error=True)
    assert r.is_error is True


# --------------------------------------------------------------------------- #
# FinishReason                                                                 #
# --------------------------------------------------------------------------- #
def test_finish_reason_members():
    names = {m.name for m in FinishReason}
    assert names == {"STOP", "TOOL_USE", "MAX_TOKENS", "LENGTH"}


def test_finish_reason_is_enum():
    import enum

    assert issubclass(FinishReason, enum.Enum)
    assert FinishReason.TOOL_USE is FinishReason.TOOL_USE


# --------------------------------------------------------------------------- #
# Usage accumulation                                                           #
# --------------------------------------------------------------------------- #
def test_usage_defaults_zero():
    u = Usage()
    assert (u.input_tokens, u.output_tokens, u.cache_read, u.cache_write) == (
        0,
        0,
        0,
        0,
    )


def test_usage_add_accumulates_all_fields():
    a = Usage(input_tokens=10, output_tokens=5, cache_read=2, cache_write=1)
    b = Usage(input_tokens=3, output_tokens=4, cache_read=5, cache_write=6)
    total = a + b
    assert total.input_tokens == 13
    assert total.output_tokens == 9
    assert total.cache_read == 7
    assert total.cache_write == 7


def test_usage_add_is_pure():
    a = Usage(input_tokens=10)
    b = Usage(input_tokens=3)
    _ = a + b
    # operands unchanged
    assert a.input_tokens == 10
    assert b.input_tokens == 3


def test_usage_add_rejects_non_usage():
    with pytest.raises(TypeError):
        _ = Usage() + 5  # type: ignore[operator]


def test_usage_sum_with_start_zero():
    usages = [Usage(input_tokens=1), Usage(input_tokens=2), Usage(input_tokens=3)]
    total = sum(usages, Usage())
    assert total.input_tokens == 6


# --------------------------------------------------------------------------- #
# CompletionResult                                                             #
# --------------------------------------------------------------------------- #
def test_completion_result_fields():
    usage = Usage(input_tokens=1)
    res = CompletionResult(
        text="all good",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=usage,
        raw={"provider": "fake"},
    )
    assert res.text == "all good"
    assert res.tool_calls == []
    assert res.finish_reason is FinishReason.STOP
    assert res.usage is usage
    assert res.raw == {"provider": "fake"}


def test_completion_result_with_tool_calls():
    call = ToolCall(id="c1", name="emit_findings", args={"findings": []})
    res = CompletionResult(
        text="",
        tool_calls=[call],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
        raw=None,
    )
    assert res.finish_reason is FinishReason.TOOL_USE
    assert res.tool_calls[0].name == "emit_findings"


def test_domain_types_are_dataclasses():
    for cls in (Message, ToolSpec, ToolCall, ToolResult, Usage, CompletionResult):
        assert dataclasses.is_dataclass(cls)
