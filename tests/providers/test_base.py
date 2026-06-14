"""Tests for the abstract Provider contract and the FakeProvider test double.

No network: FakeProvider is the whole point — it lets every downstream module's
unit tests drive the agentic loop deterministically with ZERO external deps.
"""

from __future__ import annotations

import inspect

import pytest

from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from openrabbit.providers.base import FakeProvider, Provider, ProviderError


def _result(text: str = "ok", **kw) -> CompletionResult:
    return CompletionResult(
        text=text,
        tool_calls=kw.get("tool_calls", []),
        finish_reason=kw.get("finish_reason", FinishReason.STOP),
        usage=kw.get("usage", Usage(input_tokens=1, output_tokens=1)),
        raw=kw.get("raw"),
    )


# --------------------------------------------------------------------------- #
# Provider ABC contract                                                        #
# --------------------------------------------------------------------------- #
def test_provider_is_abstract():
    import abc

    assert issubclass(Provider, abc.ABC)
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


def test_provider_complete_signature():
    sig = inspect.signature(Provider.complete)
    params = list(sig.parameters)
    # self + the documented contract parameters
    assert params[:6] == [
        "self",
        "system",
        "messages",
        "tools",
        "max_tokens",
        "cache_prefix",
    ]
    # accepts arbitrary provider options
    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def test_provider_declares_name_and_model():
    assert hasattr(Provider, "name")
    assert hasattr(Provider, "model")
    assert isinstance(Provider.name, property)
    assert isinstance(Provider.model, property)


def test_subclass_must_implement_complete():
    class Incomplete(Provider):
        @property
        def name(self) -> str:
            return "x"

        @property
        def model(self) -> str:
            return "m"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_fully_implemented_subclass_instantiates():
    class Mini(Provider):
        @property
        def name(self) -> str:
            return "mini"

        @property
        def model(self) -> str:
            return "mini-1"

        def complete(self, system, messages, tools, max_tokens, cache_prefix, **opts):
            return _result()

    p = Mini()
    assert isinstance(p, Provider)
    assert p.name == "mini"
    assert p.model == "mini-1"


# --------------------------------------------------------------------------- #
# FakeProvider                                                                 #
# --------------------------------------------------------------------------- #
def test_fake_provider_is_a_provider():
    fp = FakeProvider([_result()])
    assert isinstance(fp, Provider)


def test_fake_provider_default_name_and_model():
    fp = FakeProvider([_result()])
    assert fp.name == "fake"
    assert isinstance(fp.model, str)


def test_fake_provider_custom_name_and_model():
    fp = FakeProvider([_result()], name="nova", model="amazon.nova-pro-v1:0")
    assert fp.name == "nova"
    assert fp.model == "amazon.nova-pro-v1:0"


def test_fake_provider_returns_queued_results_in_order():
    r1 = _result("first")
    r2 = _result("second")
    fp = FakeProvider([r1, r2])
    out1 = fp.complete("sys", [Message("user", "a")], None, 100, None)
    out2 = fp.complete("sys", [Message("user", "b")], None, 100, None)
    assert out1 is r1
    assert out2 is r2


def test_fake_provider_records_calls():
    fp = FakeProvider([_result()])
    tools = [ToolSpec("emit", "d", {"type": "object"})]
    msgs = [Message("user", "review this")]
    fp.complete("SYSTEM", msgs, tools, 256, "prefix-key", reasoning_effort="medium")

    assert len(fp.calls) == 1
    call = fp.calls[0]
    assert call.system == "SYSTEM"
    assert call.messages == msgs
    assert call.tools == tools
    assert call.max_tokens == 256
    assert call.cache_prefix == "prefix-key"
    assert call.opts == {"reasoning_effort": "medium"}


def test_fake_provider_raises_when_exhausted():
    fp = FakeProvider([_result()])
    fp.complete("s", [], None, 10, None)
    with pytest.raises(ProviderError):
        fp.complete("s", [], None, 10, None)


def test_fake_provider_agentic_loop_contract():
    """Simulate a bounded agentic loop: model asks for a tool, then finishes."""
    tool_step = _result(
        "",
        tool_calls=[ToolCall(id="c1", name="grep", args={"pattern": "foo"})],
        finish_reason=FinishReason.TOOL_USE,
    )
    final_step = _result("done", finish_reason=FinishReason.STOP)
    fp = FakeProvider([tool_step, final_step])

    messages: list[Message] = [Message("user", "find callers")]
    turns = 0
    while turns < 3:
        turns += 1
        res = fp.complete("sys", messages, [ToolSpec("grep", "d", {})], 512, "pfx")
        if res.finish_reason is FinishReason.TOOL_USE:
            # feed a tool result back and continue the loop
            messages.append(Message("assistant", res.text))
            messages.append(Message("tool", "match at line 10"))
            continue
        break

    assert turns == 2
    assert res.text == "done"
    assert len(fp.calls) == 2
    # the second call saw the appended tool-result messages
    assert len(fp.calls[1].messages) == 3


def test_fake_provider_usage_accumulates_across_loop():
    steps = [
        _result(usage=Usage(input_tokens=10, output_tokens=2, cache_read=5)),
        _result(usage=Usage(input_tokens=3, output_tokens=1, cache_write=4)),
    ]
    fp = FakeProvider(steps)
    total = Usage()
    for _ in range(2):
        res = fp.complete("s", [], None, 10, None)
        total = total + res.usage
    assert total.input_tokens == 13
    assert total.output_tokens == 3
    assert total.cache_read == 5
    assert total.cache_write == 4


def test_provider_error_is_exception():
    assert issubclass(ProviderError, Exception)
