"""Tests for ``ConverseAdapter`` (AWS Bedrock Converse API).

NO NETWORK: boto3 is monkeypatched so ``ConverseAdapter`` builds and parses
real Converse request/response shapes against a fully in-memory fake client.
The adapter imports boto3 lazily, so importing this module needs zero AWS deps.

Covered:
* module imports with boto3 absent / never touched at import time
* request mapping: system blocks, message content, inferenceConfig, toolConfig
* plain completion -> CompletionResult(text, STOP)
* tool-use round-trip: stopReason==tool_use -> ToolCall parsing + sending
  ToolResult blocks back
* forced-tool structured output (emit_findings, toolChoice={tool:{name}})
* Usage accounting from cacheRead/cacheWrite/input/output tokens
* cachePoint insertion into system + final message (NOT tools) when cache_prefix
* FinishReason normalization for every Converse stopReason
* region + modelId wired from constructor into boto3.client + converse()
"""

from __future__ import annotations

import sys

import pytest

from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Message,
    ToolResult,
    ToolSpec,
)
from openrabbit.providers.base import Provider, ProviderError


# --------------------------------------------------------------------------- #
# Fake boto3                                                                   #
# --------------------------------------------------------------------------- #
class FakeBedrockClient:
    """In-memory stand-in for a ``bedrock-runtime`` client."""

    def __init__(self, responses=None, *, error=None):
        self._responses = list(responses or [])
        self._cursor = 0
        self._error = error
        self.calls: list[dict] = []  # captured converse() kwargs

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if self._cursor >= len(self._responses):
            raise AssertionError("FakeBedrockClient: no scripted response left")
        resp = self._responses[self._cursor]
        self._cursor += 1
        return resp


class FakeBoto3Module:
    """Captures ``boto3.client(...)`` calls and hands back a fake client."""

    def __init__(self, client: FakeBedrockClient):
        self._client = client
        self.client_calls: list[dict] = []

    def client(self, service_name, **kwargs):
        self.client_calls.append({"service_name": service_name, **kwargs})
        return self._client


@pytest.fixture
def install_boto3(monkeypatch):
    """Install a fake ``boto3`` module; return a setup callable.

    Usage: ``boto3, client = install_boto3(responses=[...])``.
    """

    def _setup(responses=None, *, error=None):
        client = FakeBedrockClient(responses=responses, error=error)
        fake = FakeBoto3Module(client)
        monkeypatch.setitem(sys.modules, "boto3", fake)
        return fake, client

    return _setup


# --------------------------------------------------------------------------- #
# Response builders (real Converse shapes)                                     #
# --------------------------------------------------------------------------- #
def _resp(
    *,
    content,
    stop_reason="end_turn",
    usage=None,
):
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": usage or {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def _text_resp(text="hello", **kw):
    return _resp(content=[{"text": text}], **kw)


def _tooluse_resp(tool_use_id="tu-1", name="grep", inp=None, text=None, **kw):
    blocks = []
    if text is not None:
        blocks.append({"text": text})
    blocks.append(
        {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": inp or {}}}
    )
    kw.setdefault("stop_reason", "tool_use")
    return _resp(content=blocks, **kw)


# --------------------------------------------------------------------------- #
# Import-time hygiene                                                          #
# --------------------------------------------------------------------------- #
def test_module_imports_without_boto3(monkeypatch):
    """Importing the adapter must not require boto3 at module top-level."""
    monkeypatch.setitem(sys.modules, "boto3", None)  # poison import
    import importlib

    mod = importlib.import_module("openrabbit.providers.converse")
    importlib.reload(mod)
    assert hasattr(mod, "ConverseAdapter")


def test_constructing_adapter_does_not_create_client(install_boto3):
    """Lazy: no boto3.client() until the first complete() call."""
    fake, _client = install_boto3(responses=[])
    from openrabbit.providers.converse import ConverseAdapter

    ConverseAdapter(model_id="amazon.nova-pro-v1:0", region="ap-northeast-2")
    assert fake.client_calls == []


# --------------------------------------------------------------------------- #
# Provider contract / identity                                                 #
# --------------------------------------------------------------------------- #
def test_adapter_is_a_provider(install_boto3):
    install_boto3(responses=[])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="amazon.nova-pro-v1:0", region="ap-northeast-2")
    assert isinstance(a, Provider)
    assert a.name == "converse"
    assert a.model == "amazon.nova-pro-v1:0"


def test_region_and_model_wired_into_client_and_request(install_boto3):
    fake, client = install_boto3(responses=[_text_resp("ok")])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="amazon.nova-lite-v1:0", region="us-east-2")
    a.complete("sys", [Message("user", "hi")], None, 128, None)

    assert fake.client_calls[0]["service_name"] == "bedrock-runtime"
    assert fake.client_calls[0]["region_name"] == "us-east-2"
    assert client.calls[0]["modelId"] == "amazon.nova-lite-v1:0"


def test_client_is_reused_across_calls(install_boto3):
    fake, _client = install_boto3(responses=[_text_resp("a"), _text_resp("b")])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="amazon.nova-pro-v1:0", region="ap-northeast-2")
    a.complete("s", [Message("user", "1")], None, 64, None)
    a.complete("s", [Message("user", "2")], None, 64, None)
    assert len(fake.client_calls) == 1  # only one client created


# --------------------------------------------------------------------------- #
# Request mapping                                                              #
# --------------------------------------------------------------------------- #
def test_system_mapped_to_text_blocks(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("You are a reviewer.", [Message("user", "x")], None, 100, None)
    sysblocks = client.calls[0]["system"]
    assert sysblocks == [{"text": "You are a reviewer."}]


def test_empty_system_omitted(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("", [Message("user", "x")], None, 100, None)
    assert "system" not in client.calls[0]


def test_string_message_becomes_text_block(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "review this diff")], None, 100, None)
    msgs = client.calls[0]["messages"]
    assert msgs == [{"role": "user", "content": [{"text": "review this diff"}]}]


def test_list_message_content_passed_through(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    blocks = [{"text": "part 1"}, {"text": "part 2"}]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", blocks)], None, 100, None)
    msgs = client.calls[0]["messages"]
    assert msgs[0]["content"] == blocks


def test_tool_result_message_converted_to_tool_result_block(install_boto3):
    """A Message carrying a ToolResult-shaped block round-trips to Converse."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    tr = ToolResult(id="tu-1", content="match at line 10")
    a.complete("s", [Message("user", [tr])], None, 100, None)
    block = client.calls[0]["messages"][0]["content"][0]
    assert "toolResult" in block
    assert block["toolResult"]["toolUseId"] == "tu-1"
    # content is a list of blocks per Converse spec
    assert block["toolResult"]["content"] == [{"text": "match at line 10"}]
    assert "status" not in block["toolResult"]


def test_tool_result_list_content_passed_through(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    blocks = [{"text": "line 1"}, {"text": "line 2"}]
    tr = ToolResult(id="tu-2", content=blocks)
    a.complete("s", [Message("user", [tr])], None, 100, None)
    block = client.calls[0]["messages"][0]["content"][0]
    assert block["toolResult"]["content"] == blocks


def test_tool_result_object_content_wrapped_as_json(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    tr = ToolResult(id="tu-3", content={"matches": 2})
    a.complete("s", [Message("user", [tr])], None, 100, None)
    block = client.calls[0]["messages"][0]["content"][0]
    assert block["toolResult"]["content"] == [{"json": {"matches": 2}}]


def test_dict_tool_choice_passed_through(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec("emit", "d", {"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    choice = {"tool": {"name": "emit"}}
    a.complete("s", [Message("user", "x")], tools, 100, None, tool_choice=choice)
    assert client.calls[0]["toolConfig"]["toolChoice"] == choice


def test_tool_result_error_sets_status(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    tr = ToolResult(id="tu-9", content="boom", is_error=True)
    a.complete("s", [Message("user", [tr])], None, 100, None)
    block = client.calls[0]["messages"][0]["content"][0]
    assert block["toolResult"]["status"] == "error"


def test_inference_config_carries_max_tokens(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "x")], None, 777, None)
    assert client.calls[0]["inferenceConfig"]["maxTokens"] == 777


def test_inference_config_extra_opts(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s",
        [Message("user", "x")],
        None,
        100,
        None,
        temperature=0.0,
        top_p=0.9,
    )
    cfg = client.calls[0]["inferenceConfig"]
    assert cfg["maxTokens"] == 100
    assert cfg["temperature"] == 0.0
    assert cfg["topP"] == 0.9


# --------------------------------------------------------------------------- #
# Nova 2 extended-thinking (reasoning_effort)                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("effort", ["low", "medium"])
def test_reasoning_effort_injects_reasoning_config(install_boto3, effort):
    """reasoning_effort=low/medium injects the exact Nova 2 reasoningConfig shape."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s",
        [Message("user", "x")],
        None,
        100,
        None,
        reasoning_effort=effort,
    )
    call = client.calls[0]
    assert call["additionalModelRequestFields"] == {
        "reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}
    }


def test_reasoning_effort_not_leaked_into_inference_config(install_boto3):
    """reasoning_effort is a Converse top-level field, never an inferenceConfig key."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "x")], None, 100, None, reasoning_effort="low")
    cfg = client.calls[0]["inferenceConfig"]
    assert "reasoning_effort" not in cfg
    assert "reasoningConfig" not in cfg
    assert "maxReasoningEffort" not in cfg


@pytest.mark.parametrize("disabled", [None, "none", "off"])
def test_reasoning_effort_disabled_values_omit_reasoning_config(
    install_boto3, disabled
):
    """Absent/None/'none'/'off' => reasoning disabled (no additionalModelRequestFields)."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s", [Message("user", "x")], None, 100, None, reasoning_effort=disabled
    )
    assert "additionalModelRequestFields" not in client.calls[0]


def test_reasoning_effort_absent_omits_reasoning_config(install_boto3):
    """No reasoning_effort opt at all => no additionalModelRequestFields."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "x")], None, 100, None)
    assert "additionalModelRequestFields" not in client.calls[0]


def test_reasoning_effort_high_omits_temperature_top_p_top_k(install_boto3):
    """High effort MUST omit temperature/topP/topK (else Nova 2 ValidationException)."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s",
        [Message("user", "x")],
        None,
        100,
        None,
        reasoning_effort="high",
        temperature=0.2,
        top_p=0.9,
        top_k=40,
    )
    call = client.calls[0]
    assert call["additionalModelRequestFields"] == {
        "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
    }
    cfg = call["inferenceConfig"]
    assert "temperature" not in cfg
    assert "topP" not in cfg
    assert "topK" not in cfg
    # maxTokens is unaffected by the high-effort omission.
    assert cfg["maxTokens"] == 100


def test_reasoning_effort_low_keeps_temperature_and_top_p(install_boto3):
    """Only HIGH omits sampling params; low/medium keep temperature/topP as usual."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s",
        [Message("user", "x")],
        None,
        100,
        None,
        reasoning_effort="low",
        temperature=0.0,
        top_p=0.9,
    )
    cfg = client.calls[0]["inferenceConfig"]
    assert cfg["temperature"] == 0.0
    assert cfg["topP"] == 0.9


def test_reasoning_content_block_not_leaked_into_text(install_boto3):
    """A reasoningContent block must NOT be concatenated into result.text."""
    reasoning_block = {
        "reasoningContent": {
            "reasoningText": {"text": "[REDACTED] chain of thought", "signature": "s"}
        }
    }
    install_boto3(
        responses=[
            _resp(
                content=[reasoning_block, {"text": "The code looks correct."}],
                stop_reason="end_turn",
            )
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert res.text == "The code looks correct."
    assert "REDACTED" not in res.text
    assert "chain of thought" not in res.text
    assert res.finish_reason is FinishReason.STOP


def test_reasoning_content_block_with_tool_use(install_boto3):
    """reasoningContent + toolUse: parse the toolUse, drop the reasoning text."""
    reasoning_block = {
        "reasoningContent": {
            "reasoningText": {"text": "thinking...", "signature": "sig"}
        }
    }
    install_boto3(
        responses=[
            _resp(
                content=[
                    reasoning_block,
                    {
                        "toolUse": {
                            "toolUseId": "tu-7",
                            "name": "emit_findings",
                            "input": {"findings": []},
                        }
                    },
                ],
                stop_reason="tool_use",
            )
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec("emit_findings", "emit", {"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete(
        "s", [Message("user", "x")], tools, 100, None, tool_choice="emit_findings"
    )
    assert res.text == ""
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "emit_findings"
    assert res.tool_calls[0].args == {"findings": []}
    assert res.finish_reason is FinishReason.TOOL_USE


# --------------------------------------------------------------------------- #
# Tool mapping                                                                 #
# --------------------------------------------------------------------------- #
def test_tools_mapped_to_tool_config(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [
        ToolSpec(
            name="grep",
            description="search files",
            json_schema={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
            },
        )
    ]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "x")], tools, 100, None)
    tc = client.calls[0]["toolConfig"]
    spec = tc["tools"][0]["toolSpec"]
    assert spec["name"] == "grep"
    assert spec["description"] == "search files"
    assert spec["inputSchema"]["json"] == tools[0].json_schema
    assert "toolChoice" not in tc  # no forcing by default


def test_no_tool_config_when_no_tools(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "x")], None, 100, None)
    assert "toolConfig" not in client.calls[0]


def test_forced_tool_choice_via_opts(install_boto3):
    _fake, client = install_boto3(
        responses=[
            _tooluse_resp(
                name="emit_findings", inp={"findings": []}, stop_reason="tool_use"
            )
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    tools = [
        ToolSpec(
            name="emit_findings", description="emit", json_schema={"type": "object"}
        )
    ]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s",
        [Message("user", "x")],
        tools,
        100,
        None,
        tool_choice="emit_findings",
    )
    tc = client.calls[0]["toolConfig"]
    assert tc["toolChoice"] == {"tool": {"name": "emit_findings"}}


def test_bare_verify_tool_name_forces_converse_tool_choice(install_boto3):
    """The canonical neutral bare tool name used by the verifier/judge must map
    to Converse's ``toolChoice={"tool":{"name":..}}`` — never the OpenAI shape."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [
        ToolSpec(name="verify_finding", description="d", json_schema={"type": "object"})
    ]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete(
        "s", [Message("user", "x")], tools, 100, None, tool_choice="verify_finding"
    )
    tc = client.calls[0]["toolConfig"]["toolChoice"]
    assert tc == {"tool": {"name": "verify_finding"}}
    # The invalid OpenAI shape must never reach the wire.
    assert "type" not in tc


def test_tool_choice_auto_and_any(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp(), _text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec(name="t", description="d", json_schema={"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete("s", [Message("user", "x")], tools, 100, None, tool_choice="auto")
    assert client.calls[0]["toolConfig"]["toolChoice"] == {"auto": {}}
    a.complete("s", [Message("user", "x")], tools, 100, None, tool_choice="any")
    assert client.calls[1]["toolConfig"]["toolChoice"] == {"any": {}}


# --------------------------------------------------------------------------- #
# Response parsing                                                             #
# --------------------------------------------------------------------------- #
def test_plain_completion(install_boto3):
    install_boto3(
        responses=[_text_resp("The code looks fine.", stop_reason="end_turn")]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert isinstance(res, CompletionResult)
    assert res.text == "The code looks fine."
    assert res.tool_calls == []
    assert res.finish_reason is FinishReason.STOP
    assert res.raw is not None


def test_multiple_text_blocks_concatenated(install_boto3):
    install_boto3(responses=[_resp(content=[{"text": "alpha"}, {"text": "beta"}])])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert res.text == "alphabeta"


def test_tool_use_parsed_into_tool_calls(install_boto3):
    install_boto3(
        responses=[
            _tooluse_resp(
                tool_use_id="tu-42",
                name="read_file",
                inp={"path": "src/agent.py", "start": 1},
                text="Let me look.",
            )
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete(
        "s", [Message("user", "x")], [ToolSpec("read_file", "d", {})], 100, None
    )
    assert res.finish_reason is FinishReason.TOOL_USE
    assert res.text == "Let me look."
    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert call.id == "tu-42"
    assert call.name == "read_file"
    assert call.args == {"path": "src/agent.py", "start": 1}


def test_tool_use_round_trip(install_boto3):
    """Model asks for a tool, we send ToolResult back, then it finishes."""
    _fake, client = install_boto3(
        responses=[
            _tooluse_resp(tool_use_id="tu-1", name="grep", inp={"pattern": "foo"}),
            _text_resp("Found it. Looks correct.", stop_reason="end_turn"),
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    tools = [ToolSpec("grep", "search", {"type": "object"})]

    first = a.complete("s", [Message("user", "find foo")], tools, 100, None)
    assert first.finish_reason is FinishReason.TOOL_USE
    tc = first.tool_calls[0]

    # Feed assistant turn + tool result back as neutral messages.
    follow = [
        Message("user", "find foo"),
        Message(
            "assistant",
            [{"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.args}}],
        ),
        Message("user", [ToolResult(id=tc.id, content="src/x.py:10: foo")]),
    ]
    second = a.complete("s", follow, tools, 100, None)
    assert second.finish_reason is FinishReason.STOP
    assert second.text == "Found it. Looks correct."

    # The 2nd request carried a toolResult block.
    second_msgs = client.calls[1]["messages"]
    tr_block = second_msgs[-1]["content"][0]
    assert tr_block["toolResult"]["toolUseId"] == "tu-1"


def test_forced_tool_structured_output(install_boto3):
    """emit_findings forced tool returns structured JSON in toolUse.input."""
    findings_payload = {
        "findings": [
            {
                "file": "a.py",
                "startLine": 1,
                "endLine": 2,
                "side": "RIGHT",
                "severity": "high",
                "category": "correctness",
                "confidence": 0.9,
                "title": "bug",
                "body": "...",
                "ruleId": "r",
                "fingerprint": "f",
            }
        ]
    }
    install_boto3(
        responses=[
            _tooluse_resp(
                tool_use_id="ef-1",
                name="emit_findings",
                inp=findings_payload,
                stop_reason="tool_use",
            )
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec("emit_findings", "emit", {"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete(
        "s", [Message("user", "review")], tools, 1000, None, tool_choice="emit_findings"
    )
    assert res.finish_reason is FinishReason.TOOL_USE
    assert res.tool_calls[0].name == "emit_findings"
    assert res.tool_calls[0].args == findings_payload


# --------------------------------------------------------------------------- #
# FinishReason normalization                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stop_reason,expected",
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("tool_use", FinishReason.TOOL_USE),
        ("max_tokens", FinishReason.MAX_TOKENS),
        ("content_filtered", FinishReason.STOP),
        ("guardrail_intervened", FinishReason.STOP),
        ("something_new", FinishReason.STOP),
    ],
)
def test_finish_reason_normalization(install_boto3, stop_reason, expected):
    # tool_use needs a toolUse block to parse cleanly
    if stop_reason == "tool_use":
        resp = _tooluse_resp(stop_reason="tool_use")
    else:
        resp = _text_resp("x", stop_reason=stop_reason)
    install_boto3(responses=[resp])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert res.finish_reason is expected


# --------------------------------------------------------------------------- #
# Usage accounting                                                             #
# --------------------------------------------------------------------------- #
def test_usage_full_accounting(install_boto3):
    install_boto3(
        responses=[
            _text_resp(
                usage={
                    "inputTokens": 100,
                    "outputTokens": 40,
                    "totalTokens": 140,
                    "cacheReadInputTokens": 25,
                    "cacheWriteInputTokens": 60,
                }
            )
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 40
    assert res.usage.cache_read == 25
    assert res.usage.cache_write == 60


def test_usage_defaults_when_cache_fields_absent(install_boto3):
    install_boto3(
        responses=[
            _text_resp(usage={"inputTokens": 7, "outputTokens": 3, "totalTokens": 10})
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert res.usage.input_tokens == 7
    assert res.usage.output_tokens == 3
    assert res.usage.cache_read == 0
    assert res.usage.cache_write == 0


def test_usage_missing_entirely(install_boto3):
    install_boto3(
        responses=[
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "stopReason": "end_turn",
            }
        ]
    )
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    res = a.complete("s", [Message("user", "x")], None, 100, None)
    assert res.usage.input_tokens == 0
    assert res.usage.output_tokens == 0


# --------------------------------------------------------------------------- #
# cachePoint insertion                                                         #
# --------------------------------------------------------------------------- #
def test_cache_point_inserted_in_system(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("rubric...", [Message("user", "diff")], None, 100, "pr-42")
    sysblocks = client.calls[0]["system"]
    assert sysblocks[0] == {"text": "rubric..."}
    assert sysblocks[-1] == {"cachePoint": {"type": "default"}}


def test_no_cache_point_in_tools_array(install_boto3):
    """The ``tools`` array must NOT carry a cachePoint, even with cache_prefix.

    Amazon Nova rejects a tool-level cache point on the real Converse API
    (ValidationException: "extraneous key [cachePoint] is not permitted"), so the
    adapter never appends one — the cacheable bytes live in system/messages only.
    """
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec("grep", "d", {"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete("sys", [Message("user", "diff")], tools, 100, "pr-1")
    tool_list = client.calls[0]["toolConfig"]["tools"]
    # No cachePoint entry anywhere in the tools array.
    assert all("cachePoint" not in t for t in tool_list)
    # The actual tool spec is present and is the only entry.
    assert tool_list == [
        {
            "toolSpec": {
                "name": "grep",
                "description": "d",
                "inputSchema": {"json": {"type": "object"}},
            }
        }
    ]


def test_cache_point_inserted_in_messages(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("sys", [Message("user", "shared PR context")], None, 100, "pr-1")
    last_msg = client.calls[0]["messages"][-1]
    assert last_msg["content"][-1] == {"cachePoint": {"type": "default"}}


def test_no_cache_point_without_prefix(install_boto3):
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec("grep", "d", {"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    a.complete("sys", [Message("user", "x")], tools, 100, None)
    call = client.calls[0]
    assert all("cachePoint" not in b for b in call.get("system", []))
    assert all("cachePoint" not in t for t in call["toolConfig"]["tools"])
    last_content = call["messages"][-1]["content"]
    assert all("cachePoint" not in b for b in last_content)


def test_cache_point_no_messages_safe(install_boto3):
    """cache_prefix with empty system + no messages must not crash."""
    _fake, client = install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    a.complete("", [], None, 100, "pr-1")
    # no system block (empty), no messages -> nothing to anchor on
    assert "system" not in client.calls[0]
    assert client.calls[0]["messages"] == []


# --------------------------------------------------------------------------- #
# Error handling                                                               #
# --------------------------------------------------------------------------- #
def test_client_error_wrapped_as_provider_error(install_boto3):
    install_boto3(responses=[], error=RuntimeError("throttled"))
    from openrabbit.providers.converse import ConverseAdapter

    a = ConverseAdapter(model_id="m", region="r")
    with pytest.raises(ProviderError):
        a.complete("s", [Message("user", "x")], None, 100, None)


def test_unsupported_tool_choice_raises(install_boto3):
    install_boto3(responses=[_text_resp()])
    from openrabbit.providers.converse import ConverseAdapter

    tools = [ToolSpec("t", "d", {"type": "object"})]
    a = ConverseAdapter(model_id="m", region="r")
    with pytest.raises(ProviderError):
        a.complete("s", [Message("user", "x")], tools, 100, None, tool_choice=123)
