"""Tests for the OpenAIResponsesAdapter (GPT-5.5 on Bedrock, Responses API).

NO network: every test mocks ``httpx`` (via ``respx`` or monkeypatch) with
canned Responses payloads. The adapter MUST import ``httpx`` lazily so this
test module — and the bare ``import openrabbit.providers.openai_responses`` —
needs zero external deps installed at import time. We verify that, too.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from openrabbit.domain import (
    FinishReason,
    Message,
    ToolSpec,
)
from openrabbit.providers.base import Provider, ProviderError
from openrabbit.providers.openai_responses import OpenAIResponsesAdapter


# --------------------------------------------------------------------------- #
# Canned payload helpers                                                       #
# --------------------------------------------------------------------------- #
def _text_payload(
    text: str = "Looks good.",
    *,
    finish: str = "completed",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached_tokens: int = 0,
) -> dict:
    """A minimal Responses API payload that finishes with plain text."""
    return {
        "id": "resp_abc",
        "status": finish,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def _tool_call_payload(name: str, args: dict, *, call_id: str = "fc_1") -> dict:
    """A Responses payload whose output is a single function_call item."""
    return {
        "id": "resp_tool",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "item_1",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(args),
            }
        ],
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class _FakeResponse:
    """Stand-in for ``httpx.Response`` used by the monkeypatch transport."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )


class _Recorder:
    """Captures the last POST so tests can assert on URL/headers/body."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.url: str | None = None
        self.headers: dict | None = None
        self.json_body: dict | None = None

    def post(self, url, *, headers=None, json=None, **kwargs):
        self.url = url
        self.headers = headers
        self.json_body = json
        return _FakeResponse(self.payload, self.status_code)


def _install_fake_httpx(monkeypatch, recorder: _Recorder) -> None:
    """Patch ``httpx.Client`` so the adapter talks to the recorder, no network."""
    import httpx

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self._recorder = recorder

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, headers=None, json=None, **kwargs):
            return recorder.post(url, headers=headers, json=json, **kwargs)

    monkeypatch.setattr(httpx, "Client", _FakeClient)


@pytest.fixture
def bearer_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return "test-bearer-token"


# --------------------------------------------------------------------------- #
# Import / lazy-dependency contract                                           #
# --------------------------------------------------------------------------- #
def test_module_imports_without_httpx():
    """Importing the adapter must NOT import httpx (lazy import contract)."""
    # Drop any cached import then re-import the module fresh.
    for mod in ("openrabbit.providers.openai_responses",):
        sys.modules.pop(mod, None)
    sys.modules.pop("httpx", None)
    importlib.import_module("openrabbit.providers.openai_responses")
    assert "httpx" not in sys.modules


def test_is_a_provider(bearer_env):
    adapter = OpenAIResponsesAdapter()
    assert isinstance(adapter, Provider)


def test_name_and_model_defaults(bearer_env):
    adapter = OpenAIResponsesAdapter()
    assert adapter.name == "openai"
    assert adapter.model == "openai.gpt-5.5"


# --------------------------------------------------------------------------- #
# Region validation                                                            #
# --------------------------------------------------------------------------- #
def test_default_region_is_valid(bearer_env):
    adapter = OpenAIResponsesAdapter()
    assert adapter.region in ("us-east-1", "us-east-2")


@pytest.mark.parametrize("region", ["us-east-1", "us-east-2"])
def test_valid_regions_accepted(bearer_env, region):
    adapter = OpenAIResponsesAdapter(region=region)
    assert adapter.region == region
    assert region in adapter.base_url
    assert adapter.base_url.startswith("https://bedrock-mantle.")
    assert adapter.base_url.endswith("/openai/v1")


@pytest.mark.parametrize("region", ["us-west-2", "ap-northeast-2", "eu-west-1", ""])
def test_invalid_region_rejected(bearer_env, region):
    with pytest.raises(ValueError):
        OpenAIResponsesAdapter(region=region)


# --------------------------------------------------------------------------- #
# Auth                                                                         #
# --------------------------------------------------------------------------- #
def test_missing_bearer_token_raises(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    with pytest.raises(ProviderError):
        adapter.complete("sys", [Message("user", "hi")], None, 100, None)


def test_bearer_token_from_aws_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "aws-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("sys", [Message("user", "hi")], None, 100, None)
    assert rec.headers["Authorization"] == "Bearer aws-token"


def test_bearer_token_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-token")
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("sys", [Message("user", "hi")], None, 100, None)
    assert rec.headers["Authorization"] == "Bearer openai-token"


# --------------------------------------------------------------------------- #
# Basic completion + request shaping                                          #
# --------------------------------------------------------------------------- #
def test_completion_returns_text(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload("All clear."))
    _install_fake_httpx(monkeypatch, rec)

    res = adapter.complete(
        "You are a reviewer.",
        [Message("user", "Review this diff.")],
        None,
        500,
        None,
    )
    assert res.text == "All clear."
    assert res.finish_reason is FinishReason.STOP
    assert res.tool_calls == []
    assert res.raw is not None


def test_request_targets_responses_endpoint(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter(region="us-east-2")
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("sys", [Message("user", "hi")], None, 100, None)
    assert rec.url == ("https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses")


def test_request_body_maps_system_and_messages(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete(
        "SYSTEM RUBRIC",
        [Message("user", "first"), Message("assistant", "ack")],
        None,
        777,
        None,
    )
    body = rec.json_body
    assert body["model"] == "openai.gpt-5.5"
    assert body["instructions"] == "SYSTEM RUBRIC"
    assert body["max_output_tokens"] == 777
    # input is the messages array
    assert isinstance(body["input"], list)
    assert body["input"][0]["role"] == "user"
    assert body["input"][0]["content"] == "first"
    assert body["input"][1]["role"] == "assistant"


def test_store_is_always_false(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    # Even if a caller tries to override store, it stays False.
    adapter.complete("s", [Message("user", "x")], None, 10, None, store=True)
    assert rec.json_body["store"] is False


# --------------------------------------------------------------------------- #
# Reasoning effort validation                                                  #
# --------------------------------------------------------------------------- #
def test_default_reasoning_effort_is_medium(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("s", [Message("user", "x")], None, 10, None)
    assert rec.json_body["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh"])
def test_valid_reasoning_efforts(bearer_env, monkeypatch, effort):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete(
        "s", [Message("user", "x")], None, 10, None, reasoning_effort=effort
    )
    assert rec.json_body["reasoning"] == {"effort": effort}


def test_minimal_effort_is_rejected(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    with pytest.raises(ValueError):
        adapter.complete(
            "s", [Message("user", "x")], None, 10, None, reasoning_effort="minimal"
        )


def test_unknown_effort_is_rejected(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    with pytest.raises(ValueError):
        adapter.complete(
            "s", [Message("user", "x")], None, 10, None, reasoning_effort="bogus"
        )


# --------------------------------------------------------------------------- #
# Tool calls                                                                   #
# --------------------------------------------------------------------------- #
def test_tools_serialized_flat_with_strict(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    tools = [
        ToolSpec(
            name="emit_findings",
            description="emit the findings",
            json_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
    ]
    adapter.complete("s", [Message("user", "x")], tools, 10, None)
    sent = rec.json_body["tools"]
    assert sent == [
        {
            "type": "function",
            "name": "emit_findings",
            "description": "emit the findings",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
            "strict": True,
        }
    ]


def test_tool_choice_passed_through(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    tools = [ToolSpec("emit_findings", "d", {"type": "object"})]
    choice = {"type": "function", "name": "emit_findings"}
    adapter.complete("s", [Message("user", "x")], tools, 10, None, tool_choice=choice)
    assert rec.json_body["tool_choice"] == choice


def test_bare_tool_name_forces_function_choice(bearer_env, monkeypatch):
    """A bare tool-name string (the canonical neutral form used by the verifier
    and judge) must serialize to the Responses forced-tool shape, NOT pass
    through verbatim."""
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    tools = [ToolSpec("verify_finding", "d", {"type": "object"})]
    adapter.complete(
        "s", [Message("user", "x")], tools, 10, None, tool_choice="verify_finding"
    )
    assert rec.json_body["tool_choice"] == {
        "type": "function",
        "name": "verify_finding",
    }


@pytest.mark.parametrize("keyword", ["auto", "none", "required"])
def test_tool_choice_keywords_passed_through(bearer_env, monkeypatch, keyword):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    tools = [ToolSpec("emit_findings", "d", {"type": "object"})]
    adapter.complete("s", [Message("user", "x")], tools, 10, None, tool_choice=keyword)
    assert rec.json_body["tool_choice"] == keyword


def test_no_tools_omits_tool_keys(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("s", [Message("user", "x")], None, 10, None)
    assert "tools" not in rec.json_body
    assert "tool_choice" not in rec.json_body


def test_function_call_output_parsed(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(
        _tool_call_payload(
            "emit_findings", {"findings": [{"title": "bug"}]}, call_id="fc_xyz"
        )
    )
    _install_fake_httpx(monkeypatch, rec)
    tools = [ToolSpec("emit_findings", "d", {"type": "object"})]
    res = adapter.complete("s", [Message("user", "x")], tools, 10, None)

    assert res.finish_reason is FinishReason.TOOL_USE
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.id == "fc_xyz"
    assert tc.name == "emit_findings"
    assert tc.args == {"findings": [{"title": "bug"}]}


def test_malformed_tool_arguments_raise_provider_error(bearer_env, monkeypatch):
    payload = {
        "id": "resp_bad",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "emit_findings",
                "arguments": "{not valid json",
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(payload)
    _install_fake_httpx(monkeypatch, rec)
    tools = [ToolSpec("emit_findings", "d", {"type": "object"})]
    with pytest.raises(ProviderError):
        adapter.complete("s", [Message("user", "x")], tools, 10, None)


# --------------------------------------------------------------------------- #
# Structured output (json_schema)                                             #
# --------------------------------------------------------------------------- #
def test_json_schema_structured_output(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    schema = {"type": "object", "properties": {"findings": {"type": "array"}}}
    adapter.complete(
        "s",
        [Message("user", "x")],
        None,
        100,
        None,
        json_schema=schema,
        schema_name="findings",
        verbosity="low",
    )
    text = rec.json_body["text"]
    assert text["verbosity"] == "low"
    assert text["format"]["type"] == "json_schema"
    assert text["format"]["strict"] is True
    assert text["format"]["schema"] == schema
    assert text["format"]["name"] == "findings"


def test_verbosity_alone_sets_text_block(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("s", [Message("user", "x")], None, 100, None, verbosity="high")
    assert rec.json_body["text"]["verbosity"] == "high"
    assert "format" not in rec.json_body["text"]


def test_no_text_block_when_no_options(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert "text" not in rec.json_body


# --------------------------------------------------------------------------- #
# Prompt caching                                                              #
# --------------------------------------------------------------------------- #
def test_cache_prefix_sets_cache_keys(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("s", [Message("user", "x")], None, 100, "pr-1234-prefix")
    assert rec.json_body["prompt_cache_key"] == "pr-1234-prefix"
    assert rec.json_body["prompt_cache_retention"] == "24h"


def test_no_cache_prefix_omits_cache_keys(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert "prompt_cache_key" not in rec.json_body
    assert "prompt_cache_retention" not in rec.json_body


# --------------------------------------------------------------------------- #
# Usage (incl. cached tokens)                                                  #
# --------------------------------------------------------------------------- #
def test_usage_maps_cached_tokens(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(
        _text_payload(input_tokens=200, output_tokens=40, cached_tokens=150)
    )
    _install_fake_httpx(monkeypatch, rec)
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.usage.input_tokens == 200
    assert res.usage.output_tokens == 40
    assert res.usage.cache_read == 150
    assert res.usage.cache_write == 0


def test_usage_handles_missing_details(bearer_env, monkeypatch):
    payload = {
        "id": "r",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(payload)
    _install_fake_httpx(monkeypatch, rec)
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.usage.input_tokens == 5
    assert res.usage.output_tokens == 2
    assert res.usage.cache_read == 0


# --------------------------------------------------------------------------- #
# Finish reason normalization                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,expected",
    [
        ("completed", FinishReason.STOP),
        ("incomplete", FinishReason.MAX_TOKENS),
    ],
)
def test_finish_reason_normalization(bearer_env, monkeypatch, status, expected):
    payload = _text_payload(finish=status)
    if status == "incomplete":
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(payload)
    _install_fake_httpx(monkeypatch, rec)
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.finish_reason is expected


# --------------------------------------------------------------------------- #
# HTTP errors                                                                  #
# --------------------------------------------------------------------------- #
def test_http_error_raises_provider_error(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder({"error": {"message": "bad request"}}, status_code=400)
    _install_fake_httpx(monkeypatch, rec)
    with pytest.raises(ProviderError):
        adapter.complete("s", [Message("user", "x")], None, 100, None)


def test_http_error_detail_extracted_from_response_json(bearer_env, monkeypatch):
    """A 4xx with a structured error body surfaces the API's message."""
    import httpx

    class _ErrResp:
        status_code = 429
        text = '{"error":{"message":"rate limited"}}'

        def json(self):
            return {"error": {"message": "rate limited"}}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "429",
                request=None,
                response=self,  # type: ignore[arg-type]
            )

    class _ErrClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **k):
            return _ErrResp()

    monkeypatch.setattr(httpx, "Client", _ErrClient)
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(ProviderError) as ei:
        adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert "rate limited" in str(ei.value)


def test_http_error_detail_is_bounded(bearer_env, monkeypatch):
    """A huge upstream error message is truncated so it cannot flood CI logs."""
    import httpx

    big = "y" * 5000 + "\nleaked line"

    class _ErrResp:
        status_code = 400
        text = big

        def json(self):
            return {"error": {"message": big}}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "400",
                request=None,
                response=self,  # type: ignore[arg-type]
            )

    class _ErrClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **k):
            return _ErrResp()

    monkeypatch.setattr(httpx, "Client", _ErrClient)
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(ProviderError) as ei:
        adapter.complete("s", [Message("user", "x")], None, 100, None)
    msg = str(ei.value)
    assert len(msg) < 300
    assert "\n" not in msg


def test_http_error_detail_falls_back_to_text(bearer_env, monkeypatch):
    """When the error body has no error.message, fall back to raw text."""
    import httpx

    class _ErrResp:
        status_code = 500
        text = "internal server error"

        def json(self):
            return {"unexpected": "shape"}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500",
                request=None,
                response=self,  # type: ignore[arg-type]
            )

    class _ErrClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **k):
            return _ErrResp()

    monkeypatch.setattr(httpx, "Client", _ErrClient)
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(ProviderError) as ei:
        adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert "internal server error" in str(ei.value)


def test_missing_usage_block_yields_zero_usage(bearer_env, monkeypatch):
    payload = {
        "id": "r",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(payload)
    _install_fake_httpx(monkeypatch, rec)
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.usage.input_tokens == 0
    assert res.usage.output_tokens == 0
    assert res.usage.cache_read == 0


def test_incomplete_without_max_tokens_reason_is_length(bearer_env, monkeypatch):
    payload = _text_payload(finish="incomplete")
    payload["incomplete_details"] = {"reason": "content_filter"}
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(payload)
    _install_fake_httpx(monkeypatch, rec)
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.finish_reason is FinishReason.LENGTH


def test_transport_error_wrapped_as_provider_error(bearer_env, monkeypatch):
    import httpx

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **k):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(ProviderError):
        adapter.complete("s", [Message("user", "x")], None, 100, None)


# --------------------------------------------------------------------------- #
# Message content blocks (list form)                                          #
# --------------------------------------------------------------------------- #
def test_list_content_messages_passed_through(bearer_env, monkeypatch):
    adapter = OpenAIResponsesAdapter()
    rec = _Recorder(_text_payload())
    _install_fake_httpx(monkeypatch, rec)
    blocks = [{"type": "input_text", "text": "hello"}]
    adapter.complete("s", [Message("user", blocks)], None, 100, None)
    assert rec.json_body["input"][0]["content"] == blocks
