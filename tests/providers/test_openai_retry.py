"""Retry/backoff + auth-hardening + content-filter for OpenAIResponsesAdapter.

All OFFLINE: ``httpx.Client`` is monkeypatched to a scripted transport so a 429
then 200 sequence can be replayed deterministically. ``time.sleep`` and the
jitter source are patched so the test never actually waits and the backoff is
deterministic.

Findings under test:
* [HIGH] one POST with no retry on 429/5xx/transient timeouts -> add bounded
  exponential-backoff-with-jitter that respects Retry-After.
* [LOW] OPENAI_API_KEY must be dropped from the bearer env vars so an OpenAI key
  is never sent to the AWS mantle endpoint (fail fast on AWS token only).
* [LOW] incomplete + content_filter must not be mislabeled as LENGTH.
"""

from __future__ import annotations

import json

import pytest

from openrabbit.domain import FinishReason, Message
from openrabbit.providers.base import ProviderError
from openrabbit.providers.openai_responses import OpenAIResponsesAdapter


def _text_payload(text: str = "ok") -> dict:
    return {
        "id": "r",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class _ScriptedResponse:
    """Stand-in for ``httpx.Response``; raises HTTPStatusError on >=400."""

    def __init__(self, status_code: int, payload: dict, *, headers=None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=self,  # type: ignore[arg-type]
            )


class _ScriptedClient:
    """Replays a list of (response | exception) on successive POSTs."""

    script: list = []
    calls: int = 0

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, *, headers=None, json=None, **kwargs):
        idx = _ScriptedClient.calls
        _ScriptedClient.calls += 1
        item = _ScriptedClient.script[idx]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def aws_only_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "aws-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def deterministic_backoff(monkeypatch):
    """Patch sleep + jitter so retries are instant and deterministic."""
    slept: list[float] = []
    monkeypatch.setattr(
        "openrabbit.providers.openai_responses.time.sleep",
        lambda s: slept.append(s),
    )
    # Pin jitter to a fixed multiplier so computed sleeps are predictable.
    monkeypatch.setattr(
        "openrabbit.providers.openai_responses.random.random",
        lambda: 0.0,
    )
    return slept


def _install_scripted(monkeypatch, script):
    import httpx

    _ScriptedClient.script = script
    _ScriptedClient.calls = 0
    monkeypatch.setattr(httpx, "Client", _ScriptedClient)


# --------------------------------------------------------------------------- #
# Retry / backoff                                                              #
# --------------------------------------------------------------------------- #
def test_retries_on_429_then_succeeds(aws_only_env, deterministic_backoff, monkeypatch):
    _install_scripted(
        monkeypatch,
        [
            _ScriptedResponse(429, {"error": {"message": "slow down"}}),
            _ScriptedResponse(200, _text_payload("recovered")),
        ],
    )
    adapter = OpenAIResponsesAdapter()
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.text == "recovered"
    assert _ScriptedClient.calls == 2, "must retry the 429 and then succeed"
    assert deterministic_backoff, "a backoff sleep must occur between attempts"


def test_retries_on_500_then_succeeds(aws_only_env, deterministic_backoff, monkeypatch):
    _install_scripted(
        monkeypatch,
        [
            _ScriptedResponse(503, {"error": {"message": "unavailable"}}),
            _ScriptedResponse(200, _text_payload("ok2")),
        ],
    )
    adapter = OpenAIResponsesAdapter()
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.text == "ok2"
    assert _ScriptedClient.calls == 2


def test_retries_on_transient_timeout_then_succeeds(
    aws_only_env, deterministic_backoff, monkeypatch
):
    import httpx

    _install_scripted(
        monkeypatch,
        [
            httpx.ConnectTimeout("timed out"),
            _ScriptedResponse(200, _text_payload("ok3")),
        ],
    )
    adapter = OpenAIResponsesAdapter()
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.text == "ok3"
    assert _ScriptedClient.calls == 2


def test_respects_retry_after_header(aws_only_env, deterministic_backoff, monkeypatch):
    _install_scripted(
        monkeypatch,
        [
            _ScriptedResponse(
                429, {"error": {"message": "slow"}}, headers={"Retry-After": "7"}
            ),
            _ScriptedResponse(200, _text_payload("done")),
        ],
    )
    adapter = OpenAIResponsesAdapter()
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.text == "done"
    # The first (and only) sleep must honor the server's Retry-After value.
    assert deterministic_backoff[0] >= 7.0


def test_gives_up_after_max_attempts(aws_only_env, deterministic_backoff, monkeypatch):
    # Always 429 -> exhausts retries and raises (bounded, not infinite).
    _install_scripted(
        monkeypatch,
        [_ScriptedResponse(429, {"error": {"message": "nope"}}) for _ in range(10)],
    )
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(ProviderError):
        adapter.complete("s", [Message("user", "x")], None, 100, None)
    # Bounded: it must not retry forever.
    assert 1 < _ScriptedClient.calls <= 6


def test_4xx_non_retryable_does_not_retry(
    aws_only_env, deterministic_backoff, monkeypatch
):
    # A 400 (bad request) is a client error -> NO retry, fail immediately.
    _install_scripted(
        monkeypatch,
        [
            _ScriptedResponse(400, {"error": {"message": "bad request"}}),
            _ScriptedResponse(200, _text_payload("should-not-reach")),
        ],
    )
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(ProviderError):
        adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert _ScriptedClient.calls == 1, "client errors (4xx != 429) must not retry"


# --------------------------------------------------------------------------- #
# Auth hardening: OPENAI_API_KEY must not be used.                            #
# --------------------------------------------------------------------------- #
def test_openai_api_key_is_not_used_as_bearer(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-sent")
    adapter = OpenAIResponsesAdapter()
    # No AWS token present + OpenAI key should be ignored -> fail fast.
    with pytest.raises(ProviderError):
        adapter.complete("s", [Message("user", "x")], None, 100, None)


def test_bearer_env_vars_excludes_openai_api_key():
    from openrabbit.providers.openai_responses import _BEARER_ENV_VARS

    assert "OPENAI_API_KEY" not in _BEARER_ENV_VARS
    assert "AWS_BEARER_TOKEN_BEDROCK" in _BEARER_ENV_VARS


# --------------------------------------------------------------------------- #
# Content-filter finish reason.                                               #
# --------------------------------------------------------------------------- #
def test_content_filter_incomplete_distinct_from_length(aws_only_env, monkeypatch):
    payload = _text_payload("")
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": "content_filter"}
    _install_scripted(monkeypatch, [_ScriptedResponse(200, payload)])
    adapter = OpenAIResponsesAdapter()
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.finish_reason is not FinishReason.LENGTH


def test_unknown_incomplete_reason_still_length(aws_only_env, monkeypatch):
    payload = _text_payload("")
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": "some_future_reason"}
    _install_scripted(monkeypatch, [_ScriptedResponse(200, payload)])
    adapter = OpenAIResponsesAdapter()
    res = adapter.complete("s", [Message("user", "x")], None, 100, None)
    assert res.finish_reason is FinishReason.LENGTH
