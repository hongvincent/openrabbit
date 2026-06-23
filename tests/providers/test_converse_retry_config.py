"""ConverseAdapter must build its boto3 client with an adaptive-retry Config.

[HIGH] Bedrock throttling: boto3's default retry mode is too weak for the
verifier/finder under load. The adapter must pass a botocore ``Config`` with
``retries={"max_attempts": N, "mode": "adaptive"}`` so throttling (429-style
ThrottlingException) is retried with backoff inside boto3.

OFFLINE: a fake ``boto3`` module captures the ``client(...)`` kwargs; we assert
the ``config`` it received carries an adaptive retry policy. No network, no AWS.
"""

from __future__ import annotations

import sys

import pytest


class FakeBedrockClient:
    def __init__(self):
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }


class FakeBoto3Module:
    def __init__(self, client: FakeBedrockClient):
        self._client = client
        self.client_calls: list[dict] = []

    def client(self, service_name, **kwargs):
        self.client_calls.append({"service_name": service_name, **kwargs})
        return self._client


@pytest.fixture
def install_boto3(monkeypatch):
    def _setup():
        client = FakeBedrockClient()
        fake = FakeBoto3Module(client)
        monkeypatch.setitem(sys.modules, "boto3", fake)
        return fake, client

    return _setup


def _complete_once(adapter):
    from openrabbit.domain import Message

    adapter.complete("s", [Message("user", "x")], None, 10, None)


def test_client_built_with_adaptive_retry_config(install_boto3):
    from openrabbit.providers.converse import ConverseAdapter

    fake, _client = install_boto3()
    adapter = ConverseAdapter(model_id="amazon.nova-pro-v1:0", region="us-east-2")
    _complete_once(adapter)

    assert fake.client_calls, "the adapter must build a boto3 client"
    config = fake.client_calls[0].get("config")
    assert config is not None, "boto3.client(...) must receive a Config object"
    # botocore Config exposes retries as an attribute dict.
    retries = getattr(config, "retries", None)
    assert retries is not None, "Config must set a retries policy"
    assert retries.get("mode") == "adaptive"
    assert int(retries.get("max_attempts", 0)) >= 3
