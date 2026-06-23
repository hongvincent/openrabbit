"""Live smoke: ConverseAdapter against REAL Bedrock Nova (ap-northeast-2).

Proves the Converse REQUEST shape we build is accepted by the real endpoint AND
that the response keys :class:`ConverseAdapter` parses (``output.message.content``
text blocks, ``stopReason``, ``usage.inputTokens``/``outputTokens``) actually
exist on the wire. A mock can fake those keys; only a real call confirms them.
"""

from __future__ import annotations

import pytest

from openrabbit.domain import FinishReason, Message
from openrabbit.providers.converse import ConverseAdapter

from .conftest import NOVA_MODEL, NOVA_REGION


@pytest.mark.live
def test_live_converse_smoke(aws_profile_env: str) -> None:
    adapter = ConverseAdapter(model_id=NOVA_MODEL, region=NOVA_REGION)

    result = adapter.complete(
        system="You are a terse assistant. Answer in one short word.",
        messages=[Message(role="user", content="Reply with the single word: ok")],
        tools=None,
        max_tokens=16,
        cache_prefix=None,
    )

    # finish_reason normalizes from the real ``stopReason`` key. A tiny prompt
    # well under max_tokens must terminate naturally (end_turn -> STOP).
    assert result.finish_reason == FinishReason.STOP, (
        f"expected STOP from a complete reply, got {result.finish_reason} "
        f"(raw stopReason={result.raw.get('stopReason')!r})"
    )

    # Non-empty text proves ``output.message.content[].text`` parsing is correct.
    assert result.text.strip(), "expected non-empty assistant text from Nova"

    # usage.input_tokens > 0 proves the parser reads the REAL usage key name
    # (``inputTokens``). If Bedrock had renamed it, this would be 0 and a mock
    # would never have caught it.
    assert result.usage.input_tokens > 0, (
        "usage.input_tokens must be > 0 — proves the Converse 'usage.inputTokens' "
        f"key exists on the wire (raw usage={result.raw.get('usage')!r})"
    )
    assert result.usage.output_tokens > 0, "expected output tokens for a reply"
