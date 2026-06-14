"""Neutral provider domain model (SPEC section 7.1).

These provider-agnostic types are the *only* thing the spine, routing, and
aggregation layers ever see. Concrete adapters (``ConverseAdapter`` for
Nova/Claude, ``OpenAIResponsesAdapter`` for GPT-5.5) translate Bedrock/OpenAI
payloads to and from this model so no provider detail leaks upward.

Pure stdlib — no cloud SDK imports.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Optional, Union

# A message's content may be a plain string or a list of typed content blocks
# (text, tool_use, tool_result, image, ...). Adapters interpret the blocks.
ContentBlocks = Union[str, list[dict[str, Any]]]


@dataclass
class Message:
    """A single conversation message.

    ``role`` is typically ``system`` / ``user`` / ``assistant`` / ``tool``.
    ``content`` is either a string or a list of content blocks.
    """

    role: str
    content: ContentBlocks


@dataclass
class ToolSpec:
    """A tool the model may call.

    ``json_schema`` is the JSON Schema for the tool's input parameters. Adapters
    map this to the provider's tool format (OpenAI flat function schema /
    Converse ``toolSpec.inputSchema.json``).
    """

    name: str
    description: str
    json_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A model request to invoke a tool with parsed arguments."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """The result returned to the model for a prior :class:`ToolCall`."""

    id: str
    content: Any
    is_error: bool = False


class FinishReason(enum.Enum):
    """Why a completion stopped (normalized across providers)."""

    STOP = "stop"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    LENGTH = "length"


@dataclass(frozen=True)
class Usage:
    """Token usage, including prompt-cache reads/writes (SPEC 7.3 cost tracking).

    Frozen and additive: ``+`` accumulates two ``Usage`` values into a new one,
    so ``sum(usages, Usage())`` totals a whole PR's spend without mutation.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


@dataclass
class CompletionResult:
    """The normalized result of a single provider ``complete`` call.

    ``text`` is the concatenated assistant text, ``tool_calls`` are any parsed
    tool invocations, ``finish_reason`` is normalized, ``usage`` is token usage,
    and ``raw`` is the untouched provider payload for debugging/telemetry.
    """

    text: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: Usage
    raw: Optional[Any] = None
