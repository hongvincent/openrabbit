"""Abstract ``Provider`` contract + a network-free ``FakeProvider`` (SPEC 7.1).

Concrete adapters (``ConverseAdapter``, ``OpenAIResponsesAdapter``) subclass
:class:`Provider`. They MUST import their cloud SDKs lazily inside methods so
this module — and every downstream unit test that mocks a provider — imports
with zero external dependencies.

:class:`FakeProvider` is the standard test double: it records every call and
returns pre-scripted :class:`~openrabbit.domain.CompletionResult` objects in
order, which is exactly what is needed to exercise the deterministic spine and
the bounded agentic escalation loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from openrabbit.domain import CompletionResult, Message, ToolSpec


class ProviderError(Exception):
    """Raised for provider-layer failures (transport, schema, exhaustion)."""


class Provider(ABC):
    """Model-neutral provider interface.

    The spine only ever talks to this interface; everything provider-specific is
    hidden behind concrete implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider family name, e.g. ``"converse"`` / ``"openai"``."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The concrete model/inference-profile id this instance targets."""

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolSpec]],
        max_tokens: int,
        cache_prefix: Optional[str],
        **opts: Any,
    ) -> CompletionResult:
        """Run one completion turn and return a normalized result.

        Parameters
        ----------
        system:
            System prompt (rubric, taxonomy, output contract, conventions).
        messages:
            Conversation so far as neutral :class:`Message` objects.
        tools:
            Optional tool specs the model may call (e.g. ``emit_findings``,
            ``grep``, ``read_file``, ``git_*``). ``None`` means no tools.
        max_tokens:
            Output token budget for this turn.
        cache_prefix:
            Optional prompt-cache key for the byte-stable cacheable prefix
            (SPEC section 6, step 3). ``None`` disables explicit caching.
        **opts:
            Provider-specific knobs (e.g. ``reasoning_effort``, ``store``,
            ``tool_choice``), ignored by providers that do not support them.
        """


@dataclass
class RecordedCall:
    """A single recorded :meth:`FakeProvider.complete` invocation."""

    system: str
    messages: list[Message]
    tools: Optional[list[ToolSpec]]
    max_tokens: int
    cache_prefix: Optional[str]
    opts: dict[str, Any] = field(default_factory=dict)


class FakeProvider(Provider):
    """A deterministic, no-network :class:`Provider` for tests.

    Returns ``scripted_results`` in order on successive ``complete`` calls and
    records each call in :attr:`calls`. Raises :class:`ProviderError` once the
    script is exhausted, which keeps tests honest about how many model calls a
    pipeline makes.
    """

    def __init__(
        self,
        scripted_results: list[CompletionResult],
        *,
        name: str = "fake",
        model: str = "fake-model-0",
    ) -> None:
        self._results: list[CompletionResult] = list(scripted_results)
        self._cursor = 0
        self._name = name
        self._model = model
        self.calls: list[RecordedCall] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolSpec]],
        max_tokens: int,
        cache_prefix: Optional[str],
        **opts: Any,
    ) -> CompletionResult:
        self.calls.append(
            RecordedCall(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                cache_prefix=cache_prefix,
                opts=dict(opts),
            )
        )
        if self._cursor >= len(self._results):
            raise ProviderError(
                "FakeProvider exhausted: no scripted result for call "
                f"#{self._cursor + 1} (scripted {len(self._results)})"
            )
        result = self._results[self._cursor]
        self._cursor += 1
        return result
