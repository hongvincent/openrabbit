"""GPT-5.5 adapter via the OpenAI-compatible Responses API on Bedrock (SPEC 7.1).

``OpenAIResponsesAdapter`` is the production verifier/judge provider. It targets
``openai.gpt-5.5`` through Bedrock's OpenAI-compatible endpoint
``https://bedrock-mantle.{region}.api.aws/openai/v1`` and uses the Responses API
exclusively (``POST /responses`` — no Converse / ChatCompletions here).

Design constraints (all from the spec, enforced here):

* ``store: false`` on **every** call — this is private source code; no 30-day
  server-side retention is ever permitted.
* Reasoning ``effort`` is constrained to ``{none, low, medium, high, xhigh}``
  (default ``medium``). The ChatCompletions-only ``minimal`` value is rejected.
* Tools are serialized in the flat Responses shape
  ``{type: "function", name, parameters, strict: true}``.
* Structured findings use ``text.format = {type: "json_schema", strict: true}``.
* Prompt caching is requested with ``prompt_cache_key`` +
  ``prompt_cache_retention: "24h"`` whenever a ``cache_prefix`` is supplied.
* The diff/PR text is **untrusted data**; this layer only transports it and
  never interprets its content as instructions.

``httpx`` is imported **lazily** inside methods so importing this module (and
every unit test that mocks the HTTP layer) needs zero external dependencies.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Optional

from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from openrabbit.providers.base import Provider, ProviderError

#: Regions where the Bedrock OpenAI-compatible (mantle) endpoint is available.
SUPPORTED_REGIONS: tuple[str, ...] = ("us-east-1", "us-east-2")

#: Reasoning effort levels accepted by the Responses API for GPT-5.5.
#: NOTE: ``minimal`` is a ChatCompletions-only value and is deliberately absent.
REASONING_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")

#: Default model id and reasoning effort.
DEFAULT_MODEL = "openai.gpt-5.5"
DEFAULT_REGION = "us-east-2"
DEFAULT_REASONING_EFFORT = "medium"

#: Environment variables holding the Bearer token.
#:
#: ONLY ``AWS_BEARER_TOKEN_BEDROCK`` is accepted. The OpenAI-shaped wire format
#: is incidental — this endpoint is the AWS Bedrock *mantle*, NOT api.openai.com,
#: so ``OPENAI_API_KEY`` is deliberately excluded: silently falling back to an
#: OpenAI key would ship that secret to the AWS endpoint (and mask a missing AWS
#: token). Fail fast on the one correct credential instead.
_BEARER_ENV_VARS: tuple[str, ...] = ("AWS_BEARER_TOKEN_BEDROCK",)

#: Bounded exponential-backoff-with-jitter for retryable transport failures
#: (HTTP 429 / 5xx and transient timeouts). Bedrock throttles under load, so a
#: single un-retried POST drops whole findings batches.
_MAX_ATTEMPTS = 4  # 1 initial try + up to 3 retries
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 20.0
#: HTTP statuses worth retrying (429 throttling + transient server errors).
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _base_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/openai/v1"


class OpenAIResponsesAdapter(Provider):
    """Provider for GPT-5.5 over Bedrock's OpenAI-compatible Responses API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        region: str = DEFAULT_REGION,
        timeout: float = 120.0,
    ) -> None:
        if region not in SUPPORTED_REGIONS:
            raise ValueError(
                f"unsupported region {region!r}; "
                f"the Bedrock mantle endpoint requires one of {SUPPORTED_REGIONS}"
            )
        self._model = model
        self._region = region
        self._timeout = timeout
        self._base_url = _base_url(region)

    # ------------------------------------------------------------------ #
    # Provider identity                                                  #
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def region(self) -> str:
        return self._region

    @property
    def base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------ #
    # Completion                                                         #
    # ------------------------------------------------------------------ #
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolSpec]],
        max_tokens: int,
        cache_prefix: Optional[str],
        **opts: Any,
    ) -> CompletionResult:
        """Run one Responses-API turn and return a normalized result.

        Recognized ``opts``: ``reasoning_effort`` (str), ``tool_choice`` (Any),
        ``verbosity`` (str), ``json_schema`` (dict), ``schema_name`` (str).
        ``store`` is ignored — it is forced to ``False`` always.
        """
        body = self._build_body(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            cache_prefix=cache_prefix,
            opts=opts,
        )
        payload = self._post(body)
        return self._parse(payload)

    # ------------------------------------------------------------------ #
    # Request building                                                   #
    # ------------------------------------------------------------------ #
    def _build_body(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolSpec]],
        max_tokens: int,
        cache_prefix: Optional[str],
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        effort = opts.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
        if effort not in REASONING_EFFORTS:
            raise ValueError(
                f"invalid reasoning effort {effort!r}; "
                f"expected one of {REASONING_EFFORTS} (note: 'minimal' is not "
                "supported by the Responses API)"
            )

        body: dict[str, Any] = {
            "model": self._model,
            "instructions": system,
            "input": [self._encode_message(m) for m in messages],
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": effort},
            # Private source code: never retain server-side. Always False.
            "store": False,
        }

        # Tools (flat Responses shape) + optional tool_choice.
        if tools:
            body["tools"] = [self._encode_tool(t) for t in tools]
            tool_choice = opts.get("tool_choice")
            if tool_choice is not None:
                body["tool_choice"] = self._encode_tool_choice(tool_choice)

        # Structured output / verbosity via the text block.
        text_block = self._build_text_block(opts)
        if text_block:
            body["text"] = text_block

        # Prompt caching when a stable prefix key is supplied.
        if cache_prefix:
            body["prompt_cache_key"] = cache_prefix
            body["prompt_cache_retention"] = "24h"

        return body

    @staticmethod
    def _encode_message(message: Message) -> dict[str, Any]:
        # Responses ``input`` accepts message items with role + content, where
        # content is either a plain string or a list of typed content blocks.
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _encode_tool(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema,
            "strict": True,
        }

    # Responses-mode tool_choice keywords that are passed through verbatim
    # (everything else string is treated as a bare tool name to force).
    _TOOL_CHOICE_KEYWORDS = frozenset({"auto", "none", "required"})

    @classmethod
    def _encode_tool_choice(cls, tool_choice: Any) -> Any:
        """Translate the neutral tool_choice into the Responses wire shape.

        * a dict is already a Responses ``tool_choice`` and is passed through;
        * the keywords ``auto`` / ``none`` / ``required`` are passed through;
        * any other bare string is a tool name to force, becoming
          ``{"type": "function", "name": <name>}`` (the canonical neutral
          forced-single-tool form shared with :class:`ConverseAdapter`).
        """
        if (
            isinstance(tool_choice, str)
            and tool_choice not in cls._TOOL_CHOICE_KEYWORDS
        ):
            return {"type": "function", "name": tool_choice}
        return tool_choice

    @staticmethod
    def _build_text_block(opts: dict[str, Any]) -> dict[str, Any]:
        text_block: dict[str, Any] = {}

        verbosity = opts.get("verbosity")
        if verbosity is not None:
            text_block["verbosity"] = verbosity

        json_schema = opts.get("json_schema")
        if json_schema is not None:
            fmt: dict[str, Any] = {
                "type": "json_schema",
                "strict": True,
                "schema": json_schema,
            }
            schema_name = opts.get("schema_name")
            if schema_name is not None:
                fmt["name"] = schema_name
            text_block["format"] = fmt

        return text_block

    # ------------------------------------------------------------------ #
    # Transport                                                          #
    # ------------------------------------------------------------------ #
    def _bearer_token(self) -> str:
        for var in _BEARER_ENV_VARS:
            token = os.environ.get(var)
            if token:
                return token
        raise ProviderError(
            "no Bedrock bearer token found; set one of "
            f"{_BEARER_ENV_VARS} in the environment"
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        # Lazy import keeps the module import dependency-free for unit tests.
        import httpx

        token = self._bearer_token()
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in _RETRYABLE_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                    last_exc = exc
                    self._sleep_before_retry(attempt, exc)
                    continue
                # Non-retryable (4xx != 429) or retries exhausted.
                detail = self._error_detail(exc)
                raise ProviderError(
                    f"Responses API returned an error status: {detail}"
                ) from exc
            except httpx.TimeoutException as exc:
                # Transient timeouts are retryable.
                if attempt < _MAX_ATTEMPTS - 1:
                    last_exc = exc
                    self._sleep_before_retry(attempt, exc)
                    continue
                raise ProviderError(f"Responses API request failed: {exc}") from exc
            except httpx.HTTPError as exc:  # other transport-level errors
                raise ProviderError(f"Responses API request failed: {exc}") from exc

        # Defensive: loop only exits via return/raise above, but if every
        # attempt was retryable and we fell through, surface the last error.
        raise ProviderError(  # pragma: no cover - unreachable guard
            f"Responses API request failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
        )

    @classmethod
    def _sleep_before_retry(cls, attempt: int, exc: Any) -> None:
        """Sleep with exponential backoff + jitter, honoring Retry-After.

        ``attempt`` is the zero-based attempt index that just failed. The base
        backoff is ``_BACKOFF_BASE_SECONDS * 2**attempt`` plus full jitter,
        clamped to ``_BACKOFF_MAX_SECONDS``. A server ``Retry-After`` (seconds)
        sets a floor so we never poll faster than the server asked.
        """
        backoff = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2**attempt))
        # Full jitter in [0, backoff]; random is patched in tests for determinism.
        delay = random.random() * backoff
        retry_after = cls._retry_after_seconds(exc)
        if retry_after is not None:
            delay = max(delay, retry_after)
        time.sleep(delay)

    @staticmethod
    def _retry_after_seconds(exc: Any) -> Optional[float]:
        """Parse a ``Retry-After`` header (seconds) off the error response."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            return None  # HTTP-date form is not honored; fall back to backoff.
        return max(0.0, seconds)

    #: Upstream error bodies may reflect untrusted request fragments; bound the
    #: length we interpolate so a verbose/echoed body cannot flood CI logs.
    _MAX_DETAIL_CHARS = 200

    @classmethod
    def _error_detail(cls, exc: Any) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return cls._truncate_detail(str(exc))
        try:
            data = response.json()
            message = data.get("error", {}).get("message")
            if message:
                return cls._truncate_detail(message)
        except Exception:  # pragma: no cover - defensive
            pass
        return cls._truncate_detail(getattr(response, "text", str(exc)))

    @classmethod
    def _truncate_detail(cls, detail: Any) -> str:
        text = str(detail).replace("\n", " ").replace("\r", " ").strip()
        if len(text) > cls._MAX_DETAIL_CHARS:
            return text[: cls._MAX_DETAIL_CHARS] + "…"
        return text

    # ------------------------------------------------------------------ #
    # Response parsing                                                   #
    # ------------------------------------------------------------------ #
    def _parse(self, payload: dict[str, Any]) -> CompletionResult:
        output_items = payload.get("output", []) or []

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in output_items:
            item_type = item.get("type")
            if item_type == "message":
                text_parts.append(self._extract_message_text(item))
            elif item_type == "function_call":
                tool_calls.append(self._parse_function_call(item))
            # Reasoning / other item types carry no user-facing text here.

        usage = self._parse_usage(payload.get("usage"))
        finish_reason = self._normalize_finish(payload, tool_calls)

        return CompletionResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=payload,
        )

    @staticmethod
    def _extract_message_text(item: dict[str, Any]) -> str:
        parts: list[str] = []
        for block in item.get("content", []) or []:
            block_type = block.get("type")
            if block_type in ("output_text", "text"):
                parts.append(block.get("text", ""))
            elif block_type == "refusal":
                # A refusal block carries the model's decline reason under
                # ``refusal``. Surfacing it (instead of dropping it) is critical:
                # otherwise a refused turn looks like a clean EMPTY completion
                # (no text, no tool_calls, STOP), which downstream silently
                # interprets as "no issues" and can zero every finding.
                parts.append(block.get("refusal", "") or "")
        return "".join(parts)

    @staticmethod
    def _parse_function_call(item: dict[str, Any]) -> ToolCall:
        raw_args = item.get("arguments", "")
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProviderError(
                f"could not parse function_call arguments as JSON: {exc}"
            ) from exc
        call_id = item.get("call_id") or item.get("id") or ""
        return ToolCall(id=call_id, name=item.get("name", ""), args=args)

    @staticmethod
    def _parse_usage(usage: Optional[dict[str, Any]]) -> Usage:
        if not usage:
            return Usage()
        details = usage.get("input_tokens_details") or {}
        return Usage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read=int(details.get("cached_tokens", 0) or 0),
            cache_write=0,
        )

    #: ``incomplete_details.reason`` values that mean a safety/content filter
    #: stopped the turn (NOT a length truncation). The precise reason stays in
    #: ``CompletionResult.raw['incomplete_details']`` for callers that need it.
    _CONTENT_FILTER_REASONS = frozenset({"content_filter", "content_filtered"})

    @classmethod
    def _normalize_finish(
        cls, payload: dict[str, Any], tool_calls: list[ToolCall]
    ) -> FinishReason:
        if tool_calls:
            return FinishReason.TOOL_USE

        status = payload.get("status")
        if status == "incomplete":
            reason = (payload.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                return FinishReason.MAX_TOKENS
            if reason in cls._CONTENT_FILTER_REASONS:
                # A safety stop is terminal, NOT a length truncation. The domain
                # FinishReason enum has no content-filter member, so map it to
                # STOP (matching ConverseAdapter's content_filtered -> STOP) and
                # keep LENGTH reserved for genuine truncation. The exact reason
                # remains available on the raw payload.
                return FinishReason.STOP
            return FinishReason.LENGTH
        # "completed" and any other terminal status -> STOP.
        return FinishReason.STOP
