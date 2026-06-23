"""``ConverseAdapter`` — AWS Bedrock Converse API provider (SPEC 7.1).

Maps the neutral domain model (:mod:`openrabbit.domain`) to and from the
Bedrock ``Converse`` request/response shape, covering Amazon Nova and
Claude-on-Bedrock behind the single :class:`~openrabbit.providers.base.Provider`
interface used by the spine.

What this adapter handles
-------------------------
* **System** -> ``system=[{"text": ...}]`` (omitted when empty).
* **Messages** -> ``messages=[{"role", "content":[block,...]}]``; string
  content becomes one ``{"text": ...}`` block, list content is passed through,
  and any embedded :class:`~openrabbit.domain.ToolResult` is rendered as a
  ``{"toolResult": {...}}`` block.
* **Inference** -> ``inferenceConfig={"maxTokens": ..., ...}`` (extra opts such
  as ``temperature``/``top_p`` are folded in with Converse key names).
* **Tools** -> ``toolConfig.tools[].toolSpec.inputSchema.json`` and, when a
  ``tool_choice`` opt is given, ``toolConfig.toolChoice`` — including forced
  structured output via a single ``emit_findings`` tool
  (``toolChoice={"tool": {"name": ...}}``).
* **Prompt caching** -> ``cachePoint`` blocks inserted into ``system`` and the
  final ``messages`` block when ``cache_prefix`` is supplied (SPEC 6 step 3 /
  7.1). The ``tools`` array is intentionally NOT cached: Amazon Nova rejects a
  tool-level cache point on the real Converse API, and the tools list is tiny
  and stable, so the cacheable bytes live entirely in system/messages.
* **Parsing** -> concatenated text, parsed ``toolUse`` blocks into
  :class:`~openrabbit.domain.ToolCall`, normalized :class:`FinishReason` from
  ``stopReason``, and :class:`Usage` from
  ``inputTokens``/``outputTokens``/``cacheReadInputTokens``/``cacheWriteInputTokens``.

boto3 is imported **lazily** inside methods so importing this module needs zero
AWS dependencies and unit tests can monkeypatch a fake ``boto3``.
"""

from __future__ import annotations

from typing import Any, Optional

from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from openrabbit.providers.base import Provider, ProviderError

# Converse ``stopReason`` -> neutral FinishReason. Anything unknown (or a safety
# stop such as content_filtered/guardrail_intervened) collapses to STOP so the
# spine treats the turn as terminal rather than retrying blindly.
_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_USE,
    "max_tokens": FinishReason.MAX_TOKENS,
}

# inferenceConfig keys use camelCase; map the common neutral opt names.
_INFERENCE_OPT_KEYS: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "topP",
    "topP": "topP",
    "top_k": "topK",
    "topK": "topK",
    "stop_sequences": "stopSequences",
    "stopSequences": "stopSequences",
}

# Nova 2 extended-thinking (reasoning) effort levels accepted via the Converse
# ``additionalModelRequestFields.reasoningConfig.maxReasoningEffort``.
_REASONING_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high"})

# Explicit "reasoning disabled" sentinels. ``None`` / "none" / "off" / "" mean no
# reasoningConfig is injected (the model default ``type: "disabled"``). Anything
# OUTSIDE both this set and ``_REASONING_EFFORTS`` is a typo and must RAISE — a
# silent disable would burn a non-thinking pass while the operator believed
# extended thinking was on (parity with the Responses adapter, which raises).
_REASONING_DISABLE_VALUES: frozenset[str] = frozenset({"none", "off", ""})

# inferenceConfig sampling keys that MUST be omitted when reasoning effort is
# "high": Nova 2 raises a ValidationException if temperature/topP/topK are sent
# alongside high-effort reasoning. (low/medium tolerate them.)
_HIGH_EFFORT_BANNED_INFERENCE_KEYS: frozenset[str] = frozenset(
    {"temperature", "topP", "topK"}
)

_CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}

#: Bedrock throttles (``ThrottlingException``) under load. boto3's default retry
#: policy is too weak for the finder/verifier batch workload, so build the
#: client with an ADAPTIVE retry mode and a bounded attempt count. ``adaptive``
#: adds client-side rate limiting on top of exponential backoff.
_RETRY_MAX_ATTEMPTS = 5
_RETRY_MODE = "adaptive"

#: Pin SigV4 for the ``bedrock-runtime`` client.
#:
#: WHY THIS MATTERS: the GPT-5.5 verifier path requires ``AWS_BEARER_TOKEN_BEDROCK``
#: in the environment. Recent boto3 will AUTO-PREFER that bearer token for ANY
#: bearer-capable service — including ``bedrock-runtime`` — UNLESS a signature
#: version is set "in code" on the client config. In a real review the Nova
#: finder and the GPT-5.5 verifier run in the SAME process with BOTH credentials
#: present, so without this pin the Converse (Nova) client would silently pick up
#: the mantle bearer token and fail with ``AccessDeniedException`` ("Invalid API
#: Key format"). Setting ``signature_version`` here marks an explicit in-code
#: auth choice (``handlers._should_prefer_bearer_auth``), so SigV4 (profile/role)
#: creds are used and the ambient bearer token is correctly ignored on this path.
_SIGNATURE_VERSION = "v4"


class ConverseAdapter(Provider):
    """Provider backed by Bedrock ``bedrock-runtime`` :meth:`converse`.

    Parameters
    ----------
    model_id:
        Bedrock model id or inference-profile id, e.g.
        ``"amazon.nova-pro-v1:0"`` or a Claude-on-Bedrock profile.
    region:
        AWS region for the ``bedrock-runtime`` client, e.g.
        ``"ap-northeast-2"`` (Seoul) or ``"us-east-2"``.
    """

    def __init__(self, *, model_id: str, region: str) -> None:
        self._model_id = model_id
        self._region = region
        self._client: Any = None  # lazily built on first complete()

    # ------------------------------------------------------------------ #
    # Provider identity                                                  #
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "converse"

    @property
    def model(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------ #
    # Public API                                                         #
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
        client = self._get_client()
        request = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            cache_prefix=cache_prefix,
            opts=opts,
        )
        try:
            response = client.converse(**request)
        except ProviderError:
            raise
        except Exception as exc:  # boto3 ClientError, transport, etc.
            raise ProviderError(f"Bedrock converse failed: {exc}") from exc
        return self._parse_response(response)

    # ------------------------------------------------------------------ #
    # Client (lazy boto3 import)                                         #
    # ------------------------------------------------------------------ #
    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # lazy: keeps module import AWS-free
            from botocore.config import Config  # lazy: AWS-free module import

            config = Config(
                retries={
                    "max_attempts": _RETRY_MAX_ATTEMPTS,
                    "mode": _RETRY_MODE,
                },
                # Pin SigV4 so an ambient AWS_BEARER_TOKEN_BEDROCK (the GPT-5.5
                # verifier's credential) is NOT hijacked for the Nova Converse
                # client when both paths run in one process. See _SIGNATURE_VERSION.
                signature_version=_SIGNATURE_VERSION,
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=config,
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Request building                                                   #
    # ------------------------------------------------------------------ #
    def _build_request(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: Optional[list[Any]],
        max_tokens: int,
        cache_prefix: Optional[str],
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        use_cache = cache_prefix is not None
        opts = dict(opts)
        tool_choice = opts.pop("tool_choice", None)
        # ``reasoning_effort`` is a Converse top-level concern (it maps to
        # additionalModelRequestFields), never an inferenceConfig key — pop it
        # out before the inference config is built so it cannot leak through.
        effort = self._normalize_reasoning_effort(opts.pop("reasoning_effort", None))

        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": self._build_messages(messages, use_cache=use_cache),
            "inferenceConfig": self._build_inference_config(
                max_tokens, opts, effort=effort
            ),
        }

        system_blocks = self._build_system(system, use_cache=use_cache)
        if system_blocks:
            request["system"] = system_blocks

        tool_config = self._build_tool_config(tools, tool_choice, use_cache=use_cache)
        if tool_config is not None:
            request["toolConfig"] = tool_config

        if effort is not None:
            request["additionalModelRequestFields"] = {
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}
            }

        return request

    @staticmethod
    def _normalize_reasoning_effort(value: Any) -> Optional[str]:
        """Map a neutral ``reasoning_effort`` opt to a Nova 2 effort or None.

        ``"low"``/``"medium"``/``"high"`` enable extended thinking; absent /
        ``None`` / ``"none"`` / ``"off"`` / ``""`` (case-insensitively) mean
        disabled, returning ``None`` so no ``reasoningConfig`` is injected.

        Any OTHER value is a typo (e.g. ``"minimal"``, ``"xhigh"``) and RAISES a
        :class:`ProviderError` rather than silently disabling reasoning — a silent
        disable would burn a non-thinking pass while the operator believed
        extended thinking was on. This matches the Responses adapter, which also
        raises on an unrecognized effort.
        """
        if value is None:
            return None
        effort = str(value).strip().lower()
        if effort in _REASONING_DISABLE_VALUES:
            return None
        if effort in _REASONING_EFFORTS:
            return effort
        raise ProviderError(
            f"invalid reasoning effort {value!r}; expected one of "
            f"{sorted(_REASONING_EFFORTS)} or a disable sentinel "
            f"({sorted(_REASONING_DISABLE_VALUES - {''})}/None)"
        )

    @staticmethod
    def _build_system(system: str, *, use_cache: bool) -> list[dict[str, Any]]:
        if not system:
            return []
        blocks: list[dict[str, Any]] = [{"text": system}]
        if use_cache:
            blocks.append(dict(_CACHE_POINT))
        return blocks

    def _build_messages(
        self, messages: list[Message], *, use_cache: bool
    ) -> list[dict[str, Any]]:
        out = [self._build_message(m) for m in messages]
        # Anchor the cacheable prefix on the final message's content so the
        # shared PR context is cached and only the per-file suffix varies.
        if use_cache and out:
            out[-1]["content"].append(dict(_CACHE_POINT))
        return out

    def _build_message(self, message: Message) -> dict[str, Any]:
        return {
            "role": message.role,
            "content": self._build_content(message.content),
        }

    def _build_content(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, ToolResult):
                blocks.append(self._tool_result_block(item))
            else:
                blocks.append(item)  # already a Converse content block
        return blocks

    @staticmethod
    def _tool_result_block(result: ToolResult) -> dict[str, Any]:
        content = result.content
        if isinstance(content, str):
            content_blocks: list[dict[str, Any]] = [{"text": content}]
        elif isinstance(content, list):
            content_blocks = content
        else:
            content_blocks = [{"json": content}]
        block: dict[str, Any] = {
            "toolResult": {
                "toolUseId": result.id,
                "content": content_blocks,
            }
        }
        if result.is_error:
            block["toolResult"]["status"] = "error"
        return block

    @staticmethod
    def _build_inference_config(
        max_tokens: int, opts: dict[str, Any], *, effort: Optional[str] = None
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {"maxTokens": max_tokens}
        # When reasoning effort is "high", Nova 2 rejects sampling params
        # (temperature/topP/topK) with a ValidationException, so drop them.
        banned = (
            _HIGH_EFFORT_BANNED_INFERENCE_KEYS if effort == "high" else frozenset()
        )
        for key, value in opts.items():
            mapped = _INFERENCE_OPT_KEYS.get(key)
            if mapped is not None and mapped not in banned:
                cfg[mapped] = value
        return cfg

    def _build_tool_config(
        self,
        tools: Optional[list[Any]],
        tool_choice: Any,
        *,
        use_cache: bool,
    ) -> Optional[dict[str, Any]]:
        if not tools:
            return None
        # NOTE: we deliberately do NOT append a ``cachePoint`` block to the
        # ``tools`` array. Amazon Nova (the finder this adapter primarily serves)
        # REJECTS a tool-level cache point on the real Converse API with a
        # ValidationException ("extraneous key [cachePoint] is not permitted"),
        # and tool-block caching is not portable across the model families here.
        # The large, byte-stable rubric/context lives in ``system``/``messages``
        # (which DO accept cachePoint and carry ~all the cacheable bytes), so the
        # tiny, stable tools list is left uncached — correctness over a negligible
        # token saving. ``use_cache`` is intentionally unused for tools.
        del use_cache
        tool_entries: list[dict[str, Any]] = [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.json_schema},
                }
            }
            for t in tools
        ]
        config: dict[str, Any] = {"tools": tool_entries}
        if tool_choice is not None:
            config["toolChoice"] = self._build_tool_choice(tool_choice)
        return config

    @staticmethod
    def _build_tool_choice(tool_choice: Any) -> dict[str, Any]:
        if isinstance(tool_choice, dict):
            return tool_choice
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"auto": {}}
            if tool_choice == "any":
                return {"any": {}}
            # A bare tool name -> forced single-tool structured output.
            return {"tool": {"name": tool_choice}}
        raise ProviderError(f"Unsupported tool_choice: {tool_choice!r}")

    # ------------------------------------------------------------------ #
    # Response parsing                                                   #
    # ------------------------------------------------------------------ #
    def _parse_response(self, response: dict[str, Any]) -> CompletionResult:
        message = response.get("output", {}).get("message", {})
        blocks = message.get("content", []) or []

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in blocks:
            # ``reasoningContent`` blocks (Nova 2 extended thinking) are NOT
            # user-facing — their text is redacted chain-of-thought billed as
            # output. Skip them entirely so reasoning never leaks into the
            # result text; only the normal ``text``/``toolUse`` blocks count.
            if "reasoningContent" in block:
                continue
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_calls.append(self._parse_tool_use(block["toolUse"]))

        finish_reason = _STOP_REASONS.get(
            response.get("stopReason", ""), FinishReason.STOP
        )

        return CompletionResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=self._parse_usage(response.get("usage")),
            raw=response,
        )

    @staticmethod
    def _parse_tool_use(tool_use: dict[str, Any]) -> ToolCall:
        return ToolCall(
            id=tool_use.get("toolUseId", ""),
            name=tool_use.get("name", ""),
            args=tool_use.get("input", {}) or {},
        )

    @staticmethod
    def _parse_usage(usage: Optional[dict[str, Any]]) -> Usage:
        if not usage:
            return Usage()
        return Usage(
            input_tokens=usage.get("inputTokens", 0) or 0,
            output_tokens=usage.get("outputTokens", 0) or 0,
            cache_read=usage.get("cacheReadInputTokens", 0) or 0,
            cache_write=usage.get("cacheWriteInputTokens", 0) or 0,
        )
