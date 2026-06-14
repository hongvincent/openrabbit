"""Known Bedrock model ids, their provider family, and region allow-lists.

This module is the single source of truth for *which* Bedrock models openrabbit
recognizes, *what kind of adapter* each is driven by, and *where* each model is
allowed to run (SPEC 7.1 / 7.2 / 8.2). The config loader
(:mod:`openrabbit.config`) consults it to validate a repo's ``model_roles`` so a
mis-region'd or mistyped model id fails fast in CI instead of at first Bedrock
call.

Two provider families (SPEC 7.1):

* :data:`ADAPTER_RESPONSES` — GPT-5.5 (``openai.*``) over Bedrock's
  OpenAI-compatible *mantle* Responses endpoint. The endpoint only exists in
  ``us-east-1`` / ``us-east-2``, so a GPT-5.5 role outside those regions is a
  **hard error** — there is nowhere for the call to land.
* :data:`ADAPTER_CONVERSE` — Amazon Nova (``amazon.*``) and Claude
  (``anthropic.*``) via ``bedrock-runtime`` Converse. Their region availability
  is broader and varies by inference profile; an off-allow-list region is a
  **soft warning** (the model may still be reachable via a cross-region profile)
  rather than a hard error.

Pure data + small pure lookups — no cloud SDKs, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# --------------------------------------------------------------------------- #
# adapter families (SPEC 7.1)                                                  #
# --------------------------------------------------------------------------- #
#: GPT-5.5 over the Bedrock OpenAI-compatible (mantle) Responses endpoint.
ADAPTER_RESPONSES = "responses"
#: Amazon Nova / Claude over ``bedrock-runtime`` Converse.
ADAPTER_CONVERSE = "converse"


class Severity(str, Enum):
    """Verdict severity for a model/region check."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Verdict:
    """The outcome of validating a single ``(model, region)`` pair.

    A ``None`` verdict means "no problem". A non-``None`` verdict carries a
    :class:`Severity` (``ERROR`` = hard, load fails; ``WARNING`` = soft,
    collected and surfaced) plus a human-readable, untrusted-input-free message.
    """

    severity: Severity
    message: str


@dataclass(frozen=True)
class ModelInfo:
    """Static facts about one recognized Bedrock model id.

    ``allowed_regions`` is the set of regions we *expect* the model in. For
    Responses-family models the list is authoritative (the mantle endpoint
    physically only exists there); for Converse-family models it is advisory
    (cross-region inference profiles can widen reach).
    """

    model_id: str
    adapter: str
    allowed_regions: frozenset[str] = field(default_factory=frozenset)
    #: ``True`` -> off-allow-list region is a hard ERROR (no endpoint exists);
    #: ``False`` -> off-allow-list region is a soft WARNING.
    region_strict: bool = False


# --------------------------------------------------------------------------- #
# region allow-lists (SPEC 7.2)                                               #
# --------------------------------------------------------------------------- #
#: GPT-5.5 mantle endpoint regions — the ONLY places GPT-5.5 can run. Mirrors
#: ``openrabbit.providers.openai_responses.SUPPORTED_REGIONS``.
GPT_REGIONS: frozenset[str] = frozenset({"us-east-1", "us-east-2"})

#: Amazon Nova regions (Seoul + common US/EU) plus where cross-region profiles
#: route. Seoul (``ap-northeast-2``) is the default finder/triage home.
NOVA_REGIONS: frozenset[str] = frozenset(
    {"ap-northeast-2", "us-east-1", "us-east-2", "us-west-2", "eu-central-1"}
)

#: Claude-on-Bedrock common regions (premium, optional, cost-gated role).
CLAUDE_REGIONS: frozenset[str] = frozenset(
    {"us-east-1", "us-east-2", "us-west-2", "eu-central-1", "ap-northeast-2"}
)

# --------------------------------------------------------------------------- #
# cross-region inference-profile prefixes                                     #
# --------------------------------------------------------------------------- #
#: A model id like ``us.openai.gpt-5.5`` or ``global.anthropic.claude-...`` is a
#: cross-region inference profile wrapping a base model id. We strip these to
#: resolve the underlying model. Kept in sync with
#: ``openrabbit.pricing._PROFILE_PREFIXES``.
PROFILE_PREFIXES: tuple[str, ...] = (
    "us.",
    "use1.",
    "use2.",
    "apac.",
    "eu.",
    "global.",
)

# --------------------------------------------------------------------------- #
# known-model registry                                                         #
# --------------------------------------------------------------------------- #
KNOWN_MODELS: dict[str, ModelInfo] = {
    # Verifier / judge — GPT-5.5 via mantle Responses (strict region).
    "openai.gpt-5.5": ModelInfo(
        model_id="openai.gpt-5.5",
        adapter=ADAPTER_RESPONSES,
        allowed_regions=GPT_REGIONS,
        region_strict=True,
    ),
    # Finder / triage — Amazon Nova via Converse (soft region).
    "amazon.nova-pro-v1:0": ModelInfo(
        model_id="amazon.nova-pro-v1:0",
        adapter=ADAPTER_CONVERSE,
        allowed_regions=NOVA_REGIONS,
    ),
    "amazon.nova-lite-v1:0": ModelInfo(
        model_id="amazon.nova-lite-v1:0",
        adapter=ADAPTER_CONVERSE,
        allowed_regions=NOVA_REGIONS,
    ),
    "amazon.nova-micro-v1:0": ModelInfo(
        model_id="amazon.nova-micro-v1:0",
        adapter=ADAPTER_CONVERSE,
        allowed_regions=NOVA_REGIONS,
    ),
    # Premium — Claude on Bedrock via Converse (soft region).
    "anthropic.claude-opus-4-6-v1": ModelInfo(
        model_id="anthropic.claude-opus-4-6-v1",
        adapter=ADAPTER_CONVERSE,
        allowed_regions=CLAUDE_REGIONS,
    ),
    "anthropic.claude-sonnet-4-5-v1": ModelInfo(
        model_id="anthropic.claude-sonnet-4-5-v1",
        adapter=ADAPTER_CONVERSE,
        allowed_regions=CLAUDE_REGIONS,
    ),
}

#: Family prefix -> adapter, for resolving *unregistered* (but recognizably
#: shaped) model ids to an adapter family. Order doesn't matter — prefixes are
#: disjoint.
_FAMILY_ADAPTERS: dict[str, str] = {
    "openai.": ADAPTER_RESPONSES,
    "amazon.": ADAPTER_CONVERSE,
    "anthropic.": ADAPTER_CONVERSE,
}


# --------------------------------------------------------------------------- #
# lookups                                                                      #
# --------------------------------------------------------------------------- #
def normalize_model_id(model: str) -> str:
    """Strip a known cross-region inference-profile prefix from ``model``.

    ``us.openai.gpt-5.5`` -> ``openai.gpt-5.5``; a bare model id is returned
    unchanged. Only one prefix is stripped (profile ids never nest).
    """
    for prefix in PROFILE_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def lookup_model(model: str) -> Optional[ModelInfo]:
    """Return :class:`ModelInfo` for ``model`` (profile-aware), else ``None``.

    Tries the bare id, then retries after stripping a cross-region profile
    prefix so ``us.openai.gpt-5.5`` resolves to ``openai.gpt-5.5``.
    """
    info = KNOWN_MODELS.get(model)
    if info is not None:
        return info
    base = normalize_model_id(model)
    if base != model:
        return KNOWN_MODELS.get(base)
    return None


def adapter_for_model(model: str) -> Optional[str]:
    """Return the adapter family for ``model``, or ``None`` if unrecognizable.

    Resolves a registered model first; otherwise falls back to matching the
    family prefix (``openai.``/``amazon.``/``anthropic.``) of the
    profile-normalized id, so a not-yet-registered model of a known family still
    routes to the right adapter.
    """
    info = lookup_model(model)
    if info is not None:
        return info.adapter
    base = normalize_model_id(model)
    for prefix, adapter in _FAMILY_ADAPTERS.items():
        if base.startswith(prefix):
            return adapter
    return None


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
def validate_model_region(model: str, region: Optional[str]) -> Optional[Verdict]:
    """Check that ``model`` is allowed in ``region``; return a :class:`Verdict`.

    Returns ``None`` when there is no problem. Otherwise:

    * **unknown model** -> WARNING (we can't vouch for it, but don't block).
    * **known model, no region** -> WARNING (can't region-check it).
    * **strict-region model off its allow-list** (GPT-5.5 outside us-east-1/2)
      -> ERROR (the endpoint physically does not exist there).
    * **soft-region model off its allow-list** -> WARNING (a cross-region
      profile may still reach it).
    """
    info = lookup_model(model)
    if info is None:
        return Verdict(
            severity=Severity.WARNING,
            message=(
                f"unknown model id {model!r}; openrabbit cannot validate its "
                "region or adapter family"
            ),
        )

    if region is None:
        return Verdict(
            severity=Severity.WARNING,
            message=(
                f"model {model!r} has no region declared; cannot validate "
                "against its allow-list"
            ),
        )

    if region in info.allowed_regions:
        return None

    allowed = ", ".join(sorted(info.allowed_regions))
    if info.region_strict:
        return Verdict(
            severity=Severity.ERROR,
            message=(
                f"model {model!r} is not available in region {region!r}; "
                f"allowed regions are: {allowed}"
            ),
        )
    return Verdict(
        severity=Severity.WARNING,
        message=(
            f"model {model!r} region {region!r} is outside its known "
            f"allow-list ({allowed}); a cross-region inference profile may be "
            "required"
        ),
    )
