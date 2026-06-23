"""Tests for the Bedrock model-allowlist constants (SPEC 7.2, 8.2).

``openrabbit.bedrock_models`` knows the small set of Bedrock model ids
openrabbit ships defaults for, which provider *family* (Converse vs the
OpenAI-compatible Responses/mantle endpoint) each maps to, and which AWS
regions each model is allowed in. ``validate_model_region`` turns a
``(model, region)`` pair into a structured verdict the config loader uses to
emit hard errors (GPT-5.5 outside us-east-1/2) and soft warnings (unknown
model, off-allowlist region for non-GPT models).

Pure constants + lookups — no cloud SDKs, no network.
"""

from __future__ import annotations

from openrabbit.bedrock_models import (
    ADAPTER_CONVERSE,
    ADAPTER_RESPONSES,
    DEPRECATED_MODELS,
    ModelInfo,
    Severity,
    adapter_for_model,
    is_deprecated_model,
    lookup_model,
    normalize_model_id,
    validate_model_region,
)


# --------------------------------------------------------------------------- #
# adapter-family mapping                                                       #
# --------------------------------------------------------------------------- #
def test_gpt_maps_to_responses_adapter():
    assert adapter_for_model("openai.gpt-5.5") == ADAPTER_RESPONSES


def test_nova_maps_to_converse_adapter():
    assert adapter_for_model("amazon.nova-pro-v1:0") == ADAPTER_CONVERSE
    assert adapter_for_model("amazon.nova-lite-v1:0") == ADAPTER_CONVERSE


def test_claude_maps_to_converse_adapter():
    assert adapter_for_model("anthropic.claude-opus-4-6-v1") == ADAPTER_CONVERSE


def test_adapter_for_inference_profile_id():
    # cross-region inference-profile prefixes resolve to the base family
    assert adapter_for_model("global.anthropic.claude-opus-4-6-v1") == ADAPTER_CONVERSE
    assert adapter_for_model("us.openai.gpt-5.5") == ADAPTER_RESPONSES


def test_adapter_for_unknown_prefix_is_none():
    assert adapter_for_model("totally.unknown.model") is None


def test_adapter_for_unregistered_known_family_resolves_by_prefix():
    # A not-yet-registered model of a known family still routes correctly
    # (forward-compat): unknown openai.* -> Responses, unknown amazon.* ->
    # Converse.
    assert adapter_for_model("openai.gpt-6-future") == ADAPTER_RESPONSES
    assert adapter_for_model("amazon.nova-2-pro-v1:0") == ADAPTER_CONVERSE
    assert adapter_for_model("anthropic.claude-5-v1") == ADAPTER_CONVERSE


# --------------------------------------------------------------------------- #
# decided model switch — GPT-5.4 verifier + Nova 2 Lite finder/triage          #
# (LIVE-VERIFIED: global.amazon.nova-2-lite-v1:0 on Converse @ ap-northeast-2;  #
#  openai.gpt-5.4 on mantle Responses @ us-east-2)                              #
# --------------------------------------------------------------------------- #
def test_gpt_5_4_is_registered_responses_strict():
    info = lookup_model("openai.gpt-5.4")
    assert isinstance(info, ModelInfo)
    assert info.adapter == ADAPTER_RESPONSES
    assert info.region_strict is True
    assert "us-east-2" in info.allowed_regions
    assert "us-east-1" in info.allowed_regions


def test_gpt_5_4_maps_to_responses_adapter():
    assert adapter_for_model("openai.gpt-5.4") == ADAPTER_RESPONSES


def test_gpt_5_4_in_seoul_is_hard_error():
    verdict = validate_model_region("openai.gpt-5.4", "ap-northeast-2")
    assert verdict is not None
    assert verdict.severity == Severity.ERROR


def test_gpt_5_4_in_us_east_2_ok():
    assert validate_model_region("openai.gpt-5.4", "us-east-2") is None


def test_nova_2_lite_is_registered_converse():
    info = lookup_model("amazon.nova-2-lite-v1:0")
    assert isinstance(info, ModelInfo)
    assert info.adapter == ADAPTER_CONVERSE
    assert info.region_strict is False
    assert "ap-northeast-2" in info.allowed_regions


def test_nova_2_lite_maps_to_converse_adapter():
    assert adapter_for_model("amazon.nova-2-lite-v1:0") == ADAPTER_CONVERSE


def test_nova_2_lite_global_profile_resolves_to_base():
    # The "global." prefix (NOT "apac.") is correct for Nova 2; it must
    # normalize to the base id, resolve to its ModelInfo, and route to Converse.
    assert normalize_model_id("global.amazon.nova-2-lite-v1:0") == (
        "amazon.nova-2-lite-v1:0"
    )
    info = lookup_model("global.amazon.nova-2-lite-v1:0")
    assert info is not None
    assert info.adapter == ADAPTER_CONVERSE
    assert adapter_for_model("global.amazon.nova-2-lite-v1:0") == ADAPTER_CONVERSE


def test_nova_2_lite_global_profile_in_seoul_ok():
    # global.amazon.nova-2-lite-v1:0 @ ap-northeast-2 is the finder/triage home.
    assert (
        validate_model_region("global.amazon.nova-2-lite-v1:0", "ap-northeast-2")
        is None
    )


def test_global_prefix_is_recognized_profile_prefix():
    from openrabbit.bedrock_models import PROFILE_PREFIXES

    assert "global." in PROFILE_PREFIXES


# --------------------------------------------------------------------------- #
# deprecated legacy models — nova-pro Gen-1 (backward-compat, not removed)      #
# --------------------------------------------------------------------------- #
def test_nova_pro_still_resolves_for_backward_compat():
    # nova-pro is DEPRECATED but MUST keep resolving so existing configs that
    # still reference it don't fail to load (we annotate, we don't remove).
    info = lookup_model("amazon.nova-pro-v1:0")
    assert isinstance(info, ModelInfo)
    assert info.adapter == ADAPTER_CONVERSE


def test_nova_pro_is_in_deprecated_models():
    # The legacy Gen-1 finder is flagged deprecated (5K output cap, no
    # reasoning path, no global profile, EOL signal).
    assert "amazon.nova-pro-v1:0" in DEPRECATED_MODELS
    assert is_deprecated_model("amazon.nova-pro-v1:0") is True


def test_nova_2_lite_is_not_deprecated():
    # nova-2-lite is the *preferred* replacement finder; it must NOT be flagged.
    assert "amazon.nova-2-lite-v1:0" not in DEPRECATED_MODELS
    assert is_deprecated_model("amazon.nova-2-lite-v1:0") is False


def test_nova_2_lite_is_preferred_registered_finder():
    # The decided default finder/triage is registered and Converse-driven.
    info = lookup_model("amazon.nova-2-lite-v1:0")
    assert isinstance(info, ModelInfo)
    assert info.adapter == ADAPTER_CONVERSE
    assert info.region_strict is False


def test_is_deprecated_model_resolves_inference_profile():
    # A cross-region profile id wrapping a deprecated base is also deprecated.
    assert is_deprecated_model("global.amazon.nova-pro-v1:0") is True
    # ...and a profile over the preferred finder is not.
    assert is_deprecated_model("global.amazon.nova-2-lite-v1:0") is False


def test_is_deprecated_model_unknown_is_false():
    assert is_deprecated_model("amazon.titan-imaginary-v9:0") is False


# --------------------------------------------------------------------------- #
# normalization                                                               #
# --------------------------------------------------------------------------- #
def test_normalize_strips_known_region_profile_prefix():
    assert normalize_model_id("us.openai.gpt-5.5") == "openai.gpt-5.5"
    assert (
        normalize_model_id("global.anthropic.claude-opus-4-6-v1")
        == "anthropic.claude-opus-4-6-v1"
    )


def test_normalize_leaves_bare_model_untouched():
    assert normalize_model_id("amazon.nova-pro-v1:0") == "amazon.nova-pro-v1:0"


# --------------------------------------------------------------------------- #
# model lookup                                                                 #
# --------------------------------------------------------------------------- #
def test_lookup_known_model_returns_info():
    info = lookup_model("openai.gpt-5.5")
    assert isinstance(info, ModelInfo)
    assert info.adapter == ADAPTER_RESPONSES
    assert "us-east-1" in info.allowed_regions
    assert "us-east-2" in info.allowed_regions


def test_lookup_resolves_inference_profile_to_base():
    info = lookup_model("us.openai.gpt-5.5")
    assert info is not None
    assert info.adapter == ADAPTER_RESPONSES


def test_lookup_unknown_model_returns_none():
    assert lookup_model("amazon.titan-imaginary-v9:0") is None


# --------------------------------------------------------------------------- #
# region validation — GPT-5.5 (hard allow-list)                               #
# --------------------------------------------------------------------------- #
def test_gpt_in_us_east_2_ok():
    verdict = validate_model_region("openai.gpt-5.5", "us-east-2")
    assert verdict is None  # no problem


def test_gpt_in_us_east_1_ok():
    assert validate_model_region("openai.gpt-5.5", "us-east-1") is None


def test_gpt_in_seoul_is_hard_error():
    verdict = validate_model_region("openai.gpt-5.5", "ap-northeast-2")
    assert verdict is not None
    assert verdict.severity == Severity.ERROR
    assert "ap-northeast-2" in verdict.message
    assert "us-east-1" in verdict.message or "us-east-2" in verdict.message


def test_gpt_via_profile_in_unsupported_region_is_hard_error():
    verdict = validate_model_region("us.openai.gpt-5.5", "ap-northeast-2")
    assert verdict is not None
    assert verdict.severity == Severity.ERROR


# --------------------------------------------------------------------------- #
# region validation — Nova / Claude (soft allow-list)                         #
# --------------------------------------------------------------------------- #
def test_nova_in_seoul_ok():
    assert validate_model_region("amazon.nova-pro-v1:0", "ap-northeast-2") is None
    assert validate_model_region("amazon.nova-lite-v1:0", "ap-northeast-2") is None


def test_nova_in_off_allowlist_region_is_warning():
    verdict = validate_model_region("amazon.nova-pro-v1:0", "eu-west-3")
    assert verdict is not None
    assert verdict.severity == Severity.WARNING


def test_claude_in_common_region_ok():
    assert validate_model_region("anthropic.claude-opus-4-6-v1", "us-east-1") is None


def test_claude_profile_region_ok():
    assert (
        validate_model_region("global.anthropic.claude-opus-4-6-v1", "us-east-1")
        is None
    )


# --------------------------------------------------------------------------- #
# region validation — unknown model                                           #
# --------------------------------------------------------------------------- #
def test_unknown_model_is_warning_not_error():
    verdict = validate_model_region("amazon.titan-imaginary-v9:0", "ap-northeast-2")
    assert verdict is not None
    assert verdict.severity == Severity.WARNING
    assert (
        "unknown" in verdict.message.lower()
        or "unrecognized" in verdict.message.lower()
    )


def test_unknown_model_with_unknown_prefix_is_warning():
    verdict = validate_model_region("mystery.model.x", "us-east-1")
    assert verdict is not None
    assert verdict.severity == Severity.WARNING


def test_missing_region_for_known_model_is_warning():
    # a recognized model with no region declared can't be region-checked
    verdict = validate_model_region("amazon.nova-pro-v1:0", None)
    assert verdict is not None
    assert verdict.severity == Severity.WARNING
