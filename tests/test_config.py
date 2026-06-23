"""Tests for .openrabbit.yaml loading (SPEC 8.2).

load_config accepts either a path (str/Path) or an already-parsed dict, applies
sane defaults, and validates. No network; pyyaml is the only parsing dep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openrabbit.config import (
    Config,
    ConfigError,
    ExternalTools,
    ModelRole,
    ReviewConfig,
    Telemetry,
    load_config,
    validate_model_roles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / ".openrabbit.example.yaml"

FULL = {
    "version": 1,
    "review": {
        "profile": "assertive",
        "confidence_gate": 0.9,
        "incremental": False,
        "path_filters": ["!**/dist/**"],
        "path_instructions": [
            {"path": "src/api/**", "instructions": "focus on authn/authz"}
        ],
        "lenses": ["correctness", "security"],
    },
    "model_roles": {
        "triage": {"model": "amazon.nova-lite-v1:0", "region": "ap-northeast-2"},
        "verifier": {
            "model": "openai.gpt-5.5",
            "region": "us-east-2",
            "reasoning_effort": "medium",
            "store": False,
        },
    },
    "external_tools": {"enabled": ["ruff", "semgrep"]},
    "telemetry": {"enabled": True, "mode": "opt-out"},
}


# --------------------------------------------------------------------------- #
# valid full config (dict input)                                              #
# --------------------------------------------------------------------------- #
def test_load_full_config_from_dict():
    cfg = load_config(FULL)
    assert isinstance(cfg, Config)
    assert cfg.version == 1
    assert isinstance(cfg.review, ReviewConfig)
    assert cfg.review.profile == "assertive"
    assert cfg.review.confidence_gate == 0.9
    assert cfg.review.incremental is False
    assert cfg.review.path_filters == ["!**/dist/**"]
    assert cfg.review.lenses == ["correctness", "security"]


def test_path_instructions_parsed():
    cfg = load_config(FULL)
    assert len(cfg.review.path_instructions) == 1
    pi = cfg.review.path_instructions[0]
    assert pi.path == "src/api/**"
    assert pi.instructions == "focus on authn/authz"


def test_model_roles_parsed():
    cfg = load_config(FULL)
    assert set(cfg.model_roles) == {"triage", "verifier"}
    triage = cfg.model_roles["triage"]
    assert isinstance(triage, ModelRole)
    assert triage.model == "amazon.nova-lite-v1:0"
    assert triage.region == "ap-northeast-2"
    verifier = cfg.model_roles["verifier"]
    assert verifier.model == "openai.gpt-5.5"
    # extra/unknown role keys retained as options
    assert verifier.options.get("reasoning_effort") == "medium"
    assert verifier.options.get("store") is False


def test_external_tools_parsed():
    cfg = load_config(FULL)
    assert isinstance(cfg.external_tools, ExternalTools)
    assert cfg.external_tools.enabled == ["ruff", "semgrep"]


def test_telemetry_parsed():
    cfg = load_config(FULL)
    assert isinstance(cfg.telemetry, Telemetry)
    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.mode == "opt-out"


# --------------------------------------------------------------------------- #
# defaults                                                                     #
# --------------------------------------------------------------------------- #
def test_empty_config_uses_defaults():
    cfg = load_config({})
    assert cfg.version == 1
    assert cfg.review.profile == "balanced"
    assert cfg.review.confidence_gate == 0.80
    assert cfg.review.incremental is True
    assert cfg.review.path_instructions == []
    assert cfg.review.lenses == [
        "correctness",
        "security",
        "performance",
        "tests",
        "maintainability",
    ]
    assert cfg.model_roles == {}
    assert cfg.external_tools.enabled == []
    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.mode == "opt-out"


def test_partial_review_fills_remaining_defaults():
    cfg = load_config({"review": {"profile": "chill"}})
    assert cfg.review.profile == "chill"
    # untouched fields fall back to defaults
    assert cfg.review.confidence_gate == 0.80
    assert cfg.review.incremental is True


# --------------------------------------------------------------------------- #
# file input                                                                   #
# --------------------------------------------------------------------------- #
def test_load_from_path(tmp_path):
    p = tmp_path / ".openrabbit.yaml"
    p.write_text(
        "version: 1\nreview:\n  profile: assertive\n  confidence_gate: 0.7\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.review.profile == "assertive"
    assert cfg.review.confidence_gate == 0.7


def test_load_from_str_path(tmp_path):
    p = tmp_path / ".openrabbit.yaml"
    p.write_text("review: {profile: chill}\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.review.profile == "chill"


def test_packaged_example_is_valid():
    assert EXAMPLE.exists(), "shipped .openrabbit.example.yaml must exist"
    cfg = load_config(EXAMPLE)
    assert isinstance(cfg, Config)
    assert cfg.review.profile in {"chill", "balanced", "assertive"}
    assert set(cfg.model_roles)  # example ships model role mappings


def test_packaged_example_ships_decided_model_switch():
    # The decided switch: Nova 2 Lite finder/triage + GPT-5.4 verifier
    # (replaces nova-pro finder + gpt-5.5 verifier). The example must parse,
    # validate clean (no hard ERROR -> no raise), and emit no soft warnings
    # since every model/region pair is LIVE-VERIFIED & on its allow-list.
    cfg = load_config(EXAMPLE)

    triage = cfg.model_roles["triage"]
    assert triage.model == "global.amazon.nova-2-lite-v1:0"
    assert triage.region == "ap-northeast-2"

    finder = cfg.model_roles["finder"]
    assert finder.model == "global.amazon.nova-2-lite-v1:0"
    assert finder.region == "ap-northeast-2"
    # NO reasoning_effort yet (Nova 2 Lite extended-thinking shape unconfirmed).
    assert "reasoning_effort" not in finder.options

    verifier = cfg.model_roles["verifier"]
    assert verifier.model == "openai.gpt-5.4"
    assert verifier.region == "us-east-2"
    assert verifier.options.get("reasoning_effort") == "medium"
    assert verifier.options.get("store") is False

    premium = cfg.model_roles["premium"]
    assert premium.model == "openai.gpt-5.4"
    assert premium.region == "us-east-2"
    assert premium.options.get("reasoning_effort") == "high"
    assert premium.options.get("enabled") is False

    # whole-config validation surfaces no problems at all
    assert validate_model_roles(cfg) == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# invalid configs                                                             #
# --------------------------------------------------------------------------- #
def test_invalid_profile_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"profile": "spicy"}})


def test_invalid_confidence_gate_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"confidence_gate": 1.5}})
    with pytest.raises(ConfigError):
        load_config({"review": {"confidence_gate": -0.2}})


def test_invalid_lens_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"lenses": ["correctness", "bogus"]}})


# --------------------------------------------------------------------------- #
# per-lens reasoning_effort (finder applies effort PER LENS)                   #
# --------------------------------------------------------------------------- #
def test_lens_reasoning_effort_defaults_empty():
    # Omitted -> no per-lens reasoning configured (every lens OFF by default).
    cfg = load_config({})
    assert cfg.review.lens_reasoning_effort == {}


def test_lens_reasoning_effort_parsed():
    cfg = load_config(
        {
            "review": {
                "lenses": ["correctness", "security", "maintainability"],
                "lens_reasoning_effort": {
                    "correctness": "low",
                    "security": "low",
                },
            }
        }
    )
    assert cfg.review.lens_reasoning_effort == {
        "correctness": "low",
        "security": "low",
    }
    # maintainability is intentionally absent -> reasoning stays OFF for it.
    assert "maintainability" not in cfg.review.lens_reasoning_effort


def test_lens_reasoning_effort_unknown_lens_rejected():
    with pytest.raises(ConfigError):
        load_config(
            {"review": {"lens_reasoning_effort": {"bogus": "low"}}}
        )


def test_lens_reasoning_effort_invalid_value_rejected():
    with pytest.raises(ConfigError):
        load_config(
            {"review": {"lens_reasoning_effort": {"correctness": "extreme"}}}
        )


def test_lens_reasoning_effort_must_be_mapping():
    with pytest.raises(ConfigError):
        load_config({"review": {"lens_reasoning_effort": ["correctness"]}})


def test_packaged_example_ships_per_lens_reasoning_effort():
    # Research-decided per-lens effort: correctness/security run LOW reasoning;
    # style/maintainability/tests stay OFF (omitted -> reasoning disabled).
    cfg = load_config(EXAMPLE)
    effort = cfg.review.lens_reasoning_effort
    assert effort.get("correctness") == "low"
    assert effort.get("security") == "low"
    assert "maintainability" not in effort
    assert "tests" not in effort


def test_verify_min_severity_default_is_high():
    # By default only HIGH/CRITICAL findings route through the expensive
    # cross-family verifier (SPEC 7.3 cost lever #3).
    cfg = load_config({})
    assert cfg.review.verify_min_severity == "high"


def test_verify_min_severity_configurable():
    cfg = load_config({"review": {"verify_min_severity": "medium"}})
    assert cfg.review.verify_min_severity == "medium"


def test_verify_min_severity_invalid_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"verify_min_severity": "spicy"}})


def test_always_verify_categories_default_is_trust_core():
    # Trust-core lenses (correctness/security) route through the verifier
    # REGARDLESS of severity by default (verify-strict thesis).
    cfg = load_config({})
    assert cfg.review.always_verify_categories == frozenset({"correctness", "security"})


def test_always_verify_categories_configurable():
    cfg = load_config(
        {"review": {"always_verify_categories": ["correctness", "security", "tests"]}}
    )
    assert cfg.review.always_verify_categories == frozenset(
        {"correctness", "security", "tests"}
    )


def test_always_verify_categories_can_be_emptied():
    cfg = load_config({"review": {"always_verify_categories": []}})
    assert cfg.review.always_verify_categories == frozenset()


def test_always_verify_categories_unknown_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"always_verify_categories": ["correctness", "bogus"]}})


def test_unverified_confidence_gate_default_is_high():
    # Findings that bypass the verifier (below severity AND not trust-core) must
    # clear a HIGHER bar than the normal gate to post unverified (low-noise).
    cfg = load_config({})
    assert cfg.review.unverified_confidence_gate == 0.9


def test_unverified_confidence_gate_configurable():
    cfg = load_config({"review": {"unverified_confidence_gate": 0.95}})
    assert cfg.review.unverified_confidence_gate == 0.95


def test_unverified_confidence_gate_must_be_ge_confidence_gate():
    # The unverified bar must never be LOOSER than the normal gate — that would
    # re-open the leak it exists to close.
    with pytest.raises(ConfigError):
        load_config(
            {"review": {"confidence_gate": 0.85, "unverified_confidence_gate": 0.80}}
        )


def test_unverified_confidence_gate_out_of_range_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"unverified_confidence_gate": 1.5}})


def test_invalid_telemetry_mode_rejected():
    with pytest.raises(ConfigError):
        load_config({"telemetry": {"mode": "always"}})


def test_model_role_missing_model_rejected():
    with pytest.raises(ConfigError):
        load_config({"model_roles": {"triage": {"region": "ap-northeast-2"}}})


def test_non_mapping_top_level_rejected():
    with pytest.raises(ConfigError):
        load_config([1, 2, 3])  # type: ignore[arg-type]


def test_path_instruction_missing_fields_rejected():
    with pytest.raises(ConfigError):
        load_config({"review": {"path_instructions": [{"path": "src/**"}]}})


# --------------------------------------------------------------------------- #
# model_roles validation against Bedrock allow-lists / regions (SPEC 7.2/8.2) #
# --------------------------------------------------------------------------- #
def test_gpt_verifier_in_us_east_2_loads_clean():
    cfg = load_config(
        {
            "model_roles": {
                "verifier": {"model": "openai.gpt-5.5", "region": "us-east-2"}
            }
        }
    )
    assert cfg.model_roles["verifier"].region == "us-east-2"
    # no hard errors, no warnings for an in-allow-list GPT-5.5
    assert validate_model_roles(cfg) == []


def test_gpt_verifier_in_seoul_is_config_error():
    with pytest.raises(ConfigError) as exc:
        load_config(
            {
                "model_roles": {
                    "verifier": {"model": "openai.gpt-5.5", "region": "ap-northeast-2"}
                }
            }
        )
    # the error names the offending region + role
    assert "ap-northeast-2" in str(exc.value)
    assert "verifier" in str(exc.value)


def test_gpt_verifier_via_profile_in_seoul_is_config_error():
    with pytest.raises(ConfigError):
        load_config(
            {
                "model_roles": {
                    "verifier": {
                        "model": "us.openai.gpt-5.5",
                        "region": "ap-northeast-2",
                    }
                }
            }
        )


def test_nova_finder_in_seoul_loads_clean():
    cfg = load_config(
        {
            "model_roles": {
                "finder": {"model": "amazon.nova-pro-v1:0", "region": "ap-northeast-2"}
            }
        }
    )
    assert validate_model_roles(cfg) == []


def test_unknown_model_id_is_warning_not_error():
    # An unknown model must NOT block load_config (soft), but should warn.
    cfg = load_config(
        {
            "model_roles": {
                "finder": {
                    "model": "amazon.titan-imaginary-v9:0",
                    "region": "us-east-1",
                }
            }
        }
    )
    warnings = validate_model_roles(cfg)
    assert len(warnings) == 1
    assert "finder" in warnings[0]
    assert "amazon.titan-imaginary-v9:0" in warnings[0]


def test_nova_off_allowlist_region_is_warning_not_error():
    cfg = load_config(
        {
            "model_roles": {
                "finder": {"model": "amazon.nova-pro-v1:0", "region": "eu-west-3"}
            }
        }
    )
    warnings = validate_model_roles(cfg)
    assert len(warnings) == 1
    assert "eu-west-3" in warnings[0]
    assert "finder" in warnings[0]


def test_claude_premium_in_us_east_1_loads_clean():
    cfg = load_config(
        {
            "model_roles": {
                "premium": {
                    "model": "global.anthropic.claude-opus-4-6-v1",
                    "region": "us-east-1",
                    "enabled": False,
                }
            }
        }
    )
    assert validate_model_roles(cfg) == []


def test_validate_model_roles_collects_multiple_warnings():
    cfg = load_config(
        {
            "model_roles": {
                "finder": {"model": "amazon.nova-pro-v1:0", "region": "eu-west-3"},
                "triage": {"model": "amazon.mystery-v1:0", "region": "us-east-1"},
            }
        }
    )
    warnings = validate_model_roles(cfg)
    assert len(warnings) == 2


def test_model_role_missing_region_is_warning():
    cfg = load_config({"model_roles": {"finder": {"model": "amazon.nova-pro-v1:0"}}})
    warnings = validate_model_roles(cfg)
    assert len(warnings) == 1
    assert "finder" in warnings[0]


def test_packaged_example_has_no_model_role_warnings():
    cfg = load_config(EXAMPLE)
    assert validate_model_roles(cfg) == []


def test_load_config_collect_warnings_surfaces_soft_issues():
    # load_config can return soft warnings to the caller without raising.
    cfg, warnings = load_config(
        {
            "model_roles": {
                "finder": {"model": "amazon.nova-pro-v1:0", "region": "eu-west-3"}
            }
        },
        collect_warnings=True,
    )
    assert isinstance(cfg, Config)
    assert len(warnings) == 1
    assert "eu-west-3" in warnings[0]


def test_load_config_collect_warnings_still_raises_hard_errors():
    with pytest.raises(ConfigError):
        load_config(
            {
                "model_roles": {
                    "verifier": {"model": "openai.gpt-5.5", "region": "ap-northeast-2"}
                }
            },
            collect_warnings=True,
        )
