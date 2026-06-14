"""Tests for .openrabbit.yaml loading (SPEC 8.2).

load_config accepts either a path (str/Path) or an already-parsed dict, applies
sane defaults, and validates. No network; pyyaml is the only parsing dep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openrabbit.config import (
    ConfigError,
    Config,
    ExternalTools,
    ModelRole,
    ReviewConfig,
    Telemetry,
    load_config,
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
