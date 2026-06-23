"""Wiring guard: the live harness must validate the SHIPPED model pair.

This is the END-TO-END / live-wiring regression test for the class of bug where
the live FP<10% gate silently validated a DEPRECATED pre-switch model pair while
production had already switched models. The live smokes are creds-gated (skipped
offline), so an offline-green suite could not catch a stale model pin — this test
closes that gap by asserting, OFFLINE and deterministically, that the constants
the live tests feed into ``model_factory`` / the real adapters are exactly the
ones production ships in ``.openrabbit.yaml`` (the source of truth the CLI loads).

It is intentionally NOT marked ``live``: it makes no network calls and must run
in the default offline suite so a future stale-pin regression fails fast here
instead of only being noticed when someone spends real money on the live gate.

Pins guarded:
* finder/triage -> ``global.amazon.nova-2-lite-v1:0`` (Seoul Converse)
* verifier/judge -> ``openai.gpt-5.4`` (us-east-2 Responses)
and the negative invariant: neither pinned model is in the deprecated registry.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from openrabbit.bedrock_models import is_deprecated_model, normalize_model_id

from .conftest import GPT_MODEL, GPT_REGION, NOVA_MODEL, NOVA_REGION

#: Repo root (this file lives at tests/live/).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The shipped config the CLI loads — the single source of truth for which
#: model pair production runs. The live gate must validate THIS pair.
SHIPPED_CONFIG = REPO_ROOT / ".openrabbit.example.yaml"


def _shipped_role(role: str) -> dict:
    data = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    return data["model_roles"][role]


def test_live_finder_matches_shipped_config() -> None:
    """The live Nova pin must equal the shipped finder/triage model + region."""
    finder = _shipped_role("finder")
    assert finder["model"] == NOVA_MODEL, (
        "live harness NOVA_MODEL is stale: it pins "
        f"{NOVA_MODEL!r} but production ships {finder['model']!r} — the live "
        "FP<10% gate would validate a model the CLI no longer runs"
    )
    assert finder["region"] == NOVA_REGION


def test_live_verifier_matches_shipped_config() -> None:
    """The live GPT pin must equal the shipped verifier model + region."""
    verifier = _shipped_role("verifier")
    assert verifier["model"] == GPT_MODEL, (
        "live harness GPT_MODEL is stale: it pins "
        f"{GPT_MODEL!r} but production ships {verifier['model']!r} — the live "
        "FP<10% gate would validate a model the CLI no longer runs"
    )
    assert verifier["region"] == GPT_REGION


def test_live_pins_are_not_deprecated() -> None:
    """Neither live-pinned model may be in the deprecated registry.

    ``is_deprecated_model`` is profile-aware, so a ``global.``/``apac.`` prefixed
    id resolves to its base before the check. The deprecated ``nova-pro-v1:0``
    finder must never be what the live gate exercises.
    """
    assert not is_deprecated_model(NOVA_MODEL), (
        f"live finder pin {NOVA_MODEL!r} (base "
        f"{normalize_model_id(NOVA_MODEL)!r}) is DEPRECATED — the live gate must "
        "exercise the shipped Nova 2 Lite finder, not the EOL pro-v1 finder"
    )
    assert not is_deprecated_model(GPT_MODEL), (
        f"live verifier pin {GPT_MODEL!r} is DEPRECATED"
    )
