"""Strict structured-outputs compliance for every forced tool / json_schema.

Under OpenAI strict mode (``strict: true`` on the forced tool / json_schema —
which ``OpenAIResponsesAdapter`` sets unconditionally) the model API REJECTS a
schema unless, recursively, EVERY ``object`` node:

* sets ``additionalProperties: false``, and
* lists ALL of its declared ``properties`` keys in ``required``.

Optional fields therefore cannot be expressed by leaving them out of
``required``; they must stay required and be made nullable via a union type
``["X", "null"]``.

These tests walk the ACTUAL schemas the live finder/verifier emit
(``_EMIT_FINDINGS_SCHEMA`` and ``_VERIFY_SCHEMA``) and assert full strict
compliance. They are pure-data assertions: no model, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from openrabbit.pipeline.run_lenses import _EMIT_FINDINGS_SCHEMA
from openrabbit.pipeline.verify import _VERDICT_SCHEMA, _VERIFY_SCHEMA


def _walk_objects(node: Any, path: str = "$"):
    """Yield ``(path, object_schema)`` for every object node in a JSON Schema."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield path, node
        # Recurse into properties.
        for key, sub in (node.get("properties") or {}).items():
            yield from _walk_objects(sub, f"{path}.{key}")
        # Recurse into array items.
        items = node.get("items")
        if items is not None:
            yield from _walk_objects(items, f"{path}[]")
        # Recurse into combinators that may nest objects.
        for comb in ("anyOf", "allOf", "oneOf"):
            for i, sub in enumerate(node.get(comb) or []):
                yield from _walk_objects(sub, f"{path}.{comb}[{i}]")


def _is_nullable(prop_schema: Any) -> bool:
    """True when a property schema admits ``null`` (so it is effectively optional)."""
    if not isinstance(prop_schema, dict):
        return False
    typ = prop_schema.get("type")
    if isinstance(typ, list):
        return "null" in typ
    if typ == "null":
        return True
    for comb in ("anyOf", "oneOf"):
        for sub in prop_schema.get(comb) or []:
            if _is_nullable(sub):
                return True
    return False


def assert_strict_compliant(schema: dict[str, Any]) -> None:
    """Assert a schema satisfies OpenAI strict structured-outputs rules.

    Every object: ``additionalProperties: false`` AND every declared property
    listed in ``required``. Any property NOT in required must be nullable (which
    in strict mode is meaningless, but we still forbid silently-optional props).
    """
    for path, obj in _walk_objects(schema):
        props = obj.get("properties")
        if not props:
            # An object with no declared properties is a free-form bag; strict
            # mode still needs additionalProperties controlled, but there is no
            # required list to check.
            continue
        assert obj.get("additionalProperties") is False, (
            f"{path}: object must set additionalProperties:false under strict mode"
        )
        required = set(obj.get("required") or [])
        for prop_name in props:
            assert prop_name in required, (
                f"{path}.{prop_name}: every property must appear in 'required' "
                f"under strict mode (got required={sorted(required)})"
            )
        # required must not reference unknown props.
        assert required <= set(props), (
            f"{path}: 'required' lists keys not in 'properties': "
            f"{sorted(required - set(props))}"
        )


@pytest.mark.parametrize(
    "schema",
    [_EMIT_FINDINGS_SCHEMA, _VERIFY_SCHEMA, _VERDICT_SCHEMA],
    ids=["emit_findings", "verify_findings", "verdict"],
)
def test_schema_is_strict_compliant(schema):
    assert_strict_compliant(schema)


def test_verdict_rationale_is_nullable_and_required():
    """``rationale`` was optional; under strict mode it must be required+nullable."""
    props = _VERDICT_SCHEMA["properties"]
    assert "rationale" in _VERDICT_SCHEMA["required"]
    assert _is_nullable(props["rationale"]), (
        "rationale must be nullable (type includes 'null') since it is optional "
        "but strict mode forces it into 'required'"
    )


def test_finding_suggestion_is_nullable_and_required():
    """The finding ``suggestion`` field is optional -> must be nullable+required."""
    items = _EMIT_FINDINGS_SCHEMA["properties"]["findings"]["items"]
    assert "suggestion" in items["required"]
    assert _is_nullable(items["properties"]["suggestion"])
