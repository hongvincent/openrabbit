"""Tests for the findings JSON contract (SPEC 8.1).

These tests are written first (TDD). They import the module with ZERO external
network deps and never touch AWS/GitHub.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from openrabbit.findings import (
    Finding,
    compute_fingerprint,
    load_schema,
    validate,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _valid_dict() -> dict:
    return {
        "file": "src/agent.py",
        "startLine": 42,
        "endLine": 47,
        "side": "RIGHT",
        "severity": "high",
        "category": "correctness",
        "confidence": 0.88,
        "title": "Unvalidated index can raise IndexError",
        "body": "...rationale (markdown)...",
        "suggestion": "```suggestion\nfix\n```",
        "ruleId": "openrabbit/correctness/bounds-check",
        "fingerprint": "deadbeef",
    }


def _valid_finding() -> Finding:
    return Finding(
        file="src/agent.py",
        start_line=42,
        end_line=47,
        side="RIGHT",
        severity="high",
        category="correctness",
        confidence=0.88,
        title="Unvalidated index can raise IndexError",
        body="...rationale (markdown)...",
        suggestion="```suggestion\nfix\n```",
        rule_id="openrabbit/correctness/bounds-check",
        fingerprint="deadbeef",
    )


# --------------------------------------------------------------------------- #
# dataclass shape                                                             #
# --------------------------------------------------------------------------- #
def test_finding_is_frozen_dataclass():
    assert dataclasses.is_dataclass(Finding)
    f = _valid_finding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.title = "mutated"  # type: ignore[misc]


def test_finding_fields_present():
    names = {f.name for f in dataclasses.fields(Finding)}
    assert names == {
        "file",
        "start_line",
        "end_line",
        "side",
        "severity",
        "category",
        "confidence",
        "title",
        "body",
        "suggestion",
        "rule_id",
        "fingerprint",
    }


def test_suggestion_is_optional():
    f = dataclasses.replace(_valid_finding(), suggestion=None)
    assert f.suggestion is None
    # round-trips through dict with null suggestion
    assert f.to_dict()["suggestion"] is None


# --------------------------------------------------------------------------- #
# to_dict / from_dict round trip (camelCase contract)                        #
# --------------------------------------------------------------------------- #
def test_to_dict_uses_camelcase_contract():
    d = _valid_finding().to_dict()
    assert d == _valid_dict()
    # JSON serializable
    json.dumps(d)


def test_from_dict_parses_camelcase():
    f = Finding.from_dict(_valid_dict())
    assert f == _valid_finding()


def test_round_trip_dict():
    f = _valid_finding()
    assert Finding.from_dict(f.to_dict()) == f


def test_from_dict_defaults_optional_suggestion():
    d = _valid_dict()
    del d["suggestion"]
    f = Finding.from_dict(d)
    assert f.suggestion is None


# --------------------------------------------------------------------------- #
# schema + validate                                                          #
# --------------------------------------------------------------------------- #
def test_load_schema_is_json_schema():
    schema = load_schema()
    assert schema["type"] == "object"
    assert "file" in schema["properties"]
    assert "startLine" in schema["properties"]


def test_schema_file_matches_generator():
    """The committed finding.schema.json must equal the canonical generator so a
    hand-edited JSON file can never silently diverge from _schema_document()."""
    from openrabbit.findings import _schema_document

    assert load_schema() == _schema_document()


def test_validate_accepts_valid_dict():
    assert validate(_valid_dict()) == []


def test_validate_accepts_finding_to_dict():
    assert validate(_valid_finding().to_dict()) == []


def test_validate_reports_missing_required_field():
    d = _valid_dict()
    del d["file"]
    errors = validate(d)
    assert errors
    assert any("file" in e for e in errors)


def test_validate_rejects_bad_severity():
    d = _valid_dict()
    d["severity"] = "blocker"
    errors = validate(d)
    assert errors


def test_validate_rejects_bad_category():
    d = _valid_dict()
    d["category"] = "style"
    assert validate(d)


def test_validate_rejects_bad_side():
    d = _valid_dict()
    d["side"] = "MIDDLE"
    assert validate(d)


def test_validate_rejects_confidence_out_of_range():
    d = _valid_dict()
    d["confidence"] = 1.5
    assert validate(d)
    d["confidence"] = -0.1
    assert validate(d)


def test_validate_returns_list_of_strings():
    d = _valid_dict()
    d["severity"] = "nope"
    del d["title"]
    errors = validate(d)
    assert isinstance(errors, list)
    assert all(isinstance(e, str) for e in errors)
    assert len(errors) >= 2


# --------------------------------------------------------------------------- #
# compute_fingerprint                                                         #
# --------------------------------------------------------------------------- #
def test_fingerprint_is_sha256_hex():
    fp = compute_fingerprint("src/a.py", "openrabbit/correctness/x", "ctx")
    assert isinstance(fp, str)
    assert len(fp) == 64
    int(fp, 16)  # valid hex


def test_fingerprint_is_deterministic():
    a = compute_fingerprint("src/a.py", "rule", "context")
    b = compute_fingerprint("src/a.py", "rule", "context")
    assert a == b


def test_fingerprint_changes_with_inputs():
    base = compute_fingerprint("src/a.py", "rule", "context")
    assert compute_fingerprint("src/b.py", "rule", "context") != base
    assert compute_fingerprint("src/a.py", "rule2", "context") != base
    assert compute_fingerprint("src/a.py", "rule", "context2") != base


def test_fingerprint_normalizes_context_whitespace():
    a = compute_fingerprint("src/a.py", "rule", "foo   bar")
    b = compute_fingerprint("src/a.py", "rule", "foo bar")
    c = compute_fingerprint("src/a.py", "rule", "  foo bar  \n")
    assert a == b == c
