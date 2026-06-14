"""Tests for the SARIF 2.1.0 output adapter (SPEC section 6 emit + section 9).

NO NETWORK, NO CLOUD DEPS. ``findings_to_sarif`` is pure data; ``write_sarif``
writes to a temp dir. These tests assert the SARIF 2.1.0 schema essentials:

- ``version`` / ``$schema`` / ``runs[0].tool.driver`` with ``rules[]``,
- ``results[]`` with ``ruleId``, ``level`` mapped from severity, ``message.text``,
- ``locations[].physicalLocation.artifactLocation.uri`` + ``region.startLine``/
  ``endLine``,
- ``partialFingerprints.primaryLocationLineHash`` from the finding fingerprint,
- a ``security-severity`` property mapped from severity per the doc band
  (Critical > 9 / High 7-8.9 / Medium 4-6.9 / Low 0.1-3.9).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from openrabbit.adapters.sarif import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    findings_to_sarif,
    level_for_severity,
    security_severity_for_severity,
    write_sarif,
)
from openrabbit.findings import Finding


def make_finding(**overrides: Any) -> Finding:
    defaults: dict[str, Any] = dict(
        file="src/agent.py",
        start_line=42,
        end_line=47,
        side="RIGHT",
        severity="high",
        category="correctness",
        confidence=0.88,
        title="Unvalidated index can raise IndexError",
        body="The index `i` is not bounds-checked before use.",
        rule_id="openrabbit/correctness/bounds-check",
        fingerprint="fp-aaa",
        suggestion=None,
    )
    defaults.update(overrides)
    return Finding(**defaults)


# --------------------------------------------------------------------------- #
# Top-level SARIF envelope                                                     #
# --------------------------------------------------------------------------- #
def test_sarif_envelope_version_and_schema():
    doc = findings_to_sarif([make_finding()], tool_version="1.2.3", repo_root=".")
    assert doc["version"] == "2.1.0"
    assert doc["version"] == SARIF_VERSION
    assert doc["$schema"] == SARIF_SCHEMA_URI
    assert isinstance(doc["runs"], list) and len(doc["runs"]) == 1


def test_sarif_driver_metadata():
    doc = findings_to_sarif([make_finding()], tool_version="1.2.3", repo_root=".")
    driver = doc["runs"][0]["tool"]["driver"]
    assert driver["name"] == "openrabbit"
    assert driver["version"] == "1.2.3"
    assert "rules" in driver and isinstance(driver["rules"], list)


# --------------------------------------------------------------------------- #
# rules[] (one rule per distinct ruleId)                                       #
# --------------------------------------------------------------------------- #
def test_rules_deduped_by_rule_id():
    f1 = make_finding(rule_id="openrabbit/correctness/bounds-check", fingerprint="a")
    f2 = make_finding(rule_id="openrabbit/correctness/bounds-check", fingerprint="b")
    f3 = make_finding(rule_id="openrabbit/security/injection", fingerprint="c",
                      severity="critical", category="security")
    doc = findings_to_sarif([f1, f2, f3], tool_version="0.1", repo_root=".")
    rules = doc["runs"][0]["tool"]["driver"]["rules"]
    ids = [r["id"] for r in rules]
    assert ids.count("openrabbit/correctness/bounds-check") == 1
    assert "openrabbit/security/injection" in ids


def test_rule_carries_security_severity_property():
    doc = findings_to_sarif(
        [make_finding(severity="critical")], tool_version="0.1", repo_root="."
    )
    rule = doc["runs"][0]["tool"]["driver"]["rules"][0]
    sec = rule["properties"]["security-severity"]
    # Critical maps to > 9.0
    assert float(sec) > 9.0


def test_rules_referenced_by_index_in_results():
    f1 = make_finding(rule_id="openrabbit/a", fingerprint="a")
    f2 = make_finding(rule_id="openrabbit/b", fingerprint="b")
    doc = findings_to_sarif([f1, f2], tool_version="0.1", repo_root=".")
    rules = doc["runs"][0]["tool"]["driver"]["rules"]
    results = doc["runs"][0]["results"]
    for res in results:
        idx = res["ruleIndex"]
        assert rules[idx]["id"] == res["ruleId"]


# --------------------------------------------------------------------------- #
# results[] shape                                                              #
# --------------------------------------------------------------------------- #
def test_result_basic_shape():
    doc = findings_to_sarif([make_finding()], tool_version="0.1", repo_root=".")
    res = doc["runs"][0]["results"][0]
    assert res["ruleId"] == "openrabbit/correctness/bounds-check"
    assert res["level"] == "error"  # high -> error
    assert res["message"]["text"]  # non-empty
    assert "Unvalidated index" in res["message"]["text"]


def test_result_location_region():
    doc = findings_to_sarif(
        [make_finding(start_line=42, end_line=47)], tool_version="0.1", repo_root="."
    )
    loc = doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "src/agent.py"
    assert loc["region"]["startLine"] == 42
    assert loc["region"]["endLine"] == 47


def test_result_partial_fingerprints():
    doc = findings_to_sarif(
        [make_finding(fingerprint="deadbeef")], tool_version="0.1", repo_root="."
    )
    res = doc["runs"][0]["results"][0]
    assert res["partialFingerprints"]["primaryLocationLineHash"] == "deadbeef"


def test_result_carries_security_severity_property():
    doc = findings_to_sarif(
        [make_finding(severity="medium")], tool_version="0.1", repo_root="."
    )
    res = doc["runs"][0]["results"][0]
    sec = float(res["properties"]["security-severity"])
    assert 4.0 <= sec <= 6.9


# --------------------------------------------------------------------------- #
# severity -> level mapping                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "severity,level",
    [
        ("critical", "error"),
        ("high", "error"),
        ("medium", "warning"),
        ("low", "note"),
        ("nit", "note"),
    ],
)
def test_level_for_severity(severity: str, level: str):
    assert level_for_severity(severity) == level


def test_level_for_unknown_severity_defaults_to_warning():
    assert level_for_severity("bogus") == "warning"


# --------------------------------------------------------------------------- #
# severity -> security-severity band mapping (per the doc)                     #
# --------------------------------------------------------------------------- #
def test_security_severity_bands():
    # Critical > 9, High 7-8.9, Medium 4-6.9, Low 0.1-3.9
    assert security_severity_for_severity("critical") > 9.0
    assert 7.0 <= security_severity_for_severity("high") <= 8.9
    assert 4.0 <= security_severity_for_severity("medium") <= 6.9
    assert 0.1 <= security_severity_for_severity("low") <= 3.9
    # nit is the lowest; within the Low band's floor
    assert 0.1 <= security_severity_for_severity("nit") <= 3.9


def test_security_severity_is_string_in_document():
    """GHAS expects security-severity as a string property."""
    doc = findings_to_sarif([make_finding()], tool_version="0.1", repo_root=".")
    rule = doc["runs"][0]["tool"]["driver"]["rules"][0]
    assert isinstance(rule["properties"]["security-severity"], str)


# --------------------------------------------------------------------------- #
# repo_root URI normalization                                                  #
# --------------------------------------------------------------------------- #
def test_uri_is_relative_to_repo_root():
    f = make_finding(file="/abs/repo/src/agent.py")
    doc = findings_to_sarif([f], tool_version="0.1", repo_root="/abs/repo")
    uri = doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"][
        "artifactLocation"
    ]["uri"]
    assert uri == "src/agent.py"


def test_uri_absolute_outside_repo_root_unchanged():
    """An absolute path that is NOT under repo_root is returned as-is."""
    f = make_finding(file="/elsewhere/src/agent.py")
    doc = findings_to_sarif([f], tool_version="0.1", repo_root="/abs/repo")
    uri = doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"][
        "artifactLocation"
    ]["uri"]
    assert uri == "/elsewhere/src/agent.py"


def test_result_message_handles_empty_body():
    """When the body is blank the message falls back to the title alone."""
    f = make_finding(body="   ")
    doc = findings_to_sarif([f], tool_version="0.1", repo_root=".")
    text = doc["runs"][0]["results"][0]["message"]["text"]
    assert text == "Unvalidated index can raise IndexError"


def test_uri_already_relative_unchanged():
    f = make_finding(file="src/agent.py")
    doc = findings_to_sarif([f], tool_version="0.1", repo_root="/abs/repo")
    uri = doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"][
        "artifactLocation"
    ]["uri"]
    assert uri == "src/agent.py"


def test_empty_findings_produces_valid_envelope():
    doc = findings_to_sarif([], tool_version="0.1", repo_root=".")
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["results"] == []
    assert doc["runs"][0]["tool"]["driver"]["rules"] == []


# --------------------------------------------------------------------------- #
# write_sarif                                                                  #
# --------------------------------------------------------------------------- #
def test_write_sarif_roundtrips(tmp_path):
    out = tmp_path / "out" / "openrabbit.sarif"
    write_sarif(
        out, [make_finding()], tool_version="9.9.9", repo_root="."
    )
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["version"] == "2.1.0"
    assert loaded["runs"][0]["tool"]["driver"]["version"] == "9.9.9"


def test_write_sarif_accepts_str_path(tmp_path):
    out = tmp_path / "openrabbit.sarif"
    write_sarif(str(out), [make_finding()], tool_version="0.1", repo_root=".")
    assert out.exists()
