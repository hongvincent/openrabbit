"""The bundled demo diff is not suppressed by the trivial-diff gate (finding 5).

The README quickstart points at a non-existent ``path/to/some.diff`` and a small
real ``git diff`` returns ``reviewed: false`` ("trivial diff (<3 changed
lines)"), hiding the demo finding. The fix bundles ``examples/sample.diff`` (a
real, >= ``DEFAULT_MIN_CHANGED_LINES`` diff that triggers the demo finding) so
the documented demo command actually produces a finding.

These tests lock that the bundled demo diff clears the gate AND that the
``--fixtures demo`` path produces the demo finding end-to-end. Offline-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from openrabbit.config import load_config
from openrabbit.pipeline import gate as gate_mod

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_DIFF = _REPO_ROOT / "examples" / "sample.diff"


def _pr_context(diff: str) -> dict:
    return {"draft": False, "state": "open", "diff": diff}


def test_bundled_sample_diff_exists():
    assert _SAMPLE_DIFF.is_file(), f"missing bundled demo diff at {_SAMPLE_DIFF}"


def test_bundled_sample_diff_clears_trivial_gate():
    diff = _SAMPLE_DIFF.read_text(encoding="utf-8")
    decision = gate_mod.evaluate_gate(
        load_config({"version": 1}), _pr_context(diff), diff
    )
    # The demo diff must be REVIEWED — never suppressed as a trivial diff.
    assert decision.should_review is True, decision.reason
    assert decision.changed_lines >= gate_mod.DEFAULT_MIN_CHANGED_LINES


def test_demo_fixtures_run_produces_a_finding(capsys):
    # End-to-end: the documented demo command (`--offline --fixtures demo` over the
    # bundled sample) must produce the demo finding, not a suppressed empty review.
    from openrabbit import cli

    rc = cli.main(
        ["review", "--offline", "--diff", str(_SAMPLE_DIFF), "--fixtures", "demo"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reviewed"] is True
    assert len(payload["findings"]) >= 1
    assert payload["findings"][0]["ruleId"] == "openrabbit/security/sqli"
