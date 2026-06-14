"""Tests for negative controls (SPEC section 10).

Negative controls are clean / no-op PR samples that measure whether the reviewer
"invents problems on unchanged code". A control sample is, by construction, a
diff with no real bug (known_bug == False); ANY finding fired on it is a false
positive.
"""

from __future__ import annotations

import json

import pytest

from openrabbit.eval.controls import (
    ControlSample,
    generate_noop_controls,
    is_noop_diff,
    make_whitespace_control,
)
from openrabbit.eval.golden_set import GoldenSample


def test_control_sample_is_never_a_bug():
    c = make_whitespace_control("src/a.py", "def f():\n    return 1\n")
    assert isinstance(c, ControlSample)
    assert c.known_bug is False
    assert c.control_kind == "whitespace"


def test_whitespace_control_diff_only_changes_whitespace(tmp_path):
    src = "def f():\n    return 1\n"
    c = make_whitespace_control("src/a.py", src)
    assert c.diff
    assert is_noop_diff(c.diff)


def test_is_noop_diff_detects_real_change():
    real = (
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    assert is_noop_diff(real) is False


def test_is_noop_diff_whitespace_only_is_noop():
    noop = (
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 1  \n"
    )
    assert is_noop_diff(noop) is True


def test_is_noop_diff_comment_reflow_is_noop():
    # adding a blank line only
    noop = "@@ -1 +1,2 @@\n def f():\n+\n"
    assert is_noop_diff(noop) is True


def test_generate_noop_controls_count_and_labels():
    files = {
        "a.py": "def a():\n    return 1\n",
        "b.py": "x = 2\n",
        "c.py": "y = 3\n",
    }
    controls = generate_noop_controls(files, count=2)
    assert len(controls) == 2
    assert all(isinstance(c, ControlSample) for c in controls)
    assert all(not c.known_bug for c in controls)
    assert all(is_noop_diff(c.diff) for c in controls)


def test_generate_noop_controls_caps_at_available_files():
    controls = generate_noop_controls({"only.py": "z = 1\n"}, count=5)
    assert len(controls) == 1


def test_generate_noop_controls_empty_input():
    assert generate_noop_controls({}, count=3) == []


def test_control_sample_is_a_golden_sample_subtype():
    c = make_whitespace_control("a.py", "z = 1\n")
    # controls flow through the same scoring pipeline as golden samples
    assert isinstance(c, GoldenSample)


def test_control_sample_serializes():
    c = make_whitespace_control("a.py", "z = 1\n")
    d = c.to_dict()
    json.dumps(d)
    assert d["knownBug"] is False
    assert d["controlKind"] == "whitespace"


def test_control_sample_unique_ids():
    files = {"a.py": "z=1\n", "b.py": "y=2\n"}
    controls = generate_noop_controls(files, count=2)
    ids = {c.sample_id for c in controls}
    assert len(ids) == 2


def test_control_sample_from_dict_roundtrip():
    c = make_whitespace_control("a.py", "z = 1\n")
    again = ControlSample.from_dict(c.to_dict())
    assert again.control_kind == "whitespace"
    assert again.known_bug is False
    assert again.sample_id == c.sample_id


def test_generate_noop_controls_zero_count():
    assert generate_noop_controls({"a.py": "z=1\n"}, count=0) == []
