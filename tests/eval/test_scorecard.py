"""Tests for the eval scorecard (SPEC section 10).

Pure math against known inputs: precision / recall / F1 / false-positive-rate
per category, plus action/addressed rate and an overall pass/fail against a
configurable FP budget. No network, no provider, no git.
"""

from __future__ import annotations

import json

import pytest

from openrabbit.eval.scorecard import (
    DEFAULT_FP_BUDGET,
    CategoryScore,
    GradedFinding,
    Scorecard,
    compute_scorecard,
)


def _gf(category: str, verdict: str, *, addressed: bool | None = None) -> GradedFinding:
    return GradedFinding(category=category, verdict=verdict, addressed=addressed)


# --------------------------------------------------------------------------- #
# CategoryScore math                                                           #
# --------------------------------------------------------------------------- #
def test_category_score_basic_precision_recall_f1():
    # tp=3 fp=1 fn=1 -> precision .75 recall .75 f1 .75 fpr=fp/(fp+tn)
    cs = CategoryScore(category="correctness", tp=3, fp=1, fn=1, tn=5)
    assert cs.precision == pytest.approx(0.75)
    assert cs.recall == pytest.approx(0.75)
    assert cs.f1 == pytest.approx(0.75)
    # false-positive rate = fp / (fp + tn) = 1 / 6
    assert cs.false_positive_rate == pytest.approx(1 / 6)


def test_category_score_zero_denominators_are_safe():
    cs = CategoryScore(category="tests", tp=0, fp=0, fn=0, tn=0)
    assert cs.precision == 0.0
    assert cs.recall == 0.0
    assert cs.f1 == 0.0
    assert cs.false_positive_rate == 0.0


def test_category_score_perfect():
    cs = CategoryScore(category="security", tp=4, fp=0, fn=0, tn=2)
    assert cs.precision == 1.0
    assert cs.recall == 1.0
    assert cs.f1 == 1.0
    assert cs.false_positive_rate == 0.0


def test_category_score_f1_with_uneven_precision_recall():
    # tp=2 fp=2 fn=0 -> precision .5 recall 1.0 f1 = 2*.5*1/(1.5)=0.6667
    cs = CategoryScore(category="performance", tp=2, fp=2, fn=0, tn=0)
    assert cs.precision == pytest.approx(0.5)
    assert cs.recall == pytest.approx(1.0)
    assert cs.f1 == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# compute_scorecard aggregation                                               #
# --------------------------------------------------------------------------- #
def test_compute_scorecard_tallies_per_category():
    graded = [
        _gf("correctness", "match"),
        _gf("correctness", "match"),
        _gf("correctness", "false-positive"),
        _gf("correctness", "miss"),
        _gf("security", "match"),
        _gf("security", "false-positive"),
    ]
    # controls contribute true-negatives when no finding fires on clean code
    sc = compute_scorecard(graded, control_count=4)
    by_cat = {c.category: c for c in sc.categories}
    assert by_cat["correctness"].tp == 2
    assert by_cat["correctness"].fp == 1
    assert by_cat["correctness"].fn == 1
    assert by_cat["security"].tp == 1
    assert by_cat["security"].fp == 1


def test_compute_scorecard_overall_counts():
    graded = [
        _gf("correctness", "match"),
        _gf("security", "match"),
        _gf("correctness", "false-positive"),
        _gf("security", "miss"),
    ]
    sc = compute_scorecard(graded, control_count=0)
    assert sc.overall.tp == 2
    assert sc.overall.fp == 1
    assert sc.overall.fn == 1
    assert (
        sc.total_findings == 3
    )  # match + false-positive (miss is not an emitted finding)


def test_false_positive_rate_uses_controls_as_true_negatives():
    # 1 false positive against 9 clean controls -> fp rate 1/10 = 0.10
    graded = [_gf("correctness", "false-positive")]
    sc = compute_scorecard(graded, control_count=9)
    assert sc.overall.false_positive_rate == pytest.approx(0.10)


def test_overall_pass_when_under_budget():
    graded = [_gf("correctness", "false-positive")]
    sc = compute_scorecard(graded, control_count=99, fp_budget=0.10)
    # 1/100 = 0.01 < 0.10
    assert sc.overall.false_positive_rate == pytest.approx(0.01)
    assert sc.passed is True


def test_overall_fail_when_over_budget():
    graded = [_gf("correctness", "false-positive"), _gf("security", "false-positive")]
    sc = compute_scorecard(graded, control_count=8, fp_budget=0.10)
    # 2/10 = 0.20 > 0.10
    assert sc.overall.false_positive_rate == pytest.approx(0.20)
    assert sc.passed is False


def test_default_fp_budget_value():
    assert pytest.approx(0.10) == DEFAULT_FP_BUDGET


# --------------------------------------------------------------------------- #
# action / addressed rate                                                      #
# --------------------------------------------------------------------------- #
def test_addressed_rate_counts_only_emitted_findings():
    graded = [
        _gf("correctness", "match", addressed=True),
        _gf("correctness", "match", addressed=False),
        _gf("security", "match", addressed=True),
        _gf("security", "false-positive", addressed=False),
        _gf("tests", "miss", addressed=None),  # not emitted; excluded
    ]
    sc = compute_scorecard(graded, control_count=0)
    # emitted findings = 4 (matches + false-positive); addressed True on 2
    assert sc.addressed_rate == pytest.approx(2 / 4)


def test_addressed_rate_zero_when_no_emitted_findings():
    sc = compute_scorecard([_gf("tests", "miss")], control_count=0)
    assert sc.addressed_rate == 0.0


# --------------------------------------------------------------------------- #
# serialization + pretty print                                                #
# --------------------------------------------------------------------------- #
def test_scorecard_to_dict_roundtrips_json():
    graded = [
        _gf("correctness", "match", addressed=True),
        _gf("security", "false-positive"),
    ]
    sc = compute_scorecard(graded, control_count=4, fp_budget=0.10)
    d = sc.to_dict()
    # json-serializable
    s = json.dumps(d)
    again = json.loads(s)
    assert again["passed"] == sc.passed
    assert again["fpBudget"] == pytest.approx(0.10)
    assert "categories" in again
    assert "overall" in again
    assert again["overall"]["falsePositiveRate"] == pytest.approx(
        sc.overall.false_positive_rate
    )


def test_scorecard_pretty_print_contains_key_fields():
    graded = [_gf("correctness", "match"), _gf("security", "false-positive")]
    sc = compute_scorecard(graded, control_count=4)
    text = sc.format_pretty()
    assert "correctness" in text
    assert "security" in text
    assert "PASS" in text or "FAIL" in text
    assert "F1" in text or "f1" in text.lower()


def test_isinstance_scorecard():
    sc = compute_scorecard([], control_count=0)
    assert isinstance(sc, Scorecard)
    assert sc.passed is True  # vacuously, no false positives
