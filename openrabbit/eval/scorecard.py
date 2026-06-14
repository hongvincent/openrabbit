"""Eval scorecard: precision / recall / F1 / FP-rate per category (SPEC 10).

Aggregates graded findings into a per-category and overall scorecard, computes
the action/addressed rate, and decides overall pass/fail against a configurable
false-positive budget (default < 0.10). Pure stdlib math — no network, no model.

Confusion-matrix mapping from judge verdicts (per category):
- ``match``           -> true positive (tp): reviewer correctly flagged a bug.
- ``false-positive``  -> false positive (fp): reviewer flagged a non-bug.
- ``miss``            -> false negative (fn): a known bug the reviewer did not flag.
- clean negative controls that produced NO finding -> true negatives (tn),
  supplied via ``control_count`` so the false-positive *rate* has a denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

#: Default false-positive budget; overall fails if exceeded (SPEC: < 0.10).
DEFAULT_FP_BUDGET = 0.10

_EMITTED_VERDICTS = ("match", "false-positive")  # verdicts that mean a finding fired


@dataclass(frozen=True)
class GradedFinding:
    """One judged finding feeding the scorecard.

    ``addressed`` is the online action signal (did a later commit change the
    flagged code). ``None`` for non-emitted findings (e.g. a ``miss``).
    """

    category: str
    verdict: str  # "match" | "miss" | "false-positive"
    addressed: Optional[bool] = None


@dataclass(frozen=True)
class CategoryScore:
    """Confusion-matrix-derived metrics for one category (or the overall roll-up)."""

    category: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "falsePositiveRate": self.false_positive_rate,
        }


@dataclass(frozen=True)
class Scorecard:
    """Full eval scorecard: per-category + overall + pass/fail."""

    categories: list[CategoryScore]
    overall: CategoryScore
    fp_budget: float
    addressed_rate: float
    total_findings: int

    @property
    def passed(self) -> bool:
        """Overall passes iff the overall false-positive rate is within budget."""
        return self.overall.false_positive_rate <= self.fp_budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "fpBudget": self.fp_budget,
            "addressedRate": self.addressed_rate,
            "totalFindings": self.total_findings,
            "overall": self.overall.to_dict(),
            "categories": [c.to_dict() for c in self.categories],
        }

    def format_pretty(self) -> str:
        """Render a human-readable scorecard table."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "openrabbit eval scorecard",
            "=" * 72,
            f"{'category':<16}{'P':>8}{'R':>8}{'F1':>8}{'FPrate':>9}"
            f"{'tp':>5}{'fp':>5}{'fn':>5}",
            "-" * 72,
        ]
        for c in self.categories:
            lines.append(
                f"{c.category:<16}{c.precision:>8.2f}{c.recall:>8.2f}"
                f"{c.f1:>8.2f}{c.false_positive_rate:>9.3f}"
                f"{c.tp:>5}{c.fp:>5}{c.fn:>5}"
            )
        o = self.overall
        lines.append("-" * 72)
        lines.append(
            f"{'OVERALL':<16}{o.precision:>8.2f}{o.recall:>8.2f}"
            f"{o.f1:>8.2f}{o.false_positive_rate:>9.3f}"
            f"{o.tp:>5}{o.fp:>5}{o.fn:>5}"
        )
        lines.append("=" * 72)
        lines.append(
            f"addressed-rate: {self.addressed_rate:.2%}   "
            f"FP budget: {self.fp_budget:.2%}   "
            f"result: {status} "
            f"(overall FP rate {o.false_positive_rate:.2%})"
        )
        return "\n".join(lines)


def compute_scorecard(
    graded: Iterable[GradedFinding],
    *,
    control_count: int = 0,
    fp_budget: float = DEFAULT_FP_BUDGET,
) -> Scorecard:
    """Aggregate graded findings into a :class:`Scorecard`.

    Parameters
    ----------
    graded:
        Judged findings (one per emitted finding plus one per missed known bug).
    control_count:
        Number of clean negative-control samples that produced NO finding. These
        become true negatives so the false-positive *rate* has a denominator. The
        controls are attributed to the overall roll-up.
    fp_budget:
        Maximum tolerated overall false-positive rate for an overall PASS.
    """
    graded = list(graded)

    # Per-category confusion tallies.
    cat_tp: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}

    emitted = 0
    addressed_hits = 0

    for g in graded:
        if g.verdict == "match":
            cat_tp[g.category] = cat_tp.get(g.category, 0) + 1
        elif g.verdict == "false-positive":
            cat_fp[g.category] = cat_fp.get(g.category, 0) + 1
        elif g.verdict == "miss":
            cat_fn[g.category] = cat_fn.get(g.category, 0) + 1

        if g.verdict in _EMITTED_VERDICTS:
            emitted += 1
            if g.addressed:
                addressed_hits += 1

    categories = sorted(
        set(cat_tp) | set(cat_fp) | set(cat_fn)
    )
    category_scores = [
        CategoryScore(
            category=cat,
            tp=cat_tp.get(cat, 0),
            fp=cat_fp.get(cat, 0),
            fn=cat_fn.get(cat, 0),
            tn=0,  # per-category TN not tracked; controls roll into overall only
        )
        for cat in categories
    ]

    overall = CategoryScore(
        category="overall",
        tp=sum(cat_tp.values()),
        fp=sum(cat_fp.values()),
        fn=sum(cat_fn.values()),
        tn=control_count,
    )

    addressed_rate = addressed_hits / emitted if emitted else 0.0

    return Scorecard(
        categories=category_scores,
        overall=overall,
        fp_budget=fp_budget,
        addressed_rate=addressed_rate,
        total_findings=emitted,
    )
