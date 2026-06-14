"""Dogfood eval RUNNER — wires the eval harness end-to-end (SPEC section 10).

This is the glue that turns the four standalone eval modules into a single
offline-runnable pipeline:

    golden_set.build_golden_set(repo)         # bug samples from git history
        +  controls.generate_noop_controls()  # clean-PR negative controls
        -> orchestrator.review(sample.diff)   # run the review pipeline per sample
        -> judge.Judge.judge(finding, sample) # grade findings vs golden labels
        -> scorecard.compute_scorecard(...)   # precision/recall/FP + <10% budget

Everything heavy is INJECTED so :func:`run_eval` runs fully offline with
``FakeProvider``: the review providers and the judge provider are supplied by the
caller, and the only git access is read-only ``git log``/``git show`` against the
target repo (via :mod:`openrabbit.eval.golden_set`). No network, no cloud SDK
import, no live credentials.

The live FP<10% measurement on a real repo needs real Bedrock providers (that is
checklist item 20); this module is provider-agnostic, so the same code path runs
online by injecting Bedrock-backed providers instead of fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from openrabbit.config import Config
from openrabbit.eval import controls as controls_mod
from openrabbit.eval.golden_set import GoldenSample, build_golden_set
from openrabbit.eval.judge import Judge
from openrabbit.eval.scorecard import (
    DEFAULT_FP_BUDGET,
    GradedFinding,
    Scorecard,
    compute_scorecard,
)
from openrabbit.findings import Finding
from openrabbit.pipeline import orchestrator as orch
from openrabbit.providers.base import Provider

#: A review provider source: either a ready :class:`Provider` instance (reused
#: across every sample) or a zero-arg factory that mints a *fresh* provider per
#: sample. A factory is preferred for stateful test doubles (``FakeProvider``
#: exhausts its script), but a generously-scripted instance also works.
ProviderSource = Union[Provider, Callable[[], Provider]]


@dataclass
class EvalReport:
    """The result of one :func:`run_eval` run.

    Bundles the :class:`Scorecard` with the corpus sizes that produced it so a
    caller (CLI / CI gate) can report and reason about the run.
    """

    scorecard: Scorecard
    golden_count: int
    control_count: int
    repo: str = ""
    #: One :class:`GradedFinding` per graded outcome (emitted findings + misses
    #: + control false-positives), retained for transcript-level inspection
    #: (SPEC 10: "read transcripts, not just aggregates").
    graded: list[GradedFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase, JSON-serializable dict.

        Includes a compact ``gradedFindings`` array (``{category, verdict}`` per
        entry) so the ``--json`` / CI consumer can inspect the per-finding
        transcript the report deliberately retains (SPEC 10: "read transcripts,
        not just aggregates"), not just the aggregate scorecard.
        """
        return {
            "repo": self.repo,
            "goldenCount": self.golden_count,
            "controlCount": self.control_count,
            "scorecard": self.scorecard.to_dict(),
            "gradedFindings": [
                {"category": g.category, "verdict": g.verdict} for g in self.graded
            ],
        }


def run_eval(
    repo_root: Union[str, Path],
    *,
    provider: ProviderSource,
    config: Config,
    verifier_provider: Optional[ProviderSource] = None,
    judge_provider: Optional[ProviderSource] = None,
    limit: Optional[int] = None,
    fp_budget: float = DEFAULT_FP_BUDGET,
) -> EvalReport:
    """Run the dogfood eval end-to-end and return an :class:`EvalReport`.

    Parameters
    ----------
    repo_root:
        Path to a local git repository to mine for the golden set. Read-only
        ``git`` access only; raises if it is not a git repo.
    provider:
        The review **finder** provider source (instance or factory) — the broad,
        report-all pass in :func:`orchestrator.review`.
    config:
        The :class:`Config` driving routing/gating/lenses (``.openrabbit.yaml``).
    verifier_provider:
        The in-pipeline **stage-2 verifier** provider source (SPEC 6 step 5). When
        omitted it defaults to ``provider`` (finder self-verifies — the original
        single-model offline behavior). Pass a distinct cross-family model (e.g.
        GPT-5.5) to measure the FP budget under the *real* routing the design
        relies on (Nova finds → GPT-5.5 verifies → drop below the gate).
    judge_provider:
        The LLM-as-judge provider source. Defaults to ``provider`` when omitted.
    limit:
        Cap the golden set to the first ``limit`` bug samples (smoke runs / CI).
    fp_budget:
        The false-positive budget the scorecard gates on (SPEC: < 0.10).
    """
    repo_path = Path(repo_root)
    repo_name = repo_path.name or str(repo_path)

    # 1. Golden set: bug samples mined from git history (revert/hotfix/fix).
    golden = build_golden_set(repo_path, max_samples=limit)

    # 2. Negative controls: clean no-op diffs over the files seen in the golden
    #    set. Any finding a reviewer emits on these is, by construction, a false
    #    positive (the code is unchanged in substance).
    control_samples = _build_controls(golden, count=_control_count(golden, limit))

    judge_source = judge_provider if judge_provider is not None else provider
    judge_factory = _as_factory(judge_source)
    # The in-pipeline verifier defaults to the finder (single-model offline run);
    # supply a distinct source to exercise the real cross-family stage-2 verifier.
    verifier_source = verifier_provider if verifier_provider is not None else provider

    graded: list[GradedFinding] = []

    # 3a. Grade the golden (bug) samples.
    for sample in golden:
        findings = _review_sample(sample, provider, verifier_source, config)
        graded.extend(_grade_golden_sample(sample, findings, judge_factory))

    # 3b. Grade the negative controls — every finding here is a false positive;
    #     controls that produce no finding become true negatives (the FP-rate
    #     denominator), counted via ``control_tn`` below.
    control_tn = 0
    for control in control_samples:
        findings = _review_sample(control, provider, verifier_source, config)
        if not findings:
            control_tn += 1
            continue
        graded.extend(_grade_control_sample(control, findings))

    scorecard = compute_scorecard(graded, control_count=control_tn, fp_budget=fp_budget)

    return EvalReport(
        scorecard=scorecard,
        golden_count=len(golden),
        control_count=len(control_samples),
        repo=repo_name,
        graded=graded,
    )


# --------------------------------------------------------------------------- #
# review + grade helpers                                                       #
# --------------------------------------------------------------------------- #
def _review_sample(
    sample: GoldenSample,
    provider: ProviderSource,
    verifier: ProviderSource,
    config: Config,
) -> list[Finding]:
    """Run the review pipeline on one sample's diff; return kept findings.

    A fresh provider is minted per sample when a source is a factory (so a
    stateful ``FakeProvider`` is not shared/exhausted across samples). ``provider``
    drives the finder pass and ``verifier`` the stage-2 verify pass — they are the
    same object when the caller did not request a distinct cross-family verifier.
    """
    review_provider = _resolve(provider)
    verifier_provider = _resolve(verifier)
    providers: Mapping[str, Provider] = {
        "finder": review_provider,
        "verifier": verifier_provider,
    }
    pr_context = {
        "draft": False,
        "state": "open",
        "head_sha": sample.commit,
        "repo": sample.repo,
        "diff": sample.diff,
        "title": sample.message,
        "body": "",
    }
    # ``emit=False``: the eval only needs the kept findings, not a rendered
    # GitHub payload. No StateStore/learnings are wired so the run is pure.
    result = orch.review(config, pr_context, providers, emit=False)
    return result.findings


def _grade_golden_sample(
    sample: GoldenSample,
    findings: list[Finding],
    judge_factory: Callable[[], Provider],
) -> list[GradedFinding]:
    """Grade findings on a bug sample, deriving a miss when nothing fired.

    Each emitted finding is judged (match / false-positive). When the sample is
    a known bug but the reviewer emitted NO finding, that is a miss (a false
    negative) recorded against the sample's bug category — derived without a
    model call (there is no finding to judge).
    """
    out: list[GradedFinding] = []
    if findings:
        judge = Judge(judge_factory())
        for finding in findings:
            verdict = judge.judge(finding, sample)
            out.append(
                GradedFinding(category=finding.category, verdict=verdict.verdict)
            )
        return out

    # No finding emitted. A known bug that went unflagged is a miss.
    if sample.known_bug:
        out.append(GradedFinding(category=sample.bug_category, verdict="miss"))
    return out


def _grade_control_sample(
    control: GoldenSample, findings: list[Finding]
) -> list[GradedFinding]:
    """Every finding on a clean control is a false positive (no model needed)."""
    return [
        GradedFinding(category=f.category, verdict="false-positive") for f in findings
    ]


# --------------------------------------------------------------------------- #
# negative-control construction                                                #
# --------------------------------------------------------------------------- #
# A control diff must clear the pipeline's trivial-diff gate
# (``DEFAULT_MIN_CHANGED_LINES``) to genuinely exercise the reviewer's
# false-positive behavior; a 1-line whitespace control would simply be skipped
# and silently counted as a true negative. We therefore re-emit several lines of
# each file as a multi-line trailing-whitespace no-op (still semantically empty —
# ``controls.is_noop_diff`` returns True — so any finding on it is a real FP).
_CONTROL_MIN_LINES = 4


def _build_controls(
    golden: list[GoldenSample], *, count: int
) -> list[controls_mod.ControlSample]:
    """Synthesize clean no-op controls from the files seen in the golden set.

    Each control is a multi-line, semantically-empty (trailing-whitespace) diff
    over a file from the golden set, sized to clear the trivial-diff gate so the
    reviewer actually runs on it. Reuses
    :class:`~openrabbit.eval.controls.ControlSample` so controls flow through the
    exact same judge + scorecard path. Falls back to a tiny synthetic file when
    no golden file content parses, so at least one control always exists.
    """
    files = _files_from_golden(golden)
    if not files:
        files = {"control.py": "x = 1\ny = 2\nz = 3\nw = 4\n"}
    controls: list[controls_mod.ControlSample] = []
    for path in sorted(files):
        if len(controls) >= count:
            break
        controls.append(_multiline_noop_control(path, files[path]))
    return controls


def _multiline_noop_control(path: str, source: str) -> controls_mod.ControlSample:
    """Build a multi-line whitespace no-op control diff for ``path``.

    Re-emits the first few content lines with trailing whitespace appended — a
    semantically-empty change that still touches enough lines to pass the gate.
    Verified clean by :func:`controls.is_noop_diff` (whitespace-only), so any
    finding a reviewer emits on it is a false positive by construction.
    """
    lines = [ln for ln in source.splitlines() if ln.strip()][:_CONTROL_MIN_LINES]
    if len(lines) < _CONTROL_MIN_LINES:
        # Pad with synthetic-but-stable filler lines so the control clears the
        # gate even for tiny files.
        lines = lines + [
            f"pad_{i} = {i}" for i in range(_CONTROL_MIN_LINES - len(lines))
        ]
    n = len(lines)
    # A leading ``diff --git`` header is required for the router to enumerate the
    # file (it keys on ``diff --git a/<a> b/<b>``); without it the control would
    # route to zero files and never reach the finder.
    diff_lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -1,{n} +1,{n} @@",
    ]
    for ln in lines:
        diff_lines.append(f"-{ln}")
    for ln in lines:
        diff_lines.append(f"+{ln}  ")  # trailing whitespace → no-op
    diff = "\n".join(diff_lines) + "\n"
    return controls_mod.ControlSample(
        sample_id=f"control:{path}",
        repo="control",
        commit="0" * 40,
        diff=diff,
        known_bug=False,
        bug_category="correctness",
        source="control",
        message=f"no-op whitespace change to {path}",
        control_kind="whitespace",
    )


def _control_count(golden: list[GoldenSample], limit: Optional[int]) -> int:
    """How many controls to generate.

    Aim for roughly one control per golden sample (a balanced negative-control
    set), with a floor of 1 so the FP-rate denominator is never empty, and
    respecting ``limit`` when supplied.
    """
    base = max(1, len(golden))
    if limit is not None:
        base = min(base, max(1, limit))
    return base


def _files_from_golden(golden: list[GoldenSample]) -> dict[str, str]:
    """Extract ``{path: representative_content}`` from golden-sample diffs.

    Parses unified-diff ``+++ b/<path>`` headers and collects the added ('+')
    lines under each as a stand-in for the file's content (enough for a
    whitespace no-op control). Pure string parsing — no git, no network.
    """
    files: dict[str, list[str]] = {}
    current: Optional[str] = None
    for sample in golden:
        for raw in sample.diff.splitlines():
            if raw.startswith("+++ b/"):
                current = raw[len("+++ b/") :].strip()
                files.setdefault(current, [])
                continue
            if raw.startswith("--- ") or raw.startswith("+++ "):
                continue
            if (
                current is not None
                and raw.startswith("+")
                and not raw.startswith("+++")
            ):
                files[current].append(raw[1:])
    return {
        path: ("\n".join(lines) + "\n" if lines else "x = 1\n")
        for path, lines in files.items()
        if path
    }


# --------------------------------------------------------------------------- #
# provider-source plumbing                                                      #
# --------------------------------------------------------------------------- #
def _as_factory(source: ProviderSource) -> Callable[[], Provider]:
    """Normalize a provider source to a zero-arg factory.

    A bare :class:`Provider` is reused (the same instance every call); a callable
    is treated as a factory and invoked to mint a fresh provider per call.
    """
    if isinstance(source, Provider):
        return lambda: source
    return source


def _resolve(source: ProviderSource) -> Provider:
    """Resolve a provider source to a concrete provider for one use."""
    if isinstance(source, Provider):
        return source
    return source()
