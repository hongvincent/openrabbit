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
from openrabbit.eval.golden_set import GoldenSample, build_golden_set, iter_jsonl
from openrabbit.eval.judge import (
    DEFAULT_AGREEMENT_THRESHOLD,
    CalibrationReport,
    Judge,
    calibrate_agreement,
)
from openrabbit.eval.scorecard import (
    DEFAULT_FP_BUDGET,
    GradedFinding,
    Scorecard,
    compute_scorecard,
)
from openrabbit.findings import Finding
from openrabbit.pipeline import orchestrator as orch
from openrabbit.providers.base import Provider

#: Run modes. ``"live"`` means real provider calls produced the numbers, so the
#: FP rate is a real measurement. Any non-live mode (``"fixture"``) is a wiring
#: smoke test: the numbers are scripted and MUST NOT be reported as a real FP
#: rate (contract coordinated with the cli agent).
LIVE_MODE = "live"

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
    #: ``"live"`` when real provider calls produced the numbers; any other value
    #: (``"fixture"``) marks a wiring smoke test whose FP rate is NOT a real
    #: measurement. :meth:`fp_rate` returns ``None`` when ``mode != "live"``.
    mode: str = LIVE_MODE
    #: Number of real provider ``complete`` calls made during the run. ``0`` is a
    #: red flag that nothing actually ran (e.g. an empty corpus).
    call_count: int = 0
    #: How many of the negative controls were REAL-clean (mined from the repo),
    #: as opposed to synthetic whitespace no-ops (finding 2). The FP gate must not
    #: be satisfiable purely on synthetic whitespace.
    real_clean_control_count: int = 0
    #: False-positive counts broken out by control kind (finding 2):
    #: ``{"synthetic": n, "real_clean": m}``. Surfaced so a reader can see the
    #: gate is not passing on whitespace alone.
    fp_by_kind: dict[str, int] = field(default_factory=dict)
    #: Judge/human agreement (finding 5). ``None`` when no human labels were
    #: supplied. When present and below threshold the run is marked untrusted.
    calibration: Optional[CalibrationReport] = None

    @property
    def trusted(self) -> bool:
        """False only when calibration was run and fell below its threshold.

        Absent calibration evidence the run is not contradicted, so it is treated
        as trusted; a failed calibration explicitly marks it untrusted.
        """
        if self.calibration is None:
            return True
        return self.calibration.calibrated

    def fp_rate(self) -> Optional[float]:
        """The overall false-positive rate, or ``None`` when not a live run.

        Contract: a non-live (fixture/offline) run's numbers are scripted, so the
        FP rate is NOT a real measurement and must be withheld (None / 'N/A').
        """
        if self.mode != LIVE_MODE:
            return None
        return self.scorecard.overall.false_positive_rate

    def fp_breakdown(self) -> dict[str, int]:
        """FP counts by control kind: ``{"synthetic": n, "real_clean": m}``."""
        return {
            "synthetic": self.fp_by_kind.get("synthetic", 0),
            "real_clean": self.fp_by_kind.get("real_clean", 0),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase, JSON-serializable dict.

        Includes a compact ``gradedFindings`` array (``{category, verdict}`` per
        entry) so the ``--json`` / CI consumer can inspect the per-finding
        transcript the report deliberately retains (SPEC 10: "read transcripts,
        not just aggregates"), not just the aggregate scorecard.

        ``scorecard.falsePositiveRate`` is the real number on a live run, or
        ``None`` (rendered ``null``) on a fixture run, so an offline smoke can
        never masquerade as a measured FP rate.
        """
        sc = self.scorecard.to_dict()
        # Override the scorecard's FP rate with the mode-gated value so consumers
        # cannot read a numeric FP rate off a fixture run.
        sc["falsePositiveRate"] = self.fp_rate()
        return {
            "repo": self.repo,
            "mode": self.mode,
            "callCount": self.call_count,
            "trusted": self.trusted,
            "goldenCount": self.golden_count,
            "controlCount": self.control_count,
            "realCleanControlCount": self.real_clean_control_count,
            "fpBreakdown": self.fp_breakdown(),
            "calibration": (
                self.calibration.to_dict() if self.calibration is not None else None
            ),
            "scorecard": sc,
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
    mode: str = LIVE_MODE,
    corpus_path: Optional[Union[str, Path]] = None,
    include_real_clean_controls: bool = False,
    human_verdicts: Optional[list[str]] = None,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
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
        May be a model distinct from the finder/verifier (finding 5).
    limit:
        Cap the golden set to the first ``limit`` bug samples (smoke runs / CI).
    fp_budget:
        The false-positive budget the scorecard gates on (SPEC: < 0.10).
    mode:
        ``"live"`` (default) means real provider calls produced the numbers, so
        the FP rate is a real measurement. Any other value (``"fixture"``) marks a
        wiring smoke test: :meth:`EvalReport.fp_rate` returns ``None`` (contract
        with the cli agent — offline runs must not emit a numeric FP rate).
    corpus_path:
        When given and the file exists, load a COMMITTED versioned golden JSONL
        instead of mining live (finding 3 — reproducibility). Live mining becomes
        an explicit corpus-build step (``build_golden_set`` + ``write_jsonl``).
    include_real_clean_controls:
        When True, augment the synthetic whitespace controls with REAL-clean
        controls mined from the repo (clean commits not followed by a fix/revert),
        so the FP denominator is not exclusively synthetic whitespace (finding 2).
    human_verdicts:
        Optional held-out human labels (aligned to the emitted judge verdicts, in
        order) to calibrate the judge (finding 5). When agreement falls below
        ``agreement_threshold`` the run is marked untrusted.
    agreement_threshold:
        Judge/human agreement required to consider the run calibrated-trusted.
    """
    repo_path = Path(repo_root)
    repo_name = repo_path.name or str(repo_path)

    # 1. Golden set: load a committed corpus when present (reproducible), else
    #    mine live from git history. Mining is the explicit corpus-build path.
    golden = _load_golden(repo_path, corpus_path=corpus_path, limit=limit)

    # 2a. Synthetic whitespace no-op controls over the golden files (the original
    #     negative-control kind). Any finding on these is a false positive.
    synthetic_controls = _build_controls(golden, count=_control_count(golden, limit))
    # 2b. Real-clean controls mined from the repo (finding 2): clean changes NOT
    #     followed by a fix/revert, so the FP denominator is not pure whitespace.
    real_clean_controls: list[controls_mod.ControlSample] = []
    if include_real_clean_controls:
        real_clean_controls = _build_real_clean_controls(
            repo_path, count=_control_count(golden, limit)
        )

    judge_source = judge_provider if judge_provider is not None else provider
    judge_factory = _as_factory(judge_source)
    # The in-pipeline verifier defaults to the finder (single-model offline run);
    # supply a distinct source to exercise the real cross-family stage-2 verifier.
    verifier_source = verifier_provider if verifier_provider is not None else provider

    # Count every real provider ``complete`` call across finder/verifier/judge so
    # the report can prove something actually ran (and the cli can refuse to gate
    # a run that made zero calls).
    counter = _CallCounter()
    provider_c = _wrap_source(provider, counter)
    verifier_c = _wrap_source(verifier_source, counter)
    judge_factory_c = _wrap_factory(judge_factory, counter)

    graded: list[GradedFinding] = []
    judge_verdicts: list[str] = []

    # 3a. Grade the golden (bug) samples.
    for sample in golden:
        findings = _review_sample(sample, provider_c, verifier_c, config)
        sample_graded = _grade_golden_sample(sample, findings, judge_factory_c)
        graded.extend(sample_graded)
        judge_verdicts.extend(
            g.verdict for g in sample_graded if findings and g.verdict != "miss"
        )

    # 3b. Grade the negative controls — every finding here is a false positive;
    #     controls that produce no finding become true negatives (the FP-rate
    #     denominator). FP counts are tracked per control kind (finding 2).
    control_tn = 0
    fp_by_kind: dict[str, int] = {"synthetic": 0, "real_clean": 0}
    all_controls = [(c, "synthetic") for c in synthetic_controls] + [
        (c, "real_clean") for c in real_clean_controls
    ]
    for control, kind in all_controls:
        findings = _review_sample(control, provider_c, verifier_c, config)
        if not findings:
            control_tn += 1
            continue
        control_graded = _grade_control_sample(control, findings)
        graded.extend(control_graded)
        fp_by_kind[kind] = fp_by_kind.get(kind, 0) + len(control_graded)

    scorecard = compute_scorecard(graded, control_count=control_tn, fp_budget=fp_budget)

    # 4. Calibration (finding 5): when human labels are supplied, measure judge
    #    agreement and mark the run untrusted below threshold.
    calibration: Optional[CalibrationReport] = None
    if human_verdicts is not None:
        n = min(len(judge_verdicts), len(human_verdicts))
        calibration = calibrate_agreement(
            judge_verdicts[:n], human_verdicts[:n], threshold=agreement_threshold
        )

    return EvalReport(
        scorecard=scorecard,
        golden_count=len(golden),
        control_count=len(all_controls),
        repo=repo_name,
        graded=graded,
        mode=mode,
        call_count=counter.count,
        real_clean_control_count=len(real_clean_controls),
        fp_by_kind=fp_by_kind,
        calibration=calibration,
    )


# --------------------------------------------------------------------------- #
# corpus loading (committed JSONL preferred over live mining) — finding 3       #
# --------------------------------------------------------------------------- #
def _load_golden(
    repo_path: Path,
    *,
    corpus_path: Optional[Union[str, Path]],
    limit: Optional[int],
) -> list[GoldenSample]:
    """Load a committed golden corpus when present, else mine live (finding 3).

    A committed JSONL is the reproducible default: live ``git log`` mining is
    irreproducible (history changes, labels are regex-only). When ``corpus_path``
    points at an existing file we load it (respecting ``limit``); otherwise we
    fall back to mining the repo's history.
    """
    if corpus_path is not None:
        path = Path(corpus_path)
        if path.exists():
            samples = list(iter_jsonl(path))
            return samples[:limit] if limit is not None else samples
    return build_golden_set(repo_path, max_samples=limit)


# --------------------------------------------------------------------------- #
# real-clean negative controls (finding 2)                                      #
# --------------------------------------------------------------------------- #
def _build_real_clean_controls(
    repo_path: Path, *, count: int
) -> list[controls_mod.ControlSample]:
    """Mine REAL-clean controls from the repo: clean commits, as actual diffs.

    A real-clean control is a genuine change that is NOT a bug (no fix/revert/
    hotfix label) — its real diff is reviewed as-is. Any finding on it is a false
    positive on real (non-synthetic) code, so it cannot be cleared by whitespace
    handling alone. Uses ``build_golden_set(include_clean=True)`` and keeps only
    the clean-labeled samples, wrapped as :class:`ControlSample` so they flow
    through the same judge + scorecard path.
    """
    if count <= 0:
        return []
    all_samples = build_golden_set(repo_path, include_clean=True)
    clean = [s for s in all_samples if not s.known_bug]
    controls: list[controls_mod.ControlSample] = []
    for sample in clean:
        if len(controls) >= count:
            break
        controls.append(
            controls_mod.ControlSample(
                sample_id=f"real-clean:{sample.sample_id}",
                repo=sample.repo,
                commit=sample.commit,
                diff=sample.diff,
                known_bug=False,
                bug_category=sample.bug_category,
                source="control",
                message=sample.message,
                control_kind="real-clean",
            )
        )
    return controls


# --------------------------------------------------------------------------- #
# provider call counting (contract: EvalReport.call_count) — finding/contract   #
# --------------------------------------------------------------------------- #
class _CallCounter:
    """A shared mutable tally of real provider ``complete`` calls."""

    def __init__(self) -> None:
        self.count = 0

    def tick(self) -> None:
        self.count += 1


class _CountingProvider(Provider):
    """Transparent proxy that counts each ``complete`` call on the wrapped provider."""

    def __init__(self, inner: Provider, counter: _CallCounter) -> None:
        self._inner = inner
        self._counter = counter

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        self._counter.tick()
        return self._inner.complete(*args, **kwargs)


def _wrap_source(source: ProviderSource, counter: _CallCounter) -> ProviderSource:
    """Wrap a provider source so every resolved provider counts its calls."""
    if isinstance(source, Provider):
        return _CountingProvider(source, counter)
    return lambda: _CountingProvider(source(), counter)


def _wrap_factory(
    factory: Callable[[], Provider], counter: _CallCounter
) -> Callable[[], Provider]:
    """Wrap a provider factory so each minted provider counts its calls."""
    return lambda: _CountingProvider(factory(), counter)


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
