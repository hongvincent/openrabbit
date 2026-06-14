"""The deterministic pipeline spine — ties stages 1-7 (SPEC section 6).

:func:`review` runs gate -> route -> context -> run_lenses -> verify -> dedup ->
emit using injected providers, so the whole spine is exercisable offline with
``FakeProvider`` (no network, no credentials).

:func:`model_factory` builds a concrete :class:`~openrabbit.providers.base.Provider`
for a :class:`~openrabbit.config.ModelRole` by model-id prefix:

* ``openai.*``                  -> :class:`OpenAIResponsesAdapter`
* ``amazon.*`` / ``anthropic.*`` (and ``*.anthropic.*``) -> :class:`ConverseAdapter`

Adapters import their cloud SDKs lazily, so importing this module pulls in no
boto3/httpx.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from openrabbit.config import Config, ModelRole
from openrabbit.domain import CompletionResult, Message, ToolSpec, Usage
from openrabbit.findings import Finding
from openrabbit.pricing import CostSummary
from openrabbit.providers.base import Provider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openrabbit.learnings import LearningsStore

from openrabbit.pipeline import context as ctx
from openrabbit.pipeline import dedup as dedup_mod
from openrabbit.pipeline import emit as emit_mod
from openrabbit.pipeline import gate as gate_mod
from openrabbit.pipeline import route as route_mod
from openrabbit.pipeline import run_lenses as run_lenses_mod
from openrabbit.pipeline import verify as verify_mod
from openrabbit.pipeline import walkthrough as walkthrough_mod


@dataclass
class ReviewResult:
    """The outcome of one :func:`review` run."""

    reviewed: bool
    reason: str
    findings: list[Finding] = field(default_factory=list)
    emitted: dict[str, Any] = field(default_factory=dict)
    raw_finding_count: int = 0
    #: Aggregated token usage across every model call in this review (SPEC 7.3).
    usage: Usage = field(default_factory=Usage)
    #: Per-PR cost roll-up (token totals + optional USD estimate). Always set —
    #: a gate skip yields a zeroed summary with ``calls == 0``.
    cost_summary: CostSummary = field(default_factory=CostSummary)


class _UsageRecordingProvider(Provider):
    """A transparent :class:`Provider` wrapper that sums :class:`Usage`.

    Wrapping every provider the spine touches lets the orchestrator aggregate
    token usage across all model calls (finder lenses + verifier batch +
    escalation) without threading a usage return value through every stage. The
    wrapper forwards ``complete`` verbatim and only accumulates
    ``result.usage`` + a call count, so behavior is otherwise unchanged.
    """

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.total_usage = Usage()
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolSpec]],
        max_tokens: int,
        cache_prefix: Optional[str],
        **opts: Any,
    ) -> CompletionResult:
        result = self._inner.complete(
            system, messages, tools, max_tokens, cache_prefix, **opts
        )
        self.total_usage = self.total_usage + result.usage
        self.call_count += 1
        return result


# --------------------------------------------------------------------------- #
# provider construction                                                        #
# --------------------------------------------------------------------------- #
def model_factory(role: ModelRole) -> Provider:
    """Build a :class:`Provider` for a :class:`ModelRole` by model-id prefix.

    ``openai.*`` -> OpenAIResponsesAdapter; ``amazon.*`` / ``anthropic.*`` (incl.
    ``global.anthropic.*``) -> ConverseAdapter. Adapters lazily import their
    cloud SDKs, so this is safe to call without network at import time (only the
    eventual ``complete()`` touches the network).
    """
    model = role.model
    if model.startswith("openai."):
        from openrabbit.providers.openai_responses import OpenAIResponsesAdapter

        kwargs: dict[str, Any] = {"model": model}
        if role.region is not None:
            kwargs["region"] = role.region
        return OpenAIResponsesAdapter(**kwargs)

    if model.startswith("amazon.") or "anthropic." in model:
        from openrabbit.providers.converse import ConverseAdapter

        if role.region is None:
            raise ValueError(
                f"ConverseAdapter requires a region for model {model!r}"
            )
        return ConverseAdapter(model_id=model, region=role.region)

    raise ValueError(
        f"cannot route model {model!r} to a provider adapter "
        "(expected an 'openai.', 'amazon.', or 'anthropic.' prefix)"
    )


def build_providers(config: Config) -> dict[str, Provider]:
    """Construct providers for every configured model role."""
    return {name: model_factory(role) for name, role in config.model_roles.items()}


# --------------------------------------------------------------------------- #
# lens prompt loading (SKILL.md = single source of review intelligence)        #
# --------------------------------------------------------------------------- #
#: The packaged review skills live in ``skills/`` alongside the ``openrabbit``
#: package (repo root). ``skills/lenses/<lens>/SKILL.md`` is the source of truth
#: for each lens's finder rubric (SPEC 3, 8.3, principle 8).
PACKAGED_LENSES_DIR = Path(__file__).resolve().parents[2] / "skills" / "lenses"


def load_packaged_lens_prompts(
    lenses_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Load ``{lens_name: system_prompt}`` from the packaged ``SKILL.md`` files.

    Returns ``{}`` (so the spine falls back to stubs) when the directory is
    absent or unreadable — the harness must never crash because a skill file is
    missing. ``openrabbit.lenses`` imports ``pyyaml`` lazily, so this stays
    import-cheap.
    """
    root = lenses_dir or PACKAGED_LENSES_DIR
    try:
        from openrabbit.lenses import load_lenses

        return {name: lens.system_prompt for name, lens in load_lenses(root).items()}
    except Exception:
        # Missing/malformed skills must not break review; stubs cover the gap.
        return {}


# --------------------------------------------------------------------------- #
# the spine                                                                    #
# --------------------------------------------------------------------------- #
def review(
    config: Config,
    pr_context: Mapping[str, Any],
    providers: Mapping[str, Provider],
    *,
    lens_prompts: Optional[Mapping[str, str]] = None,
    store: Optional[gate_mod.StateStore] = None,
    learnings_store: Optional["LearningsStore"] = None,
    prior_fingerprints: Optional[set[str]] = None,
    enclosing_fetcher: ctx.EnclosingFetcher = ctx.gather_enclosing_context,
    emit: bool = True,
) -> ReviewResult:
    """Run the full deterministic review spine.

    Parameters
    ----------
    config:
        Parsed :class:`Config` (``.openrabbit.yaml``).
    pr_context:
        Dict with at least ``diff``; optionally ``draft``, ``state``,
        ``head_sha``, ``repo``, ``number``, ``title``, ``body``.
    providers:
        ``{"finder": Provider, "verifier": Provider}`` — injected so the spine
        runs offline with ``FakeProvider``.
    lens_prompts:
        ``{lens_name: system_prompt}``. When omitted, the packaged
        ``skills/lenses/*/SKILL.md`` rubrics are loaded (the single source of
        review intelligence); a tiny built-in stub is the last-resort fallback
        for any configured lens whose skill file is missing.
    store:
        Optional :class:`StateStore` for incremental skip + post-review record.
    learnings_store:
        Optional :class:`~openrabbit.learnings.LearningsStore` (SPEC 10). When
        supplied: in-scope team learnings are injected into the cacheable prefix
        (a) and past dismissals down-weight similar findings via
        :func:`adjust_confidence` (b), so dismissed-style findings can fall below
        the gate. Omitting it leaves the spine's behavior unchanged.
    prior_fingerprints:
        Fingerprints from prior reviews to dedup against.
    enclosing_fetcher:
        Best-effort enclosing-context fetcher threaded into the finder pass
        (``run_lenses``). Defaults to the offline-safe no-op so unit/offline
        runs stay deterministic; the online CLI injects a real
        :class:`~openrabbit.pipeline.enclosing.GitEnclosingFetcher`.
    """
    diff = str(pr_context.get("diff", ""))

    # Stage 1 — gate.
    decision = gate_mod.evaluate_gate(config, pr_context, diff, store=store)
    if not decision.should_review:
        # No model calls -> a zeroed cost summary (calls == 0).
        return ReviewResult(
            reviewed=False, reason=decision.reason, cost_summary=CostSummary()
        )

    # Stage 2 — route.
    plan = route_mod.route_diff(diff, lenses=list(config.review.lenses))

    repo = pr_context.get("repo")
    number = pr_context.get("number")

    # Resolve in-scope team learnings (SPEC 10) once per PR so they fold into the
    # byte-stable cacheable prefix below. Best-effort: any failure degrades to no
    # learnings rather than breaking review.
    in_scope_learnings: list[str] = []
    if learnings_store is not None and repo is not None:
        try:
            in_scope_learnings = [
                ln.text
                for ln in learnings_store.get_in_scope_learnings(
                    str(repo), [f.path for f in plan.files]
                )
            ]
        except Exception:  # pragma: no cover - defensive
            in_scope_learnings = []

    # Stage 3 — context (byte-stable prefix once per PR; learnings included).
    prefix = ctx.build_prefix(config, pr_context, learnings=in_scope_learnings)
    # The per-PR prompt-cache key (SPEC 7.3 cost lever #1): one byte-stable
    # marker reused across every per-file lens call so the shared prefix caches
    # once and is read ~0.1x per file. Passed down as ``cache_prefix`` and
    # surfaced by both adapters (Converse cachePoint / Responses
    # prompt_cache_key).
    cache_key = ctx.build_cache_key(prefix, pr_context)

    # Resolve lens prompts: caller-supplied wins; otherwise load the packaged
    # SKILL.md rubrics (single source of review intelligence). A per-lens stub
    # is the last-resort fallback only when a lens file is missing.
    if lens_prompts is None:
        prompts = load_packaged_lens_prompts()
    else:
        prompts = dict(lens_prompts)
    for lens in config.review.lenses:
        prompts.setdefault(lens, _stub_lens_prompt(lens))

    raw_finder = providers.get("finder")
    if raw_finder is None:
        raise ValueError("providers must include a 'finder'")
    # Wrap finder + verifier so every model call's Usage is aggregated for the
    # per-PR cost summary (SPEC 7.3). The wrappers forward complete() verbatim.
    finder = _UsageRecordingProvider(raw_finder)
    raw_verifier = providers.get("verifier", raw_finder)
    verifier = _UsageRecordingProvider(raw_verifier)

    # Stage 4 — run lenses (report-all).
    raw_findings: list[Finding] = []
    for file_plan in plan.reviewable_files:
        raw_findings.extend(
            run_lenses_mod.run_lenses(
                finder,
                file_plan,
                prompts,
                prefix=prefix,
                enclosing_fetcher=enclosing_fetcher,
                cache_prefix=cache_key,
            )
        )

    # Stage 5 — verify (drop below gate). Skip entirely when nothing to verify.
    high_risk_files = {f.path for f in plan.files if f.risk == "high"}
    verified = verify_mod.verify_findings(
        verifier,
        raw_findings,
        gate=config.review.confidence_gate,
        min_severity=config.review.verify_min_severity,
        high_risk_files=high_risk_files,
    )

    # Feedback loop (SPEC 10, the trust differentiator): down-weight findings
    # whose (rule_id, category, file) shape matches a past human dismissal, then
    # re-apply the gate so dismissed-style findings drop out before emit. Done
    # AFTER verify so the learnings signal composes with the verifier's score
    # rather than racing it.
    if learnings_store is not None and verified:
        gate = config.review.confidence_gate
        downweighted: list[Finding] = []
        for finding in verified:
            try:
                adjusted = learnings_store.adjust_confidence(finding)
            except Exception:  # pragma: no cover - defensive
                adjusted = finding.confidence
            if adjusted < gate:
                continue  # dismissed-style finding now below the gate → drop
            if adjusted != finding.confidence:
                finding = dataclasses.replace(finding, confidence=adjusted)
            downweighted.append(finding)
        verified = downweighted

    # Stage 6 — dedup & rank. Union two dedup sources: GitHub-thread
    # fingerprints (``prior_fingerprints``, may be empty before threads load)
    # and LOCAL persisted fingerprints from prior reviews of this PR (so
    # incremental dedup works offline / before threads load).
    dedup_against: set[str] = set(prior_fingerprints or set())
    if store is not None and repo is not None and number is not None:
        dedup_against |= store.get_posted_fingerprints(str(repo), int(number))
    ranked = dedup_mod.dedup_and_rank(
        verified, prior_fingerprints=dedup_against
    )

    # Stage 7 — emit (offline payload by default).
    emitted: dict[str, Any] = {}
    if emit:
        # Label it "reviewable files" (not bare "files") so this count — which is
        # the routed reviewable-file count — never reads as contradicting the
        # walkthrough's changed-files table, which lists ALL files (incl. docs /
        # lockfiles / generated). The CLI online path mirrors this exactly.
        stats = {
            "reviewable files": len(plan.reviewable_files),
            "raw": len(raw_findings),
            "kept": len(ranked),
        }
        summary = emit_mod.render_summary_markdown(ranked, stats=stats)
        # Enriched sticky walkthrough: high-level summary + grouped changed-files
        # table + (conditional) Mermaid interaction diagram + findings table.
        walkthrough = walkthrough_mod.build_walkthrough(
            pr_context, plan.files, ranked, stats=stats
        )
        emitted = emit_mod.emit_console(
            ranked,
            summary_markdown=summary,
            commit_sha=pr_context.get("head_sha"),
            stats=stats,
            walkthrough_markdown=walkthrough,
        )

    # Record incremental state after a successful review. The SHA update and the
    # kept findings' fingerprints (a local dedup source so a re-review on
    # ``synchronize`` suppresses the same findings before GitHub threads load)
    # are folded into a single load/save.
    if store is not None and repo is not None and number is not None:
        head_sha = pr_context.get("head_sha")
        kept_fps = {f.fingerprint for f in ranked} if ranked else None
        if head_sha is not None:
            store.record_review(
                str(repo), int(number), str(head_sha), fingerprints=kept_fps
            )
        elif kept_fps:
            # No SHA to record but still persist fingerprints.
            store.record_posted_fingerprints(str(repo), int(number), kept_fps)

    # Per-PR cost telemetry (SPEC 7.3): sum Usage across every model call and
    # roll it up. The dollar estimate is attributed to the configured finder
    # model (the broad pass that dominates token volume); the *configured* model
    # id is used (not the provider instance's, which a FakeProvider stubs) so the
    # price table resolves. Unpriced models still report token totals with no $.
    total_usage = finder.total_usage + verifier.total_usage
    total_calls = finder.call_count + verifier.call_count
    cost_summary = CostSummary.from_usage(
        total_usage, model=_cost_model_id(config, raw_finder), calls=total_calls
    )

    return ReviewResult(
        reviewed=True,
        reason=decision.reason,
        findings=ranked,
        emitted=emitted,
        raw_finding_count=len(raw_findings),
        usage=total_usage,
        cost_summary=cost_summary,
    )


def _cost_model_id(config: Config, finder: Provider) -> Optional[str]:
    """Pick the model id used to price the per-PR cost summary.

    Prefer the configured ``finder`` role's model id (the broad pass that
    dominates token volume and whose price is in the table). Fall back to the
    provider instance's ``model`` only when no finder role is configured (so an
    ad-hoc/offline run still names *something*).
    """
    finder_role = config.model_roles.get("finder")
    if finder_role is not None and finder_role.model:
        return finder_role.model
    return finder.model


def _stub_lens_prompt(lens: str) -> str:
    return (
        f"Lens: {lens}. Apply the {lens} review rubric to the diff and emit all "
        "findings via emit_findings. Report every issue; do not self-filter."
    )
