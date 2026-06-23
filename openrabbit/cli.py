"""``openrabbit`` command-line entrypoint.

Subcommand ``review`` has two modes:

* ``--offline`` — read a unified diff from ``--diff <file>`` or stdin and run the
  full deterministic spine with deterministic *fixture* providers (no model
  calls, no AWS/GitHub credentials). Prints the findings + the would-be GitHub
  review payload as JSON. This is the demo path that runs anywhere.

* online (default) — read the PR diff via the GitHub adapter using
  ``GITHUB_TOKEN`` and run the spine with real Bedrock providers built from
  ``.openrabbit.yaml`` ``model_roles``. Requires live credentials.

Cloud SDKs are only ever reached through the provider/adapter layer (which
imports them lazily), so importing this module needs zero external deps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from openrabbit.config import Config, load_config
from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    ToolCall,
    Usage,
)
from openrabbit.pipeline import orchestrator as orch
from openrabbit.providers.base import FakeProvider, Provider

DEFAULT_CONFIG_NAMES = (".openrabbit.yaml", ".openrabbit.yml")


# --------------------------------------------------------------------------- #
# config resolution                                                           #
# --------------------------------------------------------------------------- #
def _load_config_or_default(config_path: Optional[str]) -> Config:
    if config_path:
        return load_config(config_path)
    for name in DEFAULT_CONFIG_NAMES:
        candidate = Path.cwd() / name
        if candidate.exists():
            return load_config(candidate)
    # No config on disk: fall back to a sane default (Config has defaults).
    return load_config({"version": 1})


# --------------------------------------------------------------------------- #
# config-as-policy trust boundary (untrusted PR head)                          #
# --------------------------------------------------------------------------- #
def _base_ref_candidates(base_ref: str) -> list[str]:
    """Resolution-order candidate refs for the trusted base config.

    A CI checkout is usually detached HEAD with NO local ``<base>`` branch, so a
    BARE branch name (``git show main:...``) fails and the run would silently
    fall back to the untrusted PR-head config. The remote-tracking refs
    (``origin/<base>`` / ``refs/remotes/origin/<base>``) almost always resolve on
    a fetched CI checkout, so they are tried before giving up. An already
    qualified ref (``origin/...`` / ``refs/...``) is used as-is (no double-prefix).
    """
    if base_ref.startswith("refs/") or base_ref.startswith("origin/"):
        return [base_ref]
    return [
        base_ref,
        f"origin/{base_ref}",
        f"refs/remotes/origin/{base_ref}",
    ]


def _load_base_config(
    base_ref: str, repo_root: Optional[Path] = None
) -> Optional[Config]:
    """Load ``.openrabbit.yaml`` from the trusted BASE git ref (best-effort).

    In CI the working tree is the PR HEAD (attacker-controlled for a fork /
    external PR), so the review POLICY must come from the base ref instead. Reads
    the config blob via read-only ``git show <ref>:<name>`` — no network, no
    write.

    A BARE branch name fails on a detached-HEAD CI checkout (where no local
    ``<base>`` branch exists), so we RETRY the remote-tracking refs
    (``origin/<base>``, ``refs/remotes/origin/<base>``) before giving up — a
    detached checkout must NOT silently fall back to trusting the PR-head config.
    Any unresolved-everywhere case (no base config, not a git repo, parse error)
    returns ``None`` and the caller warns + degrades loudly.
    """
    import subprocess

    root = repo_root or Path.cwd()
    for ref in _base_ref_candidates(base_ref):
        for name in DEFAULT_CONFIG_NAMES:
            try:
                proc = subprocess.run(
                    ["git", "show", f"{ref}:{name}"],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (
                OSError,
                subprocess.SubprocessError,
            ):  # pragma: no cover - defensive
                return None
            if proc.returncode != 0 or not proc.stdout.strip():
                continue
            try:
                import yaml

                data = yaml.safe_load(proc.stdout) or {}
                return load_config(data)
            except Exception:  # pragma: no cover - defensive
                return None
    return None


def _apply_policy_trust_boundary(head: Config, base: Optional[Config]) -> Config:
    """Anchor the review POLICY to the trusted BASE config (SPEC 12 trust boundary).

    ``.openrabbit.yaml`` is config-as-policy. A PR head can self-weaken review by
    raising ``confidence_gate`` (suppressing findings), dropping ``lenses``,
    adding ``path_filters`` that exclude the changed file, or injecting
    ``path_instructions`` that tell the finder to ignore the changed file's
    issues (finding-suppression guidance baked into the cacheable finder prefix).
    Those policy fields are therefore taken from the trusted ``base`` config;
    everything else (BYO ``model_roles``, telemetry, profile, ...) stays from the
    head so a fork that lacks model wiring still runs. ``base is None`` (no base
    config found) is a no-op — the caller warns separately — so this never
    crashes the run.
    """
    if base is None:
        return head
    boundary_review = dataclasses.replace(
        head.review,
        confidence_gate=base.review.confidence_gate,
        lenses=list(base.review.lenses),
        path_filters=list(base.review.path_filters),
        path_instructions=list(base.review.path_instructions),
    )
    return dataclasses.replace(head, review=boundary_review)


def _emit_model_role_warnings(config: Config) -> None:
    """Print soft ``model_roles`` validation warnings to stderr (advisory).

    ``config.validate_model_roles`` returns human-readable, severity-prefixed
    soft problems (unknown model id, off-allow-list region for a Converse model,
    missing region). Hard errors already fail :func:`load_config`, so anything
    surfaced here is non-fatal — we warn but never change the exit code.
    """
    from openrabbit.config import validate_model_roles

    for warning in validate_model_roles(config):
        print(f"openrabbit: {warning}", file=sys.stderr)


def _warn_existing_config(repo: str) -> None:
    """Surface soft ``model_roles`` warnings for an existing config under ``repo``.

    Best-effort: if a ``.openrabbit.yaml``/``.yml`` already exists at the repo
    root, load it and print any soft warnings (so re-running ``init`` flags a
    mis-region'd model). A missing or unparsable config is silently skipped —
    ``init``'s job is to scaffold, not to validate. Hard errors are swallowed
    here so onboarding is never blocked by a pre-existing broken config.
    """
    root = Path(repo)
    for name in DEFAULT_CONFIG_NAMES:
        candidate = root / name
        if candidate.exists():
            try:
                config = load_config(candidate)
            except Exception:
                return
            _emit_model_role_warnings(config)
            return


def _read_diff(diff_arg: Optional[str], stdin: Any) -> str:
    if diff_arg:
        return Path(diff_arg).read_text(encoding="utf-8")
    data = stdin.read()
    if not data:
        raise SystemExit("no diff provided: pass --diff <file> or pipe a diff to stdin")
    return data


# --------------------------------------------------------------------------- #
# offline fixture providers (no network, no creds)                            #
# --------------------------------------------------------------------------- #
def _demo_finding_dict() -> dict[str, Any]:
    return {
        "file": "src/api/auth.py",
        "startLine": 12,
        "endLine": 14,
        "side": "RIGHT",
        "severity": "high",
        "category": "security",
        "confidence": 92,
        "title": "Possible SQL injection from concatenated user input",
        "body": "User-controlled `token` is concatenated into a SQL string.",
        "suggestion": "db.execute(query, [token])",
        "ruleId": "openrabbit/security/sqli",
    }


def _emit_findings_result(findings: list[dict[str, Any]]) -> CompletionResult:
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(id="f", name="emit_findings", args={"findings": findings})
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )


def _verify_result(confidence: float) -> CompletionResult:
    """A batched verify result keeping the single demo finding (verdict id 0)."""
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v",
                name="verify_findings",
                args={
                    "verdicts": [
                        {
                            "id": 0,
                            "keep": True,
                            "confidence": confidence,
                            "rationale": "demo",
                        }
                    ]
                },
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )


def _offline_providers(
    config: Config, fixtures: Optional[str], num_lens_calls: int
) -> dict[str, Provider]:
    """Build deterministic fixture providers for offline mode.

    ``--fixtures demo`` scripts one sample finding (so the demo shows a real
    finding flow); otherwise the finder emits nothing for every lens call (a
    clean, creds-free dry-run of routing/gating/emitting).
    """
    if fixtures == "demo":
        finder_scripts: list[CompletionResult] = []
        for i in range(max(1, num_lens_calls)):
            # Emit the demo finding only on the first lens call; empty after.
            finder_scripts.append(
                _emit_findings_result([_demo_finding_dict()] if i == 0 else [])
            )
        verifier = FakeProvider([_verify_result(0.92)], name="fixture-verifier")
        finder = FakeProvider(finder_scripts, name="fixture-finder")
    else:
        finder = FakeProvider(
            [_emit_findings_result([]) for _ in range(max(1, num_lens_calls))],
            name="fixture-finder",
        )
        verifier = FakeProvider([], name="fixture-verifier")
    return {"finder": finder, "verifier": verifier}


def _count_lens_calls(config: Config, diff: str) -> int:
    plan = orch.route_mod.route_diff(diff, lenses=list(config.review.lenses))
    return sum(len(f.lenses) for f in plan.reviewable_files)


# --------------------------------------------------------------------------- #
# review command                                                              #
# --------------------------------------------------------------------------- #
def _open_learnings_store(args: argparse.Namespace) -> Optional[Any]:
    """Build a :class:`LearningsStore` if ``--learnings-store`` was given."""
    path = getattr(args, "learnings_store", None)
    if not path:
        return None
    from openrabbit.learnings import LearningsStore

    return LearningsStore(path)


def _cmd_review(args: argparse.Namespace) -> int:
    config = _load_config_or_default(args.config)
    # Surface soft model_roles warnings (advisory; never fails the run).
    _emit_model_role_warnings(config)
    learnings_store = _open_learnings_store(args)

    if args.offline:
        diff = _read_diff(args.diff, sys.stdin)
        num_calls = _count_lens_calls(config, diff)
        providers = _offline_providers(config, args.fixtures, num_calls)
        pr_context: dict[str, Any] = {
            "draft": False,
            "state": "open",
            "head_sha": args.commit or "OFFLINE",
            "repo": args.repo or "",
            "diff": diff,
            "title": args.title or "",
            "body": args.body or "",
        }
        result = orch.review(
            config, pr_context, providers, learnings_store=learnings_store
        )
        _print_result(result)
        return 0

    return _cmd_review_online(args, config, learnings_store=learnings_store)


def _cmd_review_online(
    args: argparse.Namespace,
    config: Config,
    *,
    learnings_store: Optional[Any] = None,
) -> int:
    """Online review: GitHub adapter + real Bedrock providers (needs creds)."""
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN is required for online review", file=sys.stderr)
        return 2
    if not args.repo or not args.pr:
        print(
            "error: --repo OWNER/NAME and --pr N are required online", file=sys.stderr
        )
        return 2
    # Posting a review pins it to a commit; without --commit the old code sent the
    # literal string "None" as the commit id (GitHub 422 / a mislabeled review).
    # Require an explicit head SHA for --post so "None" can never reach the adapter.
    if args.post and not args.commit:
        print(
            "error: --post requires --commit <head SHA> (the commit the review is "
            "pinned to); pass the PR head SHA explicitly.",
            file=sys.stderr,
        )
        return 2

    # Trust boundary (SPEC 12): the working tree is the PR HEAD (attacker-controlled
    # for a fork/external PR), so the review POLICY must be anchored to the trusted
    # BASE ref. Re-anchor gate/lenses/path_filters from the base config when a base
    # ref is known (``--base-ref`` or ``GITHUB_BASE_REF``); if the base ref names a
    # config that cannot be read, warn but keep the head policy (best-effort).
    base_ref = getattr(args, "base_ref", None) or os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        base_config = _load_base_config(base_ref, Path.cwd())
        if base_config is None:
            tried = ", ".join(_base_ref_candidates(base_ref))
            print(
                "openrabbit: WARNING [policy-trust]: could not read .openrabbit.yaml "
                f"from any trusted base ref (tried: {tried}); FALLING BACK to the PR "
                "HEAD config as policy — this is UNTRUSTED for a fork/external PR and "
                "a head-side weakening of confidence_gate/lenses/path_filters/"
                "path_instructions WILL be honored. Ensure the base ref is fetched "
                "(e.g. `git fetch origin <base>`) so the trusted base policy loads.",
                file=sys.stderr,
            )
        else:
            config = _apply_policy_trust_boundary(config, base_config)

    from openrabbit.adapters.github import GitHubAdapter, GitHubRepo

    owner, _, name = args.repo.partition("/")
    repo = GitHubRepo(owner=owner, repo=name)
    adapter = GitHubAdapter(repo, int(args.pr), token, bot_login=args.bot_login)
    try:
        diff = adapter.fetch_pr_diff()
        providers = orch.build_providers(config)
        prior_threads = adapter.list_bot_review_threads()
        prior_fps = {t.fingerprint for t in prior_threads if t.fingerprint}
        pr_context = {
            "draft": False,
            "state": "open",
            "head_sha": args.commit,
            "repo": args.repo,
            "number": int(args.pr),
            "diff": diff,
            "title": args.title or "",
            "body": args.body or "",
        }
        # Inject the real enclosing-context fetcher. The CI runner has the PR
        # head checked out, so reading from the working tree (or the head ref)
        # gives the finder lenses a bounded slice of surrounding code. This is
        # best-effort and offline-safe: any failure degrades to a diff-only
        # message. Untrusted diff paths are contained to the repo root by the
        # fetcher's path/symlink checks.
        from openrabbit.pipeline.enclosing import GitEnclosingFetcher

        fetcher = GitEnclosingFetcher(
            repo_root=Path.cwd(),
            ref=args.commit if args.commit else None,
        )
        result = orch.review(
            config,
            pr_context,
            providers,
            learnings_store=learnings_store,
            prior_fingerprints=prior_fps,
            enclosing_fetcher=fetcher,
            emit=False,
        )
        if result.reviewed and args.post:
            from openrabbit.pipeline import emit as emit_mod
            from openrabbit.pipeline import route as route_mod
            from openrabbit.pipeline import walkthrough as walkthrough_mod

            # Recompute the same stats the offline orchestrator embeds so the
            # shipped GitHub walkthrough matches the offline demo (parity): the
            # reviewable-file count, the raw finder count, and the kept count.
            plan = route_mod.route_diff(diff, lenses=list(config.review.lenses))
            stats = {
                "reviewable files": len(plan.reviewable_files),
                "raw": result.raw_finding_count,
                "kept": len(result.findings),
            }
            summary = emit_mod.render_summary_markdown(result.findings, stats=stats)
            # Build the enriched sticky walkthrough (grouped changed-files table
            # + conditional Mermaid + findings table) from the routed diff. The
            # inline review body stays the minimal summary; the walkthrough
            # comment carries the richer content.
            walkthrough = walkthrough_mod.build_walkthrough(
                pr_context, plan.files, result.findings, stats=stats
            )
            # Diff-anchor guard (route.py): forward the per-file valid
            # ``{(side, line)}`` anchor sets + the PR's real changed-file set so
            # the adapter DROPS any out-of-diff (hallucinated) comment BEFORE the
            # single batched createReview. Without these, one bad (side, line)
            # 422s the whole batch and posts zero comments.
            emit_mod.emit_github(
                adapter,
                result.findings,
                summary_markdown=summary,
                commit_sha=str(args.commit),
                walkthrough_markdown=walkthrough,
                prior_threads=prior_threads,
                valid_positions=route_mod.valid_positions_by_file(plan),
                changed_files={f.path for f in plan.files},
            )
        _print_result(result)
        return 0
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# learn command — offline feedback-capture hook (SPEC 10, item 7c)            #
# --------------------------------------------------------------------------- #
def _cmd_learn(args: argparse.Namespace) -> int:
    """Capture feedback into the local learnings store (offline, no creds).

    Two sub-actions, mirroring the two memory kinds (SPEC 10):

    * ``--text`` records a team learning (provenance + category) for a scope
      (a repo ``OWNER/NAME`` or an org ``OWNER``); injected into future reviews'
      cacheable prefix.
    * ``--dismiss`` records a dismissal (negative signal) for a finding shape
      (``--rule-id`` + ``--category`` + ``--file``) under ``--repo``; future
      similar findings are down-weighted below the gate.

    Online feedback (GitHub thread resolve/dismiss state) is derivable later
    without changing this offline hook.
    """
    from openrabbit.learnings import LearningsStore

    store = LearningsStore(args.store)

    if args.dismiss:
        if not (args.repo and args.rule_id and args.category and args.file):
            print(
                "error: --dismiss needs --repo, --rule-id, --category, --file",
                file=sys.stderr,
            )
            return 2
        # Build a minimal Finding shape; only (rule_id, category, file) matter
        # for the dismissal signal. confidence/lines/title are placeholders.
        from openrabbit.findings import Finding, compute_fingerprint

        fp = compute_fingerprint(args.file, args.rule_id, args.category)
        finding = Finding(
            file=args.file,
            start_line=1,
            end_line=1,
            side="RIGHT",
            severity="low",
            category=args.category,
            confidence=0.0,
            title=f"dismissed: {args.rule_id}",
            body="",
            rule_id=args.rule_id,
            fingerprint=fp,
        )
        store.record_dismissal(args.repo, finding)
        print(json.dumps({"recorded": "dismissal", "ruleId": args.rule_id}))
        return 0

    if args.text:
        scope = args.scope or args.repo
        if not scope:
            print(
                "error: --text needs --scope OWNER[/NAME] (or --repo)", file=sys.stderr
            )
            return 2
        provenance = {
            "pr": args.pr if args.pr is not None else "",
            "file": args.file or "",
            "user": args.user or "",
        }
        learning = store.add_learning(
            scope=scope,
            text=args.text,
            provenance=provenance,
            category=args.category or "maintainability",
        )
        print(json.dumps({"recorded": "learning", "id": learning.id}))
        return 0

    print("error: pass --text <learning> or --dismiss", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- #
# eval command — dogfood eval runner (SPEC §10, item 17)                        #
# --------------------------------------------------------------------------- #
class _EvalFixtureProvider(Provider):
    """A deterministic, network-free provider for the offline ``eval`` fixtures.

    It answers by the OFFERED tool (stable regardless of call order/count):
    ``emit_findings`` -> a small scripted finding set, ``verify_findings`` ->
    keep-all verdicts, ``emit_verdict`` -> a ``match`` judge verdict. This lets
    ``openrabbit eval`` produce a real scorecard with zero credentials.
    """

    def __init__(self, *, emit: bool = True) -> None:
        self._emit = emit

    @property
    def name(self) -> str:
        return "eval-fixture"

    @property
    def model(self) -> str:
        return "eval-fixture-0"

    def complete(
        self,
        system: str,
        messages: list[Any],
        tools: Optional[list[Any]],
        max_tokens: int,
        cache_prefix: Optional[str],
        **opts: Any,
    ) -> CompletionResult:
        names = {getattr(t, "name", None) for t in (tools or [])}
        if "emit_findings" in names:
            findings = [_eval_fixture_finding()] if self._emit else []
            return _emit_findings_result(findings)
        if "verify_findings" in names:
            verdicts = [
                {"id": i, "keep": True, "confidence": 0.92, "rationale": "fixture"}
                for i in range(64)
            ]
            return CompletionResult(
                text="",
                tool_calls=[
                    ToolCall(
                        id="v", name="verify_findings", args={"verdicts": verdicts}
                    )
                ],
                finish_reason=FinishReason.TOOL_USE,
                usage=Usage(),
            )
        if "emit_verdict" in names:
            return CompletionResult(
                text="",
                tool_calls=[
                    ToolCall(
                        id="j",
                        name="emit_verdict",
                        args={
                            "verdict": "match",
                            "confidence": 0.9,
                            "rationale": "fixture",
                        },
                    )
                ],
                finish_reason=FinishReason.TOOL_USE,
                usage=Usage(),
            )
        return CompletionResult(
            text="", tool_calls=[], finish_reason=FinishReason.STOP, usage=Usage()
        )


def _eval_fixture_finding() -> dict[str, Any]:
    """The single scripted finding the offline eval fixtures emit."""
    return {
        "file": "src/app.py",
        "startLine": 1,
        "endLine": 3,
        "side": "RIGHT",
        "severity": "high",
        "category": "correctness",
        "confidence": 90,
        "title": "Potential defect flagged by the offline fixture reviewer",
        "body": "Scripted offline finding (no model call).",
        "ruleId": "openrabbit/correctness/fixture",
    }


def _have_bedrock_creds() -> bool:
    """Best-effort pre-flight for usable Bedrock credentials (no network call).

    This gates whether ``--online`` *attempts* a live call; it is NOT a guarantee
    the creds are valid. ``AWS_BEARER_TOKEN_BEDROCK`` / explicit access keys are
    strong signals; ``AWS_PROFILE`` is best-effort only — it is commonly exported
    in shells even when the underlying SSO session is expired, so a truthy
    ``AWS_PROFILE`` clears this gate but the real call may still fail downstream in
    boto3 (where the actual credential resolution happens).
    """
    import os

    return any(
        os.environ.get(k)
        for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE")
    )


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the dogfood eval harness end-to-end and print the scorecard.

    Offline by default: a small scripted-fixture provider drives the review +
    judge with no network and no credentials. ``--online`` switches to real
    Bedrock providers built from ``.openrabbit.yaml`` and is REQUIRED for a real
    false-positive measurement — it is gated on credentials (checklist item 20).
    """
    from openrabbit.eval.runner import LIVE_MODE, run_eval

    config = _load_config_or_default(args.config)
    repo = args.repo or "."

    # A fixture run's numbers are scripted (a flagging finder + an always-'match'
    # judge), so it is a wiring smoke test — NEVER a real scorecard. ``--require-pass``
    # offline would let such a run be cited as a passing CI gate, so we refuse it
    # outright. The real gate is reserved strictly for ``--online`` (item 20).
    if args.require_pass and not args.online:
        print(
            "error: --require-pass is only meaningful with --online: the offline "
            "default runs scripted fixtures (a flagging finder + always-'match' "
            "judge), so its FP rate is not a real measurement and must never gate "
            "CI. Run `openrabbit eval --online --require-pass` with Bedrock creds "
            "(item 20).",
            file=sys.stderr,
        )
        return 2

    if args.online:
        # Real FP measurement needs live Bedrock creds (item 20). We never make a
        # network call without them; fail fast with a clear, actionable message.
        if not _have_bedrock_creds():
            print(
                "error: --online requires real Bedrock credentials (item 20): set "
                "AWS_BEARER_TOKEN_BEDROCK or run `aws sso login`. The offline "
                "default (no flag) runs with scripted fixtures and no creds.",
                file=sys.stderr,
            )
            return 2
        roles = config.model_roles
        finder_role = roles.get("finder")
        verifier_role = roles.get("verifier", finder_role)
        if finder_role is None:
            print(
                "error: --online needs a 'finder' model role in .openrabbit.yaml",
                file=sys.stderr,
            )
            return 2
        # Build the finder provider + the cross-family verifier (GPT-5.5). The
        # verifier drives BOTH the in-pipeline stage-2 verify pass (so the FP
        # budget reflects the real Nova-finds → GPT-5.5-verifies routing the
        # design relies on) AND the LLM-as-judge grading. Adapters import their
        # cloud SDKs lazily here.
        review_provider = orch.model_factory(finder_role)
        verifier_provider = (
            orch.model_factory(verifier_role) if verifier_role else review_provider
        )
        # The verifier is a DISTINCT cross-family model (GPT-5.5) from the finder
        # (Nova), so passing it as ``judge_provider`` means the verifier never
        # self-grades the very findings it produced — the judge is a different
        # family from the finder. When no separate verifier role exists we fall
        # back to the finder (single-model run).
        judge_provider = verifier_provider
        # Real-clean controls (mined from the repo's clean commits) are ON by
        # default for a live run so the FP denominator is NOT synthetic
        # trailing-whitespace no-ops (which collapse FP to ~0 and always pass).
        # ``--no-real-clean-controls`` opts out (e.g. a repo with no minable
        # clean history) but then the gate measures whitespace only.
        include_real_clean = getattr(args, "real_clean_controls", True)
        report = run_eval(
            repo,
            provider=review_provider,
            verifier_provider=verifier_provider,
            judge_provider=judge_provider,
            config=config,
            limit=args.limit,
            mode=LIVE_MODE,
            corpus_path=getattr(args, "corpus", None),
            include_real_clean_controls=include_real_clean,
        )
    else:
        # Offline scripted fixtures: a flagging reviewer + match judge. ``mode``
        # stamps the report as a fixture run so the FP rate is withheld
        # (EvalReport.fp_rate() -> None) and can never be cited as a measurement.
        report = run_eval(
            repo,
            provider=lambda: _EvalFixtureProvider(emit=True),
            config=config,
            limit=args.limit,
            mode="fixture",
            corpus_path=getattr(args, "corpus", None),
        )

    if args.json:
        payload = report.to_dict()
        if not args.online:
            # Human-facing top-level marker alongside the machine-readable
            # null FP rate the report already emits (scorecard.falsePositiveRate),
            # so no consumer can read a numeric FP rate off a fixture run.
            payload["falsePositiveRate"] = "N/A — fixture"
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if not args.online:
            print(
                "openrabbit eval — FIXTURE MODE (offline scripted fixtures): the "
                "FP rate is N/A — not a measurement.",
            )
        print(report.scorecard.format_pretty())
        print(
            f"\ngolden samples: {report.golden_count}   "
            f"negative controls: {report.control_count}",
        )
        if not args.online:
            print(
                "(fixture run: a flagging finder + always-'match' judge — for a "
                "real FP<10% number run `openrabbit eval --online --require-pass` "
                "with Bedrock creds; item 20)",
            )
    # The pass/fail gate is only honored for a live (--online) run; offline
    # --require-pass already hard-errored above, so a fixture run can never pass.
    if args.require_pass:
        # Degenerate-corpus guard: ``false_positive_rate`` is ``fp/(fp+tn)`` and
        # returns 0.0 when the denominator is empty — so a run with NO negative
        # controls (or that made zero provider calls) would VACUOUSLY "pass"
        # without measuring anything. A --require-pass gate must never pass on an
        # empty/near-empty corpus; fail closed and say why.
        overall = report.scorecard.overall
        fp_denominator = overall.fp + overall.tn
        if fp_denominator == 0 or report.control_count == 0 or report.call_count == 0:
            print(
                "error: --require-pass cannot pass on a degenerate/near-empty "
                f"corpus (controls={report.control_count}, "
                f"fp+tn denominator={fp_denominator}, provider calls="
                f"{report.call_count}). The false-positive rate has no denominator, "
                "so it is not a real measurement. Point --repo / --corpus at a repo "
                "with reviewable history (and keep real-clean controls on).",
                file=sys.stderr,
            )
            return 1
        return 0 if report.scorecard.passed else 1
    return 0


# --------------------------------------------------------------------------- #
# init command — one-command onboarding (PRD §11, item 11)                     #
# --------------------------------------------------------------------------- #
def _cmd_init(args: argparse.Namespace) -> int:
    """Detect the repo stack and plan/write the onboarding artifacts.

    ``--dry-run`` (default) prints the plan + file contents and writes nothing;
    ``--write`` writes ``.openrabbit.yaml`` + the thin caller workflow to disk.
    No ``gh``/AWS/network mutation ever happens here — the printed wiring plan is
    advisory; the ``gh`` extension shell wrapper performs the real gh calls.
    """
    from openrabbit.init import scaffold

    repo = args.path or "."
    # If the repo already has a config, surface any soft model_roles warnings so
    # a re-run of init flags a mis-region'd/unknown model (advisory; never fails).
    _warn_existing_config(repo)
    dry_run = not args.write
    plan = scaffold(
        repo,
        dry_run=dry_run,
        force=args.force,
        aws_region=args.aws_region,
    )

    if args.json:
        payload = {
            "wrote": plan.wrote,
            "stack": {
                "languages": plan.stack.languages,
                "frameworks": plan.stack.frameworks,
                "testCmd": plan.stack.test_cmd,
                "externalTools": plan.stack.external_tools,
            },
            "files": [{"path": f.path, "content": f.content} for f in plan.files],
            "wiringPlan": plan.wiring_plan,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(plan.render())
    if dry_run:
        # Show the file bodies in dry-run so the user can review before writing.
        for f in plan.files:
            print(f"\n--- {f.path} ---")
            print(f.content)
    return 0


def _print_result(result: orch.ReviewResult) -> None:
    cost = result.cost_summary.to_dict()
    payload = {
        "reviewed": result.reviewed,
        "reason": result.reason,
        "rawFindingCount": result.raw_finding_count,
        "findings": [f.to_dict() for f in result.findings],
        "emitted": result.emitted,
        # Per-PR cost telemetry (SPEC 7.3): token totals + optional $ estimate.
        "cost": cost,
    }
    # Log the cost line to stderr so CI logs surface it without polluting the
    # JSON on stdout (which downstream tooling parses).
    usd = cost.get("usdEstimate")
    usd_str = f"${usd}" if usd is not None else "n/a"
    print(
        "openrabbit cost: "
        f"calls={cost['calls']} "
        f"in={cost['inputTokens']} out={cost['outputTokens']} "
        f"cacheRead={cost['cacheRead']} cacheWrite={cost['cacheWrite']} "
        f"estimate={usd_str}",
        file=sys.stderr,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# argument parsing                                                            #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openrabbit", description="openrabbit AI code review"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rev = sub.add_parser("review", help="review a PR / diff")
    rev.add_argument(
        "--offline", action="store_true", help="run offline with fixtures (no creds)"
    )
    rev.add_argument("--diff", help="path to a unified diff file (offline; else stdin)")
    rev.add_argument("--config", help="path to .openrabbit.yaml")
    rev.add_argument("--fixtures", help="offline fixture set (e.g. 'demo')")
    rev.add_argument("--repo", help="OWNER/NAME (online)")
    rev.add_argument("--pr", help="PR number (online)")
    rev.add_argument("--commit", help="head commit SHA")
    rev.add_argument("--title", help="PR title")
    rev.add_argument("--body", help="PR body")
    rev.add_argument(
        "--bot-login", dest="bot_login", help="bot login for dedup (online)"
    )
    rev.add_argument(
        "--base-ref",
        dest="base_ref",
        help=(
            "trusted base git ref for the review POLICY (online); defaults to "
            "$GITHUB_BASE_REF. The PR head .openrabbit.yaml can never weaken the "
            "gate/lenses/path_filters relative to this ref."
        ),
    )
    rev.add_argument(
        "--post", action="store_true", help="actually post the review (online)"
    )
    rev.add_argument(
        "--learnings-store",
        dest="learnings_store",
        help="path to a local learnings JSON store (enables memory + feedback loop)",
    )
    rev.set_defaults(func=_cmd_review)

    learn = sub.add_parser(
        "learn", help="record feedback into the learnings store (offline, no creds)"
    )
    learn.add_argument(
        "--store",
        required=True,
        help="path to the learnings JSON store (created if absent)",
    )
    learn.add_argument("--text", help="a team learning to record (with --scope/--repo)")
    learn.add_argument(
        "--dismiss",
        action="store_true",
        help="record a dismissal (needs --repo, --rule-id, --category, --file)",
    )
    learn.add_argument(
        "--scope", help="learning scope: OWNER (org) or OWNER/NAME (repo)"
    )
    learn.add_argument("--repo", help="OWNER/NAME (dismissal repo / learning scope)")
    learn.add_argument("--rule-id", dest="rule_id", help="finding ruleId (dismissal)")
    learn.add_argument("--category", help="finding category")
    learn.add_argument("--file", help="finding file path / learning provenance file")
    learn.add_argument("--pr", type=int, help="PR number (learning provenance)")
    learn.add_argument("--user", help="author (learning provenance)")
    learn.set_defaults(func=_cmd_learn)

    ev = sub.add_parser(
        "eval",
        help="run the dogfood eval harness on a repo and print the scorecard",
    )
    ev.add_argument(
        "--repo",
        help="path to a local git repo to mine for the golden set (default: cwd)",
    )
    ev.add_argument("--config", help="path to .openrabbit.yaml")
    ev.add_argument(
        "--limit",
        type=int,
        help="cap the golden set to the first N bug samples (smoke/CI)",
    )
    ev.add_argument(
        "--corpus",
        help=(
            "path to a COMMITTED golden JSONL corpus to load instead of live "
            "mining (reproducible). When the file exists it is used as-is."
        ),
    )
    ev.add_argument(
        "--json",
        action="store_true",
        help="emit the EvalReport (scorecard + corpus sizes) as JSON",
    )
    ev.add_argument(
        "--online",
        action="store_true",
        help=(
            "use real Bedrock providers for a real FP measurement (requires "
            "creds; item 20). Default is offline scripted fixtures."
        ),
    )
    # Real-clean negative controls are ON by default for the --online FP
    # measurement: without them the FP denominator is synthetic
    # trailing-whitespace no-ops, so FP collapses to ~0 and --require-pass always
    # passes. ``--no-real-clean-controls`` opts out (then the gate measures
    # whitespace only).
    ev.add_argument(
        "--real-clean-controls",
        dest="real_clean_controls",
        action="store_true",
        default=True,
        help=(
            "augment the synthetic whitespace controls with REAL-clean controls "
            "mined from the repo so the FP denominator is not whitespace-only "
            "(default ON for --online)"
        ),
    )
    ev.add_argument(
        "--no-real-clean-controls",
        dest="real_clean_controls",
        action="store_false",
        help="disable real-clean controls (FP gate then measures whitespace only)",
    )
    ev.add_argument(
        "--require-pass",
        dest="require_pass",
        action="store_true",
        help="exit non-zero when the FP budget is exceeded (CI gate)",
    )
    ev.set_defaults(func=_cmd_eval)

    init = sub.add_parser(
        "init",
        help="onboard a repo: detect stack + scaffold .openrabbit.yaml + caller workflow",
    )
    init.add_argument(
        "--path",
        help="repo path to onboard (default: current directory)",
    )
    init.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="print the plan + file contents without writing (default)",
    )
    init.add_argument(
        "--write",
        action="store_true",
        help="write the scaffolded files to disk (else dry-run)",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing files when writing",
    )
    init.add_argument(
        "--aws-region",
        dest="aws_region",
        default="us-east-2",
        help="primary AWS region for OIDC/STS + the verifier (default: us-east-2)",
    )
    init.add_argument(
        "--json",
        action="store_true",
        help="emit the plan as JSON (for the gh extension wrapper)",
    )
    init.set_defaults(func=_cmd_init)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
