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
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from openrabbit.config import Config, load_config
from openrabbit.domain import (
    CompletionResult,
    FinishReason,
    ToolCall,
    Usage,
)
from openrabbit.providers.base import FakeProvider, Provider

from openrabbit.pipeline import orchestrator as orch

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
        tool_calls=[ToolCall(id="f", name="emit_findings", args={"findings": findings})],
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
                        {"id": 0, "keep": True, "confidence": confidence, "rationale": "demo"}
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
        print("error: --repo OWNER/NAME and --pr N are required online", file=sys.stderr)
        return 2

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
            emit_mod.emit_github(
                adapter,
                result.findings,
                summary_markdown=summary,
                commit_sha=str(args.commit),
                walkthrough_markdown=walkthrough,
                prior_threads=prior_threads,
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
            print("error: --text needs --scope OWNER[/NAME] (or --repo)", file=sys.stderr)
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
    parser = argparse.ArgumentParser(prog="openrabbit", description="openrabbit AI code review")
    sub = parser.add_subparsers(dest="command", required=True)

    rev = sub.add_parser("review", help="review a PR / diff")
    rev.add_argument("--offline", action="store_true", help="run offline with fixtures (no creds)")
    rev.add_argument("--diff", help="path to a unified diff file (offline; else stdin)")
    rev.add_argument("--config", help="path to .openrabbit.yaml")
    rev.add_argument("--fixtures", help="offline fixture set (e.g. 'demo')")
    rev.add_argument("--repo", help="OWNER/NAME (online)")
    rev.add_argument("--pr", help="PR number (online)")
    rev.add_argument("--commit", help="head commit SHA")
    rev.add_argument("--title", help="PR title")
    rev.add_argument("--body", help="PR body")
    rev.add_argument("--bot-login", dest="bot_login", help="bot login for dedup (online)")
    rev.add_argument("--post", action="store_true", help="actually post the review (online)")
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
    learn.add_argument("--scope", help="learning scope: OWNER (org) or OWNER/NAME (repo)")
    learn.add_argument("--repo", help="OWNER/NAME (dismissal repo / learning scope)")
    learn.add_argument("--rule-id", dest="rule_id", help="finding ruleId (dismissal)")
    learn.add_argument("--category", help="finding category")
    learn.add_argument("--file", help="finding file path / learning provenance file")
    learn.add_argument("--pr", type=int, help="PR number (learning provenance)")
    learn.add_argument("--user", help="author (learning provenance)")
    learn.set_defaults(func=_cmd_learn)

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
