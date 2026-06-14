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
    return CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v",
                name="verify_finding",
                args={"keep": True, "confidence": confidence, "rationale": "demo"},
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
def _cmd_review(args: argparse.Namespace) -> int:
    config = _load_config_or_default(args.config)

    if args.offline:
        diff = _read_diff(args.diff, sys.stdin)
        num_calls = _count_lens_calls(config, diff)
        providers = _offline_providers(config, args.fixtures, num_calls)
        pr_context: dict[str, Any] = {
            "draft": False,
            "state": "open",
            "head_sha": args.commit or "OFFLINE",
            "diff": diff,
            "title": args.title or "",
            "body": args.body or "",
        }
        result = orch.review(config, pr_context, providers)
        _print_result(result)
        return 0

    return _cmd_review_online(args, config)


def _cmd_review_online(args: argparse.Namespace, config: Config) -> int:
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


def _print_result(result: orch.ReviewResult) -> None:
    payload = {
        "reviewed": result.reviewed,
        "reason": result.reason,
        "rawFindingCount": result.raw_finding_count,
        "findings": [f.to_dict() for f in result.findings],
        "emitted": result.emitted,
    }
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
    rev.set_defaults(func=_cmd_review)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
