"""``gh openrabbit init`` onboarding logic — pure, testable Python (PRD §11).

This module mirrors the ``claude /install-github-app`` one-command UX: it
**detects** a repo's stack from on-disk manifests and **scaffolds** the three
onboarding artifacts a repo needs to start getting openrabbit reviews:

1. ``.openrabbit.yaml`` — config-as-code with sensible defaults derived from the
   detected stack (lenses + ``model_roles`` defaulting to a Nova-Pro finder and a
   GPT-5.5 verifier per SPEC §7.2; ``external_tools`` chosen for the language).
2. ``.github/workflows/openrabbit.yml`` — a **thin caller** workflow that
   ``uses:`` the SHA-pinned central reusable workflow (SPEC §11 control plane)
   with least-privilege ``permissions``.
3. a printed **plan** of the GitHub wiring steps — the OIDC trust-policy snippet
   and the required secrets — **without performing any network mutation**.

Strict separation of concerns (SPEC §12, advisory-only): this module is pure and
side-effect-bounded. ``dry_run=True`` returns the plan + file contents and writes
nothing; ``dry_run=False`` writes the files only. It **never** calls ``gh``, the
AWS CLI, or the network — the real ``gh`` mutations (secret set, ruleset) live
exclusively in the ``cli/gh-openrabbit/gh-openrabbit`` shell wrapper, which the
unit tests never exercise.

No third-party imports at module load (``pyyaml`` is imported lazily inside the
scaffold helper) so importing this module is dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# --------------------------------------------------------------------------- #
# SHA pins — keep in sync with .github/workflows/reusable-workflow.yml (§12)    #
# A single central SHA bump rolls the fleet forward; Dependabot tracks these.  #
# --------------------------------------------------------------------------- #
#: Owner/repo + path of the central reusable workflow this caller invokes.
#: GitHub only resolves a reusable workflow at ``owner/repo/.github/workflows/
#: <file>@<ref>`` — the file MUST live under ``.github/workflows/`` in the
#: referenced repo (it cannot live in an arbitrary directory like ``actions/``).
#: The ``<OWNER>``/``<PINNED_SHA>`` tokens are DELIBERATE, human-unmistakable
#: placeholders: ``gh openrabbit init`` emits a caller a human must edit before
#: it runs (replace the owner + pin to the vetted openrabbit release SHA). The
#: trailing ``# REPLACE`` comment makes the required edit impossible to miss.
REUSABLE_WORKFLOW_REF = (
    "<OWNER>/openrabbit/.github/workflows/reusable-workflow.yml@<PINNED_SHA>"
)

#: Inline guidance appended to the caller's reusable-workflow ``uses:`` line so a
#: scaffolded workflow never silently ships an unresolvable placeholder ref.
REUSABLE_WORKFLOW_REPLACE_HINT = (
    "# REPLACE <OWNER>/<PINNED_SHA> with your org + the vetted openrabbit "
    "release SHA (bump centrally to roll the fleet)"
)

#: The AWS region the reusable workflow defaults the verifier/OIDC to (SPEC §7.2
#: places GPT-5.5 in us-east-2).
DEFAULT_AWS_REGION = "us-east-2"

#: The secret name the caller workflow passes through to the reusable workflow.
AWS_ROLE_ARN_SECRET = "AWS_ROLE_ARN"


# --------------------------------------------------------------------------- #
# stack detection                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectedStack:
    """The result of :func:`detect_stack`.

    ``languages`` and ``frameworks`` are de-duplicated, stably ordered. The
    ``test_cmd`` is the best-guess canonical test command for the primary stack
    (empty string when nothing is detected — never ``None``, never raises).
    ``external_tools`` is the deterministic-grader allow-list for the stack
    (fed into the scaffolded ``.openrabbit.yaml``).
    """

    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    test_cmd: str = ""
    external_tools: list[str] = field(default_factory=list)


def detect_stack(repo_path: PathLike) -> DetectedStack:
    """Detect languages/frameworks/test-command from a repo's manifests.

    Pure filesystem heuristics (no execution, no network): the presence of a
    manifest file implies a language. Order of precedence for ``test_cmd`` is
    python -> node -> go (the first detected wins) so a polyglot repo still gets
    a single sensible default the user can edit.

    Raises ``FileNotFoundError`` if ``repo_path`` does not exist.
    """
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(f"repo path not found: {root}")

    languages: list[str] = []
    frameworks: list[str] = []
    external_tools: list[str] = []

    has_python = (root / "pyproject.toml").is_file() or (
        root / "requirements.txt"
    ).is_file() or (root / "setup.py").is_file() or (root / "setup.cfg").is_file()
    has_node = (root / "package.json").is_file()
    has_go = (root / "go.mod").is_file()

    if has_python:
        languages.append("python")
        external_tools.extend(["ruff", "semgrep"])
    if has_node:
        languages.append("node")
        node_frameworks, node_tools = _inspect_package_json(root / "package.json")
        frameworks.extend(node_frameworks)
        external_tools.extend(node_tools)
    if has_go:
        languages.append("go")

    # gitleaks is a language-agnostic secret scanner — useful on any repo with a
    # detected stack.
    if languages:
        external_tools.append("gitleaks")

    test_cmd = _pick_test_cmd(root, has_python, has_node, has_go)

    return DetectedStack(
        languages=languages,
        frameworks=_dedupe(frameworks),
        test_cmd=test_cmd,
        external_tools=_dedupe(external_tools),
    )


def _inspect_package_json(path: Path) -> tuple[list[str], list[str]]:
    """Return ``(frameworks, external_tools)`` inferred from ``package.json``.

    Best-effort: a malformed/unreadable manifest degrades to "node, eslint" so a
    broken file never crashes onboarding.
    """
    frameworks: list[str] = []
    external_tools: list[str] = ["eslint"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frameworks, external_tools

    deps: dict[str, object] = {}
    for key in ("dependencies", "devDependencies"):
        block = data.get(key)
        if isinstance(block, dict):
            deps.update(block)

    if "typescript" in deps:
        frameworks.append("typescript")
    for fw in ("react", "vue", "next", "svelte"):
        if fw in deps:
            frameworks.append(fw)
    return frameworks, external_tools


def _pick_test_cmd(root: Path, py: bool, node: bool, go: bool) -> str:
    if py:
        return "pytest"
    if node:
        return _node_test_cmd(root / "package.json")
    if go:
        return "go test ./..."
    return ""


def _node_test_cmd(path: Path) -> str:
    """``npm test`` when package.json declares a test script, else a clear hint.

    The two branches differ: a declared ``scripts.test`` yields the runnable
    ``npm test``; a missing/unreadable manifest yields an honest placeholder hint
    the user is expected to edit (so the scaffold never claims a test command the
    repo cannot actually run).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        scripts = data.get("scripts")
        if isinstance(scripts, dict) and scripts.get("test"):
            return "npm test"
    except (OSError, ValueError):
        pass
    return "npm test  # configure a `test` script in package.json"


def _dedupe(items: list[str]) -> list[str]:
    """Stable de-duplication preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# scaffold plan                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlannedFile:
    """A single file the scaffold would write: repo-relative path + contents."""

    path: str
    content: str


@dataclass(frozen=True)
class ScaffoldPlan:
    """The full onboarding plan: files to write + the GitHub wiring narrative.

    ``files`` are repo-relative (POSIX) paths with full contents. ``wiring_plan``
    is advisory text describing the GitHub side (OIDC trust policy + required
    secrets) that a human (or the ``gh`` shell wrapper) performs — this Python
    layer never mutates anything remote.
    """

    stack: DetectedStack
    files: list[PlannedFile]
    wiring_plan: str
    wrote: bool = False

    def render(self) -> str:
        """Render a human-readable summary of the whole plan (CLI output)."""
        verb = "Wrote" if self.wrote else "Would write"
        lines = ["openrabbit init plan", "=" * 20, ""]
        langs = ", ".join(self.stack.languages) or "(none detected)"
        lines.append(f"Detected stack: {langs}")
        if self.stack.frameworks:
            lines.append(f"Frameworks: {', '.join(self.stack.frameworks)}")
        if self.stack.test_cmd:
            lines.append(f"Test command: {self.stack.test_cmd}")
        lines.append("")
        lines.append(f"{verb} files:")
        for f in self.files:
            lines.append(f"  - {f.path}")
        lines.append("")
        lines.append("Next: GitHub wiring (not performed automatically)")
        lines.append("-" * 20)
        lines.append(self.wiring_plan)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# scaffold                                                                    #
# --------------------------------------------------------------------------- #
def scaffold(
    repo_path: PathLike,
    *,
    dry_run: bool = True,
    force: bool = False,
    reusable_ref: str = REUSABLE_WORKFLOW_REF,
    aws_region: str = DEFAULT_AWS_REGION,
) -> ScaffoldPlan:
    """Plan (and optionally write) the onboarding artifacts for ``repo_path``.

    Detects the stack, then builds the ``.openrabbit.yaml`` + thin caller
    workflow + GitHub wiring plan.

    * ``dry_run=True`` (default): return the :class:`ScaffoldPlan` (files +
      wiring) and write **nothing**.
    * ``dry_run=False``: write the files to disk, then return the plan (with
      ``wrote=True``). Existing files are NOT clobbered unless ``force=True``
      (a pre-existing file raises ``FileExistsError``).

    Never calls ``gh``/the AWS CLI/the network: the wiring plan is text only.
    """
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(f"repo path not found: {root}")

    stack = detect_stack(root)

    config_yaml = _render_config_yaml(stack)
    workflow_yaml = _render_caller_workflow(reusable_ref=reusable_ref, aws_region=aws_region)

    files = [
        PlannedFile(path=".openrabbit.yaml", content=config_yaml),
        PlannedFile(
            path=".github/workflows/openrabbit.yml", content=workflow_yaml
        ),
    ]
    wiring = _render_wiring_plan(aws_region=aws_region)

    wrote = False
    if not dry_run:
        _write_files(root, files, force=force)
        wrote = True

    return ScaffoldPlan(stack=stack, files=files, wiring_plan=wiring, wrote=wrote)


def _write_files(root: Path, files: list[PlannedFile], *, force: bool) -> None:
    # Pre-flight: refuse to clobber any existing file unless forced, so an
    # accidental re-run never silently overwrites a tuned config.
    if not force:
        for f in files:
            target = root / f.path
            if target.exists():
                raise FileExistsError(
                    f"{f.path} already exists; pass force=True (or --force) to overwrite"
                )
    for f in files:
        target = root / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# renderers (deterministic strings; pyyaml only used to validate, not emit)   #
# --------------------------------------------------------------------------- #
def _render_config_yaml(stack: DetectedStack) -> str:
    """Render a stack-aware ``.openrabbit.yaml`` (SPEC §8.2 / §7.2 defaults).

    Hand-rendered (not yaml.dump) for stable comments + ordering; the result is
    validated against :func:`openrabbit.config.load_config` shape implicitly via
    tests (it round-trips). ``model_roles`` default to Nova-Pro finder +
    GPT-5.5 verifier per SPEC §7.2.
    """
    tools = stack.external_tools or ["gitleaks"]
    tools_list = "[" + ", ".join(tools) + "]"
    return f"""\
# .openrabbit.yaml — generated by `gh openrabbit init` (SPEC §8.2).
# Config-as-code: additive / versioned. Edit freely; model ids are BYO Bedrock
# model/inference-profile ids. Defaults follow SPEC §7.2 (Nova finder, GPT-5.5
# verifier). See docs for the full reference.
version: 1

review:
  profile: balanced            # chill | balanced | assertive
  confidence_gate: 0.80        # drop findings below this calibrated confidence
  verify_min_severity: high    # route only >= this severity through the verifier
  incremental: true            # review diff since last_reviewed_sha when available
  path_filters:
    - "!**/dist/**"
    - "!**/*.lock"
    - "!**/generated/**"
  lenses: [correctness, security, performance, tests, maintainability]

model_roles:                   # role -> {{ model, region, ...provider opts }}
  triage:
    model: amazon.nova-lite-v1:0
    region: ap-northeast-2
  finder:
    model: amazon.nova-pro-v1:0
    region: ap-northeast-2
  verifier:
    model: openai.gpt-5.5
    region: us-east-2
    reasoning_effort: medium
    store: false

external_tools:                # deterministic graders fed into context
  enabled: {tools_list}

telemetry:
  enabled: true
  mode: opt-out                # opt-in | opt-out
"""


def _render_caller_workflow(*, reusable_ref: str, aws_region: str) -> str:
    """Render the thin per-repo caller workflow (SPEC §11 control plane).

    All review logic lives in the SHA-pinned central reusable workflow; this
    caller only wires the trigger + passes the AWS role ARN secret. Top-level
    ``permissions`` are least-privilege ``contents: read`` (the reusable workflow
    elevates ``pull-requests``/``id-token`` on its own review job).

    The reusable-workflow ``uses:`` carries human-unmistakable placeholders
    (``<OWNER>``/``<PINNED_SHA>``) plus a trailing ``# REPLACE`` comment so the
    scaffold never silently ships an unresolvable (e.g. all-zeros) pin.
    """
    replace_hint = REUSABLE_WORKFLOW_REPLACE_HINT
    return f"""\
# openrabbit review — thin caller workflow (generated by `gh openrabbit init`).
#
# All review logic lives in the SHA-pinned central reusable workflow; a single
# central SHA bump rolls this repo forward (track with Dependabot). The PR diff
# is untrusted data; the reasoning layer holds no write credentials (advisory).
name: openrabbit review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]

# Least privilege at the top level; the reusable workflow's review job elevates
# only pull-requests:write + id-token:write where it actually posts the review.
permissions:
  contents: read

jobs:
  openrabbit:
    # SHA-pinned reusable workflow (SPEC §12): bump centrally to roll forward.
    # The reference below carries DELIBERATE <OWNER>/<PINNED_SHA> placeholders —
    # edit them before this workflow can run (see the trailing REPLACE comment).
    uses: {reusable_ref}  {replace_hint}
    with:
      pr: ${{{{ github.event.pull_request.number }}}}
      commit: ${{{{ github.event.pull_request.head.sha }}}}
      config: ".openrabbit.yaml"
      aws_region: "{aws_region}"
    secrets:
      aws_role_arn: ${{{{ secrets.{AWS_ROLE_ARN_SECRET} }}}}
"""


def _render_wiring_plan(*, aws_region: str) -> str:
    """Render the advisory GitHub-wiring narrative (no mutation performed).

    Explains the keyless OIDC -> STS trust policy and the one required secret.
    The ``gh`` commands are shown so a human (or the shell wrapper) can run them;
    this Python layer never executes them.
    """
    trust_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Federated": (
                            "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/"
                            "token.actions.githubusercontent.com"
                        )
                    },
                    "Action": "sts:AssumeRoleWithWebIdentity",
                    "Condition": {
                        "StringEquals": {
                            "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                        },
                        "StringLike": {
                            "token.actions.githubusercontent.com:sub": "repo:<OWNER>/<REPO>:*"
                        },
                    },
                }
            ],
        },
        indent=2,
    )
    return f"""\
GitHub wiring (keyless OIDC -> AWS STS; no long-lived secrets):

1. Create (or reuse) an IAM role for Bedrock with this OIDC trust policy
   (replace <ACCOUNT_ID>/<OWNER>/<REPO>):

{trust_policy}

   Scope its permissions to the exact Bedrock model/inference-profile ARNs the
   .openrabbit.yaml model_roles use (least privilege).

2. Store the role ARN as the repo secret consumed by the caller workflow:

   gh secret set {AWS_ROLE_ARN_SECRET} --body "arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>"

3. (Optional, recommended) require the review check before merge via an org
   ruleset pinned to the reusable workflow's SHA (Evaluate mode first):

   gh api -X POST /repos/<OWNER>/<REPO>/rulesets --input ruleset.json

OIDC region for STS / the verifier: {aws_region}

openrabbit never runs these commands for you — review and run them yourself, or
use `gh openrabbit init --write` (the gh extension wrapper performs the guarded
gh mutations in a non-dry-run path).
"""


# --------------------------------------------------------------------------- #
# module entrypoint — `python -m openrabbit.init` (called by the gh wrapper)   #
# --------------------------------------------------------------------------- #
def main(argv: object = None) -> int:
    """Delegate to the ``openrabbit init`` CLI subcommand.

    The ``gh openrabbit`` shell extension invokes ``python -m openrabbit.init``;
    we forward to the shared CLI so there is one argument surface. Pure delegation
    (no network); the real ``gh`` mutations live in the shell wrapper.
    """
    from openrabbit.cli import build_parser

    args = ["init"]
    if argv is not None:
        args.extend(argv)  # type: ignore[arg-type]
    else:  # pragma: no cover - exercised only via the module entrypoint
        import sys

        args.extend(sys.argv[1:])
    parser = build_parser()
    ns = parser.parse_args(args)
    return int(ns.func(ns))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
