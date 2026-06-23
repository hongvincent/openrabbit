"""Tests for ``gh openrabbit init`` onboarding logic (PRD §11, checklist item 11).

The init module is **pure, testable Python**: it detects a repo's stack from
on-disk manifests, then PLANS (dry-run) or WRITES the onboarding artifacts —
``.openrabbit.yaml``, a SHA-pinned thin caller workflow, and a printed plan of
the GitHub wiring (OIDC trust policy + required secrets). It performs **no**
network mutation: the real ``gh`` calls live only in the shell extension
wrapper, which these unit tests never exercise.

All tests are offline: temp-dir fixtures (no network, no live gh/AWS creds).
The scaffolded ``.openrabbit.yaml`` must round-trip through
:func:`openrabbit.config.load_config`, and the caller workflow must be
SHA-pinned (40-hex) — the same supply-chain invariant the reusable workflow
holds (SPEC §12).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from openrabbit.config import Config, load_config
from openrabbit.init import (
    DetectedStack,
    ScaffoldPlan,
    detect_stack,
    scaffold,
)

SHA_RE = re.compile(r"@[0-9a-f]{40}\b")


# --------------------------------------------------------------------------- #
# fixtures: temp repos for each stack                                          #
# --------------------------------------------------------------------------- #
def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path / "src" / "app.py", "x = 1\n")
    return tmp_path


@pytest.fixture
def python_requirements_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "requirements.txt", "pytest\n")
    return tmp_path


@pytest.fixture
def node_repo(tmp_path: Path) -> Path:
    _write(
        tmp_path / "package.json",
        '{"name":"x","scripts":{"test":"jest"},"devDependencies":{"typescript":"5"}}',
    )
    return tmp_path


@pytest.fixture
def go_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "go.mod", "module example.com/x\n\ngo 1.22\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# detect_stack                                                                 #
# --------------------------------------------------------------------------- #
def test_detect_python_from_pyproject(python_repo: Path) -> None:
    stack = detect_stack(python_repo)
    assert isinstance(stack, DetectedStack)
    assert "python" in stack.languages
    # pyproject-based python projects use pytest as the canonical test command.
    assert "pytest" in stack.test_cmd


def test_detect_python_from_requirements(python_requirements_repo: Path) -> None:
    stack = detect_stack(python_requirements_repo)
    assert "python" in stack.languages


def test_detect_node_and_typescript(node_repo: Path) -> None:
    stack = detect_stack(node_repo)
    assert "node" in stack.languages
    # TypeScript devDependency promotes the typescript framework signal.
    assert "typescript" in stack.frameworks
    # package.json declares a `test` script -> npm test.
    assert "npm" in stack.test_cmd or "test" in stack.test_cmd


def test_detect_go(go_repo: Path) -> None:
    stack = detect_stack(go_repo)
    assert "go" in stack.languages
    assert "go test" in stack.test_cmd


def test_detect_empty_repo_is_unknown(tmp_path: Path) -> None:
    stack = detect_stack(tmp_path)
    assert stack.languages == []
    # No detected stack still yields a usable (empty) test command, never raises.
    assert isinstance(stack.test_cmd, str)


def test_detect_multi_language(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path / "go.mod", "module x\n")
    stack = detect_stack(tmp_path)
    assert "python" in stack.languages
    assert "go" in stack.languages


def test_detect_stack_accepts_str_path(python_repo: Path) -> None:
    stack = detect_stack(str(python_repo))
    assert "python" in stack.languages


def test_detect_react_framework(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        '{"name":"x","dependencies":{"react":"18"}}',
    )
    stack = detect_stack(tmp_path)
    assert "react" in stack.frameworks


def test_detect_malformed_package_json_degrades_gracefully(tmp_path: Path) -> None:
    _write(tmp_path / "package.json", "{ this is not json")
    stack = detect_stack(tmp_path)
    # node is still detected (manifest present); eslint default still applied.
    assert "node" in stack.languages
    assert "eslint" in stack.external_tools
    # test_cmd falls back to the npm hint (now carrying a "configure a test
    # script" note for the no-script case), never raises.
    assert stack.test_cmd.startswith("npm test")


def test_node_plan_render_lists_frameworks(node_repo: Path) -> None:
    plan = scaffold(node_repo, dry_run=True)
    rendered = plan.render()
    assert "Frameworks: typescript" in rendered
    assert "Test command: npm test" in rendered


def test_setup_py_detected_as_python(tmp_path: Path) -> None:
    _write(tmp_path / "setup.py", "from setuptools import setup\nsetup()\n")
    stack = detect_stack(tmp_path)
    assert "python" in stack.languages


def test_detect_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        detect_stack(tmp_path / "does-not-exist")


# --------------------------------------------------------------------------- #
# scaffold — dry run returns a plan and writes nothing                         #
# --------------------------------------------------------------------------- #
def test_scaffold_dry_run_returns_plan(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    assert isinstance(plan, ScaffoldPlan)
    # The plan enumerates the files it WOULD write, with their contents.
    paths = {f.path for f in plan.files}
    assert ".openrabbit.yaml" in paths
    assert ".github/workflows/openrabbit.yml" in paths


def test_scaffold_dry_run_writes_nothing(python_repo: Path) -> None:
    scaffold(python_repo, dry_run=True)
    assert not (python_repo / ".openrabbit.yaml").exists()
    assert not (python_repo / ".github" / "workflows" / "openrabbit.yml").exists()


def test_scaffold_plan_has_wiring_steps(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    text = plan.wiring_plan
    # The printed plan must explain the GitHub wiring WITHOUT performing it:
    # the OIDC trust policy snippet + the required secret(s).
    assert "OIDC" in text or "oidc" in text.lower()
    assert "AssumeRoleWithWebIdentity" in text or "sts" in text.lower()
    # The AWS role ARN secret the reusable workflow consumes must be named.
    assert "AWS_ROLE_ARN" in text.upper() or "aws_role_arn" in text


def test_scaffold_wiring_plan_does_not_mutate(python_repo: Path) -> None:
    """The wiring plan is advisory text only — no gh/aws is invoked from code."""
    plan = scaffold(python_repo, dry_run=True)
    # It mentions the gh commands a human (or the shell wrapper) would run, but
    # the Python layer never executes them.
    assert "gh " in plan.wiring_plan


# --------------------------------------------------------------------------- #
# scaffold — generated .openrabbit.yaml is valid and stack-aware               #
# --------------------------------------------------------------------------- #
def test_scaffolded_config_round_trips(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    cfg_file = next(f for f in plan.files if f.path == ".openrabbit.yaml")
    parsed = yaml.safe_load(cfg_file.content)
    cfg = load_config(parsed)
    assert isinstance(cfg, Config)
    # Sensible model_roles defaults per the LIVE-VERIFIED model switch (SPEC §7.2):
    # a Nova 2 Lite finder (Seoul) + a GPT-5.4 verifier (us-east-2). These are the
    # CORRECTED expectations — the registry, pricing, and .openrabbit.example.yaml
    # all reflect Nova 2 Lite + GPT-5.4 now, so the init scaffold must match them
    # (the old nova-pro / gpt-5.5 defaults would scaffold an outdated config).
    assert "finder" in cfg.model_roles
    assert "verifier" in cfg.model_roles
    assert cfg.model_roles["triage"].model == "global.amazon.nova-2-lite-v1:0"
    assert cfg.model_roles["finder"].model == "global.amazon.nova-2-lite-v1:0"
    assert cfg.model_roles["verifier"].model == "openai.gpt-5.4"
    assert cfg.model_roles["verifier"].region == "us-east-2"
    # The new finder must NOT default to the superseded nova-pro id, and the
    # verifier must NOT default to the superseded gpt-5.5 id.
    assert "nova-pro" not in cfg.model_roles["finder"].model.lower()
    assert "gpt-5.5" not in cfg.model_roles["verifier"].model.lower()
    # The Nova 2 Lite finder has NO reasoning_effort yet (extended-thinking API
    # shape is TBD), and the verifier keeps reasoning_effort: medium + store: false.
    assert "reasoning_effort" not in cfg.model_roles["finder"].options
    assert cfg.model_roles["verifier"].options.get("reasoning_effort") == "medium"
    assert cfg.model_roles["verifier"].options.get("store") is False
    # The raw scaffold text carries the Nova 2 extended-thinking TODO marker so a
    # reader knows the finder reasoning shape is deliberately deferred, not missed.
    assert "Nova 2" in cfg_file.content and "TODO" in cfg_file.content
    # All five lenses on by default.
    assert "correctness" in cfg.review.lenses
    assert "security" in cfg.review.lenses


def test_scaffolded_config_default_gate(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    cfg_file = next(f for f in plan.files if f.path == ".openrabbit.yaml")
    cfg = load_config(yaml.safe_load(cfg_file.content))
    assert cfg.review.confidence_gate == pytest.approx(0.80)


def test_scaffolded_config_external_tools_reserved_not_advertised(
    python_repo: Path,
) -> None:
    """external_tools is RESERVED / not yet wired (finding #3): the scaffolded
    config must NOT advertise a dead feature. The pipeline never runs these graders
    or injects their output, so the active `enabled` list must be EMPTY — not a
    populated list that falsely implies ruff/semgrep run.

    (Corrected from the old behavior, which scaffolded ``enabled: [ruff, semgrep,
    ...]`` and thereby advertised a feature the harness does not execute.)
    """
    plan = scaffold(python_repo, dry_run=True)
    cfg_file = next(f for f in plan.files if f.path == ".openrabbit.yaml")
    cfg = load_config(yaml.safe_load(cfg_file.content))
    # The honest contract: no tool is advertised as active.
    assert cfg.external_tools.enabled == [], (
        "scaffold must not advertise external_tools as active while the runtime "
        f"plumbing is unwired; got {cfg.external_tools.enabled!r}"
    )
    # The block is clearly marked reserved so a reader is not misled.
    assert "reserved" in cfg_file.content.lower()
    assert "not yet wired" in cfg_file.content.lower()


def test_scaffold_still_detects_external_tools_for_future_wiring(
    python_repo: Path, node_repo: Path
) -> None:
    """Detection is NOT lost: the detected graders are still surfaced on the stack
    (for the JSON payload + the reserved comment + future wiring) — they are just
    not written into the active `enabled` list."""
    py = scaffold(python_repo, dry_run=True)
    assert "ruff" in py.stack.external_tools
    # ...and the reserved comment names them so the user can opt in once wired.
    cfg = next(f for f in py.files if f.path == ".openrabbit.yaml").content
    assert "ruff" in cfg  # present in the reserved comment, not in `enabled`
    nd = scaffold(node_repo, dry_run=True)
    assert "eslint" in nd.stack.external_tools


# --------------------------------------------------------------------------- #
# scaffold — generated caller workflow is SHA-pinned & calls the reusable wf   #
# --------------------------------------------------------------------------- #
#: An all-zeros 40-hex SHA: regex-valid but resolves to no commit (a false
#: "SHA-pinned" pass). The scaffold must never ship this.
_ALL_ZEROS_SHA = "@" + "0" * 40


def test_scaffolded_workflow_ref_is_placeholder_not_resolvable_pin(
    python_repo: Path,
) -> None:
    """The reusable-workflow ref is a human-unmistakable placeholder, NOT a
    regex-valid-but-unresolvable concrete pin (all-zeros / foreign SHA)."""
    plan = scaffold(python_repo, dry_run=True)
    wf_file = next(
        f for f in plan.files if f.path == ".github/workflows/openrabbit.yml"
    )
    parsed = yaml.safe_load(wf_file.content)
    assert isinstance(parsed, dict)
    content = wf_file.content
    # The pin is a placeholder token a human must replace before it can run.
    assert "@<PINNED_SHA>" in content, (
        "reusable-workflow ref must use the <PINNED_SHA> placeholder"
    )
    assert "<OWNER>" in content, "owner must be an obvious <OWNER> placeholder"
    # It must NOT carry an all-zeros (or any concrete 40-hex) SHA that would pass
    # a naive SHA-pin regex while resolving to nothing.
    assert _ALL_ZEROS_SHA not in content, "must not ship an all-zeros SHA pin"
    assert not SHA_RE.search(content), (
        "scaffold must not ship a concrete 40-hex pin (forces a human edit)"
    )


def test_scaffolded_workflow_carries_replace_me_guidance(python_repo: Path) -> None:
    """The caller must carry inline replace-me guidance (a comment), not only a
    regex-valid placeholder — so a user cannot commit it unedited by accident."""
    plan = scaffold(python_repo, dry_run=True)
    wf_file = next(
        f for f in plan.files if f.path == ".github/workflows/openrabbit.yml"
    )
    # The reusable-workflow `uses:` line carries a REPLACE comment.
    uses_line = next(
        line
        for line in wf_file.content.splitlines()
        if "reusable-workflow.yml@" in line and line.lstrip().startswith("uses:")
    )
    assert "#" in uses_line, "reusable `uses:` line must carry a trailing comment"
    assert "REPLACE" in uses_line.upper(), (
        f"reusable `uses:` line must carry replace-me guidance: {uses_line}"
    )


def test_scaffolded_workflow_calls_reusable(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    wf_file = next(
        f for f in plan.files if f.path == ".github/workflows/openrabbit.yml"
    )
    # The thin caller `uses:` the central reusable workflow at the GitHub-required
    # path (.github/workflows/), the only path a reusable workflow can live at.
    assert ".github/workflows/reusable-workflow.yml@" in wf_file.content
    # It passes the AWS role ARN as a secret and triggers on pull_request.
    assert "aws_role_arn" in wf_file.content
    assert "pull_request" in wf_file.content


def test_scaffolded_workflow_least_privilege_permissions(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    wf_file = next(
        f for f in plan.files if f.path == ".github/workflows/openrabbit.yml"
    )
    parsed = yaml.safe_load(wf_file.content)
    perms = parsed.get("permissions")
    assert isinstance(perms, dict)
    assert perms.get("contents") == "read"


# --------------------------------------------------------------------------- #
# scaffold — write mode writes files to disk                                   #
# --------------------------------------------------------------------------- #
def test_scaffold_write_mode_writes_files(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=False)
    cfg_path = python_repo / ".openrabbit.yaml"
    wf_path = python_repo / ".github" / "workflows" / "openrabbit.yml"
    assert cfg_path.exists()
    assert wf_path.exists()
    # The plan still describes what was written.
    assert isinstance(plan, ScaffoldPlan)
    # Written config round-trips.
    cfg = load_config(cfg_path)
    assert "finder" in cfg.model_roles


def test_scaffold_write_does_not_clobber_existing_config(python_repo: Path) -> None:
    cfg_path = python_repo / ".openrabbit.yaml"
    cfg_path.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scaffold(python_repo, dry_run=False)
    # The pre-existing file is untouched.
    assert cfg_path.read_text(encoding="utf-8") == "version: 1\n"


def test_scaffold_force_overwrites(python_repo: Path) -> None:
    cfg_path = python_repo / ".openrabbit.yaml"
    cfg_path.write_text("version: 1\n", encoding="utf-8")
    scaffold(python_repo, dry_run=False, force=True)
    # force re-writes the full scaffolded config.
    cfg = load_config(cfg_path)
    assert "finder" in cfg.model_roles


def test_scaffold_write_never_touches_network(python_repo: Path, monkeypatch) -> None:
    """Defensive: importing/using init must not import boto3/httpx or call gh."""
    import subprocess

    def _boom(*a, **k):  # pragma: no cover - only fires on a regression
        raise AssertionError("scaffold must not shell out (no gh/network)")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    scaffold(python_repo, dry_run=False)


# --------------------------------------------------------------------------- #
# render_plan_text — human-readable summary of the whole plan                  #
# --------------------------------------------------------------------------- #
def test_plan_render_text_lists_files_and_wiring(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=True)
    rendered = plan.render()
    assert ".openrabbit.yaml" in rendered
    assert ".github/workflows/openrabbit.yml" in rendered
    assert "gh " in rendered  # wiring steps included


def test_render_shows_written_verb_after_write(python_repo: Path) -> None:
    plan = scaffold(python_repo, dry_run=False)
    assert plan.wrote is True
    assert "Wrote files" in plan.render()


def test_render_handles_unknown_stack(tmp_path: Path) -> None:
    plan = scaffold(tmp_path, dry_run=True)
    rendered = plan.render()
    assert "(none detected)" in rendered


def test_scaffold_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scaffold(tmp_path / "nope", dry_run=True)


# --------------------------------------------------------------------------- #
# CLI: `openrabbit init` subcommand                                           #
# --------------------------------------------------------------------------- #
def test_cli_init_dry_run_prints_plan_writes_nothing(python_repo, capsys) -> None:
    from openrabbit import cli

    rc = cli.main(["init", "--path", str(python_repo), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert ".openrabbit.yaml" in out
    assert ".github/workflows/openrabbit.yml" in out
    # Dry-run shows file bodies too.
    assert "model_roles" in out
    # Nothing written.
    assert not (python_repo / ".openrabbit.yaml").exists()


def test_cli_init_default_is_dry_run(python_repo) -> None:
    from openrabbit import cli

    rc = cli.main(["init", "--path", str(python_repo)])
    assert rc == 0
    assert not (python_repo / ".openrabbit.yaml").exists()


def test_cli_init_write_creates_files(python_repo) -> None:
    from openrabbit import cli

    rc = cli.main(["init", "--path", str(python_repo), "--write"])
    assert rc == 0
    assert (python_repo / ".openrabbit.yaml").exists()
    assert (python_repo / ".github" / "workflows" / "openrabbit.yml").exists()
    # Re-running without --force refuses to clobber.
    with pytest.raises(FileExistsError):
        cli.main(["init", "--path", str(python_repo), "--write"])
    # --force overwrites.
    rc2 = cli.main(["init", "--path", str(python_repo), "--write", "--force"])
    assert rc2 == 0


def test_cli_init_json_output(python_repo, capsys) -> None:
    import json as _json

    from openrabbit import cli

    rc = cli.main(["init", "--path", str(python_repo), "--json"])
    assert rc == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["wrote"] is False
    assert "python" in payload["stack"]["languages"]
    paths = {f["path"] for f in payload["files"]}
    assert ".openrabbit.yaml" in paths
    assert "AssumeRoleWithWebIdentity" in payload["wiringPlan"]


def test_cli_init_custom_aws_region(python_repo, capsys) -> None:
    from openrabbit import cli

    rc = cli.main(["init", "--path", str(python_repo), "--aws-region", "us-east-1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "us-east-1" in out


def test_module_main_delegates_to_cli(python_repo, capsys) -> None:
    """`python -m openrabbit.init <args>` forwards to the CLI init subcommand."""
    from openrabbit import init as init_mod

    rc = init_mod.main(["--path", str(python_repo), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert ".openrabbit.yaml" in out


def test_cli_init_surfaces_soft_warning_for_existing_config(python_repo, capsys):
    """Item 2: `init` on a repo whose existing .openrabbit.yaml has a soft
    model_roles warning prints the warning to stderr without failing."""
    from openrabbit import cli

    _write(
        python_repo / ".openrabbit.yaml",
        "version: 1\nmodel_roles:\n"
        "  finder: {model: amazon.nova-pro-v1:0, region: eu-west-3}\n",
    )
    rc = cli.main(["init", "--path", str(python_repo), "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "warning" in err.lower()
    assert "eu-west-3" in err
    assert "finder" in err


def test_cli_init_no_warning_without_existing_config(python_repo, capsys):
    # A fresh repo (no .openrabbit.yaml yet) emits no model-role warning.
    from openrabbit import cli

    rc = cli.main(["init", "--path", str(python_repo), "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "model_roles" not in err


def test_cli_init_unparsable_existing_config_does_not_block(python_repo, capsys):
    # A pre-existing BROKEN config must not block onboarding: init swallows the
    # load error, emits no warning, and still scaffolds the plan (rc == 0).
    from openrabbit import cli

    _write(python_repo / ".openrabbit.yaml", "version: not-an-int\n: : :\n")
    rc = cli.main(["init", "--path", str(python_repo), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert ".openrabbit.yaml" in out


# --------------------------------------------------------------------------- #
# gh extension shell wrapper — structural invariants (not executed online)     #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
GH_EXT = REPO_ROOT / "cli" / "gh-openrabbit" / "gh-openrabbit"


def test_gh_extension_exists_and_executable() -> None:
    import os

    assert GH_EXT.is_file(), "gh extension entrypoint is missing"
    assert os.access(GH_EXT, os.X_OK), "gh extension must be executable"


def test_gh_extension_calls_python_init() -> None:
    text = GH_EXT.read_text(encoding="utf-8")
    # It drives the pure Python planner (advisory-only), not gh, for scaffolding.
    assert "openrabbit.init" in text


def test_gh_extension_guards_network_mutations_behind_apply() -> None:
    text = GH_EXT.read_text(encoding="utf-8")
    # Real gh mutations live ONLY behind an explicit --apply guard.
    assert "--apply" in text
    assert "gh secret set" in text
    # The advisory-only contract is documented in the wrapper.
    assert "advisory-only" in text.lower()


def test_gh_extension_has_shebang() -> None:
    first = GH_EXT.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("#!"), "gh extension needs a shebang"
