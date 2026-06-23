"""Structural tests for the docs, SBOM tooling, and OpenSSF Scorecard config
(PRD §12 self-supply-chain, checklist item 18).

These are *pure structural* tests — they read files from disk and assert the
docs cover the load-bearing sections and that the Scorecard workflow is
supply-chain-hardened (SHA-pinned + least-privilege). There are **no network
calls** and no live credentials: SHA pins are verified by shape (40-hex), not by
re-resolving them online.

Coverage:

* ``README.md`` is a real README (not the stub) and names the trust thesis,
  quickstart, config reference, security model, and Bedrock model routing.
* ``docs/usage.md`` / ``docs/configuration.md`` / ``docs/security.md`` /
  ``docs/onboarding.md`` exist and mention their key sections.
* ``scripts/generate_sbom.sh`` exists, is executable, and uses a lazily-installed
  CycloneDX tool (no heavy runtime dep).
* ``.github/workflows/scorecard.yml`` parses as YAML, SHA-pins every external
  ``uses:``, and grants least-privilege permissions (``security-events: write``,
  ``id-token: write``; no broad ``contents: write`` at the top level).
"""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

README = REPO_ROOT / "README.md"
USAGE = DOCS_DIR / "usage.md"
CONFIGURATION = DOCS_DIR / "configuration.md"
SECURITY = DOCS_DIR / "security.md"
ONBOARDING = DOCS_DIR / "onboarding.md"

SBOM_SCRIPT = REPO_ROOT / "scripts" / "generate_sbom.sh"
SCORECARD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "scorecard.yml"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _text(path: Path) -> str:
    assert path.is_file(), f"missing doc/file: {path}"
    return path.read_text(encoding="utf-8")


def _lower(path: Path) -> str:
    return _text(path).lower()


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing YAML file: {path}"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _on_block(workflow: dict[str, Any]) -> Any:
    """Return the ``on:`` block (PyYAML maps the bare key ``on`` -> True)."""
    if "on" in workflow:
        return workflow["on"]
    return workflow.get(True)


def _iter_uses(node: Any):
    """Yield every ``uses:`` string anywhere in a parsed YAML tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                yield value
            else:
                yield from _iter_uses(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_uses(item)


# --------------------------------------------------------------------------- #
# files exist                                                                  #
# --------------------------------------------------------------------------- #
def test_all_docs_exist() -> None:
    for path in (README, USAGE, CONFIGURATION, SECURITY, ONBOARDING):
        assert path.is_file(), f"missing doc: {path}"
        assert path.read_text(encoding="utf-8").strip(), f"empty doc: {path}"


# --------------------------------------------------------------------------- #
# README — real README, not the stub                                          #
# --------------------------------------------------------------------------- #
def test_readme_is_not_the_stub() -> None:
    """The README must be substantial (the original stub was ~30 lines)."""
    text = _text(README)
    assert len(text.splitlines()) > 60, "README looks like the old stub"


def test_readme_states_what_openrabbit_is() -> None:
    low = _lower(README)
    assert "openrabbit" in low
    # High-trust, Bedrock-only AI code reviewer.
    assert "bedrock" in low
    assert "code review" in low or "code reviewer" in low
    assert "high-trust" in low or "high trust" in low


def test_readme_has_quickstart() -> None:
    low = _lower(README)
    assert "quickstart" in low or "quick start" in low
    # uv sync; the offline demo; init.
    assert "uv sync" in low
    assert "--offline" in low
    assert "openrabbit init" in low


def test_readme_links_config_reference() -> None:
    text = _text(README)
    low = text.lower()
    assert ".openrabbit.yaml" in low
    # Link to the configuration doc.
    assert "configuration.md" in low or "docs/configuration" in low


def test_readme_documents_security_model() -> None:
    low = _lower(README)
    assert "advisory-only" in low or "advisory only" in low
    assert "untrusted" in low
    assert "oidc" in low
    assert "sha-pin" in low or "sha pin" in low or "sha-pinning" in low


def test_readme_documents_bedrock_model_routing() -> None:
    low = _lower(README)
    assert "nova" in low
    assert "gpt-5.5" in low
    assert "finder" in low and "verifier" in low


def test_readme_states_trust_thesis() -> None:
    low = _lower(README)
    # FP < 10% and find-broad / filter-strict.
    assert "fp" in low or "false positive" in low or "false-positive" in low
    assert "10%" in low or "<10" in low
    assert "find" in low and "filter" in low


def test_readme_documents_sbom() -> None:
    low = _lower(README)
    assert "sbom" in low
    assert "generate_sbom" in low or "cyclonedx" in low


def test_readme_has_apache_badge() -> None:
    text = _text(README)
    low = text.lower()
    assert "apache-2.0" in low or "apache 2.0" in low
    # A badge is a markdown image referencing a shield/badge.
    assert "![" in text and (
        "badge" in low or "shields.io" in low or "img.shields" in low
    )


# --------------------------------------------------------------------------- #
# docs/usage.md                                                               #
# --------------------------------------------------------------------------- #
def test_usage_documents_cli_commands() -> None:
    low = _lower(USAGE)
    # All four real subcommands must be documented.
    for cmd in ("review", "eval", "init", "learn"):
        assert f"openrabbit {cmd}" in low, f"usage.md must document `openrabbit {cmd}`"
    # The offline demo path is the anywhere-runnable entrypoint.
    assert "--offline" in low


# --------------------------------------------------------------------------- #
# docs/configuration.md — full .openrabbit.yaml reference                     #
# --------------------------------------------------------------------------- #
def test_configuration_documents_full_reference() -> None:
    low = _lower(CONFIGURATION)
    assert ".openrabbit.yaml" in low
    # The fields explicitly called out in the checklist.
    assert "model_roles" in low
    assert "verify_min_severity" in low
    assert "lenses" in low
    assert "path_filters" in low or "path filters" in low
    assert "confidence_gate" in low


def test_configuration_lists_all_lenses() -> None:
    low = _lower(CONFIGURATION)
    for lens in ("correctness", "security", "performance", "tests", "maintainability"):
        assert lens in low, f"configuration.md must mention the {lens} lens"


def test_configuration_documents_model_roles() -> None:
    low = _lower(CONFIGURATION)
    for role in ("triage", "finder", "verifier"):
        assert role in low, f"configuration.md must document the {role} role"


def test_configuration_marks_external_tools_reserved() -> None:
    """Finding #3: the docs must not claim external_tools is live grounding while
    the pipeline does not run those graders. The `external_tools` section must be
    explicitly flagged as reserved / not yet wired (honest docs)."""
    text = CONFIGURATION.read_text(encoding="utf-8")
    low = text.lower()
    assert "external_tools" in low
    # The section is honestly flagged as not-yet-implemented.
    assert "reserved" in low and "not yet wired" in low, (
        "configuration.md must flag external_tools as reserved / not yet wired"
    )
    # The external_tools section must say the pipeline does NOT run these graders
    # today (honest present-tense), so a reader is not misled into thinking the
    # output already grounds the review.
    section = text.split("## `external_tools`", 1)[1].split("\n## ", 1)[0].lower()
    # Markdown emphasis (e.g. `does **not** run`) must not defeat the check.
    section_plain = section.replace("*", "")
    assert "does not" in section_plain and (
        "run these graders" in section_plain
        or "run those graders" in section_plain
        or "currently run" in section_plain
        or "inject" in section_plain
    ), "external_tools section must state the pipeline does not run them yet"
    # Any 'fed into the review context' grounding claim must be framed as INTENDED
    # / future, never as a present-tense fact.
    if "fed into the review context" in section:
        assert "intended" in section or "future" in section, (
            "the 'fed into the review context' claim must be framed as intended "
            "future behavior, not a present-tense fact"
        )


# --------------------------------------------------------------------------- #
# docs/security.md — threat model                                            #
# --------------------------------------------------------------------------- #
def test_security_documents_threat_model() -> None:
    low = _lower(SECURITY)
    assert "threat model" in low
    assert "prompt injection" in low
    assert "supply chain" in low or "supply-chain" in low
    assert "advisory-only" in low or "advisory only" in low
    assert "oidc" in low
    assert "secret" in low


# --------------------------------------------------------------------------- #
# docs/onboarding.md                                                          #
# --------------------------------------------------------------------------- #
def test_onboarding_documents_init_and_rollout() -> None:
    low = _lower(ONBOARDING)
    assert "openrabbit init" in low
    # Org rollout + plugin marketplace.
    assert "org" in low
    assert "marketplace" in low or "plugin" in low


# --------------------------------------------------------------------------- #
# SBOM generation script                                                      #
# --------------------------------------------------------------------------- #
def test_sbom_script_exists_and_is_executable() -> None:
    assert SBOM_SCRIPT.is_file(), "missing scripts/generate_sbom.sh"
    mode = SBOM_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "generate_sbom.sh must be executable (chmod +x)"


def test_sbom_script_uses_lazy_cyclonedx_tool() -> None:
    text = _text(SBOM_SCRIPT)
    low = text.lower()
    # Lazily-installed tool runner (uvx / pipx run / npx) — no heavy runtime dep.
    assert "uvx" in low or "pipx run" in low or "npx" in low, (
        "SBOM script must use a lazily-installed tool runner (uvx/pipx run/npx)"
    )
    assert "cyclonedx" in low, "SBOM script must use a CycloneDX tool"
    # It emits sbom.json.
    assert "sbom.json" in low
    # POSIX-safe shell preamble.
    assert text.startswith("#!"), "generate_sbom.sh must start with a shebang"
    assert "set -e" in text or "set -eu" in text or "set -euo" in text, (
        "generate_sbom.sh should fail fast (set -e)"
    )


def test_sbom_not_a_runtime_dependency() -> None:
    """CycloneDX must NOT be pinned as a runtime/dev dependency (keep it lazy)."""
    pyproject = _text(REPO_ROOT / "pyproject.toml").lower()
    assert "cyclonedx" not in pyproject, (
        "cyclonedx must stay a lazily-run tool, not a declared dependency"
    )


# --------------------------------------------------------------------------- #
# OpenSSF Scorecard workflow                                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scorecard() -> dict[str, Any]:
    return _load_yaml(SCORECARD_WORKFLOW)


def test_scorecard_workflow_exists(scorecard: dict[str, Any]) -> None:
    assert isinstance(scorecard, dict)


def test_scorecard_uses_ossf_action(scorecard: dict[str, Any]) -> None:
    refs = " ".join(_iter_uses(scorecard))
    assert "ossf/scorecard-action@" in refs, (
        "scorecard.yml must use ossf/scorecard-action"
    )


def test_scorecard_every_uses_is_sha_pinned(scorecard: dict[str, Any]) -> None:
    uses_values = list(_iter_uses(scorecard))
    assert uses_values, "scorecard.yml declares no `uses:` steps"
    for ref in uses_values:
        if ref.startswith("./") or ref.startswith("../"):
            continue
        owner_repo, sep, ref_spec = ref.partition("@")
        assert sep == "@", f"`uses: {ref}` is not pinned (@<sha>)"
        assert SHA_RE.match(ref_spec), f"`uses: {ref}` is not pinned to a 40-hex SHA"


def test_scorecard_pins_carry_version_comment() -> None:
    text = _text(SCORECARD_WORKFLOW)
    pin_lines = [
        line
        for line in text.splitlines()
        if "uses:" in line and "@" in line and not re.search(r"uses:\s*\.", line)
    ]
    assert pin_lines, "scorecard.yml has no pinned uses lines"
    for line in pin_lines:
        assert "#" in line.split("@", 1)[1], (
            f"pinned `uses` line lacks a trailing version comment: {line.strip()}"
        )


def test_scorecard_no_floating_refs(scorecard: dict[str, Any]) -> None:
    for ref in _iter_uses(scorecard):
        if ref.startswith("./") or ref.startswith("../"):
            continue
        _, _, ref_spec = ref.partition("@")
        assert not ref_spec.startswith("v"), f"floating tag ref: {ref}"
        assert ref_spec not in {"main", "master", "HEAD"}, f"branch ref: {ref}"


def test_scorecard_top_level_permissions_least_privilege(
    scorecard: dict[str, Any],
) -> None:
    """Top-level permissions are read-only — either the ``read-all`` shorthand
    (the OpenSSF-recommended form) or an explicit ``contents: read`` map. No
    write scope may be granted at the top level."""
    perms = scorecard.get("permissions")
    if isinstance(perms, str):
        # The recommended `read-all` shorthand (or `read`) — read-only.
        assert perms in {"read-all", "read"}, (
            f"top-level permissions string must be read-only, got {perms!r}"
        )
        return
    assert isinstance(perms, dict), "scorecard.yml needs top-level permissions"
    assert perms.get("contents") == "read", "top-level must be contents: read"
    # Least privilege: no broad write grants at the top level.
    assert "write" not in set(perms.values()), (
        "no write scope may be granted at the top level"
    )


def test_scorecard_grants_security_events_only_no_id_token(
    scorecard: dict[str, Any],
) -> None:
    """The analysis job needs security-events: write (upload SARIF) but must NOT
    grant id-token: write while openrabbit is a private repo.

    Publishing the signed Scorecard result outward (`publish_results`) is gated
    off for the private repo (design spec §1.1) — it would leak repo metadata to
    the public OpenSSF API and the public badge cannot resolve — so the OIDC
    elevation that publishing requires must not be granted (least privilege)."""
    perms_blocks: list[dict[str, Any]] = []
    if isinstance(scorecard.get("permissions"), dict):
        perms_blocks.append(scorecard["permissions"])
    for job in scorecard.get("jobs", {}).values():
        if isinstance(job, dict) and isinstance(job.get("permissions"), dict):
            perms_blocks.append(job["permissions"])
    assert any(b.get("security-events") == "write" for b in perms_blocks), (
        "scorecard.yml must grant security-events: write"
    )
    assert not any(b.get("id-token") == "write" for b in perms_blocks), (
        "scorecard.yml must NOT grant id-token: write while the repo is private "
        "(publish_results is gated off; restore it only when made public)"
    )


def test_scorecard_publish_results_gated_on_public_repo(
    scorecard: dict[str, Any],
) -> None:
    """`publish_results` must be gated on the repo being public, not hard-true.

    A private repo (the design default) must not push its Scorecard analysis to
    the public OpenSSF API; the expression resolves to false until the repo is
    intentionally made public."""
    analysis = scorecard["jobs"]["analysis"]
    run_step = next(
        s
        for s in analysis["steps"]
        if isinstance(s.get("uses"), str) and "scorecard-action" in s["uses"]
    )
    publish = run_step["with"]["publish_results"]
    # Gated on visibility (an expression), never an unconditional `true`.
    assert publish is not True, "publish_results must not be unconditionally true"
    assert "github.event.repository.private" in str(publish), (
        "publish_results must be gated on the repo being public"
    )


def test_scorecard_triggers_on_schedule_and_push(scorecard: dict[str, Any]) -> None:
    on = _on_block(scorecard)
    assert isinstance(on, dict), "scorecard.yml `on:` must be a mapping"
    assert "schedule" in on, "scorecard.yml must run on a schedule"
    assert "push" in on, "scorecard.yml must run on branch push"
