"""Structural tests for org-scale rollout artifacts (PRD §11, §12, item 13).

openrabbit's control plane ships four artifacts that an org drops into its
``.github`` repo to roll out review across every repository centrally:

* ``org/.github/workflow-templates/openrabbit.yml`` — the *starter workflow*
  that appears in GitHub's "New workflow" picker. It is a thin caller that
  invokes openrabbit's **reusable** workflow (``workflow_call``), so a single
  central SHA bump rolls the whole fleet forward.
* ``org/.github/workflow-templates/openrabbit.properties.json` — the picker
  metadata (``name``/``description``/``categories``/``iconName``).
* ``org/ruleset.json`` — an **organization ruleset** that requires the
  openrabbit review workflow/check to pass before merge (target = PRs to the
  default branch), referencing the workflow pinned to a SHA/ref.
* ``org/safe-settings.yml`` — a Safe-Settings (Probot) example applying
  consistent branch protection + the required openrabbit check across repos.

These are **pure structural** tests: parse the YAML/JSON and assert the
invariants. No network calls and no live credentials — SHA pins are verified by
shape (40-hex), not by re-resolving them online.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ORG_DIR = REPO_ROOT / "org"
WORKFLOW_TEMPLATES_DIR = ORG_DIR / ".github" / "workflow-templates"
TEMPLATE_YML = WORKFLOW_TEMPLATES_DIR / "openrabbit.yml"
TEMPLATE_PROPERTIES = WORKFLOW_TEMPLATES_DIR / "openrabbit.properties.json"
RULESET_JSON = ORG_DIR / "ruleset.json"
SAFE_SETTINGS_YML = ORG_DIR / "safe-settings.yml"
ORG_README = ORG_DIR / "README.md"

SHA_RE = re.compile(r"[0-9a-f]{40}")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.exists(), f"missing org artifact: {path}"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _load_json(path: Path) -> Any:
    assert path.exists(), f"missing org artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


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


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def template() -> dict[str, Any]:
    return _load_yaml(TEMPLATE_YML)


@pytest.fixture(scope="module")
def properties() -> dict[str, Any]:
    return _load_json(TEMPLATE_PROPERTIES)


@pytest.fixture(scope="module")
def ruleset() -> dict[str, Any]:
    return _load_json(RULESET_JSON)


@pytest.fixture(scope="module")
def safe_settings() -> dict[str, Any]:
    return _load_yaml(SAFE_SETTINGS_YML)


# --------------------------------------------------------------------------- #
# files exist                                                                  #
# --------------------------------------------------------------------------- #
def test_all_org_artifacts_exist():
    for path in (
        TEMPLATE_YML,
        TEMPLATE_PROPERTIES,
        RULESET_JSON,
        SAFE_SETTINGS_YML,
        ORG_README,
    ):
        assert path.is_file(), f"missing org artifact: {path}"


def test_workflow_templates_dir_is_github_workflow_templates():
    # GitHub only surfaces starter workflows under .github/workflow-templates/.
    assert WORKFLOW_TEMPLATES_DIR.is_dir()
    assert WORKFLOW_TEMPLATES_DIR.parent.name == ".github"
    assert WORKFLOW_TEMPLATES_DIR.name == "workflow-templates"


# --------------------------------------------------------------------------- #
# workflow-template (starter workflow)                                         #
# --------------------------------------------------------------------------- #
def test_template_triggers_on_pull_request(template: dict[str, Any]) -> None:
    on = _on_block(template)
    assert isinstance(on, dict), "starter workflow `on:` must be a mapping"
    assert "pull_request" in on or "pull_request_target" in on, (
        "starter workflow must trigger on pull_request"
    )
    # Forked-PR safety (SPEC §12): never run fork code with secrets via
    # pull_request_target. The starter workflow uses plain pull_request.
    assert "pull_request_target" not in on, (
        "starter workflow must NOT use pull_request_target (forked-PR safety)"
    )


def test_template_calls_reusable_workflow(template: dict[str, Any]) -> None:
    """The starter workflow must call openrabbit's reusable workflow via uses:."""
    uses_values = list(_iter_uses(template))
    assert uses_values, "starter workflow declares no `uses:`"
    reusable_refs = [u for u in uses_values if "reusable" in u or ".yml@" in u or ".yaml@" in u]
    assert reusable_refs, (
        "starter workflow must `uses:` openrabbit's reusable workflow"
    )
    # The reference should point at an openrabbit workflow file.
    assert any("openrabbit" in u.lower() for u in uses_values), (
        "starter workflow must reference an openrabbit workflow"
    )


def test_template_calls_reusable_via_workflow_call(template: dict[str, Any]) -> None:
    """A reusable workflow is invoked at the *job* level via `uses:`."""
    jobs = template.get("jobs", {})
    assert jobs, "starter workflow declares no jobs"
    job_level_uses = [
        job["uses"]
        for job in jobs.values()
        if isinstance(job, dict) and isinstance(job.get("uses"), str)
    ]
    assert job_level_uses, (
        "starter workflow must invoke the reusable workflow at the job level "
        "(jobs.<id>.uses), not just as a step"
    )


#: The SHA of actions/checkout v6.0.3. The reusable-workflow ref must NOT be
#: pinned to this (a copy-paste error: it's a foreign action commit, not an
#: openrabbit commit). See the HIGH finding.
_CHECKOUT_V6_SHA = "df4cb1c069e1874edd31b4311f1884172cec0e10"


def test_template_uses_are_sha_pinned_or_documented_placeholder(
    template: dict[str, Any],
) -> None:
    """Every external `uses:` is either a real 40-hex SHA OR the documented
    <PINNED_SHA> placeholder (the reusable-workflow ref a human must edit)."""
    uses_values = list(_iter_uses(template))
    assert uses_values, "starter workflow declares no `uses:`"
    for ref in uses_values:
        if ref.startswith("./") or ref.startswith("../"):
            continue
        owner_repo, sep, ref_spec = ref.partition("@")
        assert sep == "@", f"`uses: {ref}` is not pinned (@<sha>)"
        if ref_spec == "<PINNED_SHA>":
            # Intentional placeholder: must be the openrabbit reusable workflow.
            assert "reusable-workflow.yml" in owner_repo, (
                f"only the reusable-workflow ref may use the placeholder: {ref}"
            )
            continue
        assert SHA_RE.fullmatch(ref_spec), (
            f"`uses: {ref}` is not pinned to a 40-hex SHA or <PINNED_SHA>"
        )


def test_template_reusable_ref_uses_placeholder_not_foreign_sha(
    template: dict[str, Any],
) -> None:
    """The reusable-workflow ref must be a <PINNED_SHA> placeholder, NOT a
    copy-pasted foreign SHA (e.g. actions/checkout's commit)."""
    reusable_refs = [
        u for u in _iter_uses(template) if "reusable-workflow.yml" in u
    ]
    assert reusable_refs, "template must reference the reusable workflow"
    for ref in reusable_refs:
        _, _, ref_spec = ref.partition("@")
        assert ref_spec == "<PINNED_SHA>", (
            f"reusable-workflow ref must use the <PINNED_SHA> placeholder, "
            f"not a concrete/foreign SHA: {ref}"
        )
        assert _CHECKOUT_V6_SHA not in ref, (
            "reusable-workflow ref must not be pinned to actions/checkout's SHA"
        )
    # Belt-and-suspenders: the literal foreign SHA appears nowhere in the file.
    assert _CHECKOUT_V6_SHA not in _text(TEMPLATE_YML), (
        "actions/checkout's SHA must not appear in the org template"
    )


def test_template_pins_carry_version_comment() -> None:
    """Each SHA-pinned `uses:` line carries a trailing `# vX`/`# ref` comment."""
    text = _text(TEMPLATE_YML)
    pin_lines = [
        line
        for line in text.splitlines()
        if "uses:" in line and "@" in line and not re.search(r"uses:\s*\.", line)
    ]
    assert pin_lines, "starter workflow has no pinned uses lines"
    for line in pin_lines:
        assert "#" in line.split("@", 1)[1], (
            f"pinned `uses` line lacks a trailing comment: {line.strip()}"
        )


def test_template_no_floating_refs(template: dict[str, Any]) -> None:
    for ref in _iter_uses(template):
        if ref.startswith("./") or ref.startswith("../"):
            continue
        _, _, ref_spec = ref.partition("@")
        assert not ref_spec.startswith("v"), f"floating tag ref: {ref}"
        assert ref_spec not in {"main", "master", "HEAD"}, f"branch ref: {ref}"


def test_template_top_level_permissions_least_privilege(
    template: dict[str, Any],
) -> None:
    """Top-level permissions are least-privilege contents: read."""
    perms = template.get("permissions")
    assert isinstance(perms, dict), "starter workflow needs top-level permissions"
    assert perms.get("contents") == "read", "top-level must be contents: read"
    assert perms.get("pull-requests") in (None, "read"), (
        "pull-requests: write must NOT be granted at the top level"
    )


def test_template_grants_oidc_and_pr_write(template: dict[str, Any]) -> None:
    """The caller passes through the perms the reusable workflow needs (OIDC + PR write)."""
    perms_blocks: list[dict[str, Any]] = []
    if isinstance(template.get("permissions"), dict):
        perms_blocks.append(template["permissions"])
    for job in template.get("jobs", {}).values():
        if isinstance(job, dict) and isinstance(job.get("permissions"), dict):
            perms_blocks.append(job["permissions"])
    assert any(b.get("id-token") == "write" for b in perms_blocks), (
        "starter workflow must grant id-token: write (OIDC -> STS Bedrock)"
    )
    assert any(b.get("pull-requests") == "write" for b in perms_blocks), (
        "starter workflow must grant pull-requests: write to post the review"
    )


def test_template_passes_required_inputs(template: dict[str, Any]) -> None:
    """The starter workflow forwards pr + commit to the reusable workflow."""
    text = _text(TEMPLATE_YML)
    # Reference the PR number + head SHA from the event context.
    assert "github.event.pull_request.number" in text or "pull_request.number" in text
    assert "github.event.pull_request.head.sha" in text or "head.sha" in text
    # The AWS role ARN is wired as a secret to the reusable workflow.
    assert "secrets" in text.lower() and "role" in text.lower()


# --------------------------------------------------------------------------- #
# workflow-template properties.json                                           #
# --------------------------------------------------------------------------- #
def test_properties_is_object(properties: dict[str, Any]) -> None:
    assert isinstance(properties, dict)


def test_properties_has_required_fields(properties: dict[str, Any]) -> None:
    # GitHub workflow-template picker metadata.
    for field in ("name", "description", "iconName", "categories"):
        assert field in properties, f"properties.json missing {field!r}"


def test_properties_name_and_description_nonempty(properties: dict[str, Any]) -> None:
    assert isinstance(properties["name"], str) and properties["name"].strip()
    assert isinstance(properties["description"], str)
    assert properties["description"].strip()
    assert "openrabbit" in properties["name"].lower() or "openrabbit" in (
        properties["description"].lower()
    )


def test_properties_categories_is_nonempty_string_list(
    properties: dict[str, Any],
) -> None:
    cats = properties["categories"]
    assert isinstance(cats, list) and cats
    assert all(isinstance(c, str) and c.strip() for c in cats)


def test_properties_iconname_is_nonempty_string(properties: dict[str, Any]) -> None:
    assert isinstance(properties["iconName"], str) and properties["iconName"].strip()


# --------------------------------------------------------------------------- #
# org ruleset.json                                                            #
# --------------------------------------------------------------------------- #
def test_ruleset_is_object(ruleset: dict[str, Any]) -> None:
    assert isinstance(ruleset, dict)


def test_ruleset_has_name_and_target(ruleset: dict[str, Any]) -> None:
    assert isinstance(ruleset.get("name"), str) and ruleset["name"].strip()
    # Org rulesets target branches.
    assert ruleset.get("target") == "branch", (
        "ruleset must target branches (PRs to the default branch)"
    )


def test_ruleset_enforcement_supports_evaluate(ruleset: dict[str, Any]) -> None:
    """Enforcement is one of GitHub's allowed values; Evaluate is documented."""
    enforcement = ruleset.get("enforcement")
    assert enforcement in {"active", "evaluate", "disabled"}, (
        f"unexpected enforcement value: {enforcement!r}"
    )
    # The "Evaluate"-mode rollout note must be present (in the file or README).
    text = (_text(RULESET_JSON) + _text(ORG_README)).lower()
    assert "evaluate" in text, "Evaluate-mode rollout note is required"


def test_ruleset_targets_default_branch(ruleset: dict[str, Any]) -> None:
    """The branch ruleset includes the repo default branch."""
    text = _text(RULESET_JSON).lower()
    assert "default" in text, (
        "ruleset must target the default branch (~DEFAULT_BRANCH)"
    )
    # The structured conditions should reference ref_name include.
    conditions = ruleset.get("conditions", {})
    ref_name = conditions.get("ref_name", {}) if isinstance(conditions, dict) else {}
    include = ref_name.get("include", []) if isinstance(ref_name, dict) else []
    assert any("default" in str(i).lower() for i in include), (
        "ruleset conditions.ref_name.include must contain ~DEFAULT_BRANCH"
    )


def test_ruleset_requires_pull_request(ruleset: dict[str, Any]) -> None:
    """A pull_request rule is present (review happens on PRs)."""
    rules = ruleset.get("rules", [])
    assert isinstance(rules, list) and rules, "ruleset declares no rules"
    types = {r.get("type") for r in rules if isinstance(r, dict)}
    assert "pull_request" in types, "ruleset must require a pull request"


def test_ruleset_requires_status_check(ruleset: dict[str, Any]) -> None:
    """A required_status_checks rule names the openrabbit check + workflow SHA."""
    rules = ruleset.get("rules", [])
    rsc = [
        r
        for r in rules
        if isinstance(r, dict) and r.get("type") == "required_status_checks"
    ]
    assert rsc, "ruleset must include a required_status_checks rule"
    params = rsc[0].get("parameters", {})
    checks = params.get("required_status_checks", [])
    assert isinstance(checks, list) and checks, "no required status checks listed"
    contexts = [c.get("context") for c in checks if isinstance(c, dict)]
    assert any("openrabbit" in str(c).lower() for c in contexts), (
        "required status checks must include the openrabbit review check"
    )


def test_ruleset_references_sha_pinned_workflow(ruleset: dict[str, Any]) -> None:
    """The required workflow is pinned to a SHA/ref (placeholder is acceptable)."""
    text = _text(RULESET_JSON)
    # Either a concrete 40-hex SHA or a clearly-marked placeholder for org/repo@SHA.
    assert SHA_RE.search(text) or "@<" in text or "@COMMIT_SHA" in text.upper() or (
        "<sha>" in text.lower() or "placeholder" in text.lower()
    ), "ruleset must reference the workflow pinned to a SHA/ref (or placeholder)"


# --------------------------------------------------------------------------- #
# required-check name: composite "<caller job> / <reusable job>" consistency   #
# --------------------------------------------------------------------------- #
REUSABLE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "reusable-workflow.yml"


def _reusable_job_name() -> str:
    data = _load_yaml(REUSABLE_WORKFLOW)
    jobs = data.get("jobs", {})
    review = jobs.get("review")
    assert isinstance(review, dict), "reusable workflow needs a `review` job"
    return str(review.get("name", "review"))


def _template_caller_job_name(template: dict[str, Any]) -> str:
    jobs = template.get("jobs", {})
    job_id, job = next(iter(jobs.items()))
    # GitHub renders the job's `name:` if set, else the job id.
    return str(job.get("name") or job_id)


def test_required_check_context_matches_composite_job_names(
    template: dict[str, Any], ruleset: dict[str, Any], safe_settings: dict[str, Any]
) -> None:
    """The ruleset/safe-settings required-check context must equal the EXACT
    composite name GitHub reports for the reusable-workflow job:
    "<caller job name> / <reusable job name>"."""
    caller = _template_caller_job_name(template)
    reusable = _reusable_job_name()
    expected = f"{caller} / {reusable}"
    assert expected == "openrabbit / review", (
        f"composite check name drifted: caller={caller!r} reusable={reusable!r}"
    )

    # Ruleset context must equal the composite name (not the bare reusable name).
    rsc = next(
        r for r in ruleset["rules"] if r.get("type") == "required_status_checks"
    )
    contexts = [
        c.get("context")
        for c in rsc["parameters"]["required_status_checks"]
        if isinstance(c, dict)
    ]
    assert expected in contexts, (
        f"ruleset context {contexts} must be the composite {expected!r}"
    )

    # Safe-Settings context must also equal the composite name.
    ss_names: list[str] = []
    for branch in safe_settings.get("branches", []):
        if not isinstance(branch, dict):
            continue
        rsc_ss = (branch.get("protection") or {}).get("required_status_checks") or {}
        if not isinstance(rsc_ss, dict):
            continue
        for c in rsc_ss.get("checks") or []:
            ss_names.append(c.get("context") if isinstance(c, dict) else c)
        ss_names.extend(rsc_ss.get("contexts") or [])
    assert expected in ss_names, (
        f"safe-settings context {ss_names} must be the composite {expected!r}"
    )


# --------------------------------------------------------------------------- #
# safe-settings.yml                                                           #
# --------------------------------------------------------------------------- #
def test_safe_settings_has_branch_protection(safe_settings: dict[str, Any]) -> None:
    """Safe-Settings org config defines branch protection."""
    # Safe-Settings shape: branches: [{name, protection: {...}}] (per-repo) or
    # under `repository`/`branches` at org level.
    branches = safe_settings.get("branches")
    assert isinstance(branches, list) and branches, (
        "safe-settings must define branches with protection"
    )
    protections = [
        b.get("protection")
        for b in branches
        if isinstance(b, dict) and isinstance(b.get("protection"), dict)
    ]
    assert protections, "safe-settings branches must declare protection blocks"


def test_safe_settings_requires_openrabbit_check(
    safe_settings: dict[str, Any],
) -> None:
    """Branch protection requires the openrabbit status check before merge."""
    text = _text(SAFE_SETTINGS_YML).lower()
    assert "required_status_checks" in text, (
        "safe-settings protection must declare required_status_checks"
    )
    assert "openrabbit" in text, (
        "safe-settings must require the openrabbit check"
    )


def test_safe_settings_required_check_is_structured(
    safe_settings: dict[str, Any],
) -> None:
    """The openrabbit check is wired into a required_status_checks structure."""
    found = False
    for branch in safe_settings.get("branches", []):
        if not isinstance(branch, dict):
            continue
        protection = branch.get("protection") or {}
        rsc = protection.get("required_status_checks") or {}
        if not isinstance(rsc, dict):
            continue
        # GitHub API shape: required_status_checks.checks[].context or .contexts[].
        checks = rsc.get("checks") or []
        contexts = rsc.get("contexts") or []
        names = [
            c.get("context") if isinstance(c, dict) else c for c in checks
        ] + list(contexts)
        if any("openrabbit" in str(n).lower() for n in names):
            found = True
    assert found, (
        "a branch protection block must list the openrabbit check in "
        "required_status_checks (checks[].context or contexts[])"
    )


# --------------------------------------------------------------------------- #
# README — deployment guidance                                                #
# --------------------------------------------------------------------------- #
def test_readme_documents_deployment(safe_settings: dict[str, Any]) -> None:
    text = _text(ORG_README).lower()
    # Drop into the org .github repo.
    assert ".github" in text
    # Evaluate -> Active rollout.
    assert "evaluate" in text and "active" in text
    # One central SHA bump rolls all repos forward.
    assert "sha" in text and ("bump" in text or "roll" in text)
