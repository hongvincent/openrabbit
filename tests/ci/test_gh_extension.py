"""Structural + behavioral tests for the ``gh-openrabbit`` shell extension.

The shell wrapper is the only place real ``gh`` mutations happen, so these tests
deliberately never trigger a network mutation: they assert the *resolution logic*
that picks how to invoke ``openrabbit init`` (finding #2 — one-command onboarding
must be runnable in a target repo that has NO openrabbit install and NO
``pyproject.toml`` of its own).

Run the actual shell script in a controlled environment with a debug hook
(``OPENRABBIT_PRINT_RUN_CMD=1``) that prints the resolved command and exits
*before* any Python/uvx/gh is invoked — so the test is offline, deterministic,
and exercises the real bash resolution rather than a re-implementation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GH_EXT = REPO_ROOT / "cli" / "gh-openrabbit" / "gh-openrabbit"

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def _coreutils_dirs() -> list[str]:
    """Dirs that hold core POSIX utils (grep/dirname/cat) but NOT `openrabbit`.

    The resolution tests restrict PATH so `openrabbit` is undiscoverable, but the
    wrapper still legitimately shells out to grep/dirname/cat — include their
    dirs (standard system bins, which never contain an `openrabbit` binary)."""
    dirs: list[str] = []
    for tool in ("grep", "dirname", "cat"):
        found = shutil.which(tool)
        if found:
            dirs.append(os.path.dirname(found))
    # Fallbacks for the standard locations in case PATH was already minimal.
    dirs.extend(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    return list(dict.fromkeys(dirs))


def _standalone_ext(tmp_path: Path) -> Path:
    """Copy the extension into an ISOLATED ``gh-openrabbit/`` dir (no openrabbit
    source tree around it) — mirroring how ``gh extension install`` clones the
    extension to ``~/.local/share/gh/extensions/gh-openrabbit/`` on a teammate's
    machine that never checked out the openrabbit source.

    Running the in-repo script directly would let the wrapper find the surrounding
    openrabbit checkout (a legitimate dev-tree resolution) and mask the
    target-repo-independent path we want to assert here.
    """
    extdir = tmp_path / "ghext" / "gh-openrabbit"
    extdir.mkdir(parents=True)
    dst = extdir / "gh-openrabbit"
    shutil.copy2(GH_EXT, dst)
    dst.chmod(0o755)
    return dst


def _run_resolve(
    tmp_path: Path, *, extra_path: str | None = None, in_repo: bool = False
) -> str:
    """Invoke ``gh-openrabbit init`` from a target repo with the print-cmd hook.

    Returns the resolved run command (stdout, stripped). ``PATH`` is reduced to a
    minimal sandbox (so ``openrabbit`` is NOT discoverable unless ``extra_path``
    explicitly adds a dir containing it). The target repo (``cwd``) lives at
    ``tmp_path/target`` and, unless a test writes one, has no ``pyproject.toml`` —
    mirroring a teammate's repo with no openrabbit install.
    """
    script = GH_EXT if in_repo else _standalone_ext(tmp_path)
    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    base_dirs = [os.path.dirname(bash or "/bin/bash"), *_coreutils_dirs()]
    # Keep uv/uvx resolvable (the wrapper may legitimately resolve to uvx), but
    # do NOT include any dir that would expose an `openrabbit` binary.
    for tool in ("uv", "uvx"):
        found = shutil.which(tool)
        if found:
            base_dirs.append(os.path.dirname(found))
    path = os.pathsep.join(dict.fromkeys(base_dirs))
    if extra_path:
        path = extra_path + os.pathsep + path
    env = {
        "PATH": path,
        "OPENRABBIT_PRINT_RUN_CMD": "1",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [bash or "/bin/bash", str(script), "init"],
        cwd=str(target),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"resolve failed: {proc.stderr}\n{proc.stdout}"
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# resolution logic — the core of finding #2                                    #
# --------------------------------------------------------------------------- #
def test_resolves_without_target_pyproject_or_install(tmp_path: Path) -> None:
    """In a repo with NO openrabbit install AND NO pyproject.toml, the wrapper
    must NOT fall back to the broken ``python3 -m openrabbit.init`` (which raises
    ModuleNotFoundError) — it must resolve openrabbit independently (uvx)."""
    assert not (tmp_path / "target" / "pyproject.toml").exists()
    cmd = _run_resolve(tmp_path)
    assert "python3 -m openrabbit.init" not in cmd, (
        "must not fall back to bare `python3 -m openrabbit.init` (ModuleNotFound "
        f"in a target repo without openrabbit); got: {cmd!r}"
    )
    # It resolves openrabbit independent of the target repo: uvx fetches the
    # `openrabbit` distribution itself.
    assert "uvx" in cmd and "--from openrabbit" in cmd, (
        f"expected a `uvx --from openrabbit` resolution, got: {cmd!r}"
    )


def test_does_not_bind_to_target_repo_pyproject(tmp_path: Path) -> None:
    """A target repo that HAS a pyproject.toml (its own, unrelated project) must
    not trick the wrapper into `uv run` against that repo (openrabbit isn't a dep
    there). Resolution must be identical to the no-pyproject case."""
    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    (target / "pyproject.toml").write_text(
        "[project]\nname = 'some-teammate-repo'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    cmd = _run_resolve(tmp_path)
    assert "uv run" not in cmd, (
        "must not `uv run` against the TARGET repo's pyproject (openrabbit is not "
        f"a dependency there); got: {cmd!r}"
    )
    assert "--from openrabbit" in cmd, f"expected uvx resolution, got: {cmd!r}"


def test_prefers_installed_openrabbit_on_path(tmp_path: Path) -> None:
    """When `openrabbit` IS on PATH, prefer the direct binary (no uvx fetch)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "openrabbit"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    cmd = _run_resolve(tmp_path, extra_path=str(bindir))
    assert cmd.split()[0].endswith("openrabbit") or cmd.startswith("openrabbit"), (
        f"expected the installed `openrabbit` binary to be preferred, got: {cmd!r}"
    )
    assert "uvx" not in cmd, f"should not uvx when openrabbit is installed: {cmd!r}"


# --------------------------------------------------------------------------- #
# doctor — OIDC trust-policy validation (finding #4)                            #
# --------------------------------------------------------------------------- #
import json  # noqa: E402

_VALID_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": (
                    "arn:aws:iam::123456789012:oidc-provider/"
                    "token.actions.githubusercontent.com"
                )
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": "repo:acme/widgets:*"
                },
            },
        }
    ],
}

# A trust policy that allows a plain IAM user to assume the role but is MISSING
# the GitHub OIDC federated principal — i.e. the keyless wiring is not in place.
_MISSING_OIDC_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def _run_doctor(tmp_path: Path, trust_policy: object) -> subprocess.CompletedProcess:
    """Run ``gh-openrabbit doctor --trust-policy-file <f>`` offline.

    The ``--trust-policy-file`` flag lets ``doctor`` validate a trust policy
    WITHOUT calling AWS (in real use it would `aws iam get-role`), so the OIDC
    check is exercised deterministically with no network.
    """
    pf = tmp_path / "trust.json"
    pf.write_text(json.dumps(trust_policy), encoding="utf-8")
    path = os.pathsep.join(
        dict.fromkeys([os.path.dirname(bash or "/bin/bash"), *_coreutils_dirs()])
    )
    env = {"PATH": path, "HOME": str(tmp_path)}
    return subprocess.run(
        [bash or "/bin/bash", str(GH_EXT), "doctor", "--trust-policy-file", str(pf)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )


def test_doctor_detects_missing_oidc_trust_policy(tmp_path: Path) -> None:
    """`doctor` must FAIL (non-zero) when the role's trust policy lacks the GitHub
    OIDC federated principal — the keyless wiring would silently not work."""
    proc = _run_doctor(tmp_path, _MISSING_OIDC_TRUST_POLICY)
    assert proc.returncode != 0, (
        "doctor must fail when the OIDC trust policy is missing; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    combined = (proc.stdout + proc.stderr).lower()
    assert "oidc" in combined or "trust policy" in combined, (
        f"doctor must explain the missing OIDC trust policy; got: {combined!r}"
    )


def test_doctor_passes_valid_oidc_trust_policy(tmp_path: Path) -> None:
    """A trust policy WITH the GitHub OIDC federated principal +
    AssumeRoleWithWebIdentity passes `doctor`."""
    proc = _run_doctor(tmp_path, _VALID_TRUST_POLICY)
    assert proc.returncode == 0, (
        f"doctor should pass a valid OIDC trust policy; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


def test_doctor_is_documented_in_help() -> None:
    text = GH_EXT.read_text(encoding="utf-8")
    assert "doctor" in text, "the gh extension must expose a `doctor` subcommand"


# --------------------------------------------------------------------------- #
# ruleset integration_id — must be documented/verifiable (finding #4)          #
# --------------------------------------------------------------------------- #
def test_ruleset_integration_id_is_documented() -> None:
    """The hardcoded `integration_id` in org/ruleset.json must be EXPLAINED (it is
    GitHub Actions' app id) with a verify instruction, not an unexplained magic
    number."""
    ruleset = REPO_ROOT / "org" / "ruleset.json"
    raw = ruleset.read_text(encoding="utf-8")
    data = json.loads(raw)
    # The id is still present and pins the check to GitHub Actions.
    checks = [r for r in data["rules"] if r.get("type") == "required_status_checks"]
    ids = [
        c.get("integration_id")
        for rule in checks
        for c in rule["parameters"]["required_status_checks"]
    ]
    assert 15368 in ids, "ruleset must pin the check to GitHub Actions' integration_id"
    # ...and the _comment block must explain WHAT 15368 is + HOW to verify it.
    comment = " ".join(data.get("_comment", [])).lower()
    assert "15368" in comment, "the integration_id must be explained in _comment"
    assert "github actions" in comment, (
        "the _comment must identify integration_id 15368 as GitHub Actions' app id"
    )
    # A concrete verify instruction (the gh api endpoint that returns the app id).
    assert "/apps/github-actions" in comment or "gh api" in comment, (
        "the _comment must give a verify instruction for the integration_id"
    )


# --------------------------------------------------------------------------- #
# structural invariants                                                        #
# --------------------------------------------------------------------------- #
def test_gh_extension_exists_and_executable() -> None:
    assert GH_EXT.is_file(), "gh extension entrypoint is missing"
    assert os.access(GH_EXT, os.X_OK), "gh extension must be executable"


def test_gh_extension_has_shebang() -> None:
    first = GH_EXT.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("#!"), "gh extension needs a shebang"


def test_gh_extension_guards_network_mutations_behind_apply() -> None:
    text = GH_EXT.read_text(encoding="utf-8")
    assert "--apply" in text
    assert "gh secret set" in text
    assert "advisory-only" in text.lower()


def test_gh_extension_manifest_present_for_install() -> None:
    """`gh extension install` needs a real extension layout: the entrypoint must
    be named ``gh-openrabbit`` and live at the repo root of its install dir.

    The directory containing the entrypoint is what `gh extension install` clones,
    so the entrypoint basename must equal the directory basename (``gh-openrabbit``)
    — otherwise `gh` cannot discover the command."""
    assert GH_EXT.name == "gh-openrabbit"
    assert GH_EXT.parent.name == "gh-openrabbit", (
        "the entrypoint must live in a `gh-openrabbit/` dir so `gh extension "
        "install <repo>` (which expects gh-<name>/gh-<name>) can consume it"
    )
