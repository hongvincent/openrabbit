"""END-TO-END: the CI base ref actually RESOLVES from a CI-shaped git checkout.

Companion to ``tests/ci/test_base_ref_wiring.py`` (which parses the workflow yaml)
and to the unit boundary tests in ``tests/test_cli_config_trust_boundary.py``
(which inject ``base`` directly and never touch git). Those unit tests pass even
when the workflow never makes the base ref resolvable — the exact wiring gap this
bucket fixes. These tests build a REAL git repo, reproduce the CI checkout
(detached HEAD on the PR head commit) + the workflow's ``git fetch origin <base>``
step, and drive the REAL ``cli._load_base_config`` call site against the SAME
``origin/<base>`` value the workflow passes to ``--base-ref``.

Offline: a local file:// clone, no network, lazy ``git`` only.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from openrabbit import cli
from openrabbit.config import load_config


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "PATH": os.environ.get("PATH", ""),
        },
    )


def _make_ci_shaped_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a base repo + a CI-style consumer clone.

    Returns ``(consumer_repo, base_branch_name, head_sha)``. The consumer repo is
    left in the exact CI state: detached HEAD on the PR head commit, with the base
    branch NOT yet fetched (so the test controls whether the fetch happened).
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    # Base commit on main carries a STRICT policy config.
    (origin / ".openrabbit.yaml").write_text(
        "version: 1\nreview:\n  confidence_gate: 0.80\n  lenses: [correctness, security]\n",
        encoding="utf-8",
    )
    _git(origin, "add", ".openrabbit.yaml")
    _git(origin, "commit", "-q", "-m", "base policy")
    # PR head on a feature branch WEAKENS the policy in the working tree.
    _git(origin, "checkout", "-q", "-b", "feature")
    (origin / ".openrabbit.yaml").write_text(
        "version: 1\nreview:\n  confidence_gate: 0.99\n  lenses: [maintainability]\n",
        encoding="utf-8",
    )
    _git(origin, "add", ".openrabbit.yaml")
    _git(origin, "commit", "-q", "-m", "weaken policy on head")
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(origin),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Consumer clone reproduces the CI checkout (actions/checkout with
    # ``ref: <head sha>``): detached HEAD on the head commit, with NO local or
    # remote-tracking base branch until the workflow's fetch runs. A plain
    # ``git clone`` would create a local ``main`` branch + ``origin/main``, which a
    # single-commit CI checkout never does, so we delete both to mirror CI
    # faithfully — otherwise a bare ``main`` would spuriously resolve and mask the
    # need for the fetch.
    consumer = tmp_path / "consumer"
    _git(tmp_path, "clone", "-q", "--no-checkout", str(origin), str(consumer))
    _git(consumer, "fetch", "-q", "origin", head_sha)
    _git(consumer, "checkout", "-q", head_sha)  # detached HEAD on PR head
    for ref in ("refs/heads/main", "refs/remotes/origin/main"):
        subprocess.run(
            ["git", "update-ref", "-d", ref],
            cwd=str(consumer),
            capture_output=True,
            text=True,
            check=False,
        )
    return consumer, "main", head_sha


def _fetch_base(consumer: Path, base: str) -> None:
    """Run the SAME fetch the workflows now perform."""
    _git(
        consumer,
        "fetch",
        "--no-tags",
        "--depth=1",
        "origin",
        f"+refs/heads/{base}:refs/remotes/origin/{base}",
    )


def test_base_ref_unresolvable_before_workflow_fetch(tmp_path):
    """RED-anchor: on the raw CI checkout the base ref does NOT resolve.

    Before the workflow's ``git fetch origin <base>`` runs, ``origin/main`` is
    absent on the detached-HEAD checkout, so ``_load_base_config('origin/main')``
    returns ``None`` — the exact silent degradation the wiring fix prevents.
    """
    consumer, base, _ = _make_ci_shaped_repo(tmp_path)
    assert cli._load_base_config(f"origin/{base}", consumer) is None


def test_base_ref_resolves_after_workflow_fetch(tmp_path):
    """GREEN: the workflow's fetch makes ``origin/<base>`` resolvable end-to-end.

    Run the SAME ``git fetch origin <base>`` the workflows now perform, then drive
    ``_load_base_config`` with the SAME ``origin/<base>`` value the workflows pass
    to ``--base-ref``. The STRICT base policy must load from the base ref even
    though the working tree (PR head) carries a WEAKENED config.
    """
    consumer, base, _ = _make_ci_shaped_repo(tmp_path)
    _fetch_base(consumer, base)

    base_config = cli._load_base_config(f"origin/{base}", consumer)
    assert base_config is not None, (
        "base config must resolve from origin/<base> after the workflow fetch"
    )
    # It is the STRICT base policy, not the head's weakened one.
    assert base_config.review.confidence_gate == pytest.approx(0.80)
    assert "security" in base_config.review.lenses

    # And the full boundary anchors the (weakened) head to this trusted base.
    head = load_config(consumer / ".openrabbit.yaml")
    assert head.review.confidence_gate == pytest.approx(0.99)  # head is weakened
    resolved = cli._apply_policy_trust_boundary(head, base_config)
    assert resolved.review.confidence_gate == pytest.approx(0.80)
    assert "security" in resolved.review.lenses


def test_bare_base_ref_resolves_via_origin_fallback_on_detached_head(tmp_path):
    """A BARE base ref fails CLOSED (resolves the trusted base) on detached HEAD.

    ``cli._load_base_config`` retries ``origin/<base>`` / ``refs/remotes/origin/<base>``
    when the bare name has no local branch, so a detached-HEAD CI checkout does
    NOT silently fall back to the untrusted PR-head config. The workflow ships the
    explicit ``origin/<base>`` form (most robust); this fallback is the
    belt-and-suspenders. Both forms must resolve to the strict base.
    """
    consumer, base, _ = _make_ci_shaped_repo(tmp_path)
    _fetch_base(consumer, base)

    # Bare branch name: no local `main` branch exists on the detached checkout,
    # but the origin/<base> fallback resolves it (fail closed, not fail open).
    bare = cli._load_base_config(base, consumer)
    assert bare is not None
    assert bare.review.confidence_gate == pytest.approx(0.80)
    assert "security" in bare.review.lenses
    # ...and the origin/<base> form the workflow actually passes resolves too.
    assert cli._load_base_config(f"origin/{base}", consumer) is not None
