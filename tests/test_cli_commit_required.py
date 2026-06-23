"""CLI ``--post`` requires an explicit ``--commit`` (cli-security-demo finding 2).

Omitting ``--commit`` on the online ``--post`` path used to send a literal
``str(None) == "None"`` as the review commit id, which GitHub rejects with a 422
(and, worse, silently mislabels the review). The CLI must fail fast with a clear
message instead of letting the string ``"None"`` reach the adapter.

Offline-safe: the command must error BEFORE any GitHub/Bedrock call, so no token
or network is required for this assertion.
"""

from __future__ import annotations

import pytest

from openrabbit import cli


def test_post_without_commit_errors(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    # Guard: if the bug regresses, the adapter would be constructed and a network
    # call attempted. Make any GitHubAdapter construction explode so a regression
    # surfaces as a hard failure rather than a silent 'None' commit id.
    def _boom(*args, **kwargs):  # pragma: no cover - only hit on regression
        raise AssertionError("GitHubAdapter must not be built when --commit is missing")

    monkeypatch.setattr("openrabbit.adapters.github.GitHubAdapter", _boom)

    rc = cli.main(["review", "--repo", "acme/repo", "--pr", "7", "--post"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "commit" in err
    # The literal mislabeled string must never be presented to the user/adapter.
    assert "none" not in err.replace("commit", "")


def test_post_with_commit_is_accepted_by_arg_validation(monkeypatch):
    # Sanity: supplying --commit clears the new guard (it must not reject a real
    # commit id). We stop before any network by failing token resolution earlier
    # is not possible here (token is set), so we patch the adapter to a no-op that
    # records it was reached with a real commit id.
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    seen: dict[str, object] = {}

    class _FakeAdapter:
        def __init__(self, repo, pr, token, **kwargs):
            seen["built"] = True

        def fetch_pr_diff(self):
            raise SystemExit("stop-after-validation")

        def close(self):
            pass

    monkeypatch.setattr("openrabbit.adapters.github.GitHubAdapter", _FakeAdapter)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "review",
                "--repo",
                "acme/repo",
                "--pr",
                "7",
                "--commit",
                "deadbeef",
                "--post",
            ]
        )
    assert seen.get("built") is True
