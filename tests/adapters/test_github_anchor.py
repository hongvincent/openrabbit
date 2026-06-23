"""Tests for diff-anchored comment validation + rate-limit handling (github.py).

NO NETWORK — a fake transport intercepts every request in-process.

Covers the github-anchor bucket findings:

1. [CRITICAL] post_review must drop/clamp findings whose (side, line) is outside
   the real diff (and normalize a swapped start>end) so one hallucinated path
   can't 422 the entire batched review.
2. [HIGH] 403/429 with Retry-After must be retried with bounded backoff instead
   of hard-failing on the first secondary-rate-limit response.
3. [MEDIUM] the comment path must be normalized (strip model-echoed ``b/``) and
   cross-checked against the PR's changed filenames.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from openrabbit.adapters.github import (
    GitHubAdapter,
    GitHubRepo,
    build_review_comment,
)
from openrabbit.findings import Finding
from openrabbit.pipeline import route as route_mod


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload


class RecordedRequest:
    def __init__(self, method: str, url: str, *, json: Any = None) -> None:
        self.method = method
        self.url = url
        self.json = json


class ScriptedClient:
    """Returns a queued sequence of responses per (method, url-substring).

    ``script`` maps a url-substring to a list of responses consumed in order.
    Every call is recorded. An exhausted/unmatched request raises so a stray
    network attempt fails loudly.
    """

    def __init__(self, script: dict[str, list[FakeResponse]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.requests: list[RecordedRequest] = []
        self.sleeps: list[float] = []

    def _handle(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.requests.append(RecordedRequest(method, url, json=kw.get("json")))
        for substr, responses in self.script.items():
            if substr in url and responses:
                return responses.pop(0)
        raise AssertionError(f"unexpected/exhausted request: {method} {url}")

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("POST", url, **kw)

    def patch(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("PATCH", url, **kw)

    def close(self) -> None:
        pass


def make_finding(**overrides: Any) -> Finding:
    defaults: dict[str, Any] = dict(
        file="src/api/auth.py",
        start_line=11,
        end_line=11,
        side="RIGHT",
        severity="high",
        category="correctness",
        confidence=0.9,
        title="t",
        body="b",
        rule_id="openrabbit/correctness/x",
        fingerprint="fp-x",
        suggestion=None,
    )
    defaults.update(overrides)
    return Finding(**defaults)


def make_adapter(client: ScriptedClient, **kw: Any) -> GitHubAdapter:
    # A near-zero sleep keeps retry tests fast while still exercising backoff.
    kw.setdefault("sleep", lambda _s: client.sleeps.append(_s))
    return GitHubAdapter(
        repo=GitHubRepo(owner="acme", repo="widget"),
        pr_number=7,
        token="ghs_tok",
        client=client,
        **kw,
    )


DIFF = """\
diff --git a/src/api/auth.py b/src/api/auth.py
--- a/src/api/auth.py
+++ b/src/api/auth.py
@@ -10,4 +10,5 @@ def login(request):
 ctx_old_10
-removed_old_11
+added_new_11
+added_new_12
 ctx_both_13
"""


def _valid_positions():
    plan = route_mod.route_diff(DIFF, lenses=["correctness"])
    return route_mod.valid_positions_by_file(plan)


# --------------------------------------------------------------------------- #
# CRITICAL: drop out-of-diff findings before posting                          #
# --------------------------------------------------------------------------- #
def test_post_review_drops_out_of_diff_findings():
    """The RED test: one finding anchored OUTSIDE the diff (and one with a
    swapped start>end) must not nuke the batch — the out-of-diff one is dropped,
    the swapped one is clamped, and a valid finding survives and is posted."""
    valid = _valid_positions()

    good = make_finding(
        file="src/api/auth.py", start_line=11, end_line=11, fingerprint="good"
    )
    # line 999 is nowhere in the diff -> would 422 the whole batch.
    hallucinated = make_finding(
        file="src/api/auth.py", start_line=999, end_line=999, fingerprint="halluc"
    )
    # swapped multi-line range (start>end), both endpoints in-diff after clamp.
    swapped = make_finding(
        file="src/api/auth.py", start_line=13, end_line=11, fingerprint="swap"
    )

    resp = FakeResponse(200, {"id": 1})
    client = ScriptedClient({"/reviews": [resp]})
    adapter = make_adapter(client)

    adapter.post_review(
        [good, hallucinated, swapped],
        "summary",
        "sha",
        valid_positions=valid,
    )

    post = [r for r in client.requests if r.method == "POST"][0]
    comments = post.json["comments"]
    bodies = [c["body"] for c in comments]
    # hallucinated dropped, good + swapped(clamped) survive.
    assert any("good" in b for b in bodies)
    assert not any("halluc" in b for b in bodies)
    # the swapped finding survives with a normalized start<=end multi-line anchor.
    swap_comment = next(c for c in comments if "swap" in c["body"])
    assert swap_comment["start_line"] <= swap_comment["line"]


def test_post_review_drops_finding_for_unchanged_file():
    """A finding whose file is not among the PR's changed files is dropped even
    if its line number happens to look plausible."""
    valid = _valid_positions()
    good = make_finding(file="src/api/auth.py", start_line=11, end_line=11)
    wrong_file = make_finding(
        file="src/other/not_in_pr.py",
        start_line=11,
        end_line=11,
        fingerprint="otherfile",
    )
    client = ScriptedClient({"/reviews": [FakeResponse(200, {"id": 2})]})
    adapter = make_adapter(client)
    adapter.post_review(
        [good, wrong_file],
        "summary",
        "sha",
        valid_positions=valid,
        changed_files={"src/api/auth.py"},
    )
    post = [r for r in client.requests if r.method == "POST"][0]
    bodies = [c["body"] for c in post.json["comments"]]
    assert not any("otherfile" in b for b in bodies)


def test_post_review_skips_when_all_findings_filtered_out():
    """If every finding is out-of-diff, NO createReview POST is fired (an empty
    review on a clean-after-filter batch is exactly the noise we avoid)."""
    valid = _valid_positions()
    only_bad = make_finding(start_line=999, end_line=999, fingerprint="bad")
    client = ScriptedClient({"/reviews": [FakeResponse(200, {"id": 9})]})
    adapter = make_adapter(client)
    out = adapter.post_review([only_bad], "summary", "sha", valid_positions=valid)
    assert not any(r.method == "POST" for r in client.requests)
    assert out.get("skipped") is True


def test_post_review_retries_without_rejected_comment_on_422():
    """Belt-and-suspenders: if GitHub still 422s on a comment position, the
    adapter retries the POST without the rejected comment rather than failing."""
    valid = _valid_positions()
    good = make_finding(start_line=11, end_line=11, fingerprint="good")
    # both pass local validation, but GitHub rejects one on the server side.
    other = make_finding(start_line=12, end_line=12, fingerprint="srvbad")
    err = FakeResponse(
        422,
        {
            "message": "Validation Failed",
            "errors": [{"resource": "PullRequestReviewComment", "field": "line"}],
        },
    )
    ok = FakeResponse(200, {"id": 3})
    client = ScriptedClient({"/reviews": [err, ok]})
    adapter = make_adapter(client)
    out = adapter.post_review([good, other], "summary", "sha", valid_positions=valid)
    posts = [r for r in client.requests if r.method == "POST"]
    # one failed POST + one retry POST
    assert len(posts) == 2
    assert out["id"] == 3


# --------------------------------------------------------------------------- #
# MEDIUM: path normalization (strip echoed a/ or b/)                          #
# --------------------------------------------------------------------------- #
def test_build_review_comment_strips_echoed_b_prefix():
    f = make_finding(file="b/src/api/auth.py")
    c = build_review_comment(f)
    assert c["path"] == "src/api/auth.py"


def test_post_review_normalizes_path_against_changed_files():
    """A model-echoed ``b/`` prefix is stripped so the finding matches the PR's
    real changed filename (and therefore survives validation)."""
    valid = _valid_positions()
    f = make_finding(file="b/src/api/auth.py", start_line=11, end_line=11)
    client = ScriptedClient({"/reviews": [FakeResponse(200, {"id": 4})]})
    adapter = make_adapter(client)
    adapter.post_review(
        [f],
        "summary",
        "sha",
        valid_positions=valid,
        changed_files={"src/api/auth.py"},
    )
    post = [r for r in client.requests if r.method == "POST"][0]
    assert post.json["comments"][0]["path"] == "src/api/auth.py"


def test_build_review_comment_clamps_swapped_start_end():
    f = make_finding(start_line=15, end_line=10)
    c = build_review_comment(f)
    assert c["line"] == 15  # the larger endpoint anchors
    assert c["start_line"] == 10
    assert c["start_line"] <= c["line"]


# --------------------------------------------------------------------------- #
# HIGH: rate-limit / secondary-limit handling                                 #
# --------------------------------------------------------------------------- #
def test_get_retries_on_429_with_retry_after():
    """A 429 with Retry-After must be retried (idempotent read), not hard-fail."""
    diff_text = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"
    client = ScriptedClient(
        {
            "/pulls/7": [
                FakeResponse(429, {"message": "rate limited"}, headers={"Retry-After": "0"}),
                FakeResponse(200, diff_text),
            ]
        }
    )
    adapter = make_adapter(client)
    out = adapter.fetch_pr_diff()
    assert out == diff_text
    gets = [r for r in client.requests if r.method == "GET"]
    assert len(gets) == 2  # retried after the 429


def test_get_retries_on_secondary_403_rate_limit():
    """A 403 secondary rate-limit (X-RateLimit-Remaining: 0) is retried too."""
    files_payload = [
        {"filename": "src/a.py", "status": "modified", "additions": 1, "deletions": 0}
    ]
    client = ScriptedClient(
        {
            "/pulls/7/files": [
                FakeResponse(
                    403,
                    {"message": "secondary rate limit"},
                    headers={"Retry-After": "0", "X-RateLimit-Remaining": "0"},
                ),
                FakeResponse(200, files_payload),
            ]
        }
    )
    adapter = make_adapter(client)
    files = adapter.fetch_changed_files()
    assert files[0].filename == "src/a.py"
    assert len([r for r in client.requests if r.method == "GET"]) == 2


def test_post_review_retries_on_429():
    """The single createReview POST is retried on a 429 (it is safe: GitHub
    creates the review only once it succeeds)."""
    client = ScriptedClient(
        {
            "/reviews": [
                FakeResponse(429, {"message": "rate limited"}, headers={"Retry-After": "0"}),
                FakeResponse(200, {"id": 7}),
            ]
        }
    )
    adapter = make_adapter(client)
    out = adapter.post_review([make_finding()], "summary", "sha")
    assert out["id"] == 7
    assert len([r for r in client.requests if r.method == "POST"]) == 2


def test_get_gives_up_after_bounded_retries():
    """Retries are bounded — a persistently rate-limited endpoint eventually
    raises GitHubError rather than looping forever."""
    client = ScriptedClient(
        {
            "/pulls/7": [
                FakeResponse(429, {"message": "rl"}, headers={"Retry-After": "0"})
                for _ in range(10)
            ]
        }
    )
    adapter = make_adapter(client, max_retries=3)
    # Catch by class *name* rather than identity: another test in the suite
    # reloads openrabbit.adapters.github (lazy-import discipline check), which
    # rebinds GitHubError to a fresh class object — a stale module-level import
    # would no longer match the raised instance. The behavior under test
    # (bounded retries, then raise) is still fully asserted.
    with pytest.raises(Exception) as ei:
        adapter.fetch_pr_diff()
    assert type(ei.value).__name__ == "GitHubError"
    # bounded: initial try + at most max_retries follow-ups
    assert len([r for r in client.requests if r.method == "GET"]) <= 4
