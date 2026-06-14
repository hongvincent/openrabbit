"""Tests for the GitHub output adapter (SPEC section 6 step 6 + section 9).

NO NETWORK. ``httpx`` is imported lazily inside the adapter; these tests inject a
fully fake transport so every REST and GraphQL request is intercepted in-process.
The fakes assert on method/URL/headers/json so we verify the *exact* wire shape:

- one batched ``POST /pulls/{n}/reviews`` with ``comments[]`` (path/line/side,
  multi-line ``start_line``/``start_side``, committable ```suggestion``` blocks),
- sticky walkthrough upsert (create when absent, update when a bot comment
  exists),
- GraphQL ``resolveReviewThread`` / ``minimizeComment(OUTDATED)`` payloads,
- bot review-thread listing + dedup-by-fingerprint.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from openrabbit.adapters.github import (
    BOT_STICKY_MARKER,
    ChangedFile,
    GitHubAdapter,
    GitHubError,
    GitHubRepo,
    ReviewThread,
    build_review_comment,
    render_suggestion_block,
)
from openrabbit.findings import Finding


# --------------------------------------------------------------------------- #
# Fake httpx transport                                                         #
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Minimal stand-in for ``httpx.Response`` used by the adapter."""

    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def json(self) -> Any:
        return self._payload

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if not self.is_success:
            raise RuntimeError(f"HTTP {self.status_code}")


class RecordedRequest:
    def __init__(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.method = method
        self.url = url
        self.json = json
        self.headers = headers or {}


class FakeClient:
    """A fake httpx.Client. Returns scripted responses keyed by (method, url-substr).

    ``routes`` is a list of ``(method, url_substr, response)``. The first route
    whose method matches and whose substring is in the URL handles the request.
    Each call is recorded for assertions. Unmatched requests raise AssertionError
    so a stray network attempt fails loudly instead of silently hitting the wire.
    """

    def __init__(self, routes: list[tuple[str, str, FakeResponse]]) -> None:
        self.routes = routes
        self.requests: list[RecordedRequest] = []
        self.closed = False

    def _handle(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.requests.append(
            RecordedRequest(method, url, json=kw.get("json"), headers=kw.get("headers"))
        )
        for m, substr, resp in self.routes:
            if m == method and substr in url:
                return resp
        raise AssertionError(f"unexpected request: {method} {url}")

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("POST", url, **kw)

    def patch(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("PATCH", url, **kw)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def make_finding(**overrides: Any) -> Finding:
    defaults: dict[str, Any] = dict(
        file="src/agent.py",
        start_line=42,
        end_line=42,
        side="RIGHT",
        severity="high",
        category="correctness",
        confidence=0.88,
        title="Unvalidated index can raise IndexError",
        body="The index `i` is not bounds-checked before use.",
        rule_id="openrabbit/correctness/bounds-check",
        fingerprint="fp-aaa",
        suggestion=None,
    )
    defaults.update(overrides)
    return Finding(**defaults)


def make_adapter(client: FakeClient, **kw: Any) -> GitHubAdapter:
    return GitHubAdapter(
        repo=GitHubRepo(owner="acme", repo="widget"),
        pr_number=7,
        token="ghs_faketoken",
        client=client,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Pure helpers: suggestion block + comment building                            #
# --------------------------------------------------------------------------- #
def test_render_suggestion_block_wraps_in_fence():
    block = render_suggestion_block("x = clamp(i)")
    assert block.startswith("```suggestion\n")
    assert block.rstrip().endswith("```")
    assert "x = clamp(i)" in block


def test_render_suggestion_block_handles_multiline():
    block = render_suggestion_block("a = 1\nb = 2")
    assert "a = 1\nb = 2" in block
    # exactly one opening and closing fence
    assert block.count("```") == 2


def test_build_review_comment_single_line_uses_line_and_side():
    f = make_finding(start_line=10, end_line=10, side="RIGHT")
    c = build_review_comment(f)
    assert c["path"] == "src/agent.py"
    assert c["line"] == 10
    assert c["side"] == "RIGHT"
    # single-line: no multi-line keys
    assert "start_line" not in c
    assert "start_side" not in c
    assert f.title in c["body"]


def test_build_review_comment_multi_line_adds_start_keys():
    f = make_finding(start_line=10, end_line=15, side="RIGHT")
    c = build_review_comment(f)
    assert c["line"] == 15  # end line is the anchor
    assert c["side"] == "RIGHT"
    assert c["start_line"] == 10
    assert c["start_side"] == "RIGHT"


def test_build_review_comment_left_side():
    f = make_finding(start_line=3, end_line=3, side="LEFT")
    c = build_review_comment(f)
    assert c["side"] == "LEFT"


def test_render_suggestion_block_strips_existing_fence():
    """Defensive: if the model already wrapped the code in a ```suggestion```
    fence, the adapter must not double-fence it (GitHub would not render it)."""
    already_fenced = "```suggestion\nx = clamp(i)\n```"
    block = render_suggestion_block(already_fenced)
    # exactly one fence pair, and the code survives intact
    assert block == "```suggestion\nx = clamp(i)\n```"
    assert block.count("```suggestion") == 1
    assert block.count("```") == 2


def test_render_suggestion_block_strips_multiline_existing_fence():
    already_fenced = "```suggestion\na = 1\nb = 2\n```"
    block = render_suggestion_block(already_fenced)
    assert block == "```suggestion\na = 1\nb = 2\n```"
    assert block.count("```suggestion") == 1


def test_build_review_comment_embeds_committable_suggestion():
    f = make_finding(suggestion="x = clamp(i)")
    c = build_review_comment(f)
    assert "```suggestion" in c["body"]
    assert "x = clamp(i)" in c["body"]
    # exactly ONE committable suggestion fence (never a nested double fence)
    assert c["body"].count("```suggestion") == 1


def test_build_review_comment_no_double_fence_when_model_rewraps():
    """End-to-end: a model-shaped finding whose suggestion field still contains a
    fence must yield exactly one suggestion fence in the comment body."""
    f = make_finding(suggestion="```suggestion\nx = 1\n```")
    c = build_review_comment(f)
    assert c["body"].count("```suggestion") == 1
    assert "x = 1" in c["body"]


def test_build_review_comment_no_suggestion_has_no_fence():
    f = make_finding(suggestion=None)
    c = build_review_comment(f)
    assert "```suggestion" not in c["body"]


# --------------------------------------------------------------------------- #
# fetch PR diff / changed files                                                #
# --------------------------------------------------------------------------- #
def test_fetch_pr_diff_requests_diff_media_type():
    diff_text = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"
    client = FakeClient(
        [("GET", "/repos/acme/widget/pulls/7", FakeResponse(200, diff_text))]
    )
    adapter = make_adapter(client)
    out = adapter.fetch_pr_diff()
    assert out == diff_text
    req = client.requests[0]
    assert req.method == "GET"
    assert "/repos/acme/widget/pulls/7" in req.url
    assert req.headers.get("Accept") == "application/vnd.github.v3.diff"


def test_fetch_changed_files_parses_into_dataclasses():
    files_payload = [
        {
            "filename": "src/a.py",
            "status": "modified",
            "additions": 3,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        },
        {
            "filename": "src/b.py",
            "status": "added",
            "additions": 9,
            "deletions": 0,
            "patch": "@@ -0,0 +1,9 @@",
        },
    ]
    client = FakeClient(
        [("GET", "/repos/acme/widget/pulls/7/files", FakeResponse(200, files_payload))]
    )
    adapter = make_adapter(client)
    files = adapter.fetch_changed_files()
    assert all(isinstance(f, ChangedFile) for f in files)
    assert files[0].filename == "src/a.py"
    assert files[0].status == "modified"
    assert files[1].additions == 9


def test_fetch_raises_githuberror_on_http_error():
    client = FakeClient(
        [
            (
                "GET",
                "/repos/acme/widget/pulls/7",
                FakeResponse(404, {"message": "Not Found"}),
            )
        ]
    )
    adapter = make_adapter(client)
    with pytest.raises(GitHubError):
        adapter.fetch_pr_diff()


# --------------------------------------------------------------------------- #
# post_review: ONE batched createReview call                                   #
# --------------------------------------------------------------------------- #
def test_post_review_single_post_with_batched_comments():
    findings = [
        make_finding(file="src/a.py", start_line=1, end_line=1, fingerprint="f1"),
        make_finding(
            file="src/b.py",
            start_line=5,
            end_line=9,
            fingerprint="f2",
            suggestion="ok = True",
        ),
    ]
    resp = FakeResponse(200, {"id": 555, "state": "COMMENTED"})
    client = FakeClient([("POST", "/repos/acme/widget/pulls/7/reviews", resp)])
    adapter = make_adapter(client)

    result = adapter.post_review(findings, "## Summary\nLooks ok", "deadbeef")

    # exactly one POST
    posts = [r for r in client.requests if r.method == "POST"]
    assert len(posts) == 1
    body = posts[0].json
    assert body["commit_id"] == "deadbeef"
    assert body["event"] == "COMMENT"  # advisory default
    assert body["body"] == "## Summary\nLooks ok"
    assert len(body["comments"]) == 2
    # first comment single-line
    assert body["comments"][0]["path"] == "src/a.py"
    assert body["comments"][0]["line"] == 1
    # second comment multi-line with suggestion
    assert body["comments"][1]["start_line"] == 5
    assert body["comments"][1]["line"] == 9
    assert "```suggestion" in body["comments"][1]["body"]
    assert result["id"] == 555


def test_post_review_never_approves_or_requests_changes():
    resp = FakeResponse(200, {"id": 1})
    client = FakeClient([("POST", "/repos/acme/widget/pulls/7/reviews", resp)])
    adapter = make_adapter(client)
    # even if a caller passes a forbidden event, adapter is advisory-only
    with pytest.raises(GitHubError):
        adapter.post_review([], "summary", "sha", event="APPROVE")
    with pytest.raises(GitHubError):
        adapter.post_review([], "summary", "sha", event="REQUEST_CHANGES")


def test_post_review_allows_explicit_comment_event():
    resp = FakeResponse(200, {"id": 2})
    client = FakeClient([("POST", "/repos/acme/widget/pulls/7/reviews", resp)])
    adapter = make_adapter(client)
    out = adapter.post_review([make_finding()], "s", "sha", event="COMMENT")
    assert out["id"] == 2


def test_post_review_sends_auth_header():
    resp = FakeResponse(200, {"id": 9})
    client = FakeClient([("POST", "/repos/acme/widget/pulls/7/reviews", resp)])
    adapter = make_adapter(client)
    adapter.post_review([make_finding()], "s", "sha")
    req = [r for r in client.requests if r.method == "POST"][0]
    assert req.headers.get("Authorization") == "Bearer ghs_faketoken"


def test_post_review_empty_findings_posts_summary_only():
    resp = FakeResponse(200, {"id": 3})
    client = FakeClient([("POST", "/repos/acme/widget/pulls/7/reviews", resp)])
    adapter = make_adapter(client)
    out = adapter.post_review([], "just a summary", "sha")
    body = [r for r in client.requests if r.method == "POST"][0].json
    assert body["comments"] == []
    assert body["body"] == "just a summary"
    assert out["id"] == 3


# --------------------------------------------------------------------------- #
# sticky walkthrough upsert                                                    #
# --------------------------------------------------------------------------- #
def test_sticky_walkthrough_creates_when_absent():
    list_resp = FakeResponse(200, [])  # no existing issue comments
    create_resp = FakeResponse(201, {"id": 101, "body": "x"})
    client = FakeClient(
        [
            ("GET", "/repos/acme/widget/issues/7/comments", list_resp),
            ("POST", "/repos/acme/widget/issues/7/comments", create_resp),
        ]
    )
    adapter = make_adapter(client)
    out = adapter.upsert_sticky_walkthrough("## Walkthrough\nbody")

    methods = [(r.method, r.url) for r in client.requests]
    assert any(m == "GET" for m, _ in methods)
    post = [r for r in client.requests if r.method == "POST"][0]
    assert BOT_STICKY_MARKER in post.json["body"]
    assert "## Walkthrough" in post.json["body"]
    assert out["id"] == 101


def test_sticky_walkthrough_updates_existing_bot_comment():
    existing = [
        {"id": 50, "body": "unrelated human comment"},
        {"id": 77, "body": f"{BOT_STICKY_MARKER}\nold walkthrough"},
    ]
    list_resp = FakeResponse(200, existing)
    update_resp = FakeResponse(200, {"id": 77, "body": "new"})
    client = FakeClient(
        [
            ("GET", "/repos/acme/widget/issues/7/comments", list_resp),
            ("PATCH", "/repos/acme/widget/issues/comments/77", update_resp),
        ]
    )
    adapter = make_adapter(client)
    out = adapter.upsert_sticky_walkthrough("## Walkthrough\nfresh")

    # no create, exactly one PATCH on comment 77
    assert not any(r.method == "POST" for r in client.requests)
    patch = [r for r in client.requests if r.method == "PATCH"][0]
    assert "/issues/comments/77" in patch.url
    assert BOT_STICKY_MARKER in patch.json["body"]
    assert "fresh" in patch.json["body"]
    assert out["id"] == 77


def test_sticky_walkthrough_marker_is_hidden_html_comment():
    # The marker must be an HTML comment so it is invisible in rendered markdown.
    assert BOT_STICKY_MARKER.startswith("<!--")
    assert BOT_STICKY_MARKER.endswith("-->")


# --------------------------------------------------------------------------- #
# GraphQL: resolve thread + minimize comment                                   #
# --------------------------------------------------------------------------- #
def test_resolve_review_thread_graphql_payload():
    resp = FakeResponse(
        200, {"data": {"resolveReviewThread": {"thread": {"isResolved": True}}}}
    )
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client)
    ok = adapter.resolve_review_thread("THREAD_abc")
    req = [r for r in client.requests if r.method == "POST"][0]
    assert "/graphql" in req.url
    assert "resolveReviewThread" in req.json["query"]
    assert req.json["variables"]["threadId"] == "THREAD_abc"
    assert ok is True


def test_minimize_comment_uses_outdated_classifier():
    resp = FakeResponse(
        200, {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}}
    )
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client)
    ok = adapter.minimize_comment("COMMENT_xyz", "OUTDATED")
    req = [r for r in client.requests if r.method == "POST"][0]
    assert "minimizeComment" in req.json["query"]
    assert req.json["variables"]["subjectId"] == "COMMENT_xyz"
    assert req.json["variables"]["classifier"] == "OUTDATED"
    assert ok is True


def test_minimize_comment_defaults_to_outdated():
    resp = FakeResponse(
        200, {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}}
    )
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client)
    adapter.minimize_comment("COMMENT_1")
    req = [r for r in client.requests if r.method == "POST"][0]
    assert req.json["variables"]["classifier"] == "OUTDATED"


def test_graphql_raises_on_errors_field():
    resp = FakeResponse(200, {"errors": [{"message": "Resource not accessible"}]})
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client)
    with pytest.raises(GitHubError):
        adapter.resolve_review_thread("THREAD_abc")


# --------------------------------------------------------------------------- #
# list bot review threads + dedup-by-fingerprint                               #
# --------------------------------------------------------------------------- #
def _threads_graphql_payload(threads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": threads,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
    }


def _thread_node(
    thread_id: str,
    fingerprint: Optional[str],
    *,
    resolved: bool = False,
    outdated: bool = False,
    login: str = "github-actions[bot]",
) -> dict[str, Any]:
    body = "some review comment"
    if fingerprint:
        body += f"\n<!-- openrabbit:fp={fingerprint} -->"
    return {
        "id": thread_id,
        "isResolved": resolved,
        "isOutdated": outdated,
        "comments": {
            "nodes": [
                {
                    "id": f"{thread_id}-c0",
                    "body": body,
                    "author": {"login": login},
                    "path": "src/a.py",
                }
            ]
        },
    }


def test_list_bot_review_threads_parses_state_and_fingerprint():
    nodes = [
        _thread_node("T1", "fp-aaa"),
        _thread_node("T2", "fp-bbb", resolved=True, outdated=True),
    ]
    resp = FakeResponse(200, _threads_graphql_payload(nodes))
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client)
    threads = adapter.list_bot_review_threads()
    assert all(isinstance(t, ReviewThread) for t in threads)
    by_id = {t.thread_id: t for t in threads}
    assert by_id["T1"].fingerprint == "fp-aaa"
    assert by_id["T1"].is_resolved is False
    assert by_id["T2"].is_resolved is True
    assert by_id["T2"].is_outdated is True


def test_list_bot_review_threads_filters_non_bot_authors():
    nodes = [
        _thread_node("T1", "fp-aaa", login="github-actions[bot]"),
        _thread_node("T2", "fp-bbb", login="some-human"),
    ]
    resp = FakeResponse(200, _threads_graphql_payload(nodes))
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client, bot_login="github-actions[bot]")
    threads = adapter.list_bot_review_threads()
    ids = {t.thread_id for t in threads}
    assert ids == {"T1"}


def test_dedup_findings_drops_already_posted_fingerprints():
    posted = [
        ReviewThread(
            thread_id="T1",
            comment_id="c1",
            fingerprint="fp-keep-open",
            is_resolved=False,
            is_outdated=False,
            path="src/a.py",
            body="b",
        ),
        ReviewThread(
            thread_id="T2",
            comment_id="c2",
            fingerprint="fp-resolved",
            is_resolved=True,
            is_outdated=False,
            path="src/a.py",
            body="b",
        ),
    ]
    findings = [
        make_finding(fingerprint="fp-keep-open"),  # already posted & open -> drop
        make_finding(
            fingerprint="fp-resolved"
        ),  # posted but resolved -> keep? no: stay suppressed
        make_finding(fingerprint="fp-new"),  # brand new -> keep
    ]
    adapter = make_adapter(FakeClient([]))
    fresh = adapter.dedup_findings(findings, posted)
    fps = {f.fingerprint for f in fresh}
    assert "fp-new" in fps
    assert "fp-keep-open" not in fps  # don't repost an open finding
    # resolved/superseded findings should not be reposted either
    assert "fp-resolved" not in fps


def test_dedup_findings_keeps_all_when_no_prior_threads():
    findings = [make_finding(fingerprint="a"), make_finding(fingerprint="b")]
    adapter = make_adapter(FakeClient([]))
    fresh = adapter.dedup_findings(findings, [])
    assert {f.fingerprint for f in fresh} == {"a", "b"}


def test_outdated_threads_for_fingerprints_selects_superseded():
    posted = [
        ReviewThread(
            thread_id="T1",
            comment_id="c1",
            fingerprint="fp-gone",
            is_resolved=False,
            is_outdated=False,
            path="src/a.py",
            body="b",
        ),
        ReviewThread(
            thread_id="T2",
            comment_id="c2",
            fingerprint="fp-still",
            is_resolved=False,
            is_outdated=False,
            path="src/a.py",
            body="b",
        ),
    ]
    current = [make_finding(fingerprint="fp-still")]
    adapter = make_adapter(FakeClient([]))
    stale = adapter.stale_threads(current, posted)
    stale_fps = {t.fingerprint for t in stale}
    assert stale_fps == {"fp-gone"}


class SequencedClient(FakeClient):
    """Like FakeClient but returns scripted responses for repeated same-route hits.

    ``sequence`` maps a url-substring to a list of responses consumed in order;
    falls back to ``routes`` for anything not in the sequence.
    """

    def __init__(self, sequence: dict[str, list[FakeResponse]]) -> None:
        super().__init__(routes=[])
        self.sequence = {k: list(v) for k, v in sequence.items()}

    def _handle(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.requests.append(
            RecordedRequest(method, url, json=kw.get("json"), headers=kw.get("headers"))
        )
        for substr, responses in self.sequence.items():
            if substr in url and responses:
                return responses.pop(0)
        raise AssertionError(f"unexpected/exhausted request: {method} {url}")


def test_list_bot_review_threads_paginates():
    page1 = _threads_graphql_payload([_thread_node("T1", "fp-1")])
    # mark page1 as having a next page
    page1["data"]["repository"]["pullRequest"]["reviewThreads"]["pageInfo"] = {
        "hasNextPage": True,
        "endCursor": "CURSOR_1",
    }
    page2 = _threads_graphql_payload([_thread_node("T2", "fp-2")])
    client = SequencedClient(
        {"/graphql": [FakeResponse(200, page1), FakeResponse(200, page2)]}
    )
    adapter = make_adapter(client)
    threads = adapter.list_bot_review_threads()
    assert {t.thread_id for t in threads} == {"T1", "T2"}
    # second graphql call passed the cursor from page 1
    second = [r for r in client.requests if r.method == "POST"][1]
    assert second.json["variables"]["cursor"] == "CURSOR_1"


def test_list_bot_review_threads_skips_threads_without_comments():
    node = {
        "id": "T0",
        "isResolved": False,
        "isOutdated": False,
        "comments": {"nodes": []},
    }
    resp = FakeResponse(200, _threads_graphql_payload([node]))
    client = FakeClient([("POST", "/graphql", resp)])
    adapter = make_adapter(client, bot_login="github-actions[bot]")
    # no author login -> filtered out when bot_login is set
    assert adapter.list_bot_review_threads() == []


def test_error_detail_falls_back_to_text_when_json_fails():
    class BadJsonResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    bad = BadJsonResponse(500, "server boom")
    bad.text = "server boom"
    client = FakeClient([("GET", "/repos/acme/widget/pulls/7", bad)])
    adapter = make_adapter(client)
    with pytest.raises(GitHubError) as ei:
        adapter.fetch_pr_diff()
    assert "500" in str(ei.value)


def test_error_detail_is_bounded_and_single_line():
    """A huge/multiline (possibly untrusted-echoing) error body is truncated and
    collapsed to one line so it cannot flood CI logs."""
    long_msg = "x" * 5000 + "\nsecond line"
    resp = FakeResponse(422, {"message": long_msg})
    client = FakeClient([("GET", "/repos/acme/widget/pulls/7", resp)])
    adapter = make_adapter(client)
    with pytest.raises(GitHubError) as ei:
        adapter.fetch_pr_diff()
    msg = str(ei.value)
    # bounded length (cap + small overhead for the prefix + ellipsis)
    assert len(msg) < 300
    assert "422" in msg
    assert "\n" not in msg


# --------------------------------------------------------------------------- #
# Lazy httpx client construction + close (real httpx, no network via respx)    #
# --------------------------------------------------------------------------- #
def test_lazy_client_constructs_and_closes_with_respx():
    respx = pytest.importorskip("respx")
    import httpx

    adapter = GitHubAdapter(
        repo=GitHubRepo(owner="acme", repo="widget"),
        pr_number=7,
        token="ghs_x",
        client=None,  # force lazy httpx.Client construction
    )
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.get("/repos/acme/widget/pulls/7").mock(
            return_value=httpx.Response(200, text="diff-text")
        )
        out = adapter.fetch_pr_diff()
    assert out == "diff-text"
    # an owned client exists now; close() tears it down
    assert adapter._owned_client is not None
    adapter.close()
    assert adapter._owned_client is None


def test_close_is_noop_without_owned_client():
    adapter = make_adapter(FakeClient([]))
    adapter.close()  # should not raise


# --------------------------------------------------------------------------- #
# Lazy import discipline                                                       #
# --------------------------------------------------------------------------- #
def test_module_imports_without_httpx(monkeypatch):
    # Importing the adapter module must not require httpx at import time.
    import builtins
    import importlib

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "httpx" or name.startswith("httpx."):
            raise AssertionError("httpx imported at module load time")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    import openrabbit.adapters.github as mod

    importlib.reload(mod)
    assert hasattr(mod, "GitHubAdapter")
