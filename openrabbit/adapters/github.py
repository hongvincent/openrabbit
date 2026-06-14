"""GitHub output adapter — REST + GraphQL (SPEC section 6 step 6 + section 9).

The adapter turns structured :class:`~openrabbit.findings.Finding` objects into
GitHub review surface:

* :meth:`GitHubAdapter.fetch_pr_diff` / :meth:`~GitHubAdapter.fetch_changed_files`
  — read PR inputs over the REST API.
* :meth:`GitHubAdapter.post_review` — ONE ``POST /repos/{o}/{r}/pulls/{n}/reviews``
  carrying every inline comment (path / line / side, multi-line
  ``start_line`` + ``start_side``, committable ```suggestion``` blocks). The
  ``event`` defaults to ``COMMENT`` and the adapter REFUSES ``APPROVE`` /
  ``REQUEST_CHANGES`` / ``DISMISS`` / merge — it is **advisory-only**.
* :meth:`GitHubAdapter.upsert_sticky_walkthrough` — find+update a single bot
  issue comment (summary + grouped changed-files table + Mermaid), else create.
* :meth:`GitHubAdapter.resolve_review_thread` /
  :meth:`~GitHubAdapter.minimize_comment` — GraphQL helpers to resolve a fixed
  thread and minimize a superseded comment as ``OUTDATED``.
* :meth:`GitHubAdapter.list_bot_review_threads` — list the bot's prior review
  threads (with ``isResolved`` / ``isOutdated`` + embedded fingerprint) so
  :meth:`~GitHubAdapter.dedup_findings` / :meth:`~GitHubAdapter.stale_threads`
  can dedup incrementally.

Security posture (SPEC section 12): the adapter holds no merge/approve/push
ability; any PR text it reads is UNTRUSTED data, never instructions. ``httpx`` is
imported lazily inside :meth:`GitHubAdapter._client` so importing this module —
and every unit test that injects a fake client — needs ZERO external deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from openrabbit.findings import Finding

# Default GitHub endpoints. Overridable for GitHub Enterprise via the adapter
# constructor; tests inject a fake client so these are never dialed in unit tests.
DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_GRAPHQL_URL = "https://api.github.com/graphql"

# Hidden HTML-comment marker that identifies the bot's single sticky walkthrough
# comment so we can find+update it instead of spamming a new one each run. It is
# an HTML comment so it renders invisibly in the GitHub UI.
BOT_STICKY_MARKER = "<!-- openrabbit:sticky-walkthrough -->"

# Per-finding fingerprint marker embedded (hidden) in each inline comment body so
# prior threads can be matched back to the finding that created them.
_FP_MARKER_RE = re.compile(r"<!--\s*openrabbit:fp=([^\s>]+)\s*-->")

# Review events. Only COMMENT (and the empty/PENDING form) are allowed — the
# adapter must never approve, request changes, or dismiss (advisory-only).
_ALLOWED_EVENTS = frozenset({"COMMENT"})
_FORBIDDEN_EVENTS = frozenset({"APPROVE", "REQUEST_CHANGES", "DISMISS"})


class GitHubError(Exception):
    """Raised for any GitHub adapter failure (HTTP error, GraphQL error, misuse)."""


@dataclass(frozen=True)
class GitHubRepo:
    """An ``owner/repo`` coordinate."""

    owner: str
    repo: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True)
class ChangedFile:
    """One entry from ``GET /pulls/{n}/files``."""

    filename: str
    status: str
    additions: int
    deletions: int
    patch: Optional[str] = None


@dataclass(frozen=True)
class ReviewThread:
    """A prior bot review thread, with dedup metadata.

    ``fingerprint`` is the openrabbit finding fingerprint parsed from the hidden
    marker in the first comment body (``None`` if absent). ``is_resolved`` /
    ``is_outdated`` mirror GitHub's GraphQL thread state.
    """

    thread_id: str
    comment_id: Optional[str]
    fingerprint: Optional[str]
    is_resolved: bool
    is_outdated: bool
    path: Optional[str]
    body: str


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O) — easy to unit test, reused by the adapter            #
# --------------------------------------------------------------------------- #
def render_suggestion_block(suggestion: str) -> str:
    """Wrap ``suggestion`` in a GitHub committable ```suggestion``` fence.

    The output contract is that the model's ``suggestion`` field holds RAW
    replacement code (no fence). As a defensive belt-and-suspenders against an
    unreliable model that re-wraps it anyway, an already-present
    leading/trailing ```suggestion``` fence is stripped first so the result is
    ALWAYS exactly one fence — never the nested double fence GitHub refuses to
    render as a committable suggestion.
    """
    body = _strip_suggestion_fence(suggestion).rstrip("\n")
    return f"```suggestion\n{body}\n```"


def _strip_suggestion_fence(suggestion: str) -> str:
    """Remove a single leading ```suggestion fence + its matching trailing ```.

    Tolerates surrounding whitespace and a trailing newline after the opening
    fence. If no wrapping fence is present, the input is returned unchanged.
    """
    stripped = suggestion.strip()
    opener = re.match(r"^```+suggestion[ \t]*\r?\n", stripped)
    if not opener or not stripped.endswith("```"):
        return suggestion
    inner = stripped[opener.end():]
    inner = inner[: inner.rfind("```")]
    return inner


def fingerprint_marker(fingerprint: str) -> str:
    """Hidden HTML-comment marker carrying a finding fingerprint."""
    return f"<!-- openrabbit:fp={fingerprint} -->"


def _truncate_detail(detail: Any, limit: int) -> str:
    """Bound an upstream error detail to ``limit`` chars (single line).

    Upstream bodies may reflect untrusted request fragments; truncating keeps
    them from flooding CI logs. Newlines are collapsed so the message stays one
    line.
    """
    text = str(detail).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _severity_badge(finding: Finding) -> str:
    return f"**[{finding.severity.upper()}/{finding.category}]**"


def build_review_comment(finding: Finding) -> dict[str, Any]:
    """Build one ``comments[]`` entry for the batched createReview call.

    Single-line findings use ``line`` + ``side``; multi-line findings additionally
    set ``start_line`` + ``start_side`` (GitHub anchors the comment on the *end*
    line). The body carries the title, rationale, an optional committable
    ```suggestion``` block, and a hidden fingerprint marker for later dedup.
    """
    lines = [f"{_severity_badge(finding)} {finding.title}", "", finding.body]
    if finding.suggestion:
        lines += ["", render_suggestion_block(finding.suggestion)]
    lines += ["", fingerprint_marker(finding.fingerprint)]
    body = "\n".join(lines)

    comment: dict[str, Any] = {
        "path": finding.file,
        "line": finding.end_line,
        "side": finding.side,
        "body": body,
    }
    if finding.end_line != finding.start_line:
        comment["start_line"] = finding.start_line
        comment["start_side"] = finding.side
    return comment


# --------------------------------------------------------------------------- #
# Adapter                                                                      #
# --------------------------------------------------------------------------- #
class GitHubAdapter:
    """Advisory-only GitHub review adapter (REST + GraphQL).

    A ``client`` may be injected (any object exposing ``get``/``post``/``patch``
    returning response-like objects with ``status_code``/``json()``); production
    builds one lazily with ``httpx``. Tests inject a fake so no network is used.
    """

    def __init__(
        self,
        repo: GitHubRepo,
        pr_number: int,
        token: str,
        *,
        client: Any = None,
        api_base: str = DEFAULT_API_BASE,
        graphql_url: str = DEFAULT_GRAPHQL_URL,
        bot_login: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.repo = repo
        self.pr_number = pr_number
        self._token = token
        self._injected_client = client
        self._owned_client: Any = None
        self.api_base = api_base.rstrip("/")
        self.graphql_url = graphql_url
        self.bot_login = bot_login
        self.timeout = timeout

    # -- HTTP plumbing ------------------------------------------------------ #
    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _client(self) -> Any:
        """Return the HTTP client, importing ``httpx`` lazily if none injected."""
        if self._injected_client is not None:
            return self._injected_client
        if self._owned_client is None:
            import httpx  # lazy: keep module import dependency-free

            self._owned_client = httpx.Client(timeout=self.timeout)
        return self._owned_client

    def close(self) -> None:
        """Close an owned httpx client (no-op for injected clients)."""
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    @staticmethod
    def _ok(resp: Any) -> bool:
        return 200 <= getattr(resp, "status_code", 0) < 300

    # Upstream error bodies can reflect untrusted PR content; bound the length
    # we interpolate so a verbose/echoed body cannot flood CI logs.
    _MAX_DETAIL_CHARS = 200

    def _raise_for(self, resp: Any, action: str) -> None:
        if not self._ok(resp):
            detail = ""
            try:
                detail = resp.json().get("message", "")  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - defensive
                detail = getattr(resp, "text", "")
            detail = _truncate_detail(detail, self._MAX_DETAIL_CHARS)
            raise GitHubError(
                f"{action} failed: HTTP {getattr(resp, 'status_code', '?')} {detail}".strip()
            )

    def _rest_url(self, path: str) -> str:
        return f"{self.api_base}{path}"

    # -- REST: read PR ------------------------------------------------------ #
    def fetch_pr_diff(self) -> str:
        """Return the unified diff for the PR (``Accept: ...v3.diff``)."""
        url = self._rest_url(f"/repos/{self.repo.slug}/pulls/{self.pr_number}")
        resp = self._client().get(
            url, headers=self._headers("application/vnd.github.v3.diff")
        )
        self._raise_for(resp, "fetch_pr_diff")
        return resp.text

    def fetch_changed_files(self) -> list[ChangedFile]:
        """Return the PR's changed files as :class:`ChangedFile` objects."""
        url = self._rest_url(
            f"/repos/{self.repo.slug}/pulls/{self.pr_number}/files"
        )
        resp = self._client().get(url, headers=self._headers())
        self._raise_for(resp, "fetch_changed_files")
        return [
            ChangedFile(
                filename=item["filename"],
                status=item.get("status", "modified"),
                additions=int(item.get("additions", 0)),
                deletions=int(item.get("deletions", 0)),
                patch=item.get("patch"),
            )
            for item in resp.json()
        ]

    # -- REST: post review (ONE call) -------------------------------------- #
    def post_review(
        self,
        findings: list[Finding],
        summary_markdown: str,
        commit_sha: str,
        *,
        event: str = "COMMENT",
    ) -> dict[str, Any]:
        """Post ONE batched review with all inline comments + a summary body.

        ``event`` must be ``COMMENT`` (advisory-only): ``APPROVE`` /
        ``REQUEST_CHANGES`` / ``DISMISS`` raise :class:`GitHubError`. The single
        ``POST /repos/{o}/{r}/pulls/{n}/reviews`` carries ``comments[]`` with
        committable suggestion blocks and correct line/side anchoring.
        """
        if event in _FORBIDDEN_EVENTS or event not in _ALLOWED_EVENTS:
            raise GitHubError(
                f"advisory-only adapter refuses review event {event!r}; "
                f"only {sorted(_ALLOWED_EVENTS)} is permitted"
            )
        comments = [build_review_comment(f) for f in findings]
        payload = {
            "commit_id": commit_sha,
            "event": event,
            "body": summary_markdown,
            "comments": comments,
        }
        url = self._rest_url(
            f"/repos/{self.repo.slug}/pulls/{self.pr_number}/reviews"
        )
        resp = self._client().post(url, headers=self._headers(), json=payload)
        self._raise_for(resp, "post_review")
        return resp.json()

    # -- REST: sticky walkthrough upsert ----------------------------------- #
    def _list_issue_comments(self) -> list[dict[str, Any]]:
        url = self._rest_url(
            f"/repos/{self.repo.slug}/issues/{self.pr_number}/comments"
        )
        resp = self._client().get(url, headers=self._headers())
        self._raise_for(resp, "list_issue_comments")
        return list(resp.json())

    def _find_sticky_comment(self) -> Optional[dict[str, Any]]:
        for comment in self._list_issue_comments():
            if BOT_STICKY_MARKER in (comment.get("body") or ""):
                return comment
        return None

    def upsert_sticky_walkthrough(self, markdown: str) -> dict[str, Any]:
        """Create or update the single sticky walkthrough comment.

        The body is prefixed with :data:`BOT_STICKY_MARKER` so subsequent runs
        find+update the same comment instead of posting duplicates.
        """
        body = f"{BOT_STICKY_MARKER}\n{markdown}"
        existing = self._find_sticky_comment()
        if existing is None:
            url = self._rest_url(
                f"/repos/{self.repo.slug}/issues/{self.pr_number}/comments"
            )
            resp = self._client().post(url, headers=self._headers(), json={"body": body})
            self._raise_for(resp, "create_sticky_walkthrough")
            return resp.json()
        comment_id = existing["id"]
        url = self._rest_url(
            f"/repos/{self.repo.slug}/issues/comments/{comment_id}"
        )
        resp = self._client().patch(url, headers=self._headers(), json={"body": body})
        self._raise_for(resp, "update_sticky_walkthrough")
        return resp.json()

    # -- GraphQL ----------------------------------------------------------- #
    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        resp = self._client().post(
            self.graphql_url,
            headers=self._headers(),
            json={"query": query, "variables": variables},
        )
        self._raise_for(resp, "graphql")
        data = resp.json()
        if data.get("errors"):
            messages = "; ".join(
                e.get("message", str(e)) for e in data["errors"]
            )
            raise GitHubError(f"graphql errors: {messages}")
        return data.get("data", {})

    def resolve_review_thread(self, thread_id: str) -> bool:
        """Resolve a review thread whose finding is now fixed."""
        query = (
            "mutation($threadId: ID!) {"
            " resolveReviewThread(input: {threadId: $threadId})"
            " { thread { id isResolved } } }"
        )
        data = self._graphql(query, {"threadId": thread_id})
        thread = (data.get("resolveReviewThread") or {}).get("thread") or {}
        return bool(thread.get("isResolved", True))

    def minimize_comment(self, subject_id: str, classifier: str = "OUTDATED") -> bool:
        """Minimize a superseded comment (default classifier ``OUTDATED``)."""
        query = (
            "mutation($subjectId: ID!, $classifier: ReportedContentClassifiers!) {"
            " minimizeComment(input: {subjectId: $subjectId, classifier: $classifier})"
            " { minimizedComment { isMinimized minimizedReason } } }"
        )
        data = self._graphql(
            query, {"subjectId": subject_id, "classifier": classifier}
        )
        minimized = (data.get("minimizeComment") or {}).get("minimizedComment") or {}
        return bool(minimized.get("isMinimized", True))

    # -- GraphQL: list bot review threads (dedup source) ------------------- #
    def list_bot_review_threads(self) -> list[ReviewThread]:
        """List the bot's prior review threads with state + fingerprint.

        Paginates through ``pullRequest.reviewThreads``. When ``bot_login`` is
        set, only threads whose first comment was authored by that login are
        returned, so human threads never get touched by dedup/auto-resolve.
        """
        query = (
            "query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {"
            " repository(owner: $owner, name: $repo) {"
            "  pullRequest(number: $number) {"
            "   reviewThreads(first: 100, after: $cursor) {"
            "    nodes { id isResolved isOutdated"
            "      comments(first: 1) { nodes { id body path author { login } } } }"
            "    pageInfo { hasNextPage endCursor } } } } }"
        )
        threads: list[ReviewThread] = []
        cursor: Optional[str] = None
        while True:
            data = self._graphql(
                query,
                {
                    "owner": self.repo.owner,
                    "repo": self.repo.repo,
                    "number": self.pr_number,
                    "cursor": cursor,
                },
            )
            container = (
                ((data.get("repository") or {}).get("pullRequest") or {}).get(
                    "reviewThreads"
                )
                or {}
            )
            for node in container.get("nodes", []):
                parsed = self._parse_thread(node)
                if parsed is None:
                    continue
                threads.append(parsed)
            page = container.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            cursor = page.get("endCursor")
        return threads

    def _parse_thread(self, node: dict[str, Any]) -> Optional[ReviewThread]:
        comments = ((node.get("comments") or {}).get("nodes")) or []
        first = comments[0] if comments else {}
        login = ((first.get("author") or {}).get("login")) or ""
        if self.bot_login is not None and login != self.bot_login:
            return None
        body = first.get("body") or ""
        fp_match = _FP_MARKER_RE.search(body)
        return ReviewThread(
            thread_id=node["id"],
            comment_id=first.get("id"),
            fingerprint=fp_match.group(1) if fp_match else None,
            is_resolved=bool(node.get("isResolved", False)),
            is_outdated=bool(node.get("isOutdated", False)),
            path=first.get("path"),
            body=body,
        )

    # -- Dedup logic -------------------------------------------------------- #
    @staticmethod
    def dedup_findings(
        findings: list[Finding], posted_threads: list[ReviewThread]
    ) -> list[Finding]:
        """Drop findings already represented by a prior bot thread.

        A finding is suppressed when its fingerprint matches *any* existing bot
        thread — whether that thread is still open (don't repost) or already
        resolved/outdated (don't resurrect a dismissed/superseded finding). Only
        brand-new fingerprints survive.
        """
        seen = {t.fingerprint for t in posted_threads if t.fingerprint}
        return [f for f in findings if f.fingerprint not in seen]

    @staticmethod
    def stale_threads(
        current_findings: list[Finding], posted_threads: list[ReviewThread]
    ) -> list[ReviewThread]:
        """Return prior bot threads whose finding no longer appears.

        These are candidates for :meth:`resolve_review_thread` +
        :meth:`minimize_comment` (the issue was fixed or superseded). Threads
        with no fingerprint, already resolved, or still present are excluded.
        """
        current = {f.fingerprint for f in current_findings}
        return [
            t
            for t in posted_threads
            if t.fingerprint
            and not t.is_resolved
            and t.fingerprint not in current
        ]
