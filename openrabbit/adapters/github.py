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

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from openrabbit.findings import Finding

_LOG = logging.getLogger("openrabbit.adapters.github")

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
    inner = stripped[opener.end() :]
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


def normalize_comment_path(path: str) -> str:
    """Strip a model-echoed ``a/`` or ``b/`` git-diff prefix from a comment path.

    The diff text the model reads carries ``a/<path>`` (old) and ``b/<path>``
    (new) headers; an unreliable model sometimes echoes the ``b/`` (or ``a/``)
    prefix into ``finding.file``. GitHub indexes changed files by their bare
    repo-relative path, so a prefixed path would never match and the comment
    would be rejected. Only a leading ``a/``/``b/`` is removed (a real path that
    legitimately starts with ``b/...`` after a normal segment is untouched).
    """
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def build_review_comment(finding: Finding) -> dict[str, Any]:
    """Build one ``comments[]`` entry for the batched createReview call.

    Single-line findings use ``line`` + ``side``; multi-line findings additionally
    set ``start_line`` + ``start_side`` (GitHub anchors the comment on the *end*
    line). The body carries the title, rationale, an optional committable
    ```suggestion``` block, and a hidden fingerprint marker for later dedup.

    Defensive normalization against an unreliable model:

    * the path is run through :func:`normalize_comment_path` (echoed ``a/``/``b/``);
    * a swapped multi-line range (``start_line > end_line``) is clamped so the
      smaller endpoint is ``start_line`` and the larger is the ``line`` anchor —
      GitHub 422s a review where ``start_line > line``.
    """
    lines = [f"{_severity_badge(finding)} {finding.title}", "", finding.body]
    if finding.suggestion:
        lines += ["", render_suggestion_block(finding.suggestion)]
    lines += ["", fingerprint_marker(finding.fingerprint)]
    body = "\n".join(lines)

    lo = min(finding.start_line, finding.end_line)
    hi = max(finding.start_line, finding.end_line)
    comment: dict[str, Any] = {
        "path": normalize_comment_path(finding.file),
        "line": hi,
        "side": finding.side,
        "body": body,
    }
    if hi != lo:
        comment["start_line"] = lo
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
        max_retries: int = 4,
        sleep: Any = time.sleep,
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
        # Bounded retry budget for GitHub primary/secondary rate limits
        # (403/429 + Retry-After / X-RateLimit-Reset). ``sleep`` is injectable so
        # unit tests exercise the backoff path without real wall-clock delay.
        self.max_retries = max_retries
        self._sleep = sleep

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

    # -- rate-limit aware request layer ------------------------------------ #
    # GitHub primary-rate (429) and secondary-rate (403 with X-RateLimit-Remaining:0
    # or a "secondary rate limit" message) responses are transient. Hard-failing
    # on the first one would lose an entire review. We retry idempotent reads and
    # the single createReview POST with a bounded budget, honoring Retry-After /
    # X-RateLimit-Reset when present.
    _RATE_LIMIT_STATUSES = frozenset({403, 429})
    _MAX_BACKOFF_SECONDS = 60.0

    @staticmethod
    def _is_rate_limited(resp: Any) -> bool:
        status = getattr(resp, "status_code", 0)
        if status == 429:
            return True
        if status != 403:
            return False
        headers = {k.lower(): v for k, v in dict(getattr(resp, "headers", {}) or {}).items()}
        # Secondary rate limit: either an explicit Retry-After, a depleted
        # X-RateLimit-Remaining, or a body that names the secondary limit.
        if "retry-after" in headers:
            return True
        if headers.get("x-ratelimit-remaining") == "0":
            return True
        try:
            message = str(resp.json().get("message", "")).lower()
        except Exception:  # pragma: no cover - defensive
            message = ""
        return "secondary rate limit" in message or "rate limit" in message

    def _retry_delay(self, resp: Any, attempt: int) -> float:
        """Compute the backoff (seconds) for a rate-limited response.

        Prefer the server's ``Retry-After`` (seconds) or ``X-RateLimit-Reset``
        (epoch seconds) hint; otherwise fall back to exponential backoff. The
        delay is bounded so a bogus header can't wedge the run.
        """
        headers = {k.lower(): v for k, v in dict(getattr(resp, "headers", {}) or {}).items()}
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(0.0, min(float(retry_after), self._MAX_BACKOFF_SECONDS))
            except (TypeError, ValueError):
                pass
        reset = headers.get("x-ratelimit-reset")
        if reset is not None:
            try:
                delta = float(reset) - time.time()
                return max(0.0, min(delta, self._MAX_BACKOFF_SECONDS))
            except (TypeError, ValueError):
                pass
        return min(2.0**attempt, self._MAX_BACKOFF_SECONDS)

    def _request(self, method: str, url: str, action: str, **kw: Any) -> Any:
        """Issue an HTTP request, retrying transient rate-limit responses.

        ``method`` is ``get``/``post``/``patch``. On a 403/429 rate-limit
        response the call is retried up to :attr:`max_retries` times with a
        bounded, Retry-After-aware backoff; any other non-2xx response is left
        for the caller's :meth:`_raise_for` to surface.
        """
        client = self._client()
        call = getattr(client, method)
        resp = call(url, **kw)
        attempt = 0
        while (
            self._is_rate_limited(resp)
            and attempt < self.max_retries
        ):
            delay = self._retry_delay(resp, attempt)
            _LOG.warning(
                "%s: GitHub rate limit (HTTP %s), retry %d/%d after %.1fs",
                action,
                getattr(resp, "status_code", "?"),
                attempt + 1,
                self.max_retries,
                delay,
            )
            self._sleep(delay)
            resp = call(url, **kw)
            attempt += 1
        return resp

    # -- REST: read PR ------------------------------------------------------ #
    def fetch_pr_diff(self) -> str:
        """Return the unified diff for the PR (``Accept: ...v3.diff``)."""
        url = self._rest_url(f"/repos/{self.repo.slug}/pulls/{self.pr_number}")
        resp = self._request(
            "get",
            url,
            "fetch_pr_diff",
            headers=self._headers("application/vnd.github.v3.diff"),
        )
        self._raise_for(resp, "fetch_pr_diff")
        return resp.text

    def fetch_changed_files(self) -> list[ChangedFile]:
        """Return the PR's changed files as :class:`ChangedFile` objects."""
        url = self._rest_url(f"/repos/{self.repo.slug}/pulls/{self.pr_number}/files")
        resp = self._request(
            "get", url, "fetch_changed_files", headers=self._headers()
        )
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
    @staticmethod
    def _comment_is_in_diff(
        comment: dict[str, Any],
        valid_positions: Optional[dict[str, set[tuple[str, int]]]],
        changed_files: Optional[set[str]],
    ) -> bool:
        """True if a built comment anchors on a real diff position.

        Cross-checks the comment's file against ``changed_files`` (the PR's
        actual filenames) and its ``(side, line)`` — and, for a multi-line
        comment, ``(side, start_line)`` — against ``valid_positions`` (the set
        derived from the parsed ``@@`` hunk ranges). When neither map is
        supplied the check is a no-op (legacy callers keep the old behavior).
        """
        path = comment["path"]
        if changed_files is not None and path not in changed_files:
            return False
        if valid_positions is None:
            return True
        allowed = valid_positions.get(path)
        if not allowed:
            return False
        side = comment["side"]
        if (side, comment["line"]) not in allowed:
            return False
        if "start_line" in comment:
            start_side = comment.get("start_side", side)
            if (start_side, comment["start_line"]) not in allowed:
                return False
        return True

    def post_review(
        self,
        findings: list[Finding],
        summary_markdown: str,
        commit_sha: str,
        *,
        event: str = "COMMENT",
        valid_positions: Optional[dict[str, set[tuple[str, int]]]] = None,
        changed_files: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """Post ONE batched review with all inline comments + a summary body.

        ``event`` must be ``COMMENT`` (advisory-only): ``APPROVE`` /
        ``REQUEST_CHANGES`` / ``DISMISS`` raise :class:`GitHubError`. The single
        ``POST /repos/{o}/{r}/pulls/{n}/reviews`` carries ``comments[]`` with
        committable suggestion blocks and correct line/side anchoring.

        **Diff-anchor validation (CRITICAL):** ``valid_positions`` maps each
        changed file to its valid ``{(side, line)}`` anchor set (see
        :func:`openrabbit.pipeline.route.valid_positions_by_file`) and
        ``changed_files`` is the PR's real filename set. Findings whose path or
        ``(side, line)`` falls outside the diff are DROPPED before the POST —
        otherwise a single hallucinated position would 422 the entire batch and
        post zero comments. If every comment is filtered out, no review is fired
        (``{"skipped": True}`` is returned). As a last-resort guard, a 422 from
        GitHub triggers ONE retry of the same POST without inline comments so the
        summary review still lands instead of being lost.
        """
        if event in _FORBIDDEN_EVENTS or event not in _ALLOWED_EVENTS:
            raise GitHubError(
                f"advisory-only adapter refuses review event {event!r}; "
                f"only {sorted(_ALLOWED_EVENTS)} is permitted"
            )
        built = [(f, build_review_comment(f)) for f in findings]
        comments: list[dict[str, Any]] = []
        for finding, comment in built:
            if self._comment_is_in_diff(comment, valid_positions, changed_files):
                comments.append(comment)
            else:
                _LOG.warning(
                    "post_review: dropping out-of-diff finding %s at %s:%s (side=%s)",
                    finding.fingerprint,
                    comment["path"],
                    comment["line"],
                    comment["side"],
                )

        # Low-noise guard: if validation filtered out every inline comment AND
        # there were inline findings to begin with, do not fire an empty review.
        if findings and not comments:
            return {"skipped": True}

        url = self._rest_url(f"/repos/{self.repo.slug}/pulls/{self.pr_number}/reviews")

        def _post(payload_comments: list[dict[str, Any]]) -> Any:
            payload = {
                "commit_id": commit_sha,
                "event": event,
                "body": summary_markdown,
                "comments": payload_comments,
            }
            return self._request(
                "post", url, "post_review", headers=self._headers(), json=payload
            )

        resp = _post(comments)
        # Belt-and-suspenders: a server-side 422 on a comment position must not
        # nuke the whole review. Retry ONCE without inline comments so the
        # summary review still lands (the offending lines are simply not posted).
        if getattr(resp, "status_code", 0) == 422 and comments:
            _LOG.warning(
                "post_review: GitHub 422 on inline comments; retrying summary-only"
            )
            resp = _post([])
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
            resp = self._client().post(
                url, headers=self._headers(), json={"body": body}
            )
            self._raise_for(resp, "create_sticky_walkthrough")
            return resp.json()
        comment_id = existing["id"]
        url = self._rest_url(f"/repos/{self.repo.slug}/issues/comments/{comment_id}")
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
            messages = "; ".join(e.get("message", str(e)) for e in data["errors"])
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
        data = self._graphql(query, {"subjectId": subject_id, "classifier": classifier})
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
            container = ((data.get("repository") or {}).get("pullRequest") or {}).get(
                "reviewThreads"
            ) or {}
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
            if t.fingerprint and not t.is_resolved and t.fingerprint not in current
        ]
