"""GitHub App webhook verification + event routing (SPEC 1.2 / 6 / 12).

* :func:`verify_signature` — constant-time HMAC-SHA256 verification of the
  ``X-Hub-Signature-256`` header against the raw request body, using
  :func:`hmac.compare_digest` (no early-exit timing leak). A missing/empty
  header, a wrong algorithm prefix, or a non-``sha256=`` shape all reject.
* :func:`handle_event` — routes a parsed webhook event. ``pull_request`` actions
  in :data:`REVIEW_ACTIONS` are translated into the neutral ``pr_context`` the
  existing :func:`openrabbit.pipeline.orchestrator.review` consumes and dispatched
  to an injected ``deps["review"]`` callback; every other event/action is ignored.

Security posture (SPEC 12): the payload is **UNTRUSTED**. Its fields are read as
*data* only — never executed as instructions — and a hostile, partial, or
wrong-typed payload degrades to an "ignored" result instead of raising. Signature
verification gates everything upstream (in :mod:`openrabbit.app.server`).

Pure stdlib (``hmac`` / ``hashlib`` / ``json``) — no network, no third-party deps.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Union

#: ``pull_request`` actions that should trigger a review (SPEC 6 step 1).
REVIEW_ACTIONS: frozenset[str] = frozenset(
    {"opened", "synchronize", "reopened", "ready_for_review"}
)

#: The signature header GitHub sends (HMAC-SHA256). We require this, not the
#: legacy sha1 ``X-Hub-Signature``.
_SIG_PREFIX = "sha256="


@dataclass(frozen=True)
class WebhookResult:
    """The structured outcome of routing one webhook event.

    Attributes
    ----------
    handled:
        ``True`` when the event was dispatched to the review callback.
    action:
        ``"review"`` when a review was triggered, else ``"ignored"``.
    event:
        The GitHub event name (``X-GitHub-Event``).
    pr_context:
        The neutral context handed to the review callback (when handled).
    review_result:
        Whatever the injected review callback returned (opaque to this layer).
    """

    handled: bool
    action: str
    event: Optional[str] = None
    pr_context: Optional[dict[str, Any]] = None
    review_result: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# signature verification                                                       #
# --------------------------------------------------------------------------- #
def verify_signature(
    secret: str, body: Union[bytes, str], sig_header: Optional[str]
) -> bool:
    """Constant-time verify ``X-Hub-Signature-256`` over ``body``.

    Returns ``True`` only when ``sig_header`` equals ``sha256=<hexdigest>`` for
    the HMAC-SHA256 of ``body`` keyed by ``secret``. An empty/unset ``secret``, a
    missing/empty header, an unexpected prefix, or any mismatch returns
    ``False``. The comparison goes through :func:`hmac.compare_digest` so it does
    not leak timing information.

    Fail closed on a falsy secret: a server misconfigured with an empty webhook
    secret would otherwise key HMAC with an empty key, letting anyone who knows
    the secret is empty forge a valid signature. We reject up front (mirroring
    :func:`openrabbit.app.auth.make_app_jwt`, which rejects an empty private key)
    so an unset ``OPENRABBIT_WEBHOOK_SECRET`` never accepts forged deliveries.
    """
    if not secret or not isinstance(secret, str):
        return False
    if not sig_header or not isinstance(sig_header, str):
        return False
    if not sig_header.startswith(_SIG_PREFIX):
        return False

    body_bytes = body.encode("utf-8") if isinstance(body, str) else body
    expected = hmac.new(
        secret.encode("utf-8"), body_bytes, hashlib.sha256
    ).hexdigest()
    provided = sig_header[len(_SIG_PREFIX):]
    # Constant-time compare of the two hex digests (equal length by construction
    # of `expected`; compare_digest also tolerates unequal length safely).
    return hmac.compare_digest(expected, provided)


# --------------------------------------------------------------------------- #
# event routing                                                                #
# --------------------------------------------------------------------------- #
#: A review callback takes the neutral ``pr_context`` mapping and returns
#: anything (opaque). In production this wraps ``orchestrator.review`` (closing
#: over config + providers); in tests it is a spy.
ReviewCallback = Callable[[Mapping[str, Any]], Any]


def handle_event(
    event: Optional[str],
    payload: Union[Mapping[str, Any], str, bytes],
    *,
    deps: Mapping[str, Any],
) -> WebhookResult:
    """Route one (already signature-verified) webhook event.

    Parameters
    ----------
    event:
        The ``X-GitHub-Event`` header value.
    payload:
        The parsed JSON body (a mapping) or the raw JSON ``str``/``bytes`` (which
        is parsed here). Treated as UNTRUSTED.
    deps:
        Dependency map; ``deps["review"]`` MUST be the review callback. Injected
        so the App reuses the existing orchestrator without importing it here
        (keeps this module pure-stdlib and trivially testable).

    Returns a :class:`WebhookResult`. Only ``pull_request`` events whose
    ``action`` is in :data:`REVIEW_ACTIONS` — and that carry enough data to build
    a ``pr_context`` — trigger the review callback; everything else (other events,
    other actions, malformed payloads) is reported as ``"ignored"`` without
    raising.

    NOTE on the diff: the webhook payload does **not** contain the PR diff, and
    :func:`openrabbit.pipeline.orchestrator.review` gates on a ``diff`` field.
    The neutral ``pr_context`` built here therefore carries the *identifiers*
    (``repo`` / ``number`` / ``head_sha`` / ``base_sha`` / ``installation_id``)
    but no ``diff``. The injected ``deps["review"]`` callback is responsible for
    fetching the diff (e.g. via the GitHub API using an installation token) and
    merging it into the context before calling the orchestrator. Keeping the
    fetch in the callback preserves this module's pure-stdlib, offline-testable
    posture (no network here).
    """
    # Resolve the review dependency up front: a misconfigured server (no review
    # callback wired) is a programming error, surfaced as KeyError.
    review: ReviewCallback = deps["review"]

    data = _coerce_payload(payload)
    if data is None:
        return WebhookResult(handled=False, action="ignored", event=event)

    if event != "pull_request":
        return WebhookResult(handled=False, action="ignored", event=event)

    action = data.get("action")
    if action not in REVIEW_ACTIONS:
        return WebhookResult(
            handled=False, action="ignored", event=event,
            detail={"pr_action": action},
        )

    pr_context = _build_pr_context(data)
    if pr_context is None:
        # Action matched but the payload lacks the fields we need → ignore
        # rather than dispatch a half-built context.
        return WebhookResult(handled=False, action="ignored", event=event)

    review_result = review(pr_context)
    return WebhookResult(
        handled=True,
        action="review",
        event=event,
        pr_context=pr_context,
        review_result=review_result,
    )


def _coerce_payload(
    payload: Union[Mapping[str, Any], str, bytes],
) -> Optional[dict[str, Any]]:
    """Return the payload as a dict, parsing JSON str/bytes; ``None`` if invalid."""
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            parsed = json.loads(payload)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _build_pr_context(data: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Translate an untrusted ``pull_request`` payload into a neutral context.

    Returns ``None`` when the required ``repository.full_name`` / ``pull_request``
    object / PR number is absent or wrong-typed (we never fabricate them). Every
    field is read defensively — a hostile payload cannot raise here.
    """
    repo = data.get("repository")
    pr = data.get("pull_request")
    if not isinstance(repo, Mapping) or not isinstance(pr, Mapping):
        return None

    repo_full = repo.get("full_name")
    if not isinstance(repo_full, str) or not repo_full:
        return None

    number = _coerce_int(data.get("number"))
    if number is None:
        number = _coerce_int(pr.get("number"))
    if number is None:
        return None

    head = pr.get("head") if isinstance(pr.get("head"), Mapping) else {}
    base = pr.get("base") if isinstance(pr.get("base"), Mapping) else {}
    installation = (
        data.get("installation")
        if isinstance(data.get("installation"), Mapping)
        else {}
    )

    # The neutral context the orchestrator.review() spine consumes. The PR
    # title/body are UNTRUSTED text carried verbatim as DATA (the downstream
    # prefix builder fences them); they are never interpreted here.
    context: dict[str, Any] = {
        "repo": repo_full,
        "number": number,
        "action": data.get("action"),
        "draft": bool(pr.get("draft", False)),
        "state": pr.get("state"),
        "title": _as_str_or_none(pr.get("title")),
        "body": _as_str_or_none(pr.get("body")),
        "head_sha": _as_str_or_none(head.get("sha")),
        "base_sha": _as_str_or_none(base.get("sha")),
        "installation_id": _coerce_int(installation.get("id")),
    }
    return context


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort int coercion; ``None`` for anything non-integral (e.g. 'seven')."""
    if isinstance(value, bool):  # bool is an int subclass — exclude it
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None
