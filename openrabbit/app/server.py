"""Framework-free GitHub App webhook handler (SPEC 1.2 — Phase-4 App option).

:func:`handle_request` is a **pure** ``Request -> Response`` function: it verifies
the HMAC signature, parses the event, dispatches via
:func:`openrabbit.app.webhook.handle_event`, and maps the outcome to an HTTP
status. Because it is just a function over plain dataclasses, the whole App is
unit-testable with **no web framework and no running server**.

App mode is NOT YET WIRED to post reviews — use the Actions path
----------------------------------------------------------------
The supported, working review path is **GitHub Actions** (the reusable workflow
``.github/workflows/reusable-workflow.yml`` / composite ``actions/action.yml``).
This webhook handler routes a verified ``pull_request`` event to an *injected*
``deps["review"]`` callback, but openrabbit does **not** yet ship a callback that
(1) fetches the PR diff via an installation token, (2) runs
``orchestrator.review(..., emit=False)``, and (3) emits the review through a
:class:`openrabbit.adapters.github.GitHubAdapter` built from that installation
token + bot login. The flag :data:`APP_MODE_REVIEW_WIRED` is therefore ``False``.
Do not deploy App mode expecting posted reviews until a real callback is wired
and that flag flips to ``True``; until then, onboard via the Actions path.

Status mapping
--------------
* non-``POST``                         -> ``405``
* body larger than :data:`MAX_BODY_BYTES` -> ``413`` (checked FIRST, before any
  HMAC is computed over attacker-controlled megabytes — cheap-DoS defense)
* missing/invalid signature            -> ``401`` (checked BEFORE parsing —
  untrusted bytes are never JSON-parsed until authenticated)
* duplicate ``X-GitHub-Delivery`` id   -> ``200`` (acknowledged, NOT re-reviewed;
  replay defense — only when a :class:`DeliveryDedup` is supplied)
* missing ``X-GitHub-Event`` header    -> ``400``
* unparseable JSON body                -> ``400``
* event ignored (non-PR / non-review)  -> ``204``
* review dispatched                    -> ``200`` (JSON summary body)
* review callback raised               -> ``500`` (generic body; never leaks the
  internal exception text into the HTTP response)

Mounting (no web-framework dependency added)
--------------------------------------------
This module imports nothing beyond the stdlib + :mod:`openrabbit.app.webhook`.
Adapt your edge to a :class:`Request` and back from a :class:`Response`:

* **FastAPI / Starlette** (framework lives in *your* app, not here)::

      from openrabbit.app.server import Request, Response, handle_request

      @app.post("/webhook")
      async def webhook(request: FastAPIRequest):
          body = await request.body()
          req = Request("POST", dict(request.headers), body)
          resp = handle_request(req, secret=SECRET, deps={"review": review_cb})
          return PlainTextResponse(resp.body, resp.status, resp.headers)

* **AWS Lambda (API Gateway proxy)**::

      def lambda_handler(event, _ctx):
          body = event["body"] or ""
          if event.get("isBase64Encoded"):
              import base64; body = base64.b64decode(body)
          req = Request(event["httpMethod"], event["headers"] or {}, body)
          resp = handle_request(req, secret=SECRET, deps={"review": review_cb})
          return {"statusCode": resp.status, "headers": resp.headers,
                  "body": resp.body}

The ``review`` dep is the existing orchestrator: typically
``lambda ctx: orchestrator.review(config, ctx, providers, ...)`` closed over your
config + providers (and, for the App, an installation-token-backed GitHub client).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from openrabbit.app.webhook import handle_event, verify_signature

_SIG_HEADER = "x-hub-signature-256"
_EVENT_HEADER = "x-github-event"
_DELIVERY_HEADER = "x-github-delivery"
_JSON_CT = {"Content-Type": "application/json"}

#: App mode does NOT yet ship a diff-fetching, review-posting callback. See the
#: module docstring: the Actions path is the supported review path. Flip this to
#: ``True`` only when a real App-mode review callback is wired end to end.
APP_MODE_REVIEW_WIRED = False

#: Hard cap on the webhook request body. GitHub's own webhook payloads are capped
#: at 25 MiB; we accept comfortably more than any real ``pull_request`` payload
#: while refusing a body large enough to be a cheap memory/CPU DoS. Checked
#: BEFORE the HMAC so we never hash attacker-controlled megabytes.
MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB

#: Default number of recently-seen delivery ids a :class:`DeliveryDedup` retains.
_DEFAULT_DEDUP_ENTRIES = 4096


class DeliveryDedup:
    """Bounded in-memory set of processed ``X-GitHub-Delivery`` ids (replay guard).

    GitHub retries deliveries and an attacker who captures one signed delivery can
    replay it. Recording each *authenticated* delivery id lets the handler ack a
    replay (so GitHub stops retrying) without reviewing the same PR event twice.

    The set is **bounded** (FIFO eviction of the oldest ids) so a flood of
    distinct deliveries cannot grow memory without limit. It is process-local and
    not shared across replicas — adequate as a best-effort guard for a single App
    process; a multi-replica deployment would back this with a shared store.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_DEDUP_ENTRIES) -> None:
        self._max = max(1, int(max_entries))
        self._ids: OrderedDict[str, None] = OrderedDict()

    def seen(self, delivery_id: str) -> bool:
        """Return ``True`` if ``delivery_id`` was already recorded (a replay)."""
        return delivery_id in self._ids

    def record(self, delivery_id: str) -> None:
        """Record ``delivery_id`` as processed, evicting the oldest if at capacity."""
        if delivery_id in self._ids:
            self._ids.move_to_end(delivery_id)
            return
        self._ids[delivery_id] = None
        while len(self._ids) > self._max:
            self._ids.popitem(last=False)

    def __len__(self) -> int:
        return len(self._ids)


@dataclass(frozen=True)
class Request:
    """A minimal, framework-neutral HTTP request.

    ``headers`` lookups are case-insensitive (see :meth:`header`). ``body`` is the
    raw request body (``bytes`` preferred so the HMAC is computed over the exact
    bytes GitHub signed; a ``str`` is accepted and UTF-8 encoded).
    """

    method: str
    headers: Mapping[str, str]
    body: Union[bytes, str]

    def header(self, name: str) -> str:
        """Case-insensitive header lookup; ``""`` when absent."""
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return ""

    @property
    def body_bytes(self) -> bytes:
        return self.body.encode("utf-8") if isinstance(self.body, str) else self.body


@dataclass(frozen=True)
class Response:
    """A minimal HTTP response (status + headers + body)."""

    status: int
    body: str = ""
    headers: dict[str, str] = field(default_factory=dict)


def handle_request(
    request: Request,
    *,
    secret: str,
    deps: Mapping[str, Any],
    dedup: Optional[DeliveryDedup] = None,
) -> Response:
    """Verify, parse, route, and respond — a pure ``Request -> Response``.

    See the module docstring for the full status mapping. Signature verification
    happens **before** the untrusted body is JSON-parsed, so unauthenticated
    bytes are never interpreted.

    ``dedup`` (optional): a :class:`DeliveryDedup` enabling replay defense via the
    ``X-GitHub-Delivery`` id. When supplied, a previously-seen delivery is
    acknowledged (``200``) but NOT re-reviewed, and only *authenticated* (signature
    + size valid) deliveries are recorded. Omitting it preserves the original
    behavior (no replay tracking).
    """
    if request.method.upper() != "POST":
        return _json_response(405, {"error": "method not allowed"})

    # 0) Size cap FIRST — refuse an oversized body before computing the HMAC over
    # attacker-controlled bytes (a huge unsigned body must not force a megabyte
    # HMAC; cheap-DoS defense). Checked even before signature verification.
    if len(request.body_bytes) > MAX_BODY_BYTES:
        return _json_response(413, {"error": "payload too large"})

    # 1) Authenticate the raw bytes (untrusted-input discipline, SPEC 12).
    if not verify_signature(secret, request.body_bytes, request.header(_SIG_HEADER)):
        return _json_response(401, {"error": "invalid signature"})

    # 1b) Replay defense: if this authenticated delivery id was already processed,
    # acknowledge (so GitHub stops retrying) without reviewing again.
    delivery_id = request.header(_DELIVERY_HEADER)
    if dedup is not None and delivery_id and dedup.seen(delivery_id):
        return _json_response(200, {"handled": False, "duplicate": True})

    # 2) Require the event header (so we know what we're routing).
    event = request.header(_EVENT_HEADER)
    if not event:
        return _json_response(400, {"error": "missing X-GitHub-Event header"})

    # 3) Parse the (now authenticated) JSON body.
    try:
        payload = json.loads(request.body_bytes)
    except (ValueError, TypeError):
        return _json_response(400, {"error": "invalid JSON body"})
    if not isinstance(payload, dict):
        return _json_response(400, {"error": "payload must be a JSON object"})

    # 4) Route. Any failure inside the review callback is contained: we return a
    # generic 500 and never echo the internal exception text into the response.
    try:
        result = handle_event(event, payload, deps=deps)
    except KeyError:
        # Missing review dep is a server misconfiguration, not client input.
        raise
    except Exception:
        return _json_response(500, {"error": "internal error"})

    if not result.handled:
        # Acknowledge ignored events with 204 (no body) so GitHub stops retrying.
        return Response(status=204, body="", headers=dict(_JSON_CT))

    # Record only a delivery that triggered a real review, so a later replay of
    # the same id is acked (not re-reviewed). Done after the callback returns so a
    # callback that raised (→ 500) is NOT recorded and GitHub's retry can succeed.
    if dedup is not None and delivery_id:
        dedup.record(delivery_id)

    return _json_response(
        200,
        {"handled": True, "action": result.action, "event": result.event},
    )


def _json_response(status: int, body: dict[str, Any]) -> Response:
    return Response(status=status, body=json.dumps(body), headers=dict(_JSON_CT))
