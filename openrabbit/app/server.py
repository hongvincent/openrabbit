"""Framework-free GitHub App webhook handler (SPEC 1.2 — Phase-4 App option).

:func:`handle_request` is a **pure** ``Request -> Response`` function: it verifies
the HMAC signature, parses the event, dispatches via
:func:`openrabbit.app.webhook.handle_event`, and maps the outcome to an HTTP
status. Because it is just a function over plain dataclasses, the whole App is
unit-testable with **no web framework and no running server**.

Status mapping
--------------
* non-``POST``                         -> ``405``
* missing/invalid signature            -> ``401`` (checked BEFORE parsing —
  untrusted bytes are never JSON-parsed until authenticated)
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Union

from openrabbit.app.webhook import handle_event, verify_signature

_SIG_HEADER = "x-hub-signature-256"
_EVENT_HEADER = "x-github-event"
_JSON_CT = {"Content-Type": "application/json"}


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
    request: Request, *, secret: str, deps: Mapping[str, Any]
) -> Response:
    """Verify, parse, route, and respond — a pure ``Request -> Response``.

    See the module docstring for the full status mapping. Signature verification
    happens **before** the untrusted body is JSON-parsed, so unauthenticated
    bytes are never interpreted.
    """
    if request.method.upper() != "POST":
        return _json_response(405, {"error": "method not allowed"})

    # 1) Authenticate the raw bytes FIRST (untrusted-input discipline, SPEC 12).
    if not verify_signature(secret, request.body_bytes, request.header(_SIG_HEADER)):
        return _json_response(401, {"error": "invalid signature"})

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

    return _json_response(
        200,
        {"handled": True, "action": result.action, "event": result.event},
    )


def _json_response(status: int, body: dict[str, Any]) -> Response:
    return Response(status=status, body=json.dumps(body), headers=dict(_JSON_CT))
