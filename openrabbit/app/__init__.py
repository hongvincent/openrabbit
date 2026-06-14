"""Optional GitHub App mode (SPEC 1.2 — a Phase-4 *option* alongside Action-first).

This package is the **code** for running openrabbit as a GitHub App (installation
auth + a webhook handler), not a deployment. It is deliberately framework-free
and dependency-light:

* :mod:`openrabbit.app.auth` — App JWT (RS256) + installation access tokens
  (``jwt`` / ``httpx`` imported lazily; tests use an in-test RSA key + mocked
  HTTP, so importing this package needs no network and no eager heavy deps).
* :mod:`openrabbit.app.webhook` — HMAC-SHA256 signature verification + event
  routing for ``pull_request`` actions to the existing ``orchestrator.review``.
* :mod:`openrabbit.app.server` — a pure ``Request -> Response`` handler that can
  be mounted behind FastAPI / AWS Lambda / any WSGI/ASGI shim WITHOUT adding a
  web-framework runtime dependency.

Security posture (SPEC 12): the webhook body is **untrusted data**; it is never
executed as instructions, and a hostile/partial payload degrades to "ignored"
rather than crashing. The reasoning layer remains advisory-only.
"""

from __future__ import annotations

__all__ = [
    "auth",
    "server",
    "webhook",
]
