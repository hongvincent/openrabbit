"""GitHub App installation auth (SPEC 1.2 — Phase-4 App option, 9 keyless auth).

Two primitives plus a small cache:

* :func:`make_app_jwt` — a short-lived **App JWT** signed RS256 with the App's
  private key. ``iat`` is backdated 60s to tolerate clock skew; ``exp`` is capped
  at GitHub's 10-minute maximum; ``iss`` is the App id (string).
* :func:`installation_token` — exchanges that JWT at
  ``POST /app/installations/{id}/access_tokens`` for a short-lived **installation
  access token** (the token used to act on a specific installation's repos).
* :class:`InstallationTokenCache` — caches installation tokens per installation
  id and transparently refreshes them shortly before they expire, so a long-lived
  App process mints far fewer tokens.

Dependency-light & offline-testable: ``jwt`` (PyJWT, which pulls in
``cryptography`` for RS256) and ``httpx`` are imported **lazily inside the
functions**, so importing this module needs neither. Unit tests generate an RSA
key in-process and mock the HTTP exchange — no network, no real App key.

Security posture (SPEC 12): this layer only *mints* credentials; it never reads
or obeys repo/PR/webhook content. The App private key is provided by the caller
(from a secret store / env), never hardcoded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional, Union

#: GitHub's default REST API base. Overridable for GitHub Enterprise Server.
DEFAULT_API_BASE = "https://api.github.com"

#: GitHub caps the App JWT lifetime at 10 minutes.
MAX_JWT_TTL_SECONDS = 600

#: Backdate ``iat`` by this many seconds to tolerate clock drift between us and
#: GitHub (GitHub's own guidance — avoids "iat in the future" rejections).
IAT_BACKDATE_SECONDS = 60

#: Default request timeout for the token exchange (seconds).
DEFAULT_TIMEOUT = 30.0

#: Refresh a cached installation token this many seconds before it actually
#: expires, so an in-flight request never races the expiry boundary.
_REFRESH_SKEW_SECONDS = 60


class AppAuthError(Exception):
    """Raised when an installation-token exchange fails or returns bad data."""


# --------------------------------------------------------------------------- #
# App JWT                                                                      #
# --------------------------------------------------------------------------- #
def make_app_jwt(
    app_id: Union[str, int],
    private_key: str,
    *,
    ttl_seconds: int = MAX_JWT_TTL_SECONDS,
    now: Optional[int] = None,
) -> str:
    """Build a short-lived RS256-signed GitHub App JWT.

    Parameters
    ----------
    app_id:
        The numeric GitHub App id (accepted as ``int`` or ``str``; serialized to
        ``str`` for the ``iss`` claim per GitHub's expectation).
    private_key:
        The App's PEM private key (PKCS#1 or PKCS#8). Provided by the caller from
        a secret store / env — never hardcoded.
    ttl_seconds:
        Requested token lifetime; clamped to ``[1, 600]`` since GitHub rejects
        JWTs whose ``exp`` is more than 10 minutes out.
    now:
        Override "current epoch seconds" (testing/determinism). Defaults to
        :func:`time.time`.

    ``jwt`` (PyJWT + ``cryptography`` for RS256) is imported lazily here.
    """
    if not private_key or not private_key.strip():
        raise ValueError("private_key must be a non-empty PEM string")

    import jwt  # lazy: PyJWT + cryptography only needed when actually signing

    issued = int(now if now is not None else time.time())
    iat = issued - IAT_BACKDATE_SECONDS
    ttl = max(1, min(int(ttl_seconds), MAX_JWT_TTL_SECONDS))
    exp = issued + ttl

    claims = {"iat": iat, "exp": exp, "iss": str(app_id)}
    token = jwt.encode(claims, private_key, algorithm="RS256")
    # PyJWT < 2 returns bytes; normalize to str for a stable public contract.
    if isinstance(token, bytes):  # pragma: no cover - PyJWT>=2 returns str
        token = token.decode("ascii")
    return token


# --------------------------------------------------------------------------- #
# Installation access token                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InstallationToken:
    """A minted installation access token + its server-provided expiry."""

    token: str
    #: ISO-8601 UTC timestamp string exactly as returned by GitHub, e.g.
    #: ``2026-06-15T01:00:00Z`` (kept verbatim — never re-derived).
    expires_at: Optional[str] = None


def installation_token(
    app_id: Union[str, int],
    private_key: str,
    installation_id: Union[str, int],
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = DEFAULT_TIMEOUT,
    jwt_token: Optional[str] = None,
) -> InstallationToken:
    """Exchange an App JWT for an installation access token.

    Calls ``POST {api_base}/app/installations/{installation_id}/access_tokens``
    authenticating with a freshly minted App JWT (or ``jwt_token`` if supplied),
    and returns the resulting :class:`InstallationToken`.

    Raises :class:`AppAuthError` on a non-2xx response or a payload missing the
    ``token`` field. ``httpx`` is imported lazily.
    """
    token_jwt = jwt_token or make_app_jwt(app_id, private_key)
    url = f"{api_base.rstrip('/')}/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {token_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    import httpx  # lazy: keep module import dependency-free for unit tests

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json={})
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise AppAuthError(
            f"installation-token exchange returned an error status: {_status_of(exc)}"
        ) from exc
    except httpx.HTTPError as exc:  # transport-level (connect/timeout/...)
        # Keep the message generic: some httpx error reprs embed the request URL,
        # which could disclose an internal/GHES endpoint if this string is logged.
        # The original exception is preserved as ``__cause__`` (``from exc``) for
        # internal/debug inspection without surfacing it in the message.
        raise AppAuthError(
            "installation-token exchange request failed (transport error)"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("token"):
        raise AppAuthError("installation-token response missing the 'token' field")
    return InstallationToken(
        token=str(payload["token"]),
        expires_at=(str(payload["expires_at"]) if payload.get("expires_at") else None),
    )


def _status_of(exc: object) -> str:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return str(code) if code is not None else "unknown"


# --------------------------------------------------------------------------- #
# Cache / refresh helper                                                       #
# --------------------------------------------------------------------------- #
class InstallationTokenCache:
    """Cache installation access tokens per installation, refreshing on expiry.

    A long-lived App process serves many webhook deliveries; minting a fresh
    installation token on every delivery is wasteful (and rate-limited). This
    helper holds one token per ``installation_id`` and re-mints only when the
    cached token is missing, force-refreshed, or within
    :data:`_REFRESH_SKEW_SECONDS` of its expiry.

    Tokens whose ``expires_at`` GitHub did not return (or that we cannot parse)
    are treated as already-expired, so they are always refreshed — fail safe.
    """

    def __init__(
        self,
        app_id: Union[str, int],
        private_key: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._app_id = app_id
        self._private_key = private_key
        self._api_base = api_base
        self._timeout = timeout
        self._cache: dict[str, InstallationToken] = {}

    def get(
        self, installation_id: Union[str, int], *, force_refresh: bool = False
    ) -> str:
        """Return a valid installation token string, minting/refreshing as needed."""
        key = str(installation_id)
        cached = self._cache.get(key)
        if not force_refresh and cached is not None and not self._is_expired(cached):
            return cached.token

        fresh = installation_token(
            self._app_id,
            self._private_key,
            installation_id,
            api_base=self._api_base,
            timeout=self._timeout,
        )
        self._cache[key] = fresh
        return fresh.token

    @staticmethod
    def _is_expired(token: InstallationToken, *, now: Optional[float] = None) -> bool:
        expiry = _parse_iso8601(token.expires_at)
        if expiry is None:
            return True  # unknown/unparseable expiry → refresh (fail safe)
        current = now if now is not None else time.time()
        return current >= (expiry - _REFRESH_SKEW_SECONDS)


def _parse_iso8601(value: Optional[str]) -> Optional[float]:
    """Parse a GitHub ISO-8601 UTC timestamp to epoch seconds, or ``None``.

    Handles the trailing ``Z`` (UTC) form GitHub emits. Any unparseable value
    yields ``None`` so the caller treats the token as expired (fail safe).
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()
