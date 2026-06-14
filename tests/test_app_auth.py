"""Tests for the optional GitHub App auth layer (SPEC 1.2 — Phase-4 App option).

NO network and NO real creds: an RSA keypair is generated *in-test* (via the
lazily-imported ``cryptography`` backend that ``PyJWT`` already pulls in), and
the installation-token HTTP exchange is mocked by monkeypatching ``httpx``. The
auth module MUST import ``httpx`` (and ``jwt``) lazily so importing it — and
this whole test module — needs zero network and no eager heavy deps.
"""

from __future__ import annotations

import importlib
import sys
import time

import pytest

from openrabbit.app import auth as auth_mod
from openrabbit.app.auth import (
    InstallationTokenCache,
    installation_token,
    make_app_jwt,
)


# --------------------------------------------------------------------------- #
# In-test RSA key (no fixtures on disk, no real App private key)              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rsa_keys() -> tuple[str, str]:
    """Generate a throwaway RSA keypair, returned as (private_pem, public_pem)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return private_pem, public_pem


@pytest.fixture
def private_pem(rsa_keys: tuple[str, str]) -> str:
    return rsa_keys[0]


@pytest.fixture
def public_pem(rsa_keys: tuple[str, str]) -> str:
    return rsa_keys[1]


# --------------------------------------------------------------------------- #
# Lazy-import contract                                                        #
# --------------------------------------------------------------------------- #
def test_module_imports_without_httpx_or_jwt():
    """Importing the auth module must NOT eagerly import httpx or jwt (lazy)."""
    for mod in ("openrabbit.app", "openrabbit.app.auth"):
        sys.modules.pop(mod, None)
    sys.modules.pop("httpx", None)
    sys.modules.pop("jwt", None)
    importlib.import_module("openrabbit.app.auth")
    assert "httpx" not in sys.modules
    assert "jwt" not in sys.modules


# --------------------------------------------------------------------------- #
# make_app_jwt                                                                 #
# --------------------------------------------------------------------------- #
def test_jwt_claims_and_alg(private_pem, public_pem):
    import jwt

    before = int(time.time())
    token = make_app_jwt("123456", private_pem)
    after = int(time.time())

    header = jwt.get_unverified_header(token)
    assert header["alg"] == "RS256"

    # Verify the signature with the public key (RS256), and check claims.
    claims = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert claims["iss"] == "123456"
    # iat is backdated 60s to tolerate clock skew (GitHub guidance).
    assert claims["iat"] <= before - 60 + 1
    assert claims["iat"] >= before - 60 - 5
    # exp is at most 10 minutes from now.
    assert claims["exp"] <= after + 600
    assert claims["exp"] > after


def test_jwt_iss_accepts_int_app_id(private_pem, public_pem):
    import jwt

    token = make_app_jwt(987, private_pem)
    claims = jwt.decode(token, public_pem, algorithms=["RS256"])
    # GitHub accepts the App id as a string; we serialize ints to str.
    assert claims["iss"] == "987"


def test_jwt_exp_capped_at_ten_minutes(private_pem, public_pem):
    import jwt

    # Even asking for a longer ttl, exp must be capped at <= 10min (600s).
    token = make_app_jwt("1", private_pem, ttl_seconds=99999)
    claims = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert claims["exp"] - claims["iat"] <= 600 + 60  # +60 for the iat backdate


def test_jwt_rejects_empty_private_key(private_pem):
    with pytest.raises(ValueError):
        make_app_jwt("1", "")


# --------------------------------------------------------------------------- #
# installation_token — mocked httpx exchange                                  #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 201) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )


class _Recorder:
    def __init__(self, payload: dict, status_code: int = 201) -> None:
        self.payload = payload
        self.status_code = status_code
        self.url: str | None = None
        self.headers: dict | None = None
        self.json_body: dict | None = None
        self.calls = 0

    def post(self, url, *, headers=None, json=None, **kwargs):  # noqa: A002
        self.calls += 1
        self.url = url
        self.headers = headers
        self.json_body = json
        return _FakeResponse(self.payload, self.status_code)


def _install_fake_httpx(monkeypatch, recorder: _Recorder) -> None:
    import httpx

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, headers=None, json=None, **kwargs):  # noqa: A002
            return recorder.post(url, headers=headers, json=json, **kwargs)

    monkeypatch.setattr(httpx, "Client", _FakeClient)


def test_installation_token_exchange(monkeypatch, private_pem):
    rec = _Recorder(
        {"token": "ghs_installationtoken", "expires_at": "2026-06-15T01:00:00Z"}
    )
    _install_fake_httpx(monkeypatch, rec)

    result = installation_token("123456", private_pem, 42)

    assert result.token == "ghs_installationtoken"
    assert result.expires_at == "2026-06-15T01:00:00Z"
    # Correct endpoint + method.
    assert rec.url.endswith("/app/installations/42/access_tokens")
    assert rec.calls == 1
    # The JWT (Bearer) authenticates the App-level call.
    assert rec.headers["Authorization"].startswith("Bearer ")
    assert rec.headers["Accept"] == "application/vnd.github+json"


def test_installation_token_custom_api_base(monkeypatch, private_pem):
    rec = _Recorder({"token": "ghs_x", "expires_at": "2026-06-15T01:00:00Z"})
    _install_fake_httpx(monkeypatch, rec)
    installation_token(
        "1", private_pem, 7, api_base="https://ghe.example.com/api/v3"
    )
    assert rec.url == (
        "https://ghe.example.com/api/v3/app/installations/7/access_tokens"
    )


def test_installation_token_http_error_raises(monkeypatch, private_pem):
    rec = _Recorder({"message": "Not Found"}, status_code=404)
    _install_fake_httpx(monkeypatch, rec)
    with pytest.raises(auth_mod.AppAuthError):
        installation_token("1", private_pem, 99)


def test_installation_token_missing_token_field_raises(monkeypatch, private_pem):
    rec = _Recorder({"expires_at": "2026-06-15T01:00:00Z"})  # no "token"
    _install_fake_httpx(monkeypatch, rec)
    with pytest.raises(auth_mod.AppAuthError):
        installation_token("1", private_pem, 5)


# --------------------------------------------------------------------------- #
# InstallationTokenCache — cache + refresh                                     #
# --------------------------------------------------------------------------- #
def test_cache_fetches_once_then_reuses(monkeypatch, private_pem):
    rec = _Recorder(
        {"token": "ghs_cached", "expires_at": "2999-01-01T00:00:00Z"}
    )
    _install_fake_httpx(monkeypatch, rec)

    cache = InstallationTokenCache("1", private_pem)
    t1 = cache.get(42)
    t2 = cache.get(42)
    assert t1 == "ghs_cached" == t2
    # Far-future expiry → only ONE exchange.
    assert rec.calls == 1


def test_cache_refreshes_when_expired(monkeypatch, private_pem):
    # An already-expired token forces a refresh on the next get().
    rec = _Recorder(
        {"token": "ghs_expired", "expires_at": "2000-01-01T00:00:00Z"}
    )
    _install_fake_httpx(monkeypatch, rec)

    cache = InstallationTokenCache("1", private_pem)
    cache.get(42)
    cache.get(42)
    assert rec.calls == 2  # expired each time → refetch each time


def test_cache_keys_by_installation(monkeypatch, private_pem):
    rec = _Recorder(
        {"token": "ghs_a", "expires_at": "2999-01-01T00:00:00Z"}
    )
    _install_fake_httpx(monkeypatch, rec)
    cache = InstallationTokenCache("1", private_pem)
    cache.get(1)
    cache.get(2)
    # Different installations → separate exchanges.
    assert rec.calls == 2


def test_cache_force_refresh(monkeypatch, private_pem):
    rec = _Recorder(
        {"token": "ghs_force", "expires_at": "2999-01-01T00:00:00Z"}
    )
    _install_fake_httpx(monkeypatch, rec)
    cache = InstallationTokenCache("1", private_pem)
    cache.get(1)
    cache.get(1, force_refresh=True)
    assert rec.calls == 2


def test_cache_refreshes_when_expiry_missing(monkeypatch, private_pem):
    # No expires_at at all → treated as expired (fail safe) → refetch each time.
    rec = _Recorder({"token": "ghs_noexp"})  # no expires_at
    _install_fake_httpx(monkeypatch, rec)
    cache = InstallationTokenCache("1", private_pem)
    cache.get(1)
    cache.get(1)
    assert rec.calls == 2


# --------------------------------------------------------------------------- #
# Transport error + expiry parsing edge cases                                  #
# --------------------------------------------------------------------------- #
def test_installation_token_transport_error_raises(monkeypatch, private_pem):
    import httpx

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **k):
            raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    with pytest.raises(auth_mod.AppAuthError) as excinfo:
        installation_token("1", private_pem, 5)
    # The message must stay generic — it MUST NOT embed the raw httpx exception
    # text (which can carry the request URL / internal host). The original
    # exception is preserved as __cause__ for internal/debug inspection only.
    message = str(excinfo.value)
    assert "no route to host" not in message
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


def test_parse_iso8601_handles_naive_and_bad_values():
    # Trailing-Z UTC form parses to a positive epoch.
    assert auth_mod._parse_iso8601("2026-06-15T01:00:00Z") > 0
    # A naive (no-offset) timestamp is assumed UTC, not rejected.
    assert auth_mod._parse_iso8601("2026-06-15T01:00:00") > 0
    # Garbage / empty → None (caller treats token as expired).
    assert auth_mod._parse_iso8601("not-a-date") is None
    assert auth_mod._parse_iso8601(None) is None
    assert auth_mod._parse_iso8601("") is None
