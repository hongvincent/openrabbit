"""Tests for the framework-free GitHub App webhook handler (SPEC 1.2 / 12).

``handle_request`` is a pure ``Request -> Response`` function so the App can be
unit-tested with no web framework and no running server. It verifies the HMAC
signature, dispatches the event, and maps the outcome to an HTTP status. NO
network. The body is UNTRUSTED — a bad signature/JSON must be rejected, never
executed.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from openrabbit.app.server import Request, Response, handle_request


def _sign(secret: str, body: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={mac}"


def _pr_payload(action: str = "opened") -> dict:
    return {
        "action": action,
        "number": 7,
        "repository": {"full_name": "octo/repo", "id": 999},
        "pull_request": {
            "number": 7,
            "draft": False,
            "state": "open",
            "title": "T",
            "body": "B",
            "head": {"sha": "abc"},
            "base": {"sha": "def"},
        },
        "installation": {"id": 42},
    }


def _request(secret, event, payload_obj, *, sign_with=None, delivery="d1"):
    body = json.dumps(payload_obj).encode("utf-8")
    sign_secret = sign_with if sign_with is not None else secret
    headers = {
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": _sign(sign_secret, body),
        "Content-Type": "application/json",
    }
    return Request(method="POST", headers=headers, body=body)


class _SpyReview:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def __call__(self, pr_context):
        self.contexts.append(pr_context)
        return {"reviewed": True}


def test_valid_pr_event_returns_200_and_reviews():
    secret = "whsec"
    spy = _SpyReview()
    req = _request(secret, "pull_request", _pr_payload("opened"))
    resp = handle_request(req, secret=secret, deps={"review": spy})

    assert isinstance(resp, Response)
    assert resp.status == 200
    assert len(spy.contexts) == 1
    body = json.loads(resp.body)
    assert body["handled"] is True
    assert body["action"] == "review"


def test_bad_signature_returns_401_and_does_not_review():
    secret = "whsec"
    spy = _SpyReview()
    req = _request(secret, "pull_request", _pr_payload(), sign_with="attacker")
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 401
    assert spy.contexts == []


def test_missing_signature_returns_401():
    secret = "whsec"
    spy = _SpyReview()
    body = json.dumps(_pr_payload()).encode()
    req = Request(
        method="POST",
        headers={"X-GitHub-Event": "pull_request"},  # no signature header
        body=body,
    )
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 401
    assert spy.contexts == []


def test_non_post_method_returns_405():
    secret = "whsec"
    spy = _SpyReview()
    req = Request(method="GET", headers={}, body=b"")
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 405
    assert spy.contexts == []


def test_ignored_event_returns_204():
    secret = "whsec"
    spy = _SpyReview()
    req = _request(secret, "push", {"ref": "refs/heads/main"})
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 204
    assert spy.contexts == []


def test_ignored_pr_action_returns_204():
    secret = "whsec"
    spy = _SpyReview()
    req = _request(secret, "pull_request", _pr_payload("closed"))
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 204
    assert spy.contexts == []


def test_malformed_json_returns_400():
    secret = "whsec"
    spy = _SpyReview()
    body = b"{not valid json"
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(secret, body),
    }
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 400
    assert spy.contexts == []


def test_header_lookup_is_case_insensitive():
    secret = "whsec"
    spy = _SpyReview()
    body = json.dumps(_pr_payload("opened")).encode()
    headers = {
        "x-github-event": "pull_request",  # lowercase
        "x-hub-signature-256": _sign(secret, body),
    }
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 200
    assert len(spy.contexts) == 1


def test_missing_event_header_returns_400():
    secret = "whsec"
    spy = _SpyReview()
    body = json.dumps(_pr_payload()).encode()
    headers = {"X-Hub-Signature-256": _sign(secret, body)}  # no event header
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 400
    assert spy.contexts == []


def test_review_callback_exception_returns_500():
    secret = "whsec"

    def _boom(pr_context):
        raise RuntimeError("review blew up")

    req = _request(secret, "pull_request", _pr_payload("opened"))
    resp = handle_request(req, secret=secret, deps={"review": _boom})
    assert resp.status == 500
    # Error body must not leak the internal exception text.
    assert "review blew up" not in resp.body


def test_response_is_json_content_type_on_200():
    secret = "whsec"
    spy = _SpyReview()
    req = _request(secret, "pull_request", _pr_payload("opened"))
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.headers.get("Content-Type") == "application/json"


def test_non_object_json_payload_returns_400():
    secret = "whsec"
    spy = _SpyReview()
    body = b"[1, 2, 3]"  # valid JSON, but not an object
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(secret, body),
    }
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 400
    assert spy.contexts == []


def test_missing_review_dep_propagates_as_server_error():
    """A server misconfigured with no review callback raises (not a client 4xx)."""
    secret = "whsec"
    req = _request(secret, "pull_request", _pr_payload("opened"))
    with pytest.raises(KeyError):
        handle_request(req, secret=secret, deps={})


def test_str_body_request_is_verified_and_handled():
    """A str body (not bytes) is UTF-8 encoded for the HMAC and handled."""
    secret = "whsec"
    spy = _SpyReview()
    body_str = json.dumps(_pr_payload("opened"))
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(secret, body_str.encode("utf-8")),
    }
    req = Request(method="POST", headers=headers, body=body_str)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 200
    assert len(spy.contexts) == 1
