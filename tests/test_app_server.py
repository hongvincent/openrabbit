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


# --------------------------------------------------------------------------- #
# Finding 2: App mode does NOT yet post reviews — it is honestly marked        #
# not-wired so no one ships it expecting posted reviews.                       #
# --------------------------------------------------------------------------- #
def test_app_mode_review_posting_is_marked_not_wired():
    """App mode must NOT silently pretend it posts reviews.

    The Actions path (reusable workflow / composite action) is the supported,
    working review path. App mode's webhook handler dispatches to an injected
    ``deps["review"]`` callback, but openrabbit does not yet ship a callback that
    fetches the PR diff via an installation token and posts the review. Until that
    is wired, the module must advertise that honestly via a machine-checkable
    flag so onboarding code can detect it rather than ship a no-op reviewer.
    """
    import openrabbit.app.server as server_mod

    assert hasattr(server_mod, "APP_MODE_REVIEW_WIRED"), (
        "server must expose APP_MODE_REVIEW_WIRED so callers can detect whether "
        "App mode ships a real posting review callback"
    )
    assert server_mod.APP_MODE_REVIEW_WIRED is False, (
        "App mode does not yet ship a diff-fetching, review-posting callback; "
        "the flag must stay False until it genuinely does (use the Actions path)"
    )


def test_app_mode_not_wired_is_documented_in_module_docstring():
    """The 'not wired — use the Actions path' caveat is in the module docstring.

    A reader skimming the module must not be misled by the mounting examples into
    believing a posting reviewer ships.
    """
    import openrabbit.app.server as server_mod

    doc = (server_mod.__doc__ or "").lower()
    assert "not" in doc and "wired" in doc, (
        "module docstring must state App mode is NOT yet wired to post reviews"
    )
    assert "actions" in doc, (
        "module docstring must point users at the working Actions path"
    )


# --------------------------------------------------------------------------- #
# Finding 3: payload-size cap + delivery-id replay defense                     #
# --------------------------------------------------------------------------- #
def test_oversized_body_is_rejected_before_review():
    """A body larger than the cap is rejected (413) and never reviewed/parsed."""
    from openrabbit.app.server import MAX_BODY_BYTES

    secret = "whsec"
    spy = _SpyReview()
    # Build a body just over the cap. It is otherwise valid + correctly signed,
    # so only the size cap can reject it.
    payload = _pr_payload("opened")
    payload["pull_request"]["body"] = "x" * (MAX_BODY_BYTES + 1)
    body = json.dumps(payload).encode("utf-8")
    assert len(body) > MAX_BODY_BYTES
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "big-1",
        "X-Hub-Signature-256": _sign(secret, body),
    }
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 413
    assert spy.contexts == []


def test_oversized_body_rejected_before_signature_check():
    """The size cap rejects BEFORE HMAC is computed over attacker-controlled bytes.

    A huge unsigned body must not force a full HMAC over megabytes of attacker
    input (cheap DoS). The 413 fires even with no/invalid signature.
    """
    from openrabbit.app.server import MAX_BODY_BYTES

    secret = "whsec"
    spy = _SpyReview()
    body = b"x" * (MAX_BODY_BYTES + 10)
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": "sha256=" + "0" * 64,  # invalid on purpose
    }
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 413
    assert spy.contexts == []


def test_body_at_cap_is_accepted():
    """A body exactly at the cap is allowed (boundary is inclusive)."""
    from openrabbit.app.server import MAX_BODY_BYTES

    secret = "whsec"
    spy = _SpyReview()
    base = _pr_payload("opened")
    # Pad the (untrusted) body field so the serialized body lands exactly at cap.
    # Solve for the pad length empirically (the JSON envelope around the padding
    # is constant once the field is a string), then assert we hit the cap exactly.
    base["pull_request"]["body"] = ""
    envelope_len = len(json.dumps(base).encode("utf-8"))
    pad = MAX_BODY_BYTES - envelope_len
    assert pad > 0
    base["pull_request"]["body"] = "x" * pad
    body = json.dumps(base).encode("utf-8")
    assert len(body) == MAX_BODY_BYTES
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "at-cap",
        "X-Hub-Signature-256": _sign(secret, body),
    }
    req = Request(method="POST", headers=headers, body=body)
    resp = handle_request(req, secret=secret, deps={"review": spy})
    assert resp.status == 200
    assert len(spy.contexts) == 1


def test_duplicate_delivery_id_is_ignored_for_replay_defense():
    """Re-delivering the same X-GitHub-Delivery id does not review twice."""
    from openrabbit.app.server import DeliveryDedup

    secret = "whsec"
    spy = _SpyReview()
    dedup = DeliveryDedup()
    req = _request(secret, "pull_request", _pr_payload("opened"), delivery="dup-1")

    first = handle_request(req, secret=secret, deps={"review": spy}, dedup=dedup)
    assert first.status == 200
    assert len(spy.contexts) == 1

    # Exact same signed delivery, replayed.
    second = handle_request(req, secret=secret, deps={"review": spy}, dedup=dedup)
    assert second.status == 200  # acknowledged so GitHub stops retrying
    assert len(spy.contexts) == 1  # but NOT reviewed again
    body = json.loads(second.body)
    assert body.get("duplicate") is True


def test_distinct_delivery_ids_each_reviewed():
    """Different delivery ids are independent — both are reviewed."""
    from openrabbit.app.server import DeliveryDedup

    secret = "whsec"
    spy = _SpyReview()
    dedup = DeliveryDedup()
    r1 = _request(secret, "pull_request", _pr_payload("opened"), delivery="a")
    r2 = _request(secret, "pull_request", _pr_payload("synchronize"), delivery="b")
    handle_request(r1, secret=secret, deps={"review": spy}, dedup=dedup)
    handle_request(r2, secret=secret, deps={"review": spy}, dedup=dedup)
    assert len(spy.contexts) == 2


def test_dedup_is_bounded():
    """The dedup set is bounded so a flood of deliveries cannot grow it unbounded."""
    from openrabbit.app.server import DeliveryDedup

    dedup = DeliveryDedup(max_entries=4)
    for i in range(100):
        dedup.seen(f"id-{i}")
    assert len(dedup) <= 4


def test_dedup_only_records_authenticated_deliveries():
    """A bad-signature delivery id is NOT recorded (so the real, signed retry of
    that id is still processed once)."""
    from openrabbit.app.server import DeliveryDedup

    secret = "whsec"
    spy = _SpyReview()
    dedup = DeliveryDedup()
    # Forged delivery: bad signature → 401, must not poison the dedup set.
    forged = _request(
        secret, "pull_request", _pr_payload("opened"),
        sign_with="attacker", delivery="x1",
    )
    resp = handle_request(forged, secret=secret, deps={"review": spy}, dedup=dedup)
    assert resp.status == 401
    assert dedup.seen("x1") is False  # not recorded by the rejected request

    # The genuine signed delivery with the same id is processed normally.
    real = _request(secret, "pull_request", _pr_payload("opened"), delivery="x1")
    # Reset the probe-recorded id so this asserts the request path, not our probe.
    dedup = DeliveryDedup()
    ok = handle_request(real, secret=secret, deps={"review": spy}, dedup=dedup)
    assert ok.status == 200
    assert len(spy.contexts) == 1


def test_dedup_is_optional_backward_compatible():
    """Omitting the dedup arg keeps the original behavior (no replay tracking)."""
    secret = "whsec"
    spy = _SpyReview()
    req = _request(secret, "pull_request", _pr_payload("opened"), delivery="z")
    # Same delivery twice with no dedup → reviewed twice (unchanged contract).
    handle_request(req, secret=secret, deps={"review": spy})
    handle_request(req, secret=secret, deps={"review": spy})
    assert len(spy.contexts) == 2
