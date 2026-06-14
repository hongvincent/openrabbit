"""Tests for the Check-Run output adapter (SPEC section 6 emit + section 9).

NO NETWORK. ``build_check_run`` is pure data; ``post_check_run`` reuses the
GitHub adapter's lazy HTTP layer and is exercised with an injected fake client
(same fake transport pattern as ``test_github.py``).

Covered:
- conclusion logic by blocking severity (advisory-only default = ``neutral``;
  ``failure`` only when a finding meets the configured blocking severity),
- annotation_level mapping (notice | warning | failure),
- annotation shape (path / start_line / end_line / annotation_level / message),
- the ``POST /repos/{o}/{r}/check-runs`` wire payload.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from openrabbit.adapters.checkrun import (
    annotation_level_for_severity,
    build_annotation,
    build_check_run,
    post_check_run,
)
from openrabbit.adapters.github import GitHubAdapter, GitHubError, GitHubRepo
from openrabbit.findings import Finding


def make_finding(**overrides: Any) -> Finding:
    defaults: dict[str, Any] = dict(
        file="src/agent.py",
        start_line=42,
        end_line=47,
        side="RIGHT",
        severity="high",
        category="correctness",
        confidence=0.88,
        title="Unvalidated index can raise IndexError",
        body="The index `i` is not bounds-checked before use.",
        rule_id="openrabbit/correctness/bounds-check",
        fingerprint="fp-aaa",
        suggestion=None,
    )
    defaults.update(overrides)
    return Finding(**defaults)


# --------------------------------------------------------------------------- #
# Fake httpx transport (mirrors test_github.py)                                #
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def json(self) -> Any:
        return self._payload


class RecordedRequest:
    def __init__(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.method = method
        self.url = url
        self.json = json
        self.headers = headers or {}


class FakeClient:
    def __init__(self, routes: list[tuple[str, str, FakeResponse]]) -> None:
        self.routes = routes
        self.requests: list[RecordedRequest] = []

    def _handle(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.requests.append(
            RecordedRequest(method, url, json=kw.get("json"), headers=kw.get("headers"))
        )
        for m, substr, resp in self.routes:
            if m == method and substr in url:
                return resp
        raise AssertionError(f"unexpected request: {method} {url}")

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("POST", url, **kw)

    def patch(self, url: str, **kw: Any) -> FakeResponse:
        return self._handle("PATCH", url, **kw)

    def close(self) -> None:  # pragma: no cover - parity with httpx.Client
        pass


def make_adapter(client: FakeClient) -> GitHubAdapter:
    return GitHubAdapter(
        repo=GitHubRepo(owner="acme", repo="widget"),
        pr_number=7,
        token="ghs_faketoken",
        client=client,
    )


# --------------------------------------------------------------------------- #
# annotation_level mapping                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "severity,level",
    [
        ("critical", "failure"),
        ("high", "failure"),
        ("medium", "warning"),
        ("low", "notice"),
        ("nit", "notice"),
    ],
)
def test_annotation_level_for_severity(severity: str, level: str):
    assert annotation_level_for_severity(severity) == level


def test_annotation_level_unknown_defaults_to_warning():
    assert annotation_level_for_severity("bogus") == "warning"


# --------------------------------------------------------------------------- #
# annotation shape                                                             #
# --------------------------------------------------------------------------- #
def test_build_annotation_shape():
    ann = build_annotation(make_finding(start_line=42, end_line=47, severity="high"))
    assert ann["path"] == "src/agent.py"
    assert ann["start_line"] == 42
    assert ann["end_line"] == 47
    assert ann["annotation_level"] == "failure"
    assert "Unvalidated index" in ann["message"]
    assert ann["title"]  # has a title


def test_build_annotation_single_line():
    ann = build_annotation(make_finding(start_line=10, end_line=10))
    assert ann["start_line"] == 10
    assert ann["end_line"] == 10


# --------------------------------------------------------------------------- #
# build_check_run: conclusion logic                                            #
# --------------------------------------------------------------------------- #
def test_check_run_name_and_head_sha():
    cr = build_check_run([], head_sha="deadbeef")
    assert cr["name"] == "openrabbit"
    assert cr["head_sha"] == "deadbeef"


def test_no_findings_is_success():
    cr = build_check_run([], head_sha="abc")
    assert cr["conclusion"] == "success"


def test_advisory_default_is_neutral_with_findings():
    """Default (gate=None) is advisory-only: never fail, even on critical."""
    findings = [make_finding(severity="critical")]
    cr = build_check_run(findings, head_sha="abc")
    assert cr["conclusion"] == "neutral"


def test_blocking_gate_fails_when_finding_meets_severity():
    findings = [make_finding(severity="high")]
    cr = build_check_run(findings, head_sha="abc", gate="high")
    assert cr["conclusion"] == "failure"


def test_blocking_gate_fails_on_higher_severity():
    """A critical finding meets a 'high' gate (critical >= high)."""
    findings = [make_finding(severity="critical")]
    cr = build_check_run(findings, head_sha="abc", gate="high")
    assert cr["conclusion"] == "failure"


def test_blocking_gate_neutral_when_below_severity():
    """Findings below the gate -> neutral (advisory), not failure or success."""
    findings = [make_finding(severity="medium"), make_finding(severity="low")]
    cr = build_check_run(findings, head_sha="abc", gate="high")
    assert cr["conclusion"] == "neutral"


def test_blocking_gate_success_when_no_findings():
    cr = build_check_run([], head_sha="abc", gate="high")
    assert cr["conclusion"] == "success"


def test_invalid_gate_raises():
    with pytest.raises(ValueError):
        build_check_run([make_finding()], head_sha="abc", gate="bogus")


# --------------------------------------------------------------------------- #
# build_check_run: output block                                               #
# --------------------------------------------------------------------------- #
def test_output_title_and_summary_present():
    findings = [make_finding(severity="high"), make_finding(severity="low")]
    cr = build_check_run(findings, head_sha="abc")
    out = cr["output"]
    assert out["title"]
    assert out["summary"]
    assert "2" in out["summary"]  # mentions the count


def test_output_annotations_one_per_finding():
    findings = [
        make_finding(file="src/a.py", fingerprint="f1"),
        make_finding(file="src/b.py", fingerprint="f2"),
    ]
    cr = build_check_run(findings, head_sha="abc")
    anns = cr["output"]["annotations"]
    assert len(anns) == 2
    assert {a["path"] for a in anns} == {"src/a.py", "src/b.py"}


def test_status_completed_with_conclusion():
    cr = build_check_run([make_finding()], head_sha="abc")
    assert cr["status"] == "completed"
    assert "conclusion" in cr


# --------------------------------------------------------------------------- #
# post_check_run: HTTP wiring                                                  #
# --------------------------------------------------------------------------- #
def test_post_check_run_single_post():
    resp = FakeResponse(201, {"id": 999, "conclusion": "neutral"})
    client = FakeClient([("POST", "/repos/acme/widget/check-runs", resp)])
    adapter = make_adapter(client)
    findings = [make_finding(severity="critical")]

    result = post_check_run(adapter, findings, head_sha="cafef00d")

    posts = [r for r in client.requests if r.method == "POST"]
    assert len(posts) == 1
    assert "/repos/acme/widget/check-runs" in posts[0].url
    body = posts[0].json
    assert body["name"] == "openrabbit"
    assert body["head_sha"] == "cafef00d"
    assert body["conclusion"] == "neutral"  # advisory default
    assert body["output"]["annotations"][0]["path"] == "src/agent.py"
    assert result["id"] == 999


def test_post_check_run_passes_gate_through():
    resp = FakeResponse(201, {"id": 1, "conclusion": "failure"})
    client = FakeClient([("POST", "/repos/acme/widget/check-runs", resp)])
    adapter = make_adapter(client)

    post_check_run(
        adapter, [make_finding(severity="critical")], head_sha="abc", gate="high"
    )

    body = [r for r in client.requests if r.method == "POST"][0].json
    assert body["conclusion"] == "failure"


def test_post_check_run_raises_on_http_error():
    resp = FakeResponse(422, {"message": "Validation Failed"})
    client = FakeClient([("POST", "/repos/acme/widget/check-runs", resp)])
    adapter = make_adapter(client)
    with pytest.raises(GitHubError):
        post_check_run(adapter, [make_finding()], head_sha="abc")


def test_post_check_run_chunks_annotations_over_50(monkeypatch):
    """GitHub caps annotations at 50 per Check-Run request; the adapter must
    create with the first 50 then PATCH the rest in batches of <=50."""
    findings = [
        make_finding(file=f"src/f{i}.py", fingerprint=f"fp{i}") for i in range(120)
    ]
    create = FakeResponse(201, {"id": 4242, "conclusion": "neutral"})
    patch = FakeResponse(200, {"id": 4242})
    client = FakeClient(
        [
            ("POST", "/repos/acme/widget/check-runs", create),
            ("PATCH", "/repos/acme/widget/check-runs/4242", patch),
        ]
    )
    adapter = make_adapter(client)

    post_check_run(adapter, findings, head_sha="abc")

    posts = [r for r in client.requests if r.method == "POST"]
    patches = [r for r in client.requests if r.method == "PATCH"]
    # 1 create (50) + 2 patches (50 + 20) = 120 annotations total
    assert len(posts) == 1
    assert len(patches) == 2
    assert len(posts[0].json["output"]["annotations"]) == 50
    assert len(patches[0].json["output"]["annotations"]) == 50
    assert len(patches[1].json["output"]["annotations"]) == 20
