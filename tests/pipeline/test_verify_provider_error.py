"""A terminal verifier ProviderError must NOT abort the whole review.

Adversarial scenario (found in real live e2e): the GPT verifier hits OpenAI's
cyber-safety filter on a genuine SECURITY-vulnerability diff and the Responses
adapter raises a NON-retryable ``ProviderError`` ('... This request has been
flagged for potentially high-risk cyber activity'). That 4xx is not retryable,
so it escapes the providers' bounded retry/backoff and propagates straight out
of :func:`verify_findings` — killing the entire review. The highest-value
SECURITY PRs are exactly the ones that fail.

Phase A already made ``verify_findings`` fail SAFE when the verifier returns NO
usable verdict (a refusal CONTENT block) — fall back to finder confidence
through the gate. But a RAISED ``ProviderError`` is a DIFFERENT code path that
was NOT caught. This module pins the soft-skip behaviour:

* a verifier ``ProviderError`` must be caught (no propagation / abort);
* the stage falls back to the SAME fail-safe the refusal path uses (finder
  confidence run through the gate);
* a clear WARNING is logged distinguishing 'verifier unavailable' from a normal
  verdict, carrying the (truncated, secret-free) error text.

All OFFLINE (FakeProvider-style doubles), no network.
"""

from __future__ import annotations

import logging

from openrabbit.domain import CompletionResult, FinishReason, Message, ToolSpec, Usage
from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import verify as verify_mod
from openrabbit.providers.base import Provider, ProviderError

# The real, verbatim cyber-safety filter message that aborted a live review.
_CYBER_FLAG = (
    "Responses API returned an error status: This request has been flagged for "
    "potentially high-risk cyber activity"
)


def _finding(conf: float, *, rule: str, severity: str = "high") -> Finding:
    fp = compute_fingerprint("src/api/auth.py", rule, f"ctx-{rule}")
    return Finding(
        file="src/api/auth.py",
        start_line=12,
        end_line=14,
        side="RIGHT",
        severity=severity,
        category="security",
        confidence=conf,
        title="t",
        body="b",
        rule_id=rule,
        fingerprint=fp,
    )


class _RaisingVerifier(Provider):
    """A verifier whose ``complete`` raises a given exception on every call."""

    name = "fake"
    model = "fake-model"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def complete(  # type: ignore[override]
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        max_tokens: int,
        cache_prefix: str | None,
        **opts: object,
    ) -> CompletionResult:
        self.calls += 1
        raise self._exc


def test_verifier_provider_error_does_not_abort_review(caplog):
    """A terminal verifier ProviderError must NOT propagate; fail SAFE instead.

    The verifier raises the real cyber-safety ``ProviderError``. ``verify_findings``
    must NOT re-raise it. It falls back to finder confidence through the gate, so
    the 0.90 HIGH finding (>= 0.80 gate) survives and still posts.
    """
    findings = [_finding(0.90, rule="r1")]
    verifier = _RaisingVerifier(ProviderError(_CYBER_FLAG))

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)

    assert verifier.calls == 1, "the verifier should have been called exactly once"
    assert [f.rule_id for f in kept] == ["r1"], (
        "a terminal verifier ProviderError must fall back to finder confidence "
        "through the gate, not abort the review and drop the finding"
    )


def test_verifier_provider_error_applies_gate_on_fallback(caplog):
    """Fallback is finder confidence THROUGH the gate, not unconditional keep."""
    findings = [
        _finding(0.95, rule="keep"),
        _finding(0.50, rule="drop"),  # below gate on finder confidence
    ]
    verifier = _RaisingVerifier(ProviderError(_CYBER_FLAG))

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)

    assert {f.rule_id for f in kept} == {"keep"}, (
        "the soft-skip fallback must still apply the confidence gate"
    )


def test_verifier_provider_error_logs_warning_with_reason(caplog):
    """The warning must flag 'verifier unavailable' and carry the error reason."""
    findings = [_finding(0.90, rule="r1")]
    verifier = _RaisingVerifier(ProviderError(_CYBER_FLAG))

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        verify_mod.verify_findings(verifier, findings, gate=0.80)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a soft-skipped verifier ProviderError must emit a WARNING"
    msg = "\n".join(r.getMessage() for r in warnings).lower()
    assert "verifier unavailable" in msg, (
        "the warning must distinguish a verifier-unavailable soft-skip from a "
        "normal verdict"
    )
    assert "providererror" in msg, "the warning must name the error type"
    assert "high-risk cyber activity" in msg, (
        "the (truncated, secret-free) error reason must be logged so an operator "
        "can see WHY the verifier was skipped"
    )


def test_verifier_provider_error_does_not_leak_via_long_message(caplog):
    """The logged error text is truncated (no unbounded blob / secret leakage)."""
    blob = "X" * 5000
    findings = [_finding(0.90, rule="r1")]
    verifier = _RaisingVerifier(ProviderError(blob))

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        verify_mod.verify_findings(verifier, findings, gate=0.80)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    logged = "\n".join(r.getMessage() for r in warnings)
    assert blob not in logged, (
        "the full provider error must be truncated before logging, not dumped "
        "verbatim (avoid leaking large/sensitive payloads)"
    )


def test_transient_error_is_handled_by_provider_retries_not_here(caplog):
    """Retry of transient (429/5xx) errors is the PROVIDER's job, not verify's.

    The bounded retry/backoff lives in the concrete providers
    (``openai_responses`` / ``converse``); they retry transient statuses and
    only surface a ``ProviderError`` once it is terminal (a non-retryable 4xx or
    exhausted backoff). So whatever ``ProviderError`` reaches ``verify_findings``
    — including one whose message says retries were exhausted — is, by design,
    soft-skipped here rather than retried again or re-raised. This pins that
    boundary: verify.py must NOT swallow a transient error before retries get a
    chance (it never sees one) and must NOT abort once a terminal one arrives.
    """
    exhausted = ProviderError(
        "Responses API request failed after 4 attempts: Service Unavailable"
    )
    findings = [_finding(0.90, rule="r1")]
    verifier = _RaisingVerifier(exhausted)

    with caplog.at_level(logging.WARNING, logger=verify_mod.__name__):
        kept = verify_mod.verify_findings(verifier, findings, gate=0.80)

    # Called exactly once: verify_findings does NOT add its own retry loop on top
    # of the provider's (which would double-charge / double-flag).
    assert verifier.calls == 1
    assert [f.rule_id for f in kept] == ["r1"], (
        "a terminal ProviderError (even one from exhausted retries) soft-skips "
        "the verifier rather than aborting the review"
    )


def test_non_provider_exception_still_propagates():
    """Only ProviderError is soft-skipped; a real bug must NOT be swallowed.

    Catching too broadly would hide genuine programming errors (KeyError,
    ValueError, ...) behind a silent fail-safe. Those must still surface.
    """
    import pytest

    findings = [_finding(0.90, rule="r1")]
    verifier = _RaisingVerifier(RuntimeError("boom: a real bug, not a provider issue"))

    with pytest.raises(RuntimeError, match="boom"):
        verify_mod.verify_findings(verifier, findings, gate=0.80)


def test_normal_verdict_path_unaffected_by_soft_skip():
    """The happy path is untouched: a real verdict array still filter-strict drops."""
    from openrabbit.domain import ToolCall

    result = CompletionResult(
        text="",
        tool_calls=[
            ToolCall(
                id="v1",
                name="verify_findings",
                args={
                    "verdicts": [
                        {"id": 0, "keep": True, "confidence": 0.95, "rationale": "ok"}
                    ]
                },
            )
        ],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(),
    )

    class _OnceVerifier(Provider):
        name = "fake"
        model = "fake"

        def __init__(self) -> None:
            self._r = result

        def complete(self, *a, **k):  # type: ignore[override]
            return self._r

    findings = [_finding(0.95, rule="r1"), _finding(0.95, rule="r2")]
    kept = verify_mod.verify_findings(_OnceVerifier(), findings, gate=0.80)
    # r2 has no verdict in a valid array -> dropped (filter-strict preserved).
    assert [f.rule_id for f in kept] == ["r1"]
