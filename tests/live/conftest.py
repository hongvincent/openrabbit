"""Credential-gated fixtures for the live Bedrock e2e tests.

Design contract (so the default offline suite stays green AND free):

* Every fixture here SKIPS the test (``pytest.skip``) — never fails — when the
  credential it guards is unavailable. A missing profile / un-mintable bearer
  token is an environment gap, not a product bug.
* These fixtures perform NO network calls themselves beyond the minimal STS
  identity check used to confirm the ``openrabbit`` AWS profile authenticates.
* Nothing here ever prints, logs, or returns the bearer token in a way that
  could leak it into CI logs (the token value is passed straight to the adapter
  via the environment).

The verified recipe these fixtures encode (do not change without re-validating
against real Bedrock):

* AWS profile ``openrabbit`` authorizes BOTH paths (Nova converse + GPT-5.5).
* Nova finder/triage: region ``ap-northeast-2``, model id MUST be the inference
  profile ``apac.amazon.nova-pro-v1:0`` (the bare id fails ValidationException).
* GPT-5.5 verifier/judge: region ``us-east-2``, model ``openai.gpt-5.5``, auth
  via a short-lived bearer token minted with ``aws-bedrock-token-generator`` and
  read by the adapter from ``AWS_BEARER_TOKEN_BEDROCK``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

#: The least-privilege AWS profile that authorizes both Bedrock paths.
AWS_PROFILE = "openrabbit"

#: Region/model wiring shared across the live tests (the verified recipe).
NOVA_REGION = "ap-northeast-2"
NOVA_MODEL = "apac.amazon.nova-pro-v1:0"  # inference profile — bare id is rejected
GPT_REGION = "us-east-2"
GPT_MODEL = "openai.gpt-5.5"


@pytest.fixture(scope="session")
def aws_profile() -> str:
    """Return the ``openrabbit`` profile name after proving it authenticates.

    SKIPS (does not fail) when boto3 is missing, the profile is absent, or STS
    GetCallerIdentity fails — any of which means "no live creds here", which is
    an environment gap and must not break the suite.
    """
    try:
        import boto3  # lazy: keep module import AWS-free for the offline suite
        from botocore.exceptions import BotoCoreError, ClientError
    except Exception as exc:  # pragma: no cover - exercised only without boto3
        pytest.skip(f"boto3 unavailable for live tests: {exc}")

    try:
        session = boto3.Session(profile_name=AWS_PROFILE)
    except Exception as exc:
        pytest.skip(
            f"AWS profile {AWS_PROFILE!r} not configured (skipping live tests): {exc}"
        )

    # Cheapest possible proof the profile's creds are valid & resolvable.
    try:
        session.client("sts").get_caller_identity()
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(
            f"AWS profile {AWS_PROFILE!r} failed to authenticate "
            f"(skipping live tests): {type(exc).__name__}"
        )
    return AWS_PROFILE


@pytest.fixture(scope="session")
def aws_profile_env(aws_profile: str) -> Iterator[str]:
    """Ensure ``AWS_PROFILE`` is set for the duration of the live session.

    The Converse adapter builds its boto3 client with no explicit credentials,
    so it relies on the ambient ``AWS_PROFILE``. We set it here (restoring any
    prior value on teardown) so a live test need only depend on this fixture to
    get the right profile wired in, regardless of how pytest was invoked.
    """
    previous = os.environ.get("AWS_PROFILE")
    os.environ["AWS_PROFILE"] = aws_profile
    try:
        yield aws_profile
    finally:
        if previous is None:
            os.environ.pop("AWS_PROFILE", None)
        else:
            os.environ["AWS_PROFILE"] = previous


@pytest.fixture(scope="session")
def bearer_token() -> Iterator[str]:
    """Mint a short-lived Bedrock bearer token and expose it via the environment.

    Sets ``AWS_BEARER_TOKEN_BEDROCK`` (which ``OpenAIResponsesAdapter`` reads)
    for the duration of the test, restoring any prior value on teardown. SKIPS
    when the token generator is unavailable or minting fails (no creds here).

    The token value is NEVER returned, printed, or logged — only the
    environment carries it to the adapter. The fixture yields a redacted sentinel
    so a test can assert "a token was provisioned" without ever seeing it.
    """
    try:
        from aws_bedrock_token_generator import provide_token
    except Exception as exc:
        pytest.skip(
            "aws-bedrock-token-generator not installed (skipping GPT-5.5 live "
            f"tests): {exc}"
        )

    try:
        token = provide_token(region=GPT_REGION)
    except Exception as exc:  # broad: any minting failure means "no live creds"
        pytest.skip(
            f"could not mint Bedrock bearer token (skipping GPT-5.5 live tests): "
            f"{type(exc).__name__}"
        )

    if not token:
        pytest.skip("Bedrock bearer token minting returned empty (no live creds)")

    previous = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    try:
        # Yield a redacted sentinel, NOT the token, so no test can leak it.
        yield "<provisioned>"
    finally:
        if previous is None:
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        else:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = previous
