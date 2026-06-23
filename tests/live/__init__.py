"""Live, no-cheat Bedrock e2e tests (opt-in via ``pytest -m live``).

These are the ONLY tests in the suite that make REAL, paid AWS Bedrock calls.
They exist to catch real-API-shape bugs that 1000+ mock-based tests cannot —
usage-key names, function_call wire shape, strict-schema acceptance, bearer
auth, and the eval-online wiring.

Every test here is marked ``@pytest.mark.live`` and is EXCLUDED from the default
run (``addopts = -m "not live"`` in pyproject.toml), so offline CI stays green
and free. When credentials are unavailable the fixtures :func:`pytest.skip`
(never fail), so even ``pytest -m live`` is safe without creds.
"""
