# openrabbit

A high-trust, low-noise, **model-neutral** AI code review harness.

openrabbit puts a thin, opinionated policy/calibration/eval layer on top of
trusted first-party building blocks. Production inference runs on **AWS Bedrock
only** (GPT-5.5 + Amazon Nova; Claude-on-Bedrock optional). Review intelligence
is portable across vendors/agents via `SKILL.md` + `.openrabbit.yaml` + a JSON
findings contract.

The product is **signal-to-noise**: a separate verifier scores every finding and
drops anything below a confidence gate, with an eval harness that proves a false
positive rate under 10% before any change ships.

See `docs/superpowers/specs/2026-06-14-openrabbit-design.md` for the full design.

## Status

Phase 0 — dogfood core. This repository is the Python reference implementation.

## Development

```bash
uv sync --all-extras       # install runtime + dev deps
uv run pytest -q           # run the test suite
```

Unit tests never make network calls and never require live AWS/GitHub
credentials; cloud SDKs are imported lazily inside functions.

## License

Apache-2.0 — see [LICENSE](./LICENSE).
