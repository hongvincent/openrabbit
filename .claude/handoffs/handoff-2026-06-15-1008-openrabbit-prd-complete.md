# Session Handoff

> Created: 2026-06-15 10:08

## Summary

Built **openrabbit** — a high-trust, Bedrock-only AI code reviewer (CodeRabbit/Greptile-class) — from a blank repo to a complete v1: 3 research rounds → approved design spec → **PRD items 1–19 fully implemented** by an autonomous self-paced `/loop` (6 dynamic workflows, each TDD + adversarial review, every iteration verified with real `pytest` before commit). **910 tests / 99% coverage / ruff clean, 11 commits, 117 files.** Only 2 human-gated items remain (live Bedrock smoke; private-repo push).

## Project Facts (read first)

- **Repo:** `~/SIDE/coderabbit-alternative` — git inited, **NOT pushed to any remote**. `gh` authed as `hongvincent`.
- **Active branch:** `phase-0/dogfood-core` (all code + spec; last commit `1204aa7`). Tree is CLEAN.
- **Other branch:** `spec/openrabbit-design` (`9cb8191`) — now an ancestor of phase-0, so redundant; safe to delete (`git branch -d spec/openrabbit-design`). `main` is empty.
- **Spec / PRD source of truth:** `docs/superpowers/specs/2026-06-14-openrabbit-design.md`.
- **Prior handoff (the PRD checklist, all 1–19 now ✅):** `.claude/handoffs/handoff-2026-06-14-2232-openrabbit-prd-autoloop.md`.
- **Memory:** `~/.claude/projects/-Users-sungminnetworks-SIDE-coderabbit-alternative/memory/openrabbit-project.md` (8 locked decisions + completion status).
- **Tests:** `uv run --extra dev pytest -q` (910 passed, 99%). **Lint:** `uv run --extra dev ruff check .` (clean).
- **Offline demo (no creds):** `uv run python -m openrabbit.cli review --offline --diff <f> --fixtures demo`.

## Completed Tasks (PRD items 1–19)

- [x] **Research** — 3 background workflows (OSS landscape · big-tech/eval/official-docs · Bedrock harness specifics)
- [x] **Design spec** — 8 locked decisions, committed
- [x] **Phase 0** — findings JSON contract+schema, neutral domain model, Provider base+FakeProvider, ConverseAdapter (Nova/Claude) + OpenAIResponsesAdapter (GPT-5.5), GitHub adapter (batched createReview + committable suggestions + sticky walkthrough + GraphQL resolve/OUTDATED, advisory-only), eval harness v0, review SKILL.md lenses + loader, deterministic pipeline spine, CLI offline mode
- [x] **Phase 1 (1–6)** — console-script entry; GitEnclosingFetcher (symlink-safe, wired); incremental fingerprint persistence; reusable SHA-pinned Action + OIDC; all 5 lenses (correctness/security/perf/tests/maintainability); walkthrough enrichment (grouped table + Mermaid)
- [x] **Phase 2 (7–10)** — learnings/memory + feedback (`learn` CLI, confidence down-weight); verifier batching + HIGH/CRITICAL scoping (`verify_min_severity`); prompt-caching wiring + per-PR cost telemetry (`pricing.py`); `model_roles` validation (`bedrock_models.py`)
- [x] **Phase 3 (11–13)** — `gh openrabbit init` (stack detect + scaffold, `--apply`-guarded); Claude Code plugin + private marketplace; org `.github` templates + ruleset + safe-settings
- [x] **Phase 4 (14–16)** — SARIF 2.1.0 + Check-Run adapters (GHAS tier); optional GitHub App mode (`openrabbit/app/`: RS256 JWT + HMAC webhook + framework-free server); codebase index (`openrabbit/index/`: ast+regex SymbolIndex + embeddings interface, offline-safe)
- [x] **Cross-cutting (17–19)** — dogfood eval runner + `openrabbit eval` CLI; README + docs/ + CycloneDX SBOM + OpenSSF Scorecard workflow; CI (ruff + pytest `--cov-fail-under=95`, Py 3.11/3.12, SHA-pinned) + Dependabot
- [x] **Security hardening (from adversarial reviews):** symlink-traversal leak (HIGH), untrusted-fence prompt-injection ×2 (HIGH/MED), 2 CRITICAL onboarding wiring bugs, Mermaid/markdown injection, webhook fail-closed — all fixed + regression-tested
- [x] **Small test:** real `smnt-agent-core` commit diff flows through the pipeline → valid review payload (plumbing verified, offline)

## In Progress

- None. The autonomous loop is STOPPED (all non-blocked items done).

## Next Steps (the only 2 remaining — both need the user)

1. **#20 — Live Bedrock smoke (creds-blocked).** Prove the real path end-to-end (Nova finder → GPT-5.5 verifier) and the FP<10% number on a real repo. Everything is wired; needs AWS creds:
   ```
   ! aws sso login                # or: export AWS_BEARER_TOKEN_BEDROCK=...
   openrabbit eval --repo ~/SIDE/smnt-agent-core --online --require-pass   # real FP<10% measurement
   # or a single live PR review:
   openrabbit review --repo <owner/name> --pr <n> --commit <sha> --config .openrabbit.yaml --post
   ```
   Expect to tune prompts/threshold if the real FP is high.
2. **#21 — Create + push the private GitHub repo (user-confirm).** Outward-facing; not done yet:
   ```
   gh repo create openrabbit --private --source . --remote origin
   git push -u origin phase-0/dogfood-core   # (consider merging to main first)
   ```
3. **Optional cleanup:** delete the redundant `spec/openrabbit-design` branch; decide a branch strategy (phase-0/dogfood-core → main) before pushing.

## Key Decisions

| Decision | Reason |
|----------|--------|
| Bedrock-only, multi-model BYO (Nova finder → GPT-5.5 verifier; Claude optional) | Only Bedrock credits; Claude-on-Bedrock expensive |
| Action-first hybrid + S3/DynamoDB (no public service v1) | Supply-chain caution; reusable via Actions; App is a Phase-4 option |
| Deterministic spine + bounded escalation; curated/JIT MCP (not "unlimited") | Anthropic guidance; noise/cost/attack-surface control |
| Own model-neutral harness (2 adapters: Converse + OpenAI-Responses) | GPT-5.5 only on bedrock-mantle Responses, not Converse |
| Trust > coverage; find-broad/filter-strict; FP<10% gate; eval ships day 1 | Big-tech lesson: noisy bots get muted (step function) |
| agentskills.io SKILL.md + .openrabbit.yaml + JSON findings contract | Vendor/agent/model neutrality |

## Important Context

- **Everything live is mocked in tests.** No real Bedrock or GitHub API call has been exercised — unit tests use FakeProvider / mocked httpx / temp git repos. Item #20 is the first real call.
- **Invariants baked in:** cloud SDKs lazy-imported (core imports with zero deps); no network in unit tests; advisory-only (no merge/approve/push in the reasoning path); diff/PR/webhook text fenced as untrusted data; `store:false` on Bedrock Responses; all GitHub Actions SHA-pinned; OIDC keyless.
- **Build method:** autonomous `/loop` (dynamic mode) → one dynamic `Workflow` per ~3-item iteration (sequential TDD build → verify → parallel code+security review → fix). Survived a session-limit interruption mid-iter5 (committed complete part, removed partial, resumed post-reset). To continue similarly: re-launch `/loop` with the same converge-the-PRD prompt.
- **Repo layout:** `openrabbit/{pipeline,providers,adapters,app,index,eval}` + `findings.py/config.py/learnings.py/pricing.py/bedrock_models.py/init.py/cli.py` · `skills/` (SKILL.md lenses) · `plugin/` · `actions/` + `.github/workflows/` · `org/` · `cli/gh-openrabbit/` · `docs/` · `tests/` (38 files).

## Files Modified

Entire repo (117 tracked files) — built this session. Key: `openrabbit/**`, `skills/**`, `tests/**`, `.github/workflows/{ci,scorecard}.yml`, `actions/**`, `org/**`, `plugin/**`, `cli/gh-openrabbit/**`, `docs/**`, `README.md`, `pyproject.toml`, `sbom.json`, the design spec, and both handoffs.

## Blockers

- **#20 live Bedrock smoke:** needs AWS creds (`aws sso login` / `AWS_BEARER_TOKEN_BEDROCK`).
- **#21 private repo push:** needs explicit user go-ahead (outward-facing).
