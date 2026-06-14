# Session Handoff

> Created: 2026-06-14 22:32

## Summary

Built **openrabbit** (a high-trust, Bedrock-only AI code reviewer) from scratch: 3 research rounds → approved design spec → Phase 0 "dogfood core" implemented, verified (360 tests / 98% cov), and committed. Next: an **autonomous loop** drives the PRD (the design spec) to **100% completion** (Phases 1→4 + creds-blocked live smoke).

## Project Facts (read these first)

- **Repo:** `~/SIDE/coderabbit-alternative` (git inited; NOT yet pushed to a GitHub remote). `gh` authed as `hongvincent`.
- **PRD / source of truth:** `docs/superpowers/specs/2026-06-14-openrabbit-design.md` (commit `9cb8191`).
- **Branches:** `main` (empty), `spec/openrabbit-design` (spec), `phase-0/dogfood-core` (CURRENT — spec + Phase 0 code, commit `12c1a20`).
- **Run tests:** `uv run pytest -q` (360 passed, 98% cov). **Offline demo:** `uv run python -m openrabbit.cli review --offline --diff <f> --fixtures demo`.
- **Memory:** `~/.claude/projects/-Users-sungminnetworks-SIDE-coderabbit-alternative/memory/openrabbit-project.md` has all 8 locked decisions.

## Completed Tasks

- [x] 3 research workflows (OSS landscape · big-tech/eval/official-docs · Bedrock harness specifics)
- [x] Design spec written, self-reviewed, committed (8 locked decisions)
- [x] **Phase 0 dogfood core** (55 files, 360 tests, 98% cov): findings contract+schema, neutral domain model, Provider base+FakeProvider, ConverseAdapter (Nova/Claude) + OpenAIResponsesAdapter (GPT-5.5), GitHub adapter (batched createReview + committable suggestions + sticky walkthrough + GraphQL resolve/OUTDATED, advisory-only), eval harness v0 (golden-set/clean-PR-controls/LLM-judge/scorecard), review SKILL.md lenses (correctness+security)+loader, deterministic pipeline spine, CLI with offline mode
- [x] Adversarial code+security review applied (fixed 3 HIGH: tool_choice neutrality, suggestion double-fence, lens-rubric wiring)
- [x] Small test on `smnt-agent-core`: real commit diff flows through pipeline → valid review payload (plumbing verified)

## In Progress

- Setting up the autonomous **loop** (dynamic/self-paced) to complete the PRD.

## Next Steps — PRD COMPLETION CHECKLIST (loop drives this to 100%)

Each item = a dynamic workflow (TDD + adversarial review), then `uv run pytest`, then commit. Mark `[x]` as done.

### Phase 1 — Action packaging & incremental UX
1. [ ] `[project.scripts] openrabbit = "openrabbit.cli:main"` in pyproject (bare `openrabbit` command)
2. [ ] Wire `gather_enclosing_context` to real `grep`/`read_file`(window)/`git log`/`git blame` symbol pre-fetch (behind the existing `EnclosingFetcher` hook)
3. [ ] Incremental review fingerprint persistence in StateStore (last_reviewed_sha already done) + `synchronize` re-review only new commits
4. [ ] Reusable GitHub Action (`actions/reusable-workflow.yml` + composite), SHA-pinned, `permissions: contents: read` + minimal write; OIDC→STS keyless Bedrock auth in the workflow
5. [ ] Remaining lenses as SKILL.md: performance, tests, maintainability (+ wire into route/run_lenses)
6. [ ] Walkthrough enrichment: grouped changed-files table + Mermaid for interaction changes

### Phase 2 — Trust & cost
7. [ ] Learnings/memory store + feedback loop (dismissal → confidence recalibration + auto-suppression); per-repo/org scope
8. [ ] Verifier formalization: HIGH/CRITICAL scoping + batching (kill the N+1)
9. [ ] Prompt-caching wiring end-to-end (stable prefix; Converse cachePoint + Responses prompt_cache_key) + per-PR Usage cost telemetry
10. [ ] Model routing tuning hooks + `.openrabbit.yaml` validation of model_roles against Bedrock allow-lists/regions

### Phase 3 — Onboarding & skillification
11. [ ] `gh openrabbit init` (gh extension): detect stack → write `.openrabbit.yaml` + SHA-pinned caller workflow → OIDC trust → register secrets → marketplace+plugin enable
12. [ ] `plugin/.claude-plugin/{plugin.json, marketplace.json}` (namespaced `openrabbit:review` plugin) for private marketplace distribution
13. [ ] `org/` templates: `.github` org starter workflow-template + reusable workflow + org ruleset (SHA-pinned required check) + Safe-Settings example

### Phase 4 — Enterprise (optional, gated)
14. [ ] SARIF output adapter (Security-tab tier, GHAS-gated) + Check-Run adapter (merge gating)
15. [ ] GitHub App mode (webhooks, installation tokens) as an alternative to Action-first
16. [ ] Greptile-style codebase index (tree-sitter symbol/call graph + embeddings; LanceDB-on-S3 or pgvector) behind an embeddings interface

### Cross-cutting / quality gates
17. [ ] **Dogfood eval on `smnt-agent-core`**: build golden set from merged PRs + reverts/hotfixes → run scorecard → prove **FP < 10%** at the default gate (tune prompts/threshold)
18. [ ] README + docs (usage, config reference, security model, onboarding); SBOM + OpenSSF Scorecard config
19. [ ] CI for openrabbit itself (lint + pytest + coverage gate; Dependabot for action SHAs)
20. [ ] **[CREDS-BLOCKED] Live Bedrock smoke**: one real end-to-end review (Nova finder → GPT-5.5 verify) on a real `smnt-agent-core` PR diff. Needs `AWS_BEARER_TOKEN_BEDROCK` or `aws sso login`. Skip in loop until creds present; surface as the top "needs human" item.
21. [ ] **[USER-CONFIRM] Create private GitHub repo `openrabbit` (Apache-2.0) + push.** Outward-facing — confirm with user before pushing.

## Key Decisions

| Decision | Reason |
|----------|--------|
| Bedrock-only, multi-model BYO (Nova finder → GPT-5.5 verifier; Claude optional) | User has only Bedrock credits; Claude-on-Bedrock expensive |
| Action-first hybrid + S3/DynamoDB (no public service in v1) | Supply-chain caution; reusable via GitHub Actions; App is Phase 4 |
| Deterministic spine + bounded escalation; "curated/JIT MCP" not "unlimited" | Anthropic guidance; noise/cost/attack-surface control |
| Own model-neutral harness (2 adapters: Converse + OpenAI-Responses) | GPT-5.5 only on bedrock-mantle Responses; not Converse |
| Trust > coverage; find-broad/filter-strict; FP < 10% gate; eval ships day 1 | Big-tech lesson: noisy bots get muted (step function) |
| agentskills.io SKILL.md + .openrabbit.yaml + JSON findings contract | Vendor/agent/model neutrality ("선택 가능") |

## Important Context

- **Autonomy mandate (user):** run the loop self-paced; goal = PRD 100%. Each iteration: implement one checklist item via a dynamic `Workflow` (TDD + adversarial review), run `uv run pytest` (must stay green, ≥80% cov on touched modules), commit on `phase-0/dogfood-core` (or a phase branch), update this checklist. Items 20–21 need human input — do NOT block the loop on them; do everything else first, then surface them.
- **Invariants to preserve:** cloud SDKs lazy-imported (no top-level boto3/httpx); no network in unit tests (FakeProvider/mocks); advisory-only (no merge/approve/push in reasoning path); diff fenced as untrusted data.
- **Ultracode is ON** — use `Workflow` for each substantive item; adversarially verify.

## Files Modified (this session)

- `docs/superpowers/specs/2026-06-14-openrabbit-design.md` (spec)
- `openrabbit/**` (engine, providers, adapters, pipeline, eval, findings, config, lenses, cli)
- `skills/**` (review SKILL.md lenses)
- `tests/**` (360 tests), `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`, `.openrabbit.example.yaml`, `conftest.py`, `.gitignore`

## Blockers

- **Live Bedrock test (item 20):** no creds in env. Needs `aws sso login` / `AWS_BEARER_TOKEN_BEDROCK`.
- **Private repo push (item 21):** outward-facing; needs user confirmation.
