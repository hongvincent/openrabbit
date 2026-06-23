# Tuning guide — reasoning effort, cost, and prompt cache

openrabbit routes a PR through three (optionally four) **roles** — `triage`,
`finder`, `verifier`, and an off-by-default `premium` — each a BYO Bedrock model
in `.openrabbit.yaml`'s `model_roles`. This guide is the per-role / per-lens
**reasoning-effort** decision table, the **confirmed** Nova 2 extended-thinking
request shape, the GPT-5.4/5.5 `reasoning.effort` guidance, rough cost notes, the
Nova Pro deprecation rationale, and the prompt-cache budget.

The headline lever: reasoning is **billed as output tokens**. Turning it on buys
recall/precision on hard findings but costs real money, so the defaults keep it
OFF on the cheap broad pass and ON only where it pays for itself.

## Per-role / per-lens reasoning_effort decision table

| Role | Model | Default reasoning | Per-lens nuance | Why |
|------|-------|-------------------|-----------------|-----|
| `triage` | Nova 2 Lite | **OFF** (temp 0) | n/a | near-free yes/no skip gate — a deterministic, temperature-0 pass; reasoning would just add cost. |
| `finder` (pattern / style lenses) | Nova 2 Lite | **OFF** | pattern, style → OFF | shallow surface lenses don't benefit from thinking; keep them cheap. |
| `finder` (logic / security / correctness / concurrency lenses) | Nova 2 Lite | **LOW** (opt-in) | logic, security, correctness, concurrency → `maxReasoningEffort: low` | deep lenses gain recall from a little reasoning; `low` is the cost-aware setting. Ships commented in the scaffold (default stays OFF/safe). |
| `verifier` | GPT-5.4 | **medium** | escalate to **high** for security / deep findings | the cross-family judge; medium is the calibrated default, high for the highest-stakes findings. |
| `premium` (off by default) | GPT-5.4 | **high** (→ `xhigh` available) | high for everything; `xhigh` for the hardest PRs | cost-gated highest-stakes role; only flip on when the spend is justified. |

`OFF` means **no** `reasoningConfig` / `reasoning_effort` key at all (Nova
defaults `type` to `disabled`). The finder reasoning path uses the **same
`global.amazon.nova-2-lite-v1:0` profile** the finder already runs on
(`ap-northeast-2` / Seoul) — this is **live-verified**: that global profile
accepts `reasoningConfig` and returns `reasoningContent` + `text`, so enabling
finder reasoning needs **no** `us.*` cross-region profile switch.

## Confirmed Nova 2 extended-thinking request shape

This was previously TBD; it is now **CONFIRMED** against the AWS Nova 2 docs and
**live-verified** on `global.amazon.nova-2-lite-v1:0` in `ap-northeast-2`. Nova 2
takes extended thinking through the Converse `additionalModelRequestFields`:

```python
# bedrock-runtime Converse — finder LOW-effort reasoning (logic/security lenses)
response = client.converse(
    modelId="global.amazon.nova-2-lite-v1:0",   # Seoul; live-verified
    messages=messages,
    inferenceConfig={"maxTokens": 16000},        # low → keep maxTokens >= 15000
    additionalModelRequestFields={
        "reasoningConfig": {
            "type": "enabled",                    # default is "disabled" (= OFF)
            "maxReasoningEffort": "low",          # "low" | "medium" | "high"
        }
    },
)
# The reply has a reasoningContent block ([REDACTED] thinking text) alongside the
# normal text block. Reasoning tokens are billed as OUTPUT ($2.50/MTok).
```

Rules that are load-bearing for correctness (a `ValidationException` otherwise):

- **`type`** defaults to `"disabled"`. Set `"enabled"` to turn reasoning on.
- **`maxReasoningEffort`** is one of `low`, `medium`, `high`.
- **`high` requires omitting `temperature`, `topP`, and `topK`** from
  `inferenceConfig` — passing any of them with high effort raises
  `ValidationException`. (low/medium tolerate them but temp 0 is fine.)
- **`low` recommends `inferenceConfig.maxTokens >= 15000`** so the model has room
  for the thinking budget plus the answer.
- Reasoning tokens come back in a separate `reasoningContent` block (the text is
  `[REDACTED]`) and are **billed as output tokens** ($2.50/MTok on Nova 2 Lite).

## GPT-5.4 / GPT-5.5 reasoning.effort guidance

The `openai.*` verifier/premium roles run over Bedrock's OpenAI-compatible
*mantle* Responses endpoint (`us-east-1` / `us-east-2` only) and take
`reasoning_effort` as a provider option in `.openrabbit.yaml`:

| Role | `reasoning_effort` | When |
|------|--------------------|------|
| `verifier` | **medium** (default) | the calibrated cross-family judge for normal findings. |
| `verifier` | **high** | security findings and deep/high-severity findings that need the extra rigor. |
| `premium` | **high** | the off-by-default highest-stakes role. |
| `premium` | **xhigh** | the hardest PRs — note `xhigh` is **untested on the mantle** endpoint, so treat it as experimental and watch for `ValidationException`. |

Both `openai.gpt-5.4` and `openai.gpt-5.5` accept these efforts; the
live-verified default verifier id is **gpt-5.4**.

## Rough per-role cost notes

Reasoning tokens are **billed as output tokens**, so every effort bump is a
direct output-token multiplier. Rough mental model:

- **`triage`** — Nova 2 Lite, reasoning OFF: near-free per PR; it exists to skip
  trivial diffs before any expensive role runs.
- **`finder`** — Nova 2 Lite, OFF for pattern/style, LOW for logic/security:
  cheap broad pass. Reasoning `low` adds output tokens (Nova 2 Lite output is
  ~$2.50/MTok, and the `reasoningContent` block counts as output), so enable it
  only on the deep lenses where recall matters.
- **`verifier`** — GPT-5.4 medium: the dominant per-PR cost; widening
  `verify_min_severity` (configuration.md's cost lever) routes more findings here
  and `high` effort multiplies its output-token bill on security findings.
- **`premium`** — GPT-5.4 high/`xhigh`: most expensive per call; keep it off by
  default and gate it to the highest-stakes PRs.

Cut spend with `verify_min_severity` (route fewer findings to the verifier),
prompt caching (below), and by keeping reasoning OFF on the cheap roles.

## Nova Pro deprecation

`amazon.nova-pro-v1:0` is **deprecated** for new roles — prefer
**`global.amazon.nova-2-lite-v1:0`**. Nova Pro is not weak (HumanEval ~89%), but
it is disqualified as a finder and superseded everywhere:

- **Hard 5K output cap** (vs Nova 2 Lite's 64K) — a high-recall report-all finder
  routinely needs more than 5K output tokens.
- **No reasoning path** — there is no `reasoningConfig` extended-thinking option,
  so it cannot do the LOW-effort finder reasoning above.
- **No `global.` profile** — it lacks the cross-region inference profile the rest
  of the routing relies on.
- **Higher cost**, an **Oct-2024 cutoff**, **no Korean**, and a general **EOL
  signal**.

It is kept in the registry / price table for backward-compatibility only, and is
annotated as deprecated there.

## Prompt-cache notes

Bedrock prompt caching cuts repeated-prefix cost (the skill prompts + fenced
context are stable across a PR's lenses), within these limits:

- **~20K-token cap** on what a single cache entry holds — keep the cached prefix
  (skills + system framing) under it.
- **4 checkpoints** maximum per request (cache checkpoints) — budget them across
  the stable prefix segments.
- **5-minute TTL** — the cache is short-lived, so it helps within a single PR's
  burst of lens calls but not across PRs minutes apart.
