# Configuration — `.openrabbit.yaml`

openrabbit is configured by a single `.openrabbit.yaml` at the repo root.
[`.openrabbit.example.yaml`](../.openrabbit.example.yaml) is a copy-paste
starting point and `openrabbit init` scaffolds a stack-aware one for you.

It is **config-as-code**: additive and versioned. Unknown keys are tolerated
where the format is additive, but enum-constrained values are validated strictly
by `openrabbit.config.load_config` — a typo fails fast in CI rather than silently
degrading review quality.

## Full reference

```yaml
version: 1                       # config schema version (integer; currently 1)

review:
  profile: balanced              # chill | balanced | assertive
  confidence_gate: 0.80          # 0.0–1.0; drop findings below this calibrated confidence
  verify_min_severity: high      # critical | high | medium | low | nit
  incremental: true              # review diff since last_reviewed_sha when available
  path_filters:                  # glob filters; "!" excludes a path from review
    - "!**/dist/**"
    - "!**/*.lock"
    - "!**/generated/**"
  path_instructions:             # per-path reviewer focus
    - path: "src/api/**"
      instructions: "focus on authn/authz and input validation"
  lenses: [correctness, security, performance, tests, maintainability]
  lens_reasoning_effort:         # per-lens finder reasoning (Nova 2 extended-thinking)
    correctness: low             # {lens: low|medium|high}; omitted lens => reasoning OFF
    security: low                # threaded into the finder Converse reasoningConfig

model_roles:                     # role -> { model, region, ...provider opts }
  triage:
    model: global.amazon.nova-2-lite-v1:0   # Nova 2 Lite via "global." profile
    region: ap-northeast-2                  # Seoul (live-verified on Converse)
  finder:
    model: global.amazon.nova-2-lite-v1:0
    region: ap-northeast-2
    # Reasoning OFF by default (safe/cheap). The Nova 2 extended-thinking shape is
    # now CONFIRMED + live-verified — opt in per the tuning guide by adding:
    #   reasoning_effort: low   # Converse additionalModelRequestFields →
    #                           # reasoningConfig {type: enabled, maxReasoningEffort: low}
  verifier:
    model: openai.gpt-5.4
    region: us-east-2
    reasoning_effort: medium
    store: false
  premium:                       # optional, cost-gated; off by default
    model: openai.gpt-5.4
    region: us-east-2
    reasoning_effort: high
    enabled: false

external_tools:                  # RESERVED / not yet wired (see below)
  enabled: []                    # the pipeline does not run these graders yet

telemetry:
  enabled: true
  mode: opt-out                  # opt-in | opt-out
```

## `review`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `profile` | enum | `balanced` | review temperament: `chill`, `balanced`, or `assertive`. |
| `confidence_gate` | float `[0,1]` | `0.80` | findings with calibrated confidence below this are dropped. |
| `verify_min_severity` | severity | `high` | minimum severity routed through the (expensive, cross-family) **verifier**; less-severe findings take the cheaper finder-confidence path. One of `critical`, `high`, `medium`, `low`, `nit`. |
| `incremental` | bool | `true` | when a `last_reviewed_sha` is known, review only the diff since then. |
| `path_filters` | string list | `[]` | glob filters; a leading `!` excludes a path from review (e.g. lockfiles, generated/dist code). |
| `path_instructions` | list | `[]` | per-path reviewer focus; each entry is `{ path, instructions }`. |
| `lenses` | string list | all five | which review lenses run (see below). |
| `lens_reasoning_effort` | map | `{}` | per-lens finder reasoning effort (`{lens: low\|medium\|high}`); an omitted lens runs with reasoning OFF. The **only** wired finder-reasoning knob (see below). |

### `verify_min_severity` (the cost lever)

By default only **HIGH / CRITICAL** findings route through the expensive
cross-family verifier; everything below takes the cheaper finder-confidence +
gate path. Widen it (e.g. to `medium`) to verify more findings at higher cost, or
keep it tight to control spend. This is one of the biggest cost levers alongside
prefix prompt caching and tiering.

### Lenses

openrabbit runs up to five parallel **lenses**, each a portable `SKILL.md` that
reports findings with `confidence` + `severity` and never self-filters:

| Lens | Focus |
|------|-------|
| `correctness` | bugs, logic errors, edge cases, unsafe indexing |
| `security` | injection, authn/authz, secret handling, unsafe deserialization |
| `performance` | hot-path inefficiency, N+1 queries, needless allocation |
| `tests` | missing / weak / brittle test coverage for changed behavior |
| `maintainability` | clarity, naming, duplication, dead code |

List a subset to narrow scope (e.g. `lenses: [correctness, security]`).

## `model_roles`

Each role maps to a **BYO Bedrock** model id + region + provider options. `model`
and `region` are first-class; any other keys (`reasoning_effort`, `store`,
`enabled`, …) are preserved as provider options.

| Role | Purpose | Default |
|------|---------|---------|
| `triage` | near-free yes/no on the diff (skip trivial changes) | `global.amazon.nova-2-lite-v1:0` @ `ap-northeast-2` |
| `finder` | broad, high-recall report-all first pass | `global.amazon.nova-2-lite-v1:0` @ `ap-northeast-2` |
| `verifier` | cross-family judge; scores + drops below the gate | `openai.gpt-5.4` @ `us-east-2` |
| `premium` | optional, cost-gated highest-stakes role | `openai.gpt-5.4` (high) @ `us-east-2` (off by default) |

The `finder` ships with **no** reasoning by default (the cheap/safe default).
Nova 2 Lite's extended-thinking request shape (`additionalModelRequestFields` →
`reasoningConfig`) is now **CONFIRMED** and live-verified, so enabling low-effort
reasoning on the finder is a supported, documented opt-in rather than a deferred
follow-up — see [`docs/tuning-guide.md`](tuning-guide.md) for the request shape,
the per-lens plan, and the cost notes. Crucially, the finder reasoning path uses
the **same `global.amazon.nova-2-lite-v1:0` profile** the finder already runs on
(Seoul) — that global profile is **live-verified** to accept `reasoningConfig` and
return `reasoningContent`, so no `us.*` cross-region profile switch is needed. Both
`openai.gpt-5.4` and `openai.gpt-5.5` are supported verifier ids — the
live-verified default is **gpt-5.4**.

### Finder reasoning is configured per lens: `review.lens_reasoning_effort`

The finder's reasoning effort is wired through **exactly one** mechanism:
**`review.lens_reasoning_effort`** — a `{lens: low|medium|high}` map under
`review:`. There is **no** per-role `model_roles.finder.reasoning_effort` knob; the
pipeline reads the per-lens map and threads the matching effort into the finder's
Converse call as `additionalModelRequestFields.reasoningConfig`
(`{type: enabled, maxReasoningEffort: <effort>}`). A lens **omitted** from the map
runs with reasoning **OFF** (Nova 2 default `type: "disabled"`), i.e. a plain
non-thinking pass — so you pay for reasoning only on the lenses that benefit:

```yaml
review:
  lens_reasoning_effort:
    correctness: low    # logic / correctness — LOW-effort reasoning
    security: low       # security — LOW-effort reasoning
    # performance / tests / maintainability OMITTED => reasoning OFF (cheaper pass)
```

(The `verifier` / `premium` roles keep a first-class per-role `reasoning_effort`
provider option in `model_roles`, since they are single-call roles, not lens
fan-outs.)

### Per-role `reasoning_effort` guidance

Reasoning is billed as **output tokens**, so the defaults keep it OFF on the cheap
roles and ON where it pays for itself. The `finder` row below is driven by
`review.lens_reasoning_effort` (per lens), not a `model_roles.finder` key; the
`verifier` / `premium` rows are the per-role `model_roles.<role>.reasoning_effort`
option. Full table + the confirmed Nova 2 request shape live in
[`docs/tuning-guide.md`](tuning-guide.md); the summary:

| Role | Default | Escalation |
|------|---------|------------|
| `triage` | **OFF** (temp 0) | — |
| `finder` (via `review.lens_reasoning_effort`) | **OFF** for pattern / style lenses | **`low`** for logic / security / correctness / concurrency lenses (Nova 2 `reasoningConfig` `maxReasoningEffort: low`) |
| `verifier` (`model_roles.verifier.reasoning_effort`) | **medium** | **high** for security / deep findings |
| `premium` (off by default) | **high** | **`xhigh`** for the hardest PRs (untested on the mantle endpoint) |

> **Nova Pro (`nova-pro-v1:0`) is deprecated** for new roles — prefer
> `global.amazon.nova-2-lite-v1:0`. Nova Pro has a hard 5K output cap (vs 64K), no
> reasoning path, no `global.` profile, higher cost, and an Oct-2024 cutoff; it is
> kept in the registry/price table for backward-compat only. See the tuning guide
> for the full rationale.

**Region validation.** Model ids are validated against
`openrabbit.bedrock_models`:

- **GPT-5.4 / GPT-5.5** (`openai.*`) run over Bedrock's OpenAI-compatible *mantle*
  Responses endpoint, which physically only exists in `us-east-1` / `us-east-2`.
  An `openai.*` role outside those regions is a **hard error** (the load fails).
- **Nova** (`amazon.*`) and **Claude** (`anthropic.*`) run over `bedrock-runtime`
  Converse with a broader, advisory region allow-list. An off-allow-list region
  is a **soft warning** (a cross-region inference profile may still reach it).
- An **unknown** model id is a soft warning (openrabbit can't vouch for its
  region/adapter, but won't block).

Cross-region inference-profile prefixes (`us.`, `apac.`, `eu.`, `global.`, …) are
understood — e.g. `us.openai.gpt-5.4` resolves to `openai.gpt-5.4`, and
`global.amazon.nova-2-lite-v1:0` resolves to `amazon.nova-2-lite-v1:0`.

## `external_tools`

> **Reserved / not yet wired.** This block is a forward-looking placeholder.
> openrabbit does **not** currently run these graders or inject their output into
> the review context — `enabled` ships empty and `openrabbit init` scaffolds it as
> a commented, reserved block. Listing tools here has **no effect** today; the
> field is reserved so configs stay forward-compatible once the plumbing lands.

The *intended* future behavior: a list of deterministic graders (e.g. `ruff`,
`eslint`, `semgrep`, `gitleaks`) whose output is fed into the review context as
grounding (the model never *runs* them — they are advisory signal). Until that is
implemented, leave `enabled: []`.

## `telemetry`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | bool | `true` | whether per-PR usage/cost telemetry is recorded. |
| `mode` | enum | `opt-out` | `opt-in` or `opt-out`. |

## Validation

`openrabbit.config.load_config` raises `ConfigError` on missing files, parse
errors, invalid enum values, or **hard** `model_roles` problems (an `openai.*`
verifier in an unsupported region). Soft `model_roles` issues never block the load; pass
`collect_warnings=True` (or run via the CLI) to surface them.
