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

model_roles:                     # role -> { model, region, ...provider opts }
  triage:
    model: amazon.nova-lite-v1:0
    region: ap-northeast-2
  finder:
    model: amazon.nova-pro-v1:0
    region: ap-northeast-2
  verifier:
    model: openai.gpt-5.5
    region: us-east-2
    reasoning_effort: medium
    store: false
  premium:                       # optional, cost-gated; off by default
    model: global.anthropic.claude-opus-4-6-v1
    region: us-east-1
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
| `triage` | near-free yes/no on the diff (skip trivial changes) | `amazon.nova-lite-v1:0` @ `ap-northeast-2` |
| `finder` | broad, high-recall report-all first pass | `amazon.nova-pro-v1:0` @ `ap-northeast-2` |
| `verifier` | cross-family judge; scores + drops below the gate | `openai.gpt-5.5` @ `us-east-2` |
| `premium` | optional, cost-gated highest-stakes role | Claude on Bedrock (off by default) |

**Region validation.** Model ids are validated against
`openrabbit.bedrock_models`:

- **GPT-5.5** (`openai.*`) runs over Bedrock's OpenAI-compatible *mantle*
  Responses endpoint, which physically only exists in `us-east-1` / `us-east-2`.
  A GPT-5.5 role outside those regions is a **hard error** (the load fails).
- **Nova** (`amazon.*`) and **Claude** (`anthropic.*`) run over `bedrock-runtime`
  Converse with a broader, advisory region allow-list. An off-allow-list region
  is a **soft warning** (a cross-region inference profile may still reach it).
- An **unknown** model id is a soft warning (openrabbit can't vouch for its
  region/adapter, but won't block).

Cross-region inference-profile prefixes (`us.`, `apac.`, `eu.`, `global.`, …) are
understood — e.g. `us.openai.gpt-5.5` resolves to `openai.gpt-5.5`.

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
errors, invalid enum values, or **hard** `model_roles` problems (GPT-5.5 in an
unsupported region). Soft `model_roles` issues never block the load; pass
`collect_warnings=True` (or run via the CLI) to surface them.
