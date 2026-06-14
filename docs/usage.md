# Usage

openrabbit ships a single console script, `openrabbit`, installed by `uv sync`
(`[project.scripts]` in `pyproject.toml`). It has four subcommands: `review`,
`eval`, `init`, and `learn`. Every subcommand has an **offline, credential-free**
default path so you can try the whole pipeline anywhere.

```bash
uv sync --all-extras       # install the openrabbit console script + deps
openrabbit --help
openrabbit review --help
```

## `openrabbit review`

Run the deterministic review spine over a diff. Two modes:

### Offline (demo, no creds)

Reads a unified diff from `--diff <file>` (or stdin) and runs the full spine with
deterministic **fixture** providers — no model calls, no AWS/GitHub credentials,
no network. Prints the kept findings + the would-be GitHub review payload as JSON
and a per-PR cost line on stderr.

```bash
# From a diff file, with the scripted demo finding:
openrabbit review --offline --diff some.diff --fixtures demo

# From a piped diff (clean dry-run of routing/gating/emitting, no findings):
git diff main...HEAD | openrabbit review --offline
```

`--fixtures demo` scripts one sample finding so you can see a real finding flow;
omitting it gives a clean, creds-free dry-run of routing/gating/emitting.

### Online (real review, needs creds)

Reads the PR diff via the GitHub adapter and runs the spine with real Bedrock
providers built from `.openrabbit.yaml` `model_roles`. Requires `GITHUB_TOKEN`
and Bedrock credentials (keyless OIDC → STS in CI; `aws sso login` locally).

```bash
openrabbit review \
  --repo OWNER/NAME --pr 123 --commit <head-sha> \
  --config .openrabbit.yaml \
  --post                       # actually post the advisory review (omit to dry-run)
```

| Flag | Meaning |
|------|---------|
| `--offline` | run with scripted fixtures (no creds) |
| `--diff <file>` | unified diff path (offline; else stdin) |
| `--config <path>` | path to `.openrabbit.yaml` (default: `./.openrabbit.yaml`) |
| `--fixtures demo` | offline fixture set with a sample finding |
| `--repo OWNER/NAME` / `--pr N` / `--commit SHA` | online PR targeting |
| `--post` | post the advisory review (online); omit to dry-run |
| `--bot-login` | bot login used for dedup against prior review threads |
| `--learnings-store <path>` | enable the memory + feedback loop (local JSON store) |

In CI the reusable workflow / composite action invoke exactly this command with
`--post` after keyless OIDC → STS Bedrock auth (see
[`onboarding.md`](./onboarding.md)).

## `openrabbit eval`

Run the dogfood eval harness (golden set + clean-PR negative controls + a
calibrated LLM-as-judge) and print a scorecard (precision / recall / FP / action
rate per category) — the gate that proves **FP < 10%** before a change ships.

```bash
# Offline scripted fixtures (no creds): exercises the whole harness.
openrabbit eval --repo /path/to/local/git/repo --limit 20

# JSON scorecard for tooling:
openrabbit eval --json

# CI gate — exit non-zero if the FP budget is exceeded:
openrabbit eval --require-pass

# Real FP measurement (needs Bedrock creds): builds providers from model_roles.
openrabbit eval --online
```

`--online` is the only path that yields a *real* false-positive number; it is
gated on credentials and never makes a network call without them.

## `openrabbit init`

Detect a repo's stack and scaffold the onboarding artifacts: `.openrabbit.yaml`
(stack-aware defaults) + a SHA-pinned thin caller workflow, plus a printed
GitHub-wiring plan (OIDC trust policy + required secret). No `gh`/AWS/network
mutation happens here.

```bash
openrabbit init                  # dry-run: print the plan + file contents
openrabbit init --write          # write the files to disk
openrabbit init --write --force  # overwrite existing files
openrabbit init --json           # plan as JSON (for the gh extension wrapper)
```

See [`onboarding.md`](./onboarding.md) for the full one-command onboarding flow
and the `gh openrabbit init` extension.

## `openrabbit learn`

Capture feedback into a local learnings store (offline, no creds). Two actions:

```bash
# Record a team learning, injected into future reviews' cacheable prefix:
openrabbit learn --store .openrabbit-learnings.json \
  --scope OWNER/NAME --text "Prefer `pathlib` over os.path in new code."

# Record a dismissal (negative signal): future similar findings are
# down-weighted below the gate.
openrabbit learn --store .openrabbit-learnings.json \
  --repo OWNER/NAME --dismiss \
  --rule-id openrabbit/maintainability/style --category maintainability --file src/x.py
```

Pass the same `--learnings-store` path to `openrabbit review` to apply the
learnings + dismissals during a review.

## Cost telemetry

Every `review` prints a per-PR cost line to **stderr** (token totals + an
optional USD estimate) while the JSON payload stays on stdout for downstream
tooling:

```
openrabbit cost: calls=3 in=24120 out=860 cacheRead=18004 cacheWrite=6116 estimate=$0.02
```
