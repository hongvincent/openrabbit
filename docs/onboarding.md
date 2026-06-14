# Onboarding & Org Rollout

openrabbit is designed to onboard a new repo in **one command** and to roll out
across an entire organization governed **centrally** — one SHA bump rolls every
repo forward. There are two surfaces: the per-repo `openrabbit init` / `gh
openrabbit init`, and the org control plane.

## One-command onboarding: `gh openrabbit init`

The `gh openrabbit` extension mirrors the `claude /install-github-app` UX. Under
the hood it calls the pure-Python `openrabbit init`, which **detects** the repo's
stack and **scaffolds** the onboarding artifacts.

```bash
# Plain CLI (no gh): detect + scaffold.
openrabbit init                  # dry-run — print the plan + file contents
openrabbit init --write          # write the files to disk
openrabbit init --write --force  # overwrite existing files

# gh extension (precompiled wrapper) — same flow, plus the guarded gh mutations.
gh openrabbit init
```

`init` produces:

1. **`.openrabbit.yaml`** — stack-aware config (lenses + `model_roles` defaulting
   to a Nova-Pro finder and a GPT-5.5 verifier; `external_tools` chosen for the
   detected language).
2. **`.github/workflows/openrabbit.yml`** — a **thin caller** workflow that
   `uses:` the SHA-pinned central reusable workflow, with least-privilege
   `permissions`.
3. A printed **GitHub-wiring plan** — the OIDC trust-policy snippet and the one
   required secret (`AWS_ROLE_ARN`). This Python layer **never** calls `gh` / the
   AWS CLI / the network; the real `gh` mutations live only in the
   `cli/gh-openrabbit/gh-openrabbit` shell wrapper (and run only in a guarded,
   non-dry-run path).

### Wiring the keyless Bedrock auth

The scaffolded plan walks you through keyless OIDC → AWS STS (no long-lived
secrets):

1. Create / reuse an IAM role for Bedrock whose trust policy pins the repo's
   GitHub OIDC `sub` (`repo:<OWNER>/<REPO>:*`) and scopes permissions to the exact
   Bedrock model / inference-profile ARNs your `model_roles` use.
2. Store the role ARN as the repo secret the caller workflow consumes:
   ```bash
   gh secret set AWS_ROLE_ARN --body "arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>"
   ```
3. Edit the caller workflow's `uses:` to replace the `<OWNER>` / `<PINNED_SHA>`
   placeholders with your org + the vetted openrabbit release SHA.

## Org-scale rollout (control plane)

For fleet-wide governance, openrabbit ships drop-in control-plane artifacts under
[`org/`](../org/) (see [`org/README.md`](../org/README.md) for the deployment
walkthrough):

- **Starter workflow template** (`org/.github/workflow-templates/openrabbit.yml`
  + `openrabbit.properties.json`) — appears under Actions → "New workflow" for
  every repo in the org once dropped into the org `.github` repo. A thin caller
  that invokes the reusable workflow.
- **Reusable workflow** (`.github/workflows/reusable-workflow.yml`) — holds *all*
  review logic, invoked via `workflow_call`. Bump its central SHA to roll the
  whole fleet forward (Dependabot opens the PR).
- **Org ruleset** (`org/ruleset.json`) — requires the `openrabbit / review` check
  to pass before merge to the default branch. Ships in **Evaluate** mode (reports
  would-block decisions without blocking) for safe rollout; flip to **Active**
  once the check is green and stable.
- **Safe-Settings** (`org/safe-settings.yml`) — an optional alternative that
  applies the same required-check branch protection across repos and reconciles
  drift.

> **Required check name.** The ruleset / safe-settings gate requires the status
> check **`openrabbit / review`** — the exact composite name GitHub reports for a
> reusable-workflow job (`<caller job> / <reusable job>`). Renaming either job
> requires updating both `org/ruleset.json` and `org/safe-settings.yml`.

### Fleet sync = one knob

The reusable-workflow reference is SHA-pinned. To roll the whole fleet forward,
bump that one central SHA (let Dependabot open the PR for action SHAs). Every
onboarded repo picks up the new logic on its next PR — no per-repo edits.

## Plugin marketplace (portable review skills)

openrabbit's review intelligence is also packaged as a namespaced **Claude Code
plugin** so the same skills can run for local / interactive review in Claude Code
/ Codex / Gemini — one source of truth for review intelligence.

- **Plugin** ([`plugin/.claude-plugin/plugin.json`](../plugin/.claude-plugin/plugin.json))
  bundles the `openrabbit-review` orchestrator + the five lens skills
  (correctness, security, performance, tests, maintainability) as Agent Skills.
- **Marketplace** ([`plugin/.claude-plugin/marketplace.json`](../plugin/.claude-plugin/marketplace.json))
  is a private, git-backed marketplace whose plugin `source` is pinned to a repo
  ref.

Register the marketplace and enable the plugin:

```bash
# In Claude Code, add the private marketplace and enable the plugin:
/plugin marketplace add https://github.com/<OWNER>/openrabbit.git
/plugin install openrabbit@openrabbit-marketplace
```

Stable vs latest channels are achieved with two marketplaces (two refs) plus
managed settings (`extraKnownMarketplaces` + `enabledPlugins` +
`strictKnownMarketplaces`).
