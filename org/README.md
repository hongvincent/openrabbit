# openrabbit — Org-Scale Rollout

Drop-in control-plane artifacts (PRD §11) for rolling out openrabbit's AI code
review across an entire GitHub organization, governed **centrally** so one SHA
bump rolls every repo forward.

## What's here

| File | Purpose |
|------|---------|
| `.github/workflow-templates/openrabbit.yml` | Starter workflow shown in the "New workflow" picker. A thin caller that invokes openrabbit's **reusable** workflow. |
| `.github/workflow-templates/openrabbit.properties.json` | Picker metadata (`name`, `description`, `categories`, `iconName`). |
| `ruleset.json` | Organization **ruleset** requiring the openrabbit check to pass before merge (PRs → default branch). |
| `safe-settings.yml` | [Safe-Settings](https://github.com/github/safe-settings) (Probot) example applying consistent branch protection + the required check across repos. |

These are advisory, supply-chain-hardened, and contain **no secrets**.

## How to deploy

1. **Drop into the org `.github` repo.**
   Copy `.github/workflow-templates/openrabbit.yml` and
   `openrabbit.properties.json` into your organization's special **`.github`**
   repository (path: `.github/workflow-templates/`). They then appear under
   Actions → "New workflow" → "By _your-org_" for every repo.

2. **Edit the pinned reference.**
   In `openrabbit.yml`, replace the `<OWNER>`/`<PINNED_SHA>` placeholders: set
   `<OWNER>` to your org and `<PINNED_SHA>` to the **SHA** of the openrabbit
   release you've vetted:
   `<OWNER>/openrabbit/.github/workflows/reusable-workflow.yml@<PINNED_SHA>`.
   (GitHub only resolves a reusable workflow under `.github/workflows/`, so that
   is where openrabbit's reusable workflow lives.) Set the org/repo secret
   `OPENRABBIT_AWS_ROLE_ARN` to the IAM role whose trust policy pins each repo's
   GitHub OIDC subject (keyless Bedrock — no static keys).

   > **Required check name:** the ruleset / safe-settings gate requires the
   > status check **`openrabbit / review`** — the exact name GitHub reports for
   > a reusable-workflow job (`<caller job> / <reusable job>`). The caller job is
   > named `openrabbit`; the reusable job is `review`. Rename either and you must
   > update `ruleset.json` + `safe-settings.yml` to match.

3. **Enable the ruleset in Evaluate, then Active.**
   Create the org ruleset from `ruleset.json`:

   ```bash
   gh api -X POST /orgs/<ORG>/rulesets --input org/ruleset.json
   ```

   It ships with `"enforcement": "evaluate"` so it **reports** would-block
   decisions without blocking any merge. Once the openrabbit check is green and
   stable across the fleet, flip `enforcement` to `"active"` to make it a hard
   merge gate.

4. **(Optional) Safe-Settings instead of / alongside rulesets.**
   If your org standardizes on Safe-Settings, install it and place
   `safe-settings.yml` at `.github/settings.yml` in the Safe-Settings admin repo.
   It applies the same required-check branch protection to every repo's default
   branch and continuously reconciles drift.

## Fleet sync = one knob

The reusable-workflow reference is **SHA-pinned**. To roll the whole fleet
forward, **bump that one central SHA** (let Dependabot open the PR for the action
SHAs). Every onboarded repo picks up the new logic on its next PR — no per-repo
edits. To pin a check to a specific workflow version, update both the caller's
pinned SHA and, if used, the ruleset's `required_status_checks` reference.
