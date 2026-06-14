# openrabbit — Claude Code plugin

This directory packages openrabbit's review intelligence as a **Claude Code
plugin** for private [marketplace](https://docs.claude.com/en/docs/claude-code/plugins)
distribution. Installing it makes the openrabbit review skills available inside
Claude Code as **namespaced skills** (`openrabbit:review`, `openrabbit:correctness`,
`openrabbit:security`, …).

## What's in the plugin

Skills are **auto-discovered** by Claude Code from the conventional `skills/`
directory *inside the plugin root* (there is no `skills` field in the plugin
manifest — `commands`, `agents`, `hooks`, and `mcpServers` are the only
component-path fields). To keep one source of truth for review intelligence,
`plugin/skills` is a relative symlink to the repo-level `skills/` tree (`skills
-> ../skills`), so the authored files live in exactly one place yet resolve
*under* the plugin root where the loader expects them. The advertised skills are:

| Namespaced skill            | Source `SKILL.md`                              | Role                                  |
| --------------------------- | ---------------------------------------------- | ------------------------------------- |
| `openrabbit:review`         | `skills/openrabbit-review/SKILL.md`            | Orchestrator: contract + output shape |
| `openrabbit:correctness`    | `skills/lenses/correctness/SKILL.md`           | Correctness / bug lens                |
| `openrabbit:security`       | `skills/lenses/security/SKILL.md`              | Security lens                         |
| `openrabbit:performance`    | `skills/lenses/performance/SKILL.md`           | Performance lens                      |
| `openrabbit:tests`          | `skills/lenses/tests/SKILL.md`                 | Test-quality lens                     |
| `openrabbit:maintainability`| `skills/lenses/maintainability/SKILL.md`       | Maintainability lens                  |

> Skills resolve from `plugin/skills/` (a symlink to the repo-level `skills/`
> directory); the plugin does not copy them. Clone (or install) the whole repo so
> the symlink resolves.

## Install (private marketplace)

From inside Claude Code:

```text
# 1. Register this repo as a plugin marketplace (any pinnable git ref works):
/plugin marketplace add hongvincent/openrabbit

# 2. Install the openrabbit plugin from that marketplace:
/plugin install openrabbit@openrabbit-marketplace
```

The marketplace name (`openrabbit-marketplace`) and the plugin name
(`openrabbit`) come from
[`.claude-plugin/marketplace.json`](./.claude-plugin/marketplace.json). The
plugin's git `source` is **pinned to a ref** (`v0.0.0`) so a fleet upgrade is a
single ref bump — bump the ref, re-run `/plugin marketplace add`, and every
onboarded environment rolls forward together.

> **Placeholder ref:** `v0.0.0` is a pre-release placeholder — no such git tag
> exists yet and the repo is not yet published to the `source` URL (roadmap item
> 21). Before publishing, create and push a real tag (e.g. `v0.0.1`) and update
> `marketplace.json`'s `ref` to it (or temporarily a branch/commit that exists).

To update later:

```text
/plugin marketplace update openrabbit-marketplace
/plugin update openrabbit@openrabbit-marketplace
```

## These are the open Agent Skills standard

Each skill is a plain [agentskills.io](https://agentskills.io) `SKILL.md`: YAML
frontmatter (`name`, `description`, `allowed-tools`) plus a markdown body used
verbatim as the model's system prompt. Nothing here is Claude-specific — the
same files drive openrabbit's Bedrock harness and run directly in **Codex**,
**Gemini**, or any agent that speaks the standard. The Claude Code plugin is
just one convenient distribution channel for portable, vendor-neutral review
intelligence.

## Security

The skills are **advisory-only**: they declare a least-privilege
`allowed-tools` list (read / grep / `git diff|log|blame|show` only) and have no
ability to merge, approve, push, or run arbitrary commands. The diff under
review is treated as untrusted data, never as instructions.

See the [top-level README](../README.md) and the
[design spec](../docs/superpowers/specs/2026-06-14-openrabbit-design.md)
(§11 Onboarding & Distribution) for the full picture.
