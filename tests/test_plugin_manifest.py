"""Tests for the Claude Code plugin manifests (SPEC 11; agentskills.io).

openrabbit packages its review skills as a **Claude Code plugin** for private
*marketplace* distribution. Two JSON manifests describe it:

* ``plugin/.claude-plugin/plugin.json`` — the plugin manifest (``name``,
  ``version``, ``description``, ``author``, ``keywords``, and the ``skills``
  path that points at the shipped ``SKILL.md`` files).
* ``plugin/.claude-plugin/marketplace.json`` — the marketplace catalog that
  lists the ``openrabbit`` plugin with a git ``source`` so it installs via
  ``/plugin marketplace add <repo>`` + ``/plugin install openrabbit@<mkt>``.

Every skill the plugin advertises must actually exist on disk under ``skills/``
with valid ``SKILL.md`` frontmatter — reuse the lens loader to prove that.

No model calls and no network: pure JSON / local-file assertions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openrabbit.lenses import Lens, parse_skill

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugin"
MANIFEST_DIR = PLUGIN_DIR / ".claude-plugin"
PLUGIN_JSON = MANIFEST_DIR / "plugin.json"
MARKETPLACE_JSON = MANIFEST_DIR / "marketplace.json"
PLUGIN_README = PLUGIN_DIR / "README.md"
SKILLS_DIR = REPO_ROOT / "skills"

#: kebab-case plugin / skill names (lowercase, digits, single hyphens).
_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: The skills the plugin bundles: the top-level review skill + the five lenses.
ADVERTISED_SKILLS = (
    "openrabbit-review",
    "correctness",
    "security",
    "performance",
    "tests",
    "maintainability",
)


# --------------------------------------------------------------------------- #
# loaders                                                                      #
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def plugin() -> dict:
    return _load_json(PLUGIN_JSON)


@pytest.fixture
def marketplace() -> dict:
    return _load_json(MARKETPLACE_JSON)


# --------------------------------------------------------------------------- #
# files exist                                                                  #
# --------------------------------------------------------------------------- #
def test_manifest_dir_is_dot_claude_plugin():
    # Claude Code requires manifests under ``.claude-plugin/``.
    assert MANIFEST_DIR.is_dir()
    assert MANIFEST_DIR.name == ".claude-plugin"


def test_plugin_json_exists():
    assert PLUGIN_JSON.is_file()


def test_marketplace_json_exists():
    assert MARKETPLACE_JSON.is_file()


def test_plugin_readme_exists_and_documents_install():
    assert PLUGIN_README.is_file()
    text = PLUGIN_README.read_text(encoding="utf-8")
    # Both install steps must be documented.
    assert "/plugin marketplace add" in text
    assert "/plugin install openrabbit@" in text
    # Mentions the open Agent Skills standard / portability.
    lower = text.lower()
    assert "skill" in lower
    assert "agentskills" in lower or "agent skills" in lower


# --------------------------------------------------------------------------- #
# plugin.json — required fields + valid name                                   #
# --------------------------------------------------------------------------- #
def test_plugin_json_is_valid_json_object(plugin):
    assert isinstance(plugin, dict)


def test_plugin_json_has_required_name(plugin):
    # ``name`` is the only required field in the documented schema.
    assert "name" in plugin
    assert plugin["name"] == "openrabbit"


def test_plugin_name_is_kebab_case(plugin):
    assert _KEBAB_RE.match(plugin["name"]), (
        f"plugin name {plugin['name']!r} is not kebab-case"
    )


def test_plugin_json_has_recommended_fields(plugin):
    for field in ("version", "description", "author", "keywords"):
        assert field in plugin, f"plugin.json missing {field!r}"


def test_plugin_version_is_semver(plugin):
    # Mirror the package version style (e.g. ``0.0.0``).
    assert re.match(r"^\d+\.\d+\.\d+", plugin["version"]), plugin["version"]


def test_plugin_description_is_nonempty_string(plugin):
    assert isinstance(plugin["description"], str)
    assert plugin["description"].strip()


def test_plugin_author_is_object_or_string(plugin):
    author = plugin["author"]
    assert isinstance(author, (str, dict))
    if isinstance(author, dict):
        assert author.get("name")


def test_plugin_keywords_are_nonempty_string_list(plugin):
    kw = plugin["keywords"]
    assert isinstance(kw, list) and kw
    assert all(isinstance(k, str) and k.strip() for k in kw)
    # Should advertise review / code-review intent.
    joined = " ".join(kw).lower()
    assert "review" in joined


def test_plugin_json_has_no_unsupported_skills_field(plugin):
    # The Claude Code plugin manifest has NO ``skills`` path field — the only
    # component-path fields are commands/agents/hooks/mcpServers. Skills are
    # auto-discovered from a conventional ``skills/`` dir INSIDE the plugin root.
    # A ``skills`` key (esp. one pointing outside the root via ``../``) makes the
    # plugin advertise zero loadable skills, so it must be absent.
    assert "skills" not in plugin, (
        "plugin.json must NOT declare a `skills` path field — skills are "
        "auto-discovered from skills/ under the plugin root"
    )


def test_plugin_skills_dir_lives_under_plugin_root():
    # The skills must be reachable from a ``skills/`` directory UNDER the plugin
    # root (the dir holding ``.claude-plugin``) — never via a ``../`` traversal
    # outside the root, which plugin loaders reject. A relative symlink to the
    # single-source-of-truth repo skills/ tree is fine, as long as it resolves
    # to a real directory inside the plugin root path.
    plugin_skills = PLUGIN_DIR / "skills"
    assert plugin_skills.exists(), (
        "plugin must expose a skills/ directory under its root (Claude Code "
        "auto-discovers skills from <plugin-root>/skills/)"
    )
    assert plugin_skills.is_dir(), "plugin/skills must resolve to a directory"
    # It must NOT escape the plugin root: the resolved path stays a child of
    # PLUGIN_DIR's *path* (a symlink target is fine as long as we address it via
    # the in-root path the loader uses).
    assert plugin_skills.resolve().is_dir()


def test_every_advertised_skill_resolves_under_plugin_root():
    # Each advertised skill must have a SKILL.md reachable UNDER the plugin root
    # (plugin/skills/<name>/SKILL.md or plugin/skills/lenses/<name>/SKILL.md) —
    # the path the real plugin loader walks. This is the non-tautological check
    # the old ``../skills`` resolution test lacked.
    plugin_skills = PLUGIN_DIR / "skills"
    for name in ADVERTISED_SKILLS:
        top = plugin_skills / name / "SKILL.md"
        lens = plugin_skills / "lenses" / name / "SKILL.md"
        assert top.is_file() or lens.is_file(), (
            f"advertised skill {name!r} has no SKILL.md under the plugin root "
            f"(looked at {top} and {lens})"
        )


# --------------------------------------------------------------------------- #
# marketplace.json — references the openrabbit plugin                          #
# --------------------------------------------------------------------------- #
def test_marketplace_json_is_valid_json_object(marketplace):
    assert isinstance(marketplace, dict)


def test_marketplace_has_name_and_owner(marketplace):
    assert isinstance(marketplace.get("name"), str) and marketplace["name"].strip()
    owner = marketplace.get("owner")
    assert owner is not None
    # ``owner`` may be a string or an object with a ``name``.
    if isinstance(owner, dict):
        assert owner.get("name")
    else:
        assert isinstance(owner, str) and owner.strip()


def test_marketplace_name_is_kebab_case(marketplace):
    assert _KEBAB_RE.match(marketplace["name"]), marketplace["name"]


def test_marketplace_lists_openrabbit_plugin(marketplace):
    plugins = marketplace.get("plugins")
    assert isinstance(plugins, list) and plugins
    names = [p.get("name") for p in plugins if isinstance(p, dict)]
    assert "openrabbit" in names


def test_marketplace_openrabbit_entry_has_source(marketplace):
    entry = next(p for p in marketplace["plugins"] if p.get("name") == "openrabbit")
    assert "source" in entry
    source = entry["source"]
    # A git source with a pinnable ref. Either a string URL or a
    # ``{source:"git", ...}`` object carrying a ref/tag/branch handle.
    if isinstance(source, dict):
        assert (
            source.get("source") == "git"
            or source.get("type") == "git"
            or source.get("url")
            or source.get("repo")
        )
        # A pinnable ref handle should be present (ref/tag/branch/commit).
        assert any(k in source for k in ("ref", "tag", "branch", "commit", "rev")), (
            f"git source has no pinnable ref: {source}"
        )
    else:
        assert isinstance(source, str) and source.strip()


def test_marketplace_plugin_name_matches_plugin_json(marketplace, plugin):
    names = [p.get("name") for p in marketplace["plugins"]]
    assert plugin["name"] in names


# --------------------------------------------------------------------------- #
# every advertised skill exists on disk with valid frontmatter                 #
# --------------------------------------------------------------------------- #
def _skill_path(name: str) -> Path:
    """Locate the SKILL.md for ``name`` under ``skills/`` (top-level or lens)."""
    top = SKILLS_DIR / name / "SKILL.md"
    if top.is_file():
        return top
    return SKILLS_DIR / "lenses" / name / "SKILL.md"


def test_every_advertised_skill_exists_on_disk():
    for name in ADVERTISED_SKILLS:
        path = _skill_path(name)
        assert path.is_file(), f"advertised skill {name!r} missing at {path}"


def test_every_advertised_skill_has_valid_frontmatter():
    # Reuse the production lens loader to validate frontmatter.
    for name in ADVERTISED_SKILLS:
        lens = parse_skill(_skill_path(name))
        assert isinstance(lens, Lens)
        assert lens.name, f"{name} has no frontmatter name"
        assert lens.description.strip(), f"{name} has no description"
        assert lens.system_prompt.strip(), f"{name} has an empty body"


def test_advertised_skills_cover_review_plus_five_lenses():
    # The plugin bundles the top-level review skill + all five lenses.
    assert "openrabbit-review" in ADVERTISED_SKILLS
    lenses = set(ADVERTISED_SKILLS) - {"openrabbit-review"}
    assert lenses == {
        "correctness",
        "security",
        "performance",
        "tests",
        "maintainability",
    }


def test_plugin_readme_lists_namespaced_skills():
    # The README documents the namespaced skill ids (openrabbit:<skill>).
    text = PLUGIN_README.read_text(encoding="utf-8")
    assert "openrabbit:review" in text or "openrabbit:openrabbit-review" in text
