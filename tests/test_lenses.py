"""Tests for the lens SKILL.md loader (SPEC 8.3).

A *lens* is a portable review skill stored as an agentskills.io ``SKILL.md``
file: YAML frontmatter (``name``, ``description``, ``allowed-tools``) plus a
markdown body. The harness uses the body as the system prompt to drive any
Bedrock model, so the loader must turn a ``SKILL.md`` into a typed
:class:`~openrabbit.lenses.Lens`.

No model calls and no network: everything operates on local files / strings.
``pyyaml`` is the only (lazy) parsing dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openrabbit.lenses import (
    Lens,
    LensError,
    load_lenses,
    parse_skill,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_LENSES_DIR = REPO_ROOT / "skills" / "lenses"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _write_skill(tmp_path: Path, name: str, text: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(text, encoding="utf-8")
    return p


FULL_SKILL = """\
---
name: correctness
description: Finds correctness bugs in a diff.
allowed-tools:
  - Read
  - Grep
---

# Correctness Lens

Body line one.

Body line two.
"""


# --------------------------------------------------------------------------- #
# parse_skill — frontmatter parsing                                           #
# --------------------------------------------------------------------------- #
def test_parse_skill_reads_frontmatter_fields(tmp_path):
    p = _write_skill(tmp_path, "correctness", FULL_SKILL)
    lens = parse_skill(p)
    assert isinstance(lens, Lens)
    assert lens.name == "correctness"
    assert lens.description == "Finds correctness bugs in a diff."
    assert lens.allowed_tools == ["Read", "Grep"]


def test_parse_skill_extracts_body_without_frontmatter(tmp_path):
    p = _write_skill(tmp_path, "correctness", FULL_SKILL)
    lens = parse_skill(p)
    # The frontmatter block must not leak into the system prompt.
    assert "---" not in lens.system_prompt
    assert "name: correctness" not in lens.system_prompt
    assert "# Correctness Lens" in lens.system_prompt
    assert "Body line one." in lens.system_prompt
    assert "Body line two." in lens.system_prompt


def test_parse_skill_body_is_stripped(tmp_path):
    p = _write_skill(tmp_path, "correctness", FULL_SKILL)
    lens = parse_skill(p)
    assert lens.system_prompt == lens.system_prompt.strip()
    assert lens.system_prompt.startswith("# Correctness Lens")


# --------------------------------------------------------------------------- #
# robustness to missing optional fields                                        #
# --------------------------------------------------------------------------- #
def test_parse_skill_missing_allowed_tools_defaults_empty(tmp_path):
    text = (
        "---\n"
        "name: security\n"
        "description: Finds security issues.\n"
        "---\n\n"
        "# Security Lens\n\nBody.\n"
    )
    p = _write_skill(tmp_path, "security", text)
    lens = parse_skill(p)
    assert lens.allowed_tools == []


def test_parse_skill_missing_description_defaults_empty(tmp_path):
    text = "---\nname: security\n---\n\nBody only.\n"
    p = _write_skill(tmp_path, "security", text)
    lens = parse_skill(p)
    assert lens.description == ""
    assert lens.system_prompt == "Body only."


def test_parse_skill_missing_name_falls_back_to_dir_name(tmp_path):
    text = "---\ndescription: no name here\n---\n\nBody.\n"
    p = _write_skill(tmp_path, "performance", text)
    lens = parse_skill(p)
    # Name falls back to the containing directory's name.
    assert lens.name == "performance"


def test_parse_skill_allowed_tools_as_csv_string(tmp_path):
    # agentskills.io permits a comma-separated string for allowed-tools.
    text = (
        "---\n"
        "name: correctness\n"
        "allowed-tools: Read, Grep, Bash\n"
        "---\n\n"
        "Body.\n"
    )
    p = _write_skill(tmp_path, "correctness", text)
    lens = parse_skill(p)
    assert lens.allowed_tools == ["Read", "Grep", "Bash"]


def test_parse_skill_accepts_underscore_allowed_tools_key(tmp_path):
    text = (
        "---\n"
        "name: correctness\n"
        "allowed_tools:\n"
        "  - Read\n"
        "---\n\n"
        "Body.\n"
    )
    p = _write_skill(tmp_path, "correctness", text)
    lens = parse_skill(p)
    assert lens.allowed_tools == ["Read"]


def test_parse_skill_scalar_allowed_tools_yields_empty(tmp_path):
    # A non-list, non-string value (e.g. an int) coerces to [].
    text = "---\nname: correctness\nallowed-tools: 5\n---\n\nBody.\n"
    p = _write_skill(tmp_path, "correctness", text)
    lens = parse_skill(p)
    assert lens.allowed_tools == []


def test_parse_skill_empty_frontmatter_falls_back(tmp_path):
    # A frontmatter block that parses to None (only comments/blank) is allowed;
    # name falls back to the directory.
    text = "---\n# nothing but a comment\n---\n\nBody only.\n"
    p = _write_skill(tmp_path, "maintainability", text)
    lens = parse_skill(p)
    assert lens.name == "maintainability"
    assert lens.description == ""
    assert lens.allowed_tools == []
    assert lens.system_prompt == "Body only."


# --------------------------------------------------------------------------- #
# error handling                                                               #
# --------------------------------------------------------------------------- #
def test_parse_skill_missing_file_raises(tmp_path):
    with pytest.raises(LensError):
        parse_skill(tmp_path / "nope" / "SKILL.md")


def test_parse_skill_no_frontmatter_raises(tmp_path):
    p = _write_skill(tmp_path, "correctness", "# Just a body, no frontmatter\n")
    with pytest.raises(LensError):
        parse_skill(p)


def test_parse_skill_non_mapping_frontmatter_raises(tmp_path):
    text = "---\n- just\n- a\n- list\n---\n\nBody.\n"
    p = _write_skill(tmp_path, "correctness", text)
    with pytest.raises(LensError):
        parse_skill(p)


def test_parse_skill_malformed_yaml_raises(tmp_path):
    text = "---\nname: : : bad\n  - broken\n---\n\nBody.\n"
    p = _write_skill(tmp_path, "correctness", text)
    with pytest.raises(LensError):
        parse_skill(p)


# --------------------------------------------------------------------------- #
# load_lenses — discovery                                                      #
# --------------------------------------------------------------------------- #
def test_load_lenses_discovers_skill_dirs(tmp_path):
    _write_skill(tmp_path, "correctness", FULL_SKILL)
    _write_skill(
        tmp_path,
        "security",
        "---\nname: security\ndescription: sec\n---\n\n# Security\n\nBody.\n",
    )
    lenses = load_lenses(tmp_path)
    assert set(lenses) == {"correctness", "security"}
    assert all(isinstance(v, Lens) for v in lenses.values())
    assert lenses["security"].description == "sec"


def test_load_lenses_ignores_dirs_without_skill_md(tmp_path):
    _write_skill(tmp_path, "correctness", FULL_SKILL)
    (tmp_path / "not_a_lens").mkdir()  # no SKILL.md inside
    lenses = load_lenses(tmp_path)
    assert set(lenses) == {"correctness"}


def test_load_lenses_keyed_by_frontmatter_name_not_dir(tmp_path):
    # Directory says "dir_x" but frontmatter name wins as the key.
    text = "---\nname: correctness\ndescription: c\n---\n\nBody.\n"
    _write_skill(tmp_path, "dir_x", text)
    lenses = load_lenses(tmp_path)
    assert set(lenses) == {"correctness"}


def test_load_lenses_missing_dir_raises(tmp_path):
    with pytest.raises(LensError):
        load_lenses(tmp_path / "does-not-exist")


def test_load_lenses_accepts_str_path(tmp_path):
    _write_skill(tmp_path, "correctness", FULL_SKILL)
    lenses = load_lenses(str(tmp_path))
    assert "correctness" in lenses


def test_load_lenses_empty_dir_returns_empty_dict(tmp_path):
    lenses = load_lenses(tmp_path)
    assert lenses == {}


def test_load_lenses_ignores_top_level_files(tmp_path):
    # A loose file (not a directory) sitting in the skills dir is skipped.
    (tmp_path / "README.md").write_text("not a lens", encoding="utf-8")
    _write_skill(tmp_path, "correctness", FULL_SKILL)
    lenses = load_lenses(tmp_path)
    assert set(lenses) == {"correctness"}


# --------------------------------------------------------------------------- #
# integration with the actual shipped lens skills                              #
# --------------------------------------------------------------------------- #
def test_shipped_lenses_load_correctness_and_security():
    lenses = load_lenses(SKILLS_LENSES_DIR)
    assert "correctness" in lenses
    assert "security" in lenses


def test_shipped_lens_prompts_are_nonempty_and_report_all():
    lenses = load_lenses(SKILLS_LENSES_DIR)
    for name in ("correctness", "security"):
        lens = lenses[name]
        assert lens.system_prompt.strip(), f"{name} prompt empty"
        body = lens.system_prompt.lower()
        # The report-all contract (SPEC 3) and confidence/severity must appear.
        assert "confidence" in body
        assert "severity" in body
