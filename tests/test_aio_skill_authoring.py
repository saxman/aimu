"""Tests for runtime skill authoring (sync write_skill + SkillManager.refresh + async tool)."""

from __future__ import annotations

import pytest

from aimu.skills import SkillManager, make_skill_authoring_tool, write_skill


def test_write_skill_creates_discoverable_file(tmp_path):
    path = write_skill("greet-user", "Greet the user warmly.", "# Greet\n\nSay hello.", skills_dir=tmp_path)

    assert path == tmp_path / "greet-user" / "SKILL.md"
    content = path.read_text()
    assert content.startswith("---\nname: greet-user\n")
    assert "description: Greet the user warmly." in content

    skills = SkillManager(skill_dirs=[str(tmp_path)]).skills
    assert "greet-user" in skills
    assert skills["greet-user"].description == "Greet the user warmly."


def test_write_skill_with_metadata(tmp_path):
    path = write_skill("tagged", "A tagged skill.", "body", skills_dir=tmp_path, metadata={"author": "test"})
    assert "metadata:" in path.read_text()
    assert SkillManager(skill_dirs=[str(tmp_path)]).skills["tagged"].metadata == {"author": "test"}


@pytest.mark.parametrize("bad_name", ["../evil", "with space", "Upper", "has/slash", ""])
def test_write_skill_rejects_invalid_name(tmp_path, bad_name):
    with pytest.raises(ValueError):
        write_skill(bad_name, "desc", "body", skills_dir=tmp_path)
    # Nothing escaped the skills dir.
    assert list(tmp_path.iterdir()) == []


def test_write_skill_requires_description(tmp_path):
    with pytest.raises(ValueError):
        write_skill("noop", "   ", "body", skills_dir=tmp_path)


def test_write_skill_no_clobber(tmp_path):
    write_skill("dup", "first", "body one", skills_dir=tmp_path)
    with pytest.raises(FileExistsError):
        write_skill("dup", "second", "body two", skills_dir=tmp_path)

    write_skill("dup", "second", "body two", skills_dir=tmp_path, overwrite=True)
    assert "body two" in (tmp_path / "dup" / "SKILL.md").read_text()


def test_refresh_picks_up_new_skill(tmp_path):
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert manager.skills == {}  # cache populated empty

    write_skill("late", "Authored after first discovery.", "body", skills_dir=tmp_path)
    assert "late" not in manager.skills  # stale cache does not see it

    refreshed = manager.refresh()
    assert "late" in refreshed
    assert "late" in manager.skills


async def test_author_skill_tool_writes_and_refreshes(tmp_path):
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    author_skill = make_skill_authoring_tool(manager, tmp_path)

    assert author_skill.__tool_is_async__ is True
    assert author_skill.__tool_spec__["function"]["name"] == "author_skill"
    params = author_skill.__tool_spec__["function"]["parameters"]["properties"]
    assert set(params) == {"name", "description", "body"}

    result = await author_skill(name="from-tool", description="Made by the tool.", body="# Do it")
    assert "from-tool" in result
    assert (tmp_path / "from-tool" / "SKILL.md").exists()
    assert "from-tool" in manager.skills  # refreshed mid-run


async def test_author_skill_tool_rejects_bad_name(tmp_path):
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    author_skill = make_skill_authoring_tool(manager, tmp_path)
    with pytest.raises(ValueError):
        await author_skill(name="../escape", description="bad", body="x")


def test_authored_skill_round_trips_body(tmp_path):
    write_skill("rt", "round trip", "Line one.\n\nLine two.", skills_dir=tmp_path)
    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["rt"]
    assert skill.load_body() == "Line one.\n\nLine two."
