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


@pytest.mark.parametrize("bad_name", ["../evil", "with space", "Upper", "has/slash", "with_underscores", ""])
def test_write_skill_rejects_invalid_name(tmp_path, bad_name):
    with pytest.raises(ValueError):
        write_skill(bad_name, "desc", "body", skills_dir=tmp_path)
    # Nothing escaped the skills dir.
    assert list(tmp_path.iterdir()) == []


def test_write_skill_underscore_name_error_points_at_hyphens(tmp_path):
    # Skill names stay kebab-case; the error must steer the model toward hyphens.
    with pytest.raises(ValueError, match="hyphen"):
        write_skill("weekly_review", "desc", "body", skills_dir=tmp_path)


def test_write_skill_accepts_hyphenated_name(tmp_path):
    write_skill("weekly-review", "Weekly review.", "# Review", skills_dir=tmp_path)
    assert "weekly-review" in SkillManager(skill_dirs=[str(tmp_path)]).skills


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


# ---------------------------------------------------------------------------
# Script authoring + mid-run reload (async)
# ---------------------------------------------------------------------------


async def test_aio_reload_skills_appends_new_script_tool(tmp_path):
    from aimu import aio
    from aimu.skills import write_skill
    from helpers_aio import MockAsyncModelClient

    write_skill("grow", "Grows scripts.", "# Grow", skills_dir=tmp_path)
    client = MockAsyncModelClient([])
    client.system_message = "Base."
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = aio.SkillAgent(client, skill_manager=manager, name="a")

    await agent._setup_skills_async()
    assert "grow__added" not in [fn.__name__ for fn in client.tools]

    write_skill(
        "grow", "Grows scripts.", "# Grow", skills_dir=tmp_path, overwrite=True, scripts={"added.py": "print('x')\n"}
    )
    manager.refresh()
    await agent.reload_skills()

    assert "grow__added" in [fn.__name__ for fn in client.tools]
    assert client.system_message.count("<available_skills>") == 1  # catalog not duplicated


def test_make_skill_script_tool_spec(tmp_path):
    from aimu.skills import make_skill_script_tool

    class _StubAgent:
        skill_manager = None

    tool = make_skill_script_tool(_StubAgent(), SkillManager(skill_dirs=[str(tmp_path)]), tmp_path)
    assert tool.__tool_is_async__ is True
    assert tool.__tool_spec__["function"]["name"] == "add_skill_script"
    assert set(tool.__tool_spec__["function"]["parameters"]["properties"]) == {"skill_name", "filename", "content"}


async def test_add_skill_script_unknown_skill(tmp_path):
    from aimu.skills import make_skill_script_tool

    reloaded = []

    class _StubAgent:
        skill_manager = None

        async def reload_skills(self):
            reloaded.append(True)

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    tool = make_skill_script_tool(_StubAgent(), manager, tmp_path)
    msg = await tool(skill_name="nope", filename="x.py", content="print(1)")
    assert "not found" in msg
    assert reloaded == []  # no reload for a missing skill


async def test_add_skill_script_writes_and_reloads(tmp_path):
    from aimu.skills import make_skill_script_tool, write_skill

    write_skill("auto", "Automations.", "# Auto", skills_dir=tmp_path)
    manager = SkillManager(skill_dirs=[str(tmp_path)])

    reloaded = []

    class _StubAgent:
        def __init__(self, mgr):
            self.skill_manager = mgr

        async def reload_skills(self):
            reloaded.append(True)

    tool = make_skill_script_tool(_StubAgent(manager), manager, tmp_path)
    msg = await tool(skill_name="auto", filename="backup.sh", content="echo backing up\n")

    assert (tmp_path / "auto" / "scripts" / "backup.sh").exists()
    assert "auto__backup" in msg
    assert reloaded == [True]
    assert "auto__backup" in manager.skills["auto"].script_tool_names()
