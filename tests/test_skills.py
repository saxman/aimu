"""
Tests for aimu.skills: AgentSkill, SkillManager, and Agent skill integration.

All filesystem tests use tmp_path so no real ~/.agents/skills paths are touched.
Unit tests use MagicMock inline. The model_client fixture is available for
integration tests:
  - Default (no --client): MockBaseModelClient
  - pytest tests/test_skills.py --client=ollama --model=LLAMA_3_2_3B
"""

from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import pytest

from aimu.models import BaseModelClient, StreamChunk, StreamingContentType
from aimu.skills.manager import SkillManager
from aimu.skills.skill import AgentSkill
from helpers import create_real_model_client, resolve_model_params

_MOCK = "mock"


class _MockBaseModelClient(BaseModelClient):
    """Minimal BaseModelClient stub for skill integration tests."""

    def __init__(self):
        self.model = MagicMock()
        self.model.supports_tools = False
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools = []
        self.last_thinking = ""

    def chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, tools=None):
        if stream:
            return self._chat_streamed(user_message)
        self.messages.append({"role": "user", "content": user_message})
        response = "I can help with that."
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _chat_streamed(self, user_message):
        response = self.chat(user_message)
        yield StreamChunk(StreamingContentType.GENERATING, response)

    def generate(self, prompt, generate_kwargs=None, stream=False, include=None):
        if stream:
            return self._generate_streamed()
        return "Generated response."

    def _generate_streamed(self):
        yield StreamChunk(StreamingContentType.GENERATING, "Generated response.")

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}


def pytest_generate_tests(metafunc):
    if "model_client" not in metafunc.fixturenames:
        return
    params = resolve_model_params(metafunc.config, default_params=[_MOCK])
    metafunc.parametrize("model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[BaseModelClient]:
    if request.param == _MOCK:
        yield _MockBaseModelClient()
        return
    yield from create_real_model_client(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_skill_dir(parent: Path, name: str, description: str, body: str = "## Instructions\nDo the thing.") -> Path:
    """Create a minimal skill directory under parent and return the SKILL.md path."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}",
        encoding="utf-8",
    )
    return skill_md


# ---------------------------------------------------------------------------
# AgentSkill.load_body
# ---------------------------------------------------------------------------


def test_skill_load_body_strips_frontmatter(tmp_path):
    skill_md = make_skill_dir(tmp_path, "my-skill", "Does things.", body="## Steps\n1. Do it.")
    skill = AgentSkill(name="my-skill", description="Does things.", path=skill_md)
    body = skill.load_body()
    assert "## Steps" in body
    assert "---" not in body
    assert "name:" not in body


def test_skill_load_body_no_frontmatter(tmp_path):
    skill_dir = tmp_path / "bare-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("Just instructions, no frontmatter.", encoding="utf-8")
    skill = AgentSkill(name="bare-skill", description="x", path=skill_md)
    assert skill.load_body() == "Just instructions, no frontmatter."


def test_skill_base_dir(tmp_path):
    skill_md = make_skill_dir(tmp_path, "s", "x")
    skill = AgentSkill(name="s", description="x", path=skill_md)
    assert skill.base_dir == skill_md.parent


# ---------------------------------------------------------------------------
# SkillManager discovery
# ---------------------------------------------------------------------------


def test_skill_manager_custom_dirs_discovers_skill(tmp_path):
    make_skill_dir(tmp_path, "hello-world", "Say hello to the world.")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert "hello-world" in manager.skills


def test_skill_manager_discovers_multiple_skills(tmp_path):
    make_skill_dir(tmp_path, "skill-a", "Does A.")
    make_skill_dir(tmp_path, "skill-b", "Does B.")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert set(manager.skills.keys()) == {"skill-a", "skill-b"}


def test_skill_manager_skips_dir_without_skill_md(tmp_path):
    (tmp_path / "not-a-skill").mkdir()
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert len(manager.skills) == 0


def test_skill_manager_raises_on_missing_description(tmp_path):
    """Malformed SKILL.md is no longer silently skipped; SkillLoadError is raised."""
    import pytest

    from aimu.skills import SkillLoadError

    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: bad-skill\n---\n\nNo description.", encoding="utf-8")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    with pytest.raises(SkillLoadError, match="description"):
        _ = manager.skills


def test_skill_manager_raises_on_no_frontmatter(tmp_path):
    """SKILL.md without YAML frontmatter raises SkillLoadError."""
    import pytest

    from aimu.skills import SkillLoadError

    skill_dir = tmp_path / "raw-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("No frontmatter at all.", encoding="utf-8")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    with pytest.raises(SkillLoadError, match="frontmatter"):
        _ = manager.skills


# ---------------------------------------------------------------------------
# SkillManager name collision (project > user)
# ---------------------------------------------------------------------------


def test_skill_manager_project_overrides_user(tmp_path):
    project_dir = tmp_path / "project" / ".agents" / "skills"
    user_dir = tmp_path / "user"
    project_dir.mkdir(parents=True)
    user_dir.mkdir()

    make_skill_dir(project_dir, "shared", "Project version.")
    make_skill_dir(user_dir, "shared", "User version.")

    # Pass project dir first; it wins on collision
    manager = SkillManager(skill_dirs=[str(project_dir), str(user_dir)])
    assert manager.skills["shared"].description == "Project version."


# ---------------------------------------------------------------------------
# SkillManager.catalog_prompt
# ---------------------------------------------------------------------------


def test_skill_manager_catalog_prompt_contains_names_and_descriptions(tmp_path):
    make_skill_dir(tmp_path, "pdf-processing", "Extract and merge PDFs.")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    catalog = manager.catalog_prompt()
    assert "<available_skills>" in catalog
    assert "<name>pdf-processing</name>" in catalog
    assert "<description>Extract and merge PDFs.</description>" in catalog


def test_skill_manager_catalog_prompt_empty_when_no_skills(tmp_path):
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert manager.catalog_prompt() == ""


# ---------------------------------------------------------------------------
# SkillManager.get_skill_body
# ---------------------------------------------------------------------------


def test_skill_manager_get_skill_body_returns_body(tmp_path):
    make_skill_dir(tmp_path, "coder", "Writes code.", body="## How to code\nWrite clean code.")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    body = manager.get_skill_body("coder")
    assert "Write clean code." in body


def test_skill_manager_get_skill_body_unknown_name(tmp_path):
    """get_skill_body raises SkillNotFoundError instead of returning a sentinel string."""
    import pytest

    from aimu.skills import SkillNotFoundError

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    with pytest.raises(SkillNotFoundError, match="not found"):
        manager.get_skill_body("nonexistent")


# ---------------------------------------------------------------------------
# Agent skill integration (no real model needed)
# ---------------------------------------------------------------------------


def test_agent_setup_skills_injects_catalog(tmp_path):
    """_setup_skills() appends the skill catalog to the model client's system message."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "my-skill", "Does my thing.")

    client = MagicMock()
    client.system_message = "Be helpful."
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    assert "my-skill" in client.system_message
    assert "Does my thing." in client.system_message
    assert "activate_skill" in client.system_message


def test_agent_setup_skills_adds_tools(tmp_path):
    """_setup_skills() adds the skills server's tools (incl. activate_skill) to model_client.tools."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "my-skill", "Does my thing.")

    client = MagicMock()
    client.system_message = None
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    assert "activate_skill" in [fn.__name__ for fn in client.tools]


def test_agent_setup_skills_no_op_when_no_skills(tmp_path):
    """_setup_skills() does nothing when no skills are found."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    client = MagicMock()
    client.system_message = "Original."
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    # No skills found; no tools added
    assert client.tools == []


def test_agent_setup_skills_runs_only_once(tmp_path):
    """_setup_skills() is idempotent: calling it twice doesn't duplicate catalog or tools."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "once-skill", "Run only once.")

    client = MagicMock()
    client.system_message = ""
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)

    agent._setup_skills()
    tools_after_first = list(client.tools)
    agent._setup_skills()  # second call; should be a no-op for catalog + tools
    assert agent._skills_setup_done is True
    assert [fn.__name__ for fn in client.tools] == [fn.__name__ for fn in tools_after_first]


def test_agent_from_config_with_skill_dirs(tmp_path):
    """SkillAgent.from_config with skill_dirs creates a SkillManager."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "cfg-skill", "From config.")

    client = MagicMock()
    client.system_message = None

    agent = SkillAgent.from_config({"name": "cfg-agent", "skill_dirs": [str(tmp_path)]}, client)

    assert "cfg-skill" in agent.skill_manager.skills


def test_skill_manager_default_paths_discover_skill(tmp_path, monkeypatch):
    """SkillManager() with no skill_dirs scans the four default paths."""
    from aimu.skills import manager as skills_module

    project_skills = tmp_path / ".agents" / "skills"
    project_skills.mkdir(parents=True)
    make_skill_dir(project_skills, "auto-skill", "Found via default path.")

    # Override the default dirs so the test doesn't touch real home/project paths
    monkeypatch.setattr(skills_module, "_DEFAULT_SKILL_DIRS", [str(project_skills)])

    mgr = SkillManager()  # no skill_dirs, should use defaults
    assert "auto-skill" in mgr.skills


def test_skill_manager_custom_dirs_override_defaults(tmp_path, monkeypatch):
    """When skill_dirs is given, default paths are ignored."""
    from aimu.skills import manager as skills_module

    default_skills = tmp_path / "default_skills"
    default_skills.mkdir()
    make_skill_dir(default_skills, "default-skill", "Should not appear.")

    custom_skills = tmp_path / "custom_skills"
    custom_skills.mkdir()
    make_skill_dir(custom_skills, "custom-skill", "From explicit dirs.")

    monkeypatch.setattr(skills_module, "_DEFAULT_SKILL_DIRS", [str(default_skills)])

    mgr = SkillManager(skill_dirs=[str(custom_skills)])
    assert "custom-skill" in mgr.skills
    assert "default-skill" not in mgr.skills


# ---------------------------------------------------------------------------
# Skill scripts: discovery, execution, args, authoring, reload
# ---------------------------------------------------------------------------


def _write_script(skill_md_path: Path, filename: str, content: str) -> Path:
    scripts_dir = skill_md_path.parent / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    target = scripts_dir / filename
    target.write_text(content, encoding="utf-8")
    return target


def test_build_skills_server_registers_and_runs_py(tmp_path):
    from aimu.skills.mcp import build_skills_server
    from aimu.tools.client import MCPClient

    md = make_skill_dir(tmp_path, "tools", "Has scripts.")
    _write_script(md, "hello.py", "import sys\nprint('py:' + ' '.join(sys.argv[1:]))\n")

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    client = MCPClient(server=build_skills_server(manager))
    names = client.list_tools()
    assert "tools__hello" in [t.name for t in names]

    out = client.call_tool("tools__hello", {"args": "alpha beta"})
    assert "py:alpha beta" in out.content[0].text


@pytest.mark.skipif(__import__("shutil").which("bash") is None, reason="bash not on PATH")
def test_build_skills_server_registers_and_runs_sh(tmp_path):
    from aimu.skills.mcp import build_skills_server
    from aimu.tools.client import MCPClient

    md = make_skill_dir(tmp_path, "shtools", "Has shell scripts.")
    _write_script(md, "greet.sh", '#!/usr/bin/env bash\necho "sh:$1"\n')

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    client = MCPClient(server=build_skills_server(manager))
    assert "shtools__greet" in [t.name for t in client.list_tools()]

    out = client.call_tool("shtools__greet", {"args": "world"})
    assert "sh:world" in out.content[0].text


def test_run_script_file_nonzero_and_unsupported(tmp_path):
    from aimu.skills.mcp import run_script_file

    bad = tmp_path / "boom.py"
    bad.write_text("import sys\nsys.stderr.write('nope')\nsys.exit(3)\n", encoding="utf-8")
    out = run_script_file(bad)
    assert "exited with code 3" in out and "nope" in out

    txt = tmp_path / "note.txt"
    txt.write_text("hi", encoding="utf-8")
    assert "unsupported script extension" in run_script_file(txt)


def test_run_script_file_timeout(tmp_path, monkeypatch):
    from aimu.skills import mcp as mcp_mod

    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    monkeypatch.setattr(mcp_mod, "_SCRIPT_TIMEOUT", 0.2)
    assert "timed out" in mcp_mod.run_script_file(slow)


def test_failing_script_tool_tells_model_how_to_fix_it(tmp_path):
    """A registered script tool that errors must name its skill + filename so the model can
    overwrite the right file (avoids 'fixing' into a new file and leaving the bug)."""
    from aimu.skills.mcp import build_skills_server
    from aimu.tools.client import MCPClient

    md = make_skill_dir(tmp_path, "buggy", "Buggy.")
    _write_script(md, "go.py", "import sys\nsys.exit(2)\n")

    client = MCPClient(server=build_skills_server(SkillManager(skill_dirs=[str(tmp_path)])))
    out = client.call_tool("buggy__go", {}).content[0].text
    assert "exited with code 2" in out
    assert "go.py" in out and "buggy" in out  # the exact file + skill to overwrite


def test_script_tool_names_includes_sh_and_dedupes(tmp_path):
    md = make_skill_dir(tmp_path, "mixed", "Mixed scripts.")
    _write_script(md, "a.py", "print(1)\n")
    _write_script(md, "b.sh", "echo 2\n")
    _write_script(md, "c.py", "print(3)\n")
    _write_script(md, "c.sh", "echo 3\n")  # collides with c.py on the {skill}__c name

    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["mixed"]
    names = skill.script_tool_names()
    assert names == ["mixed__a", "mixed__b", "mixed__c"]  # c listed once


def test_write_skill_with_scripts_creates_discoverable_and_chmod(tmp_path):
    import os

    from aimu.skills import write_skill

    write_skill(
        "deploy",
        "Deploy things.",
        "# Deploy",
        skills_dir=tmp_path,
        scripts={"run.py": "print('hi')\n", "do.sh": "echo hi\n"},
    )
    sk = SkillManager(skill_dirs=[str(tmp_path)]).skills["deploy"]
    assert set(sk.script_tool_names()) == {"deploy__run", "deploy__do"}
    assert os.access(tmp_path / "deploy" / "scripts" / "do.sh", os.X_OK)


@pytest.mark.parametrize("bad", ["../x.py", "a/b.sh", "up.txt", "Bad.py", "noext"])
def test_write_skill_rejects_bad_script_filename(tmp_path, bad):
    from aimu.skills import write_skill

    with pytest.raises(ValueError):
        write_skill("s", "desc", "body", skills_dir=tmp_path, scripts={bad: "x"})


def test_write_skill_overwrite_replaces_script(tmp_path):
    from aimu.skills import write_skill

    write_skill("s", "desc", "body", skills_dir=tmp_path, scripts={"x.py": "print(1)\n"})
    # A second new skill with the same name without overwrite is refused at the SKILL.md level.
    with pytest.raises(FileExistsError):
        write_skill("s", "desc", "body", skills_dir=tmp_path, scripts={"y.py": "print(9)\n"})
    # overwrite=True replaces the script content (the add_skill_script path).
    write_skill("s", "desc", "body", skills_dir=tmp_path, overwrite=True, scripts={"x.py": "print(2)\n"})
    assert (tmp_path / "s" / "scripts" / "x.py").read_text() == "print(2)\n"


def test_reload_skills_surfaces_new_script_tool(tmp_path):
    from unittest.mock import MagicMock

    from aimu.agents.skill_agent import SkillAgent
    from aimu.skills import write_skill

    make_skill_dir(tmp_path, "grow", "Grows scripts.")
    client = MagicMock()
    client.system_message = ""
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()
    assert "grow__added" not in [fn.__name__ for fn in client.tools]

    write_skill(
        "grow", "Grows scripts.", "# Grow", skills_dir=tmp_path, overwrite=True, scripts={"added.py": "print('x')\n"}
    )
    manager.refresh()
    agent.reload_skills()

    assert "grow__added" in [fn.__name__ for fn in client.tools]


def test_reload_keeps_existing_script_tools_callable(tmp_path):
    """A pre-existing script tool stays callable after a reload (replace, don't leave stale)."""
    from unittest.mock import MagicMock

    from aimu.agents.skill_agent import SkillAgent
    from aimu.skills import write_skill

    write_skill("pre", "Pre.", "# Pre", skills_dir=tmp_path, scripts={"p.py": "print('pre ok')\n"})
    client = MagicMock()
    client.system_message = ""
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()
    pre = next(t for t in client.tools if t.__name__ == "pre__p")
    assert pre().strip() == "pre ok"

    write_skill("other", "Other.", "# Other", skills_dir=tmp_path, overwrite=True, scripts={"q.py": "print('q')\n"})
    manager.refresh()
    agent.reload_skills()

    pre_after = next(t for t in client.tools if t.__name__ == "pre__p")
    assert pre_after().strip() == "pre ok"


def test_reinject_catalog_does_not_duplicate(tmp_path):
    from unittest.mock import MagicMock

    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "cat", "Catalog skill.")
    client = MagicMock()
    client.system_message = "Base prompt."
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()
    agent.reload_skills()

    assert client.system_message.count("<available_skills>") == 1
    assert client.system_message.startswith("Base prompt.")
