"""
Tests for aimu.skills — Skill, SkillManager, and Agent skill integration.

All filesystem tests use tmp_path so no real ~/.agents/skills paths are touched.
Unit tests use MagicMock inline. The model_client fixture is available for
integration tests:
  - Default (no --client): MockModelClient
  - pytest tests/test_skills.py --client=ollama --model=LLAMA_3_2_3B
"""

from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import pytest

from aimu.models import ModelClient
from aimu.models.base import StreamChunk, StreamingContentType
from aimu.skills.manager import SkillManager
from aimu.skills.skill import Skill
from conftest import create_real_model_client, resolve_model_params

_MOCK = "mock"


class _MockModelClient(ModelClient):
    """Minimal ModelClient stub for skill integration tests."""

    def __init__(self):
        self.model = MagicMock()
        self.model.supports_tools = False
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.mcp_client = None
        self.last_thinking = ""

    def chat(self, user_message, generate_kwargs=None, use_tools=True):
        self.messages.append({"role": "user", "content": user_message})
        response = "I can help with that."
        self.messages.append({"role": "assistant", "content": response})
        return response

    def chat_streamed(self, user_message, generate_kwargs=None, use_tools=True):
        response = self.chat(user_message)
        yield StreamChunk(StreamingContentType.GENERATING, response)

    def generate(self, prompt, generate_kwargs=None):
        return "Generated response."

    def generate_streamed(self, prompt, generate_kwargs=None, include_thinking=True):
        yield StreamChunk(StreamingContentType.GENERATING, "Generated response.")

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}


def pytest_generate_tests(metafunc):
    if "model_client" not in metafunc.fixturenames:
        return
    params = resolve_model_params(metafunc.config, default_params=[_MOCK])
    metafunc.parametrize("model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[ModelClient]:
    if request.param == _MOCK:
        yield _MockModelClient()
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
# Skill.load_body
# ---------------------------------------------------------------------------


def test_skill_load_body_strips_frontmatter(tmp_path):
    skill_md = make_skill_dir(tmp_path, "my-skill", "Does things.", body="## Steps\n1. Do it.")
    skill = Skill(name="my-skill", description="Does things.", path=skill_md)
    body = skill.load_body()
    assert "## Steps" in body
    assert "---" not in body
    assert "name:" not in body


def test_skill_load_body_no_frontmatter(tmp_path):
    skill_dir = tmp_path / "bare-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("Just instructions, no frontmatter.", encoding="utf-8")
    skill = Skill(name="bare-skill", description="x", path=skill_md)
    assert skill.load_body() == "Just instructions, no frontmatter."


def test_skill_base_dir(tmp_path):
    skill_md = make_skill_dir(tmp_path, "s", "x")
    skill = Skill(name="s", description="x", path=skill_md)
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


def test_skill_manager_skips_missing_description(tmp_path):
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: bad-skill\n---\n\nNo description.", encoding="utf-8")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert "bad-skill" not in manager.skills


def test_skill_manager_skips_no_frontmatter(tmp_path):
    skill_dir = tmp_path / "raw-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("No frontmatter at all.", encoding="utf-8")
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    assert len(manager.skills) == 0


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

    # Pass project dir first — it wins on collision
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
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    result = manager.get_skill_body("nonexistent")
    assert "not found" in result.lower()


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
    client.mcp_client = None

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    assert "my-skill" in client.system_message
    assert "Does my thing." in client.system_message
    assert "activate_skill" in client.system_message


def test_agent_setup_skills_attaches_mcp_client(tmp_path):
    """_setup_skills() creates and attaches an MCPClient to the model client."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "my-skill", "Does my thing.")

    client = MagicMock()
    client.system_message = None
    client.mcp_client = None

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    assert client.mcp_client is not None


def test_agent_setup_skills_no_op_when_no_skills(tmp_path):
    """_setup_skills() does nothing when no skills are found."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    client = MagicMock()
    client.system_message = "Original."
    client.mcp_client = None

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    # No skills found — mcp_client must not have been set
    assert client.mcp_client is None


def test_agent_setup_skills_runs_only_once(tmp_path):
    """_setup_skills() is idempotent — calling it twice doesn't duplicate catalog."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "once-skill", "Run only once.")

    client = MagicMock()
    client.system_message = ""
    client.mcp_client = None

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)

    agent._setup_skills()
    agent._setup_skills()  # second call — should be a no-op
    assert agent._skills_setup_done is True


def test_agent_from_config_with_skill_dirs(tmp_path):
    """SkillAgent.from_config with skill_dirs creates a SkillManager."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "cfg-skill", "From config.")

    client = MagicMock()
    client.system_message = None

    agent = SkillAgent.from_config({"name": "cfg-agent", "skill_dirs": [str(tmp_path)]}, client)

    assert "cfg-skill" in agent.skill_manager.skills
