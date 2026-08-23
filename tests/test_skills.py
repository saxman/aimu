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

    def _resolve_generate_kwargs(self, generate_kwargs=None):
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
# Agent Skills spec validation
#
# The rules come from https://agentskills.io/specification. A skill that violates one is
# rejected on discovery rather than loaded under a name the spec forbids: a skill that loads
# but cannot be addressed is the silent failure SkillLoadError exists to prevent.
# ---------------------------------------------------------------------------


def _write_skill_md(parent: Path, dir_name: str, frontmatter: str) -> Path:
    """Write a SKILL.md with frontmatter given verbatim, so a test can violate the spec."""
    skill_dir = parent / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\n{frontmatter}\n---\n\n# Body\n", encoding="utf-8")
    return skill_md


def test_skill_manager_raises_on_missing_name(tmp_path):
    """`name` is required by the spec, so it can no longer default to the directory name."""
    from aimu.skills import SkillLoadError

    _write_skill_md(tmp_path, "nameless", "description: Has no name.")
    with pytest.raises(SkillLoadError, match="name"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


@pytest.mark.parametrize(
    "name, dir_name",
    [
        ("My Skill", "my-skill"),  # uppercase and a space
        ("under_score", "under_score"),  # underscore is not in [a-z0-9-]
        ("-leading", "-leading"),
        ("trailing-", "trailing-"),
        ("double--hyphen", "double--hyphen"),
        ("x" * 65, "x" * 65),  # over the 64-character limit
    ],
)
def test_skill_manager_raises_on_a_name_the_spec_forbids(tmp_path, name, dir_name):
    from aimu.skills import SkillLoadError

    _write_skill_md(tmp_path, dir_name, f"name: {name}\ndescription: Does a thing.")
    with pytest.raises(SkillLoadError, match="name"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


def test_skill_manager_raises_when_name_disagrees_with_its_directory(tmp_path):
    """The spec requires `name` to match the parent directory, so one addresses the other."""
    from aimu.skills import SkillLoadError

    _write_skill_md(tmp_path, "on-disk", "name: in-frontmatter\ndescription: Mismatched.")
    with pytest.raises(SkillLoadError, match="directory"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


def test_skill_manager_raises_on_an_over_long_description(tmp_path):
    from aimu.skills import SkillLoadError

    _write_skill_md(tmp_path, "wordy", f"name: wordy\ndescription: {'d' * 1025}")
    with pytest.raises(SkillLoadError, match="description"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


def test_skill_manager_raises_on_an_over_long_compatibility(tmp_path):
    from aimu.skills import SkillLoadError

    _write_skill_md(tmp_path, "picky", f"name: picky\ndescription: Fine.\ncompatibility: {'c' * 501}")
    with pytest.raises(SkillLoadError, match="compatibility"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


def test_skill_manager_raises_on_non_string_metadata_values(tmp_path):
    """The spec defines metadata as a map of string to string, so a nested value is invalid."""
    from aimu.skills import SkillLoadError

    _write_skill_md(tmp_path, "nested", "name: nested\ndescription: Fine.\nmetadata:\n  version:\n    major: 1")
    with pytest.raises(SkillLoadError, match="metadata"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


def test_skill_manager_accepts_every_optional_spec_field(tmp_path):
    """A skill using all of license, compatibility, metadata and allowed-tools still loads."""
    _write_skill_md(
        tmp_path,
        "fully-specified",
        "name: fully-specified\n"
        "description: Uses every optional field.\n"
        "license: Apache-2.0\n"
        "compatibility: Requires git and network access\n"
        "metadata:\n  author: example-org\n  version: '1.0'\n"
        "allowed-tools: Bash(git:*) Read",
    )
    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["fully-specified"]
    assert skill.license_info == "Apache-2.0"
    assert skill.compatibility == "Requires git and network access"
    assert skill.metadata == {"author": "example-org", "version": "1.0"}


# ---------------------------------------------------------------------------
# allowed-tools
# ---------------------------------------------------------------------------


def test_allowed_tools_is_split_on_whitespace(tmp_path):
    """The spec stores allowed-tools as one space-separated string; callers want the entries."""
    _write_skill_md(tmp_path, "gated", "name: gated\ndescription: Fine.\nallowed-tools: Bash(git:*) Bash(jq:*) Read")
    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["gated"]
    assert skill.allowed_tools == ("Bash(git:*)", "Bash(jq:*)", "Read")


def test_allowed_tools_defaults_to_empty_when_absent(tmp_path):
    make_skill_dir(tmp_path, "ungated", "No allowed-tools field.")
    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["ungated"]
    assert skill.allowed_tools == ()


# ---------------------------------------------------------------------------
# SkillManager include filtering
# ---------------------------------------------------------------------------


def test_include_scopes_discovery_to_the_named_skills(tmp_path):
    """A host giving one agent a subset of skills should not have to filter in three places:
    catalog_prompt and the skills server both read `skills`."""
    make_skill_dir(tmp_path, "wanted", "Keep me.")
    make_skill_dir(tmp_path, "unwanted", "Drop me.")

    manager = SkillManager(skill_dirs=[str(tmp_path)], include=["wanted"])

    assert sorted(manager.skills) == ["wanted"]
    assert "unwanted" not in manager.catalog_prompt()


def test_include_none_discovers_everything(tmp_path):
    make_skill_dir(tmp_path, "one", "First.")
    make_skill_dir(tmp_path, "two", "Second.")
    assert sorted(SkillManager(skill_dirs=[str(tmp_path)]).skills) == ["one", "two"]


def test_include_raises_on_a_name_that_was_not_discovered(tmp_path):
    """An unknown include name is a typo that would otherwise hand the agent fewer skills
    than asked for, with nothing to say so."""
    from aimu.skills import SkillLoadError

    make_skill_dir(tmp_path, "present", "Here.")
    manager = SkillManager(skill_dirs=[str(tmp_path)], include=["absent"])
    with pytest.raises(SkillLoadError, match="absent"):
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
    """_setup_skills() surfaces the skills server's tools (incl. activate_skill) via _effective_tools."""
    from unittest.mock import MagicMock
    from aimu.agents.skill_agent import SkillAgent

    make_skill_dir(tmp_path, "my-skill", "Does my thing.")

    client = MagicMock()
    client.system_message = None
    client.tools = []

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager)
    agent._setup_skills()

    assert "activate_skill" in [fn.__name__ for fn in agent._effective_tools(None)]


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


# ---------------------------------------------------------------------------
# Self-contained scripts (PEP 723) and non-interactive stdin
#
# https://agentskills.io/skill-creation/using-scripts recommends declaring a script's
# dependencies inline and running it with `uv run`, and states that a non-interactive shell is a
# hard requirement of the execution environment.
# ---------------------------------------------------------------------------


_PEP_723_SCRIPT = '# /// script\n# dependencies = [\n#   "humanize",\n# ]\n# ///\nimport humanize\nprint(humanize.intcomma(1234567))\n'


def test_a_plain_script_runs_on_the_host_interpreter(tmp_path):
    """Without an inline dependency block a script keeps using this interpreter, so a skill that
    relies on a package installed in the host environment is unaffected."""
    import sys

    from aimu.skills.mcp import _interpreter_for

    plain = tmp_path / "plain.py"
    plain.write_text("print('ok')\n", encoding="utf-8")
    assert _interpreter_for(plain) == [sys.executable]


def test_a_script_declaring_inline_dependencies_runs_through_uv(tmp_path):
    """PEP 723 dependencies need an installer; `sys.executable` alone raises ModuleNotFoundError."""
    from aimu.skills.mcp import _interpreter_for

    script = tmp_path / "inline.py"
    script.write_text(_PEP_723_SCRIPT, encoding="utf-8")

    argv = _interpreter_for(script)

    assert argv[0].endswith("uv")
    assert argv[1:] == ["run", "--script"]


def test_inline_dependencies_are_actually_installed_and_importable(tmp_path):
    """End to end, because the interpreter choice only matters if the dependency resolves."""
    from aimu.skills.mcp import run_script_file

    script = tmp_path / "inline.py"
    script.write_text(_PEP_723_SCRIPT, encoding="utf-8")

    assert "1,234,567" in run_script_file(script)


def test_a_missing_uv_names_the_fix_rather_than_failing_on_the_import(tmp_path, monkeypatch):
    from aimu.skills import mcp as mcp_mod

    script = tmp_path / "inline.py"
    script.write_text(_PEP_723_SCRIPT, encoding="utf-8")
    monkeypatch.setattr(mcp_mod.shutil, "which", lambda name: None)

    out = mcp_mod.run_script_file(script)

    assert "uv" in out
    assert "dependencies" in out


def test_a_script_never_inherits_the_parent_stdin(tmp_path, monkeypatch):
    """A non-interactive shell is a hard requirement of the spec's execution environment, and an
    inherited stdin made an interactive script burn the whole timeout, blocking the event loop with
    it. Closing stdin turns that into an immediate EOFError the agent can act on.

    Asserted on the call rather than on the timing, because the behaviour otherwise depends on
    whatever stdin the parent happens to have: under pytest it is already closed, so a script
    reading it raises EOFError whether or not this function asks for that.
    """
    import subprocess

    from aimu.skills import mcp as mcp_mod

    recorded = {}

    def fake_run(argv, **kwargs):
        recorded.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(mcp_mod.subprocess, "run", fake_run)

    script = tmp_path / "plain.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    mcp_mod.run_script_file(script)

    assert recorded["stdin"] is subprocess.DEVNULL


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


# ---------------------------------------------------------------------------
# Script tool names are slugified
#
# The name is {skill}__{stem} with both halves reduced to [a-z0-9_], so the double
# underscore stays the only separator and the result is a valid identifier. Spec validation
# already confines a skill name to [a-z0-9-], so slugifying that half only folds hyphens to
# underscores; the script stem still legitimately allows either separator.
# ---------------------------------------------------------------------------


def test_script_tool_name_slugifies_a_hyphenated_skill_name(tmp_path):
    md = make_skill_dir(tmp_path, "pdf-processing", "Work with PDFs.")
    _write_script(md, "extract_pages.py", "print(1)\n")

    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["pdf-processing"]
    assert skill.script_tool_names() == ["pdf_processing__extract_pages"]


def test_script_tool_name_slugifies_a_hyphenated_script_stem(tmp_path):
    md = make_skill_dir(tmp_path, "deploy", "Deploy things.")
    _write_script(md, "merge-all.py", "print(1)\n")

    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["deploy"]
    assert skill.script_tool_names() == ["deploy__merge_all"]


def test_script_tool_names_dedupe_when_two_stems_slugify_alike(tmp_path):
    md = make_skill_dir(tmp_path, "ops", "Ops helpers.")
    _write_script(md, "backup-db.py", "print('hyphen')\n")
    _write_script(md, "backup_db.py", "print('underscore')\n")  # same slug as backup-db

    skill = SkillManager(skill_dirs=[str(tmp_path)]).skills["ops"]
    assert skill.script_tool_names() == ["ops__backup_db"]


def test_a_skill_name_that_is_not_a_slug_is_rejected_rather_than_slugified(tmp_path):
    """A name with spaces or capitals once reached the tool name unaltered and produced an
    identifier no provider accepts. Spec validation now refuses it on discovery, which is a
    stronger guarantee than slugifying it into something callable but unaddressable.
    """
    from aimu.skills import SkillLoadError

    skill_dir = tmp_path / "shouty"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: My Skill\ndescription: Shouts.\n---\n\n# Body\n", encoding="utf-8")
    _write_script(skill_dir / "SKILL.md", "go.py", "print('ok')\n")

    with pytest.raises(SkillLoadError, match="name"):
        _ = SkillManager(skill_dirs=[str(tmp_path)]).skills


def test_every_catalog_script_tool_name_is_callable_on_the_server(tmp_path):
    """The catalogue and the skills server must agree, since three sites build the name.

    The catalogue is what the model reads, so a name advertised but not registered is a
    tool call that cannot succeed.
    """
    from aimu.skills.mcp import build_skills_server
    from aimu.tools import MCPClient

    md = make_skill_dir(tmp_path, "pdf-processing", "Work with PDFs.")
    _write_script(md, "extract-pages.py", "print('extracted')\n")
    _write_script(md, "merge.sh", "echo merged\n")

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    advertised = manager.skills["pdf-processing"].script_tool_names()
    assert advertised  # guard: an empty catalogue would pass the loop below vacuously

    client = MCPClient(server=build_skills_server(manager))
    registered = {tool.name for tool in client.list_tools()}
    assert set(advertised) <= registered

    for name in advertised:
        assert name in manager.catalog_prompt()


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
    assert "grow__added" not in [fn.__name__ for fn in agent._effective_tools(None)]

    write_skill(
        "grow", "Grows scripts.", "# Grow", skills_dir=tmp_path, overwrite=True, scripts={"added.py": "print('x')\n"}
    )
    manager.refresh()
    agent.reload_skills()

    assert "grow__added" in [fn.__name__ for fn in agent._effective_tools(None)]


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
    pre = next(t for t in agent._effective_tools(None) if t.__name__ == "pre__p")
    assert pre().strip() == "pre ok"

    write_skill("other", "Other.", "# Other", skills_dir=tmp_path, overwrite=True, scripts={"q.py": "print('q')\n"})
    manager.refresh()
    agent.reload_skills()

    pre_after = next(t for t in agent._effective_tools(None) if t.__name__ == "pre__p")
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


# ---------------------------------------------------------------------------
# Host-provided environment
#
# Environment variables are one of the three input channels the spec's script guidance names, and
# the one a host uses to hand a script context it cannot discover for itself (where to write output,
# which account to use). Merged over os.environ rather than replacing it, so PATH survives and the
# interpreter lookup keeps working.
# ---------------------------------------------------------------------------


def test_run_script_file_passes_host_environment_to_the_script(tmp_path):
    from aimu.skills.mcp import run_script_file

    script = tmp_path / "reads_env.py"
    script.write_text("import os\nprint(os.environ['HOST_PROVIDED'])\n", encoding="utf-8")

    assert "from-the-host" in run_script_file(script, env={"HOST_PROVIDED": "from-the-host"})


def test_host_environment_is_merged_over_the_inherited_one(tmp_path):
    """Replacing the environment would strip PATH, which the interpreter lookup itself needs."""
    from aimu.skills.mcp import run_script_file

    script = tmp_path / "reads_both.py"
    script.write_text("import os\nprint(bool(os.environ.get('PATH')), os.environ['EXTRA'])\n", encoding="utf-8")

    assert "True added" in run_script_file(script, env={"EXTRA": "added"})


def test_a_script_tool_receives_the_servers_environment(tmp_path):
    """The host sets it once when building the server, not per call."""
    from aimu.skills.mcp import build_skills_server
    from aimu.tools import MCPClient

    md = make_skill_dir(tmp_path, "reporter", "Writes a report.")
    _write_script(md, "where.py", "import os\nprint(os.environ['REPORT_DIR'])\n")

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    client = MCPClient(server=build_skills_server(manager, env={"REPORT_DIR": "/tmp/reports"}))
    try:
        tool = next(fn for fn in client.as_tools() if fn.__name__ == "reporter__where")
        assert "/tmp/reports" in tool()
    finally:
        client.close()


def test_skill_agents_script_tools_receive_its_environment(tmp_path):
    """A SkillAgent builds its own skills server, so a host cannot pass ``env`` to it directly.

    Without ``script_env`` the only route left is a process-wide variable, which hands one agent's
    context to every subprocess in the process.
    """
    from unittest.mock import MagicMock

    from aimu.agents.skill_agent import SkillAgent

    md = make_skill_dir(tmp_path, "reporter", "Writes a report.")
    _write_script(md, "where.py", "import os\nprint(os.environ['REPORT_DIR'])\n")

    client = MagicMock()
    client.system_message = ""
    client.tools = []
    agent = SkillAgent(
        model_client=client,
        skill_manager=SkillManager(skill_dirs=[str(tmp_path)]),
        script_env={"REPORT_DIR": "/tmp/reports"},
    )
    agent._setup_skills()

    tool = next(fn for fn in agent._effective_tools(None) if fn.__name__ == "reporter__where")
    assert "/tmp/reports" in tool()


def test_reload_skills_keeps_the_agents_environment(tmp_path):
    """``reload_skills`` rebuilds the server, which is the second place the env can be dropped."""
    from unittest.mock import MagicMock

    from aimu.agents.skill_agent import SkillAgent

    md = make_skill_dir(tmp_path, "reporter", "Writes a report.")
    _write_script(md, "where.py", "import os\nprint(os.environ['REPORT_DIR'])\n")

    client = MagicMock()
    client.system_message = ""
    client.tools = []
    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(model_client=client, skill_manager=manager, script_env={"REPORT_DIR": "/tmp/reports"})
    agent._setup_skills()
    agent.reload_skills()

    tool = next(fn for fn in agent._effective_tools(None) if fn.__name__ == "reporter__where")
    assert "/tmp/reports" in tool()
