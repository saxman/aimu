"""Tool groups on the sync Agent and SkillAgent, driven through the real tool loop (mock client)."""

from __future__ import annotations

import time

import pytest

from aimu.agents import Agent
from aimu.tools import ToolGroup, tool
from helpers import MockModelClient


@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


@tool
def remember(fact: str) -> str:
    """Remember a fact."""
    return "ok"


def _web(calls=None):
    def build():
        if calls is not None:
            calls.append(1)
        return [search_web]

    return ToolGroup("web", "Search the web.", build=build, guidance="Cite sources.")


MEMORY = ToolGroup("memory", "Remember facts.")


def _tool_results(agent):
    return [m["content"] for m in agent.model_client.messages if m.get("role") == "tool"]


def _names(agent):
    return [fn.__name__ for fn in agent._effective_tools(None)]


def test_no_core_tools_without_a_deferred_group():
    agent = Agent(MockModelClient(["hi"]), tools=[remember], tool_groups=[MEMORY])
    assert _names(agent) == ["remember"]


def test_core_tools_follow_the_agents_own_tools():
    agent = Agent(MockModelClient(["hi"]), tools=[remember], tool_groups=[MEMORY, _web()])
    assert _names(agent) == ["remember", "available_tools", "load_tools"]


def test_a_host_tool_named_like_a_core_tool_is_refused():
    @tool
    def load_tools(name: str) -> str:
        """Collides."""
        return name

    with pytest.raises(ValueError, match="'load_tools'"):
        Agent(MockModelClient(["hi"]), tools=[load_tools], tool_groups=[_web()])


def test_malformed_groups_are_refused_at_construction():
    with pytest.raises(ValueError, match="only ToolGroup instances"):
        Agent(MockModelClient(["hi"]), tool_groups=["web"])


def test_the_model_lists_then_loads_then_calls_a_loaded_tool():
    client = MockModelClient(
        [
            {"tool": "available_tools", "arguments": {}},
            {"tool": "load_tools", "arguments": {"name": "web"}},
            {"tool": "search_web", "arguments": {"query": "kokua"}},
            "done",
        ]
    )
    agent = Agent(client, tools=[remember], tool_groups=[MEMORY, _web()])
    assert agent.run("task") == "done"
    assert _tool_results(agent) == [
        "memory [resident]: Remember facts.\nweb [loadable]: Search the web.",
        "Loaded 'web': search_web.\nCite sources.",
        "results for kokua",
    ]
    assert _names(agent) == ["remember", "available_tools", "load_tools", "search_web"]


def test_calling_an_unloaded_tool_points_at_loading():
    client = MockModelClient([{"tool": "search_web", "arguments": {"query": "x"}}, "done"])
    agent = Agent(client, tool_groups=[_web()])
    agent.run("task")
    assert _tool_results(agent) == [
        "Tool 'search_web' not found. If it belongs to a tool group you have not loaded, call available_tools "
        "to see what you can load, then load_tools."
    ]


def test_an_unknown_tool_without_groups_keeps_the_plain_message():
    client = MockModelClient([{"tool": "nope", "arguments": {}}, "done"])
    agent = Agent(client, tools=[remember])
    agent.run("task")
    assert _tool_results(agent) == ["Tool 'nope' not found."]


def test_host_replay_matches_what_the_model_would_get_and_builds_once():
    calls = []
    agent = Agent(MockModelClient(["hi"]), tool_groups=[_web(calls)])
    assert agent.load_tool_group("web") == "Loaded 'web': search_web.\nCite sources."
    assert agent.load_tool_group("web") == "'web' is already loaded."
    assert calls == [1]


def test_loads_persist_across_runs_of_one_instance():
    client = MockModelClient([{"tool": "load_tools", "arguments": {"name": "web"}}, "first", "second"])
    agent = Agent(client, tool_groups=[_web()])
    agent.run("one")
    agent.run("two")
    assert "search_web" in _names(agent)


def test_instances_built_from_one_group_list_never_share_loads():
    groups = [_web()]
    first = Agent(MockModelClient(["hi"]), tool_groups=groups)
    second = Agent(MockModelClient(["hi"]), tool_groups=groups)
    first.load_tool_group("web")
    assert "search_web" in _names(first)
    assert "search_web" not in _names(second)


def test_a_build_failure_reaches_the_model_and_leaves_the_group_loadable(caplog):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("store locked")
        return [search_web]

    agent = Agent(MockModelClient(["hi"]), tool_groups=[ToolGroup("web", "Search.", build=flaky)])
    assert agent.load_tool_group("web") == "Could not load 'web': store locked"
    assert "failed to build" in caplog.text
    assert agent.load_tool_group("web") == "Loaded 'web': search_web."


def test_two_concurrent_loads_of_one_group_build_once():
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return [search_web]

    client = MockModelClient(
        [
            {"tools": [{"name": "load_tools", "arguments": {"name": "web"}}] * 2},
            "done",
        ]
    )
    agent = Agent(client, tool_groups=[ToolGroup("web", "Search.", build=slow)], concurrent_tool_calls=True)
    agent.run("task")
    assert calls == [1]
    assert sorted(_tool_results(agent)) == ["'web' is already loaded.", "Loaded 'web': search_web."]


def test_skill_agent_appends_loaded_tools_after_its_skill_tools(tmp_path):
    from aimu.agents import SkillAgent
    from aimu.skills import SkillManager

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(MockModelClient(["hi"]), tools=[remember], tool_groups=[_web()], skill_manager=manager)

    @tool
    def demo__run() -> str:
        """A skill script tool."""
        return "ran"

    agent._skills_tools = [demo__run]
    agent.load_tool_group("web")
    assert _names(agent) == ["remember", "demo__run", "available_tools", "load_tools", "search_web"]
