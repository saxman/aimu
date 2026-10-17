"""Tool groups on the async Agent and SkillAgent (mock client): the async twin of
tests/test_tool_groups_agent.py, plus the event-loop property only the async surface has."""

from __future__ import annotations

import asyncio
import time

import pytest

from aimu.aio import Agent
from aimu.tools import ToolGroup, tool
from helpers_aio import MockAsyncModelClient


@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


def _web(calls=None, delay=0.0):
    def build():
        if calls is not None:
            calls.append(1)
        if delay:
            time.sleep(delay)
        return [search_web]

    return ToolGroup("web", "Search the web.", build=build, guidance="Cite sources.")


def _tool_results(agent):
    return [m["content"] for m in agent.model_client.messages if m.get("role") == "tool"]


def _names(agent):
    return [fn.__name__ for fn in agent._effective_tools(None)]


async def test_the_model_loads_then_calls_a_loaded_tool():
    client = MockAsyncModelClient(
        [
            {"tool": "load_tools", "arguments": {"name": "web"}},
            {"tool": "search_web", "arguments": {"query": "kokua"}},
            "done",
        ]
    )
    agent = Agent(client, tool_groups=[_web()])
    assert await agent.run("task") == "done"
    assert _tool_results(agent) == ["Loaded 'web': search_web.\nCite sources.", "results for kokua"]
    assert _names(agent) == ["available_tools", "load_tools", "search_web"]


async def test_calling_an_unloaded_tool_points_at_loading():
    client = MockAsyncModelClient([{"tool": "search_web", "arguments": {"query": "x"}}, "done"])
    agent = Agent(client, tool_groups=[_web()])
    await agent.run("task")
    assert _tool_results(agent) == [
        "Tool 'search_web' not found. If it belongs to a tool group you have not loaded, call available_tools "
        "to see what you can load, then load_tools."
    ]


async def test_host_replay_matches_and_builds_once():
    calls = []
    agent = Agent(MockAsyncModelClient(["hi"]), tool_groups=[_web(calls)])
    assert await agent.load_tool_group("web") == "Loaded 'web': search_web.\nCite sources."
    assert await agent.load_tool_group("web") == "'web' is already loaded."
    assert calls == [1]


async def test_instances_built_from_one_group_list_never_share_loads():
    groups = [_web()]
    first = Agent(MockAsyncModelClient(["hi"]), tool_groups=groups)
    second = Agent(MockAsyncModelClient(["hi"]), tool_groups=groups)
    await first.load_tool_group("web")
    assert "search_web" in _names(first)
    assert "search_web" not in _names(second)


async def test_two_concurrent_loads_in_one_turn_build_once():
    calls = []
    client = MockAsyncModelClient([{"tools": [{"name": "load_tools", "arguments": {"name": "web"}}] * 2}, "done"])
    agent = Agent(client, tool_groups=[_web(calls, delay=0.1)], concurrent_tool_calls=True)
    await agent.run("task")
    assert calls == [1]
    assert sorted(_tool_results(agent)) == ["'web' is already loaded.", "Loaded 'web': search_web.\nCite sources."]


async def test_a_slow_build_does_not_block_the_event_loop():
    agent = Agent(MockAsyncModelClient(["hi"]), tool_groups=[_web(delay=0.3)])
    ticks = []

    async def ticker():
        for _ in range(5):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    start = time.monotonic()
    await asyncio.gather(agent.load_tool_group("web"), ticker())
    assert len(ticks) == 5
    # The load is scheduled first; had its build run on the event loop, the ticker could not have started
    # until the 0.3s build finished.
    assert ticks[0] - start < 0.1


async def test_a_build_failure_reaches_the_model_and_leaves_the_group_loadable():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("store locked")
        return [search_web]

    agent = Agent(MockAsyncModelClient(["hi"]), tool_groups=[ToolGroup("web", "Search.", build=flaky)])
    assert await agent.load_tool_group("web") == "Could not load 'web': store locked"
    assert await agent.load_tool_group("web") == "Loaded 'web': search_web."


async def test_a_host_tool_named_like_a_core_tool_is_refused():
    @tool
    def available_tools(filter: str = "") -> str:
        """Collides."""
        return filter

    with pytest.raises(ValueError, match="'available_tools'"):
        Agent(MockAsyncModelClient(["hi"]), tools=[available_tools], tool_groups=[_web()])


async def test_skill_agent_appends_loaded_tools_after_its_skill_tools(tmp_path):
    from aimu.aio import SkillAgent
    from aimu.skills import SkillManager

    manager = SkillManager(skill_dirs=[str(tmp_path)])
    agent = SkillAgent(MockAsyncModelClient(["hi"]), tool_groups=[_web()], skill_manager=manager)

    @tool
    def demo__run() -> str:
        """A skill script tool."""
        return "ran"

    agent._skills_tools = [demo__run]
    await agent.load_tool_group("web")
    assert _names(agent) == ["demo__run", "available_tools", "load_tools", "search_web"]


async def test_a_cancelled_load_does_not_start_a_second_build():
    # A run cancelled mid-build (a user pressing stop during a slow store open) must not leave the next
    # load to start a second build beside the one still running in its worker thread.
    starts = []

    def slow():
        starts.append(1)
        time.sleep(0.3)
        return [search_web]

    agent = Agent(MockAsyncModelClient(["hi"]), tool_groups=[ToolGroup("web", "Search.", build=slow)])
    first = asyncio.ensure_future(agent.load_tool_group("web"))
    await asyncio.sleep(0.05)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await agent.load_tool_group("web") == "Loaded 'web': search_web."
    assert starts == [1]
