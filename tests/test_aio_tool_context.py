"""Async ToolContext dependency injection + async Agent deps/schema support.

Mirrors tests/test_tool_context.py on the aio surface. Deterministic via MockAsyncModelClient.
"""

from dataclasses import dataclass, field

import pytest

from aimu.aio import Agent
from aimu.tools import ToolContext, tool
from helpers_aio import MockAsyncModelClient


# ---------------------------------------------------------------------------
# Dispatch: the async client fills ctx.deps for sync and async tools
# ---------------------------------------------------------------------------


async def test_async_dispatch_injects_context_into_sync_tool():
    @dataclass
    class Deps:
        seen: dict = field(default_factory=dict)

    captured = {}

    @tool
    def remember(ctx: ToolContext[Deps], key: str) -> str:
        """Remember a value."""
        captured["deps"] = ctx.deps
        ctx.deps.seen[key] = True
        return f"ok:{key}"

    deps = Deps()
    client = MockAsyncModelClient([])
    client.tools = [remember]
    client.tool_context_deps = deps

    await client._handle_tool_calls([{"name": "remember", "arguments": {"key": "abc"}}])

    assert captured["deps"] is deps
    assert deps.seen == {"abc": True}
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "ok:abc"


async def test_async_dispatch_injects_context_into_async_tool():
    @dataclass
    class Deps:
        calls: list = field(default_factory=list)

    @tool
    async def record(ctx: ToolContext[Deps], value: str) -> str:
        """Record a value."""
        ctx.deps.calls.append(value)
        return f"recorded:{value}"

    deps = Deps()
    client = MockAsyncModelClient([])
    client.tools = [record]
    client.tool_context_deps = deps

    await client._handle_tool_calls([{"name": "record", "arguments": {"value": "x"}}])

    assert deps.calls == ["x"]
    assert record.__tool_injected__ == ["ctx"]
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "recorded:x"


async def test_async_dispatch_deps_none_when_unset():
    captured = {}

    @tool
    async def peek(ctx: ToolContext) -> str:
        """Peek at deps."""
        captured["deps"] = ctx.deps
        return "ok"

    client = MockAsyncModelClient([])
    client.tools = [peek]
    await client._handle_tool_calls([{"name": "peek", "arguments": {}}])

    assert captured["deps"] is None


# ---------------------------------------------------------------------------
# Async Agent: deps field / run(deps=) override published; schema single-pass
# ---------------------------------------------------------------------------


async def test_async_agent_publishes_deps_field_to_client():
    @dataclass
    class Deps:
        v: int = 1

    client = MockAsyncModelClient(["done"])
    deps = Deps()
    agent = Agent(client, deps=deps)
    await agent.run("task")
    assert client.tool_context_deps is deps


async def test_async_agent_run_deps_overrides_field():
    client = MockAsyncModelClient(["done"])
    agent = Agent(client, deps="from-field")
    await agent.run("task", deps="from-run")
    assert client.tool_context_deps == "from-run"


async def test_async_agent_run_schema_returns_typed_object():
    @dataclass
    class Verdict:
        passed: bool
        feedback: str = ""

    client = MockAsyncModelClient(['{"passed": true, "feedback": "good"}'])
    client.model.supports_structured_output = False
    agent = Agent(client)

    verdict = await agent.run("judge this", schema=Verdict)
    assert isinstance(verdict, Verdict)
    assert verdict.passed is True
    assert verdict.feedback == "good"


async def test_async_agent_run_schema_rejects_stream():
    @dataclass
    class Verdict:
        passed: bool

    client = MockAsyncModelClient(["x"])
    agent = Agent(client)
    with pytest.raises(ValueError, match="mutually exclusive"):
        await agent.run("t", schema=Verdict, stream=True)
