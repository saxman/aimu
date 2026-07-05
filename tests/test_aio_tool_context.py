"""Async ToolContext dependency injection + async Agent deps/schema support.

Mirrors tests/test_tool_context.py on the aio surface. Deterministic via MockAsyncModelClient.
"""

from dataclasses import dataclass, field


from aimu.aio import Agent
from aimu.aio._tool_loop import _AsyncToolLoop
from aimu.tools import ToolContext, tool
from helpers_aio import MockAsyncModelClient


def _stage_tool_calls(client, calls):
    """Append the assistant(tool_calls) message a provider stores, so the engine can dispatch it.

    ``calls`` is a list of ``{"name": ..., "arguments": ...}`` dicts.
    """
    client.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c.get("arguments", {})},
                    "id": f"id{i}",
                }
                for i, c in enumerate(calls)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Dispatch: the async engine fills ctx.deps for sync and async tools
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
    _stage_tool_calls(client, [{"name": "remember", "arguments": {"key": "abc"}}])
    await _AsyncToolLoop(client, [remember], deps=deps)._dispatch()

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
    _stage_tool_calls(client, [{"name": "record", "arguments": {"value": "x"}}])
    await _AsyncToolLoop(client, [record], deps=deps)._dispatch()

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
    _stage_tool_calls(client, [{"name": "peek", "arguments": {}}])
    await _AsyncToolLoop(client, [peek])._dispatch()

    assert captured["deps"] is None


# ---------------------------------------------------------------------------
# Async Agent: deps field / run(deps=) override reach the tool as ctx.deps
# ---------------------------------------------------------------------------


async def test_async_agent_deps_field_reaches_tool_via_context():
    """Behavioral: the async Agent's ``deps`` field is injected as ``ctx.deps`` when a tool runs.
    (Replaces the old test asserting deps were published onto the client.)"""

    @dataclass
    class Deps:
        v: int = 1

    seen = {}

    @tool
    async def capture(ctx: ToolContext) -> str:
        """Capture the injected deps."""
        seen["deps"] = ctx.deps
        return "ok"

    client = MockAsyncModelClient(["tool", "done"])
    deps = Deps()
    agent = Agent(client, tools=[capture], deps=deps)
    assert await agent.run("task") == "done"
    assert seen["deps"] is deps


async def test_async_agent_run_deps_overrides_field():
    """Behavioral: a per-run ``run(deps=)`` override wins over the ``deps`` field at the tool."""
    seen = {}

    @tool
    async def capture(ctx: ToolContext) -> str:
        """Capture the injected deps."""
        seen["deps"] = ctx.deps
        return "ok"

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[capture], deps="from-field")
    await agent.run("task", deps="from-run")
    assert seen["deps"] == "from-run"


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


async def test_async_agent_run_schema_streams():
    @dataclass
    class Verdict:
        passed: bool

    client = MockAsyncModelClient(['{"passed": true}'])
    client.model.supports_structured_output = False
    agent = Agent(client)
    chunks = [c async for c in await agent.run("t", schema=Verdict, stream=True)]
    assert chunks[-1].is_done()
    assert chunks[-1].content["result"] == Verdict(passed=True)
