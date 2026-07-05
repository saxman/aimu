"""Tool-approval hook: async surface (mock-only).

Mirrors tests/test_tool_approval.py for ``aimu.aio``: the gate also accepts a coroutine policy
(awaited by the async dispatcher). Tool execution lives in the async tool-loop engine
(``aimu.aio._tool_loop._AsyncToolLoop``), which owns approval, so dispatch is driven directly
through the engine after staging the assistant tool-call message a provider would have stored.
"""

from __future__ import annotations

from aimu.aio import Agent
from aimu.aio._tool_loop import _AsyncToolLoop
from aimu.models import StreamChunk, StreamingContentType
from aimu.tools import tool
from helpers_aio import MockAsyncModelClient


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def _deny_all(name, arguments):
    return False


def _stage_tool_calls(client, calls):
    """Append the assistant(tool_calls) message a provider stores, so the engine can dispatch it.

    ``calls`` is a list of ``{"name": ..., "arguments": ...}`` dicts (the old ``_handle_tool_calls``
    input shape).
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


async def test_default_approves_no_behavior_change():
    client = MockAsyncModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 2, "b": 3}}])
    await _AsyncToolLoop(client, [add])._dispatch()
    assert client.messages[-1]["role"] == "tool" and client.messages[-1]["content"] == "5"


async def test_deny_skips_invocation_and_appends_refusal():
    ran = []

    @tool
    def danger(x: int) -> str:
        """Risky."""
        ran.append(x)
        return "ran"

    client = MockAsyncModelClient([])
    _stage_tool_calls(client, [{"name": "danger", "arguments": {"x": 1}}])
    await _AsyncToolLoop(client, [danger], tool_approval=_deny_all)._dispatch()

    assert client.messages[-1]["content"] == "Tool 'danger' was not approved."
    assert ran == []


async def test_async_coroutine_policy_is_awaited():
    calls = []

    async def policy(name, arguments):
        calls.append(name)
        return name != "add"  # deny add

    client = MockAsyncModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])
    await _AsyncToolLoop(client, [add], tool_approval=policy)._dispatch()

    assert calls == ["add"]
    assert client.messages[-1]["content"] == "Tool 'add' was not approved."


async def test_concurrent_deny():
    client = MockAsyncModelClient([])
    _stage_tool_calls(
        client,
        [{"name": "add", "arguments": {"a": 1, "b": 1}}, {"name": "add", "arguments": {"a": 2, "b": 2}}],
    )
    await _AsyncToolLoop(client, [add], concurrent_tool_calls=True, tool_approval=_deny_all)._dispatch()
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert all(m["content"] == "Tool 'add' was not approved." for m in tool_msgs)


async def test_streaming_deny():
    @tool
    async def streamer(x: int):
        """An async streaming tool."""
        yield StreamChunk(StreamingContentType.GENERATING, "chunk")

    client = MockAsyncModelClient([])
    _stage_tool_calls(client, [{"name": "streamer", "arguments": {"x": 1}}])
    chunks = [ch async for ch in _AsyncToolLoop(client, [streamer], tool_approval=_deny_all)._dispatch_streamed(0)]

    assert all(ch.phase != StreamingContentType.GENERATING for ch in chunks)
    assert client.messages[-1]["content"] == "Tool 'streamer' was not approved."


async def test_agent_tool_approval_field_denies_and_per_run_override_approves():
    """Behavioral: the async Agent's ``tool_approval`` field gates tool execution during a run,
    and a per-run ``run(tool_approval=)`` override wins over the field. (Replaces the old test that
    asserted the policy was published onto the client, which no longer holds it.)"""
    ran = []

    @tool
    def danger() -> str:
        """Risky."""
        ran.append(1)
        return "ran"

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[danger], tool_approval=_deny_all)
    assert await agent.run("go") == "done"
    assert ran == []
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert tool_msgs[-1]["content"] == "Tool 'danger' was not approved."

    ran.clear()
    client2 = MockAsyncModelClient(["tool", "done"])
    agent2 = Agent(client2, tools=[danger], tool_approval=_deny_all)
    assert await agent2.run("go", tool_approval=lambda name, arguments: True) == "done"
    assert ran == [1]
    tool_msgs2 = [m for m in client2.messages if m["role"] == "tool"]
    assert tool_msgs2[-1]["content"] == "ran"
