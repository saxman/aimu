"""Tool-approval hook: async surface (mock-only).

Mirrors tests/test_tool_approval.py for ``aimu.aio``: the gate also accepts a coroutine policy
(awaited by the async dispatcher).
"""

from __future__ import annotations

from aimu.aio import Agent
from aimu.models import StreamChunk, StreamingContentType
from aimu.tools import approve_all, tool
from helpers_aio import MockAsyncModelClient


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def _deny_all(name, arguments):
    return False


async def test_default_approves_no_behavior_change():
    client = MockAsyncModelClient([])
    client.tools = [add]
    await client._handle_tool_calls([{"name": "add", "arguments": {"a": 2, "b": 3}}])
    assert client.messages[-1]["role"] == "tool" and client.messages[-1]["content"] == "5"


async def test_deny_skips_invocation_and_appends_refusal():
    ran = []

    @tool
    def danger(x: int) -> str:
        """Risky."""
        ran.append(x)
        return "ran"

    client = MockAsyncModelClient([])
    client.tools = [danger]
    client.tool_approval = _deny_all
    await client._handle_tool_calls([{"name": "danger", "arguments": {"x": 1}}])

    assert client.messages[-1]["content"] == "Tool 'danger' was not approved."
    assert ran == []


async def test_async_coroutine_policy_is_awaited():
    calls = []

    async def policy(name, arguments):
        calls.append(name)
        return name != "add"  # deny add

    client = MockAsyncModelClient([])
    client.tools = [add]
    client.tool_approval = policy
    await client._handle_tool_calls([{"name": "add", "arguments": {"a": 1, "b": 1}}])

    assert calls == ["add"]
    assert client.messages[-1]["content"] == "Tool 'add' was not approved."


async def test_concurrent_deny():
    client = MockAsyncModelClient([])
    client.tools = [add]
    client.concurrent_tool_calls = True
    client.tool_approval = _deny_all
    await client._handle_tool_calls(
        [{"name": "add", "arguments": {"a": 1, "b": 1}}, {"name": "add", "arguments": {"a": 2, "b": 2}}]
    )
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert all(m["content"] == "Tool 'add' was not approved." for m in tool_msgs)


async def test_streaming_deny():
    @tool
    async def streamer(x: int):
        """An async streaming tool."""
        yield StreamChunk(StreamingContentType.GENERATING, "chunk")

    client = MockAsyncModelClient([])
    client.tools = [streamer]
    client.tool_approval = _deny_all
    chunks = [ch async for ch in client._handle_tool_calls_streamed([{"name": "streamer", "arguments": {"x": 1}}])]

    assert all(ch.phase != StreamingContentType.GENERATING for ch in chunks)
    assert client.messages[-1]["content"] == "Tool 'streamer' was not approved."


async def test_agent_publishes_policy_and_per_run_override():
    async def policy(name, arguments):
        return True

    client = MockAsyncModelClient([])
    agent = Agent(client, tools=[add], tool_approval=policy)
    agent._prepare_run()
    assert client.tool_approval is policy

    other = _deny_all
    agent._prepare_run(tool_approval=other)
    assert client.tool_approval is other

    agent_none = Agent(client, tools=[add])
    agent_none._prepare_run()
    assert client.tool_approval is approve_all
