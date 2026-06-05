"""Async Agent tests: async @tool dispatch, mixed sync/async tools, concurrent_tool_calls."""

from __future__ import annotations

import asyncio
import time

import pytest

from aimu.aio import Agent
from aimu.tools import tool
from helpers_aio import MockAsyncModelClient


async def test_agent_returns_response():
    agent = Agent(MockAsyncModelClient(["hi"]), reset_messages_on_run=True, name="a")
    result = await agent.run("greet")
    assert result == "hi"


async def test_agent_messages_after_run():
    agent = Agent(MockAsyncModelClient(["hi"]), reset_messages_on_run=True, name="a")
    await agent.run("greet")
    assert agent.messages == {"a": agent.model_client.messages}


async def test_async_tool_decorator_sets_flag():
    @tool
    async def echo_async(text: str) -> str:
        """Async echo."""
        return text.upper()

    assert echo_async.__tool_is_async__ is True
    assert "echo_async" == echo_async.__tool_spec__["function"]["name"]


async def test_sync_tool_decorator_sets_flag_false():
    @tool
    def echo_sync(text: str) -> str:
        """Sync echo."""
        return text.upper()

    assert echo_sync.__tool_is_async__ is False


async def test_sync_basemodelclient_rejects_async_tool():
    """The sync surface should reject async @tool functions cleanly."""
    from helpers import MockModelClient

    @tool
    async def async_tool(x: str) -> str:
        """Async tool."""
        return x

    client = MockModelClient(["tool", "done"])
    client.tools = [async_tool]
    # The sync _handle_tool_calls path checks __tool_is_async__ and raises.
    # MockModelClient.chat appends a tool message itself, so we directly call
    # _handle_tool_calls to verify the gate.
    with pytest.raises(ValueError, match="async function"):
        client._handle_tool_calls(
            [{"name": "async_tool", "arguments": {"x": "hi"}}],
            [async_tool.__tool_spec__],
        )


async def test_async_handle_tool_calls_awaits_async_tool():
    """The async surface should await async tools."""

    @tool
    async def doubler(n: str) -> str:
        """Async doubler."""
        return str(int(n) * 2)

    client = MockAsyncModelClient(["unused"])
    client.tools = [doubler]
    # Drive the dispatch directly so we don't need to mock a tool-calling model response.
    await client._handle_tool_calls(
        [{"name": "doubler", "arguments": {"n": "5"}}],
        [doubler.__tool_spec__],
    )
    # After dispatch: assistant tool_calls msg, then tool result msg.
    tool_msg = client.messages[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "10"


async def test_concurrent_tool_calls_overlap():
    """concurrent_tool_calls=True under async should overlap via TaskGroup."""

    @tool
    async def slow_tool(n: str) -> str:
        """Sleeps half a second."""
        await asyncio.sleep(0.5)
        return f"done-{n}"

    client = MockAsyncModelClient(["unused"])
    client.tools = [slow_tool]
    client.concurrent_tool_calls = True

    t0 = time.perf_counter()
    await client._handle_tool_calls(
        [
            {"name": "slow_tool", "arguments": {"n": "a"}},
            {"name": "slow_tool", "arguments": {"n": "b"}},
        ],
        [slow_tool.__tool_spec__],
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.9, f"expected concurrent (<0.9s), got {elapsed:.2f}s"
    tool_msgs = [m for m in client.messages if m.get("role") == "tool"]
    assert {m["content"] for m in tool_msgs} == {"done-a", "done-b"}


# ---------------------------------------------------------------------------
# Per-run tools= override on async Agent.run()
# ---------------------------------------------------------------------------


class _AsyncToolsRecordingClient(MockAsyncModelClient):
    """Records ``self.tools`` seen on each ``_chat`` call during an async run."""

    def __init__(self, responses):
        super().__init__(responses)
        self.tools_per_call = []

    async def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
        self.tools_per_call.append(list(self.tools))
        return await super()._chat(user_message, generate_kwargs, use_tools, stream, images=images)


async def test_async_agent_run_tools_override_applied_each_loop_call():
    base_tool, override_tool = object(), object()
    client = _AsyncToolsRecordingClient(["tool", "after", "done"])
    agent = Agent(client, name="t", tools=[base_tool])

    await agent.run("task", tools=[override_tool])

    assert client.tools_per_call == [[override_tool], [override_tool]]
    assert client.tools == [base_tool]  # restored to configured tools


async def test_async_agent_run_tools_none_uses_configured():
    base_tool = object()
    client = _AsyncToolsRecordingClient(["done"])
    agent = Agent(client, name="t", tools=[base_tool])

    await agent.run("task")

    assert client.tools_per_call == [[base_tool]]


async def test_async_agent_run_streamed_tools_override_applied():
    override_tool = object()
    client = _AsyncToolsRecordingClient(["tool", "after", "done"])
    agent = Agent(client, name="t", tools=[object()])

    stream = await agent.run("task", stream=True, tools=[override_tool])
    async for _ in stream:
        pass

    assert client.tools_per_call and all(seen == [override_tool] for seen in client.tools_per_call)
    assert client.tools[0] is not override_tool  # restored to configured tools
