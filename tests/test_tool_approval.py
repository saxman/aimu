"""Tool-approval hook: sync surface (mock-only).

The gate runs right before each tool invocation; default approves everything, deny appends a
refusal tool message and skips the call. Dispatch is driven directly (``_handle_tool_calls`` /
``_handle_tool_calls_streamed``) so no tool-calling model response is needed.
"""

from __future__ import annotations

import pytest

from aimu.agents import Agent
from aimu.models import StreamChunk, StreamingContentType
from aimu.tools import approve_all, tool
from helpers import MockModelClient


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def _deny_all(name, arguments):
    return False


def test_default_approves_no_behavior_change():
    client = MockModelClient([])
    client.tools = [add]
    client._handle_tool_calls([{"name": "add", "arguments": {"a": 2, "b": 3}}])
    assert client.messages[-1] == {
        "role": "tool",
        "name": "add",
        "content": "5",
        "tool_call_id": client.messages[-1]["tool_call_id"],
    }


def test_deny_skips_invocation_and_appends_refusal():
    ran = []

    @tool
    def danger(x: int) -> str:
        """Risky."""
        ran.append(x)
        return "ran"

    client = MockModelClient([])
    client.tools = [danger]
    client.tool_approval = _deny_all
    client._handle_tool_calls([{"name": "danger", "arguments": {"x": 1}}])

    msg = client.messages[-1]
    assert msg["role"] == "tool" and msg["content"] == "Tool 'danger' was not approved."
    assert ran == []  # the tool body never ran


def test_approve_runs_and_policy_sees_name_and_args():
    seen = {}

    def policy(name, arguments):
        seen["name"] = name
        seen["args"] = dict(arguments)
        return True

    client = MockModelClient([])
    client.tools = [add]
    client.tool_approval = policy
    client._handle_tool_calls([{"name": "add", "arguments": {"a": 1, "b": 1}}])

    assert seen == {"name": "add", "args": {"a": 1, "b": 1}}
    assert client.messages[-1]["content"] == "2"


def test_concurrent_deny():
    client = MockModelClient([])
    client.tools = [add]
    client.concurrent_tool_calls = True
    client.tool_approval = _deny_all
    client._handle_tool_calls(
        [{"name": "add", "arguments": {"a": 1, "b": 1}}, {"name": "add", "arguments": {"a": 2, "b": 2}}]
    )
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert all(m["content"] == "Tool 'add' was not approved." for m in tool_msgs)


def test_streaming_deny():
    @tool
    def streamer(x: int):
        """A streaming tool."""
        yield StreamChunk(StreamingContentType.GENERATING, "chunk")
        return "done"

    client = MockModelClient([])
    client.tools = [streamer]
    client.tool_approval = _deny_all
    chunks = list(client._handle_tool_calls_streamed([{"name": "streamer", "arguments": {"x": 1}}]))

    # The tool was gated, so it yields no GENERATING chunk; only the TOOL_CALLING refusal.
    assert all(ch.phase != StreamingContentType.GENERATING for ch in chunks)
    assert client.messages[-1]["content"] == "Tool 'streamer' was not approved."
    tool_chunks = [ch for ch in chunks if ch.phase == StreamingContentType.TOOL_CALLING]
    assert "was not approved" in tool_chunks[-1].content["response"]


def test_sync_coroutine_policy_raises():
    async def acoro(name, arguments):
        return True

    client = MockModelClient([])
    client.tools = [add]
    client.tool_approval = acoro
    with pytest.raises(ValueError, match="coroutine"):
        client._handle_tool_calls([{"name": "add", "arguments": {"a": 1, "b": 1}}])


def test_agent_publishes_policy_and_per_run_override():
    def policy(name, arguments):
        return True

    client = MockModelClient([])
    agent = Agent(client, tools=[add], tool_approval=policy)
    agent._prepare_run()
    assert client.tool_approval is policy

    other = lambda name, arguments: False  # noqa: E731
    agent._prepare_run(tool_approval=other)
    assert client.tool_approval is other

    # An agent with no policy publishes approve_all (never leaves a stale policy on the client).
    agent_none = Agent(client, tools=[add])
    agent_none._prepare_run()
    assert client.tool_approval is approve_all
