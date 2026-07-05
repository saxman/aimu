"""Tool-approval hook: sync surface (mock-only).

The gate runs right before each tool invocation; default approves everything, deny appends a
refusal tool message and skips the call. Tool execution now lives in the tool-loop engine
(``aimu.agents._tool_loop._ToolLoop``), which owns approval, so dispatch is driven directly
through the engine (``_dispatch`` / ``_dispatch_streamed``) after staging the assistant
tool-call message a provider would have stored.
"""

from __future__ import annotations

import pytest

from aimu.agents import Agent
from aimu.agents._tool_loop import _ToolLoop
from aimu.models import StreamChunk, StreamingContentType
from aimu.tools import tool
from helpers import MockModelClient


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


def test_default_approves_no_behavior_change():
    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 2, "b": 3}}])
    _ToolLoop(client, [add])._dispatch()
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
    _stage_tool_calls(client, [{"name": "danger", "arguments": {"x": 1}}])
    _ToolLoop(client, [danger], tool_approval=_deny_all)._dispatch()

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
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])
    _ToolLoop(client, [add], tool_approval=policy)._dispatch()

    assert seen == {"name": "add", "args": {"a": 1, "b": 1}}
    assert client.messages[-1]["content"] == "2"


def test_concurrent_deny():
    client = MockModelClient([])
    _stage_tool_calls(
        client,
        [{"name": "add", "arguments": {"a": 1, "b": 1}}, {"name": "add", "arguments": {"a": 2, "b": 2}}],
    )
    _ToolLoop(client, [add], concurrent_tool_calls=True, tool_approval=_deny_all)._dispatch()
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
    _stage_tool_calls(client, [{"name": "streamer", "arguments": {"x": 1}}])
    chunks = list(_ToolLoop(client, [streamer], tool_approval=_deny_all)._dispatch_streamed(0))

    # The tool was gated, so it yields no GENERATING chunk; only the TOOL_CALLING refusal.
    assert all(ch.phase != StreamingContentType.GENERATING for ch in chunks)
    assert client.messages[-1]["content"] == "Tool 'streamer' was not approved."
    tool_chunks = [ch for ch in chunks if ch.phase == StreamingContentType.TOOL_CALLING]
    assert "was not approved" in tool_chunks[-1].content["response"]


def test_sync_coroutine_policy_raises():
    async def acoro(name, arguments):
        return True

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])
    with pytest.raises(ValueError, match="coroutine"):
        _ToolLoop(client, [add], tool_approval=acoro)._dispatch()


def test_agent_tool_approval_field_denies_and_per_run_override_approves():
    """Behavioral: the Agent's ``tool_approval`` field gates tool execution during a run, and a
    per-run ``run(tool_approval=)`` override wins over the field. (Replaces the old test that
    asserted the policy was published onto the client, which no longer holds it.)"""
    ran = []

    @tool
    def danger() -> str:
        """Risky."""
        ran.append(1)
        return "ran"

    # Field policy denies -> the tool round parses a call, the engine skips execution.
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[danger], tool_approval=_deny_all)
    assert agent.run("go") == "done"
    assert ran == []
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert tool_msgs[-1]["content"] == "Tool 'danger' was not approved."

    # Per-run approval override -> the tool body runs.
    ran.clear()
    client2 = MockModelClient(["tool", "done"])
    agent2 = Agent(client2, tools=[danger], tool_approval=_deny_all)
    assert agent2.run("go", tool_approval=lambda name, arguments: True) == "done"
    assert ran == [1]
    tool_msgs2 = [m for m in client2.messages if m["role"] == "tool"]
    assert tool_msgs2[-1]["content"] == "ran"
