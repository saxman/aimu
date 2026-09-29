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
from aimu.tools import Denied, tool
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
    tool_msg = {key: value for key, value in client.messages[-1].items() if key != "timestamp"}
    assert tool_msg == {
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


def test_denied_reason_reaches_the_model():
    """A refusal the model can act on: without the reason its next move is the same call again."""
    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])

    def policy(name, arguments):
        return Denied("only api.example.com is allowed")

    _ToolLoop(client, [add], tool_approval=policy)._dispatch()
    assert client.messages[-1]["content"] == "Tool 'add' was not approved: only api.example.com is allowed"


def test_denied_skips_invocation_like_false():
    ran = []

    @tool
    def danger(x: int) -> str:
        """Risky."""
        ran.append(x)
        return "ran"

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "danger", "arguments": {"x": 1}}])
    _ToolLoop(client, [danger], tool_approval=lambda n, a: Denied("nope"))._dispatch()
    assert ran == []


def test_a_truthy_string_still_approves():
    """Pins the compatibility decision behind Denied: a policy returning a non-empty string
    approves today (bool("x") is True), so a bare string could not be repurposed as a refusal
    without silently inverting such a policy. Denied is a type that cannot collide.
    """
    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])
    _ToolLoop(client, [add], tool_approval=lambda n, a: "sure")._dispatch()
    assert client.messages[-1]["content"] == "2"


def test_denied_with_an_empty_reason_falls_back_to_the_plain_refusal():
    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])
    _ToolLoop(client, [add], tool_approval=lambda n, a: Denied(""))._dispatch()
    assert client.messages[-1]["content"] == "Tool 'add' was not approved."


def test_denied_reason_is_reported_on_the_event():
    from aimu.events import ToolDenied

    events = []
    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "add", "arguments": {"a": 1, "b": 1}}])
    _ToolLoop(client, [add], tool_approval=lambda n, a: Denied("host not allowed"), events=events.append)._dispatch()

    denied = [e for e in events if isinstance(e, ToolDenied)]
    assert len(denied) == 1
    assert denied[0].reason == "host not allowed"


def test_denied_reason_reaches_a_streamed_dispatch():
    @tool
    def streamer(x: int):
        """A streaming tool."""
        yield StreamChunk(StreamingContentType.GENERATING, "chunk")
        return "done"

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "streamer", "arguments": {"x": 1}}])
    chunks = list(_ToolLoop(client, [streamer], tool_approval=lambda n, a: Denied("why not"))._dispatch_streamed(0))

    assert client.messages[-1]["content"] == "Tool 'streamer' was not approved: why not"
    tool_chunks = [ch for ch in chunks if ch.phase == StreamingContentType.TOOL_CALLING]
    assert "why not" in tool_chunks[-1].content["response"]


def test_denied_reason_reaches_a_concurrent_dispatch():
    client = MockModelClient([])
    _stage_tool_calls(
        client,
        [{"name": "add", "arguments": {"a": 1, "b": 1}}, {"name": "add", "arguments": {"a": 2, "b": 2}}],
    )
    policy = lambda name, arguments: Denied(f"no calls with a={arguments['a']}")  # noqa: E731
    _ToolLoop(client, [add], concurrent_tool_calls=True, tool_approval=policy)._dispatch()

    contents = [m["content"] for m in client.messages if m["role"] == "tool"]
    assert sorted(contents) == [
        "Tool 'add' was not approved: no calls with a=1",
        "Tool 'add' was not approved: no calls with a=2",
    ]


def test_denied_is_exported_from_the_package_roots():
    import aimu
    import aimu.tools

    assert aimu.Denied is Denied
    assert aimu.tools.Denied is Denied


def test_denied_is_falsy():
    """The contract a host relies on when composing verdicts: Denied substitutes for False."""
    assert not Denied("because")
    assert not Denied("")
    assert bool(Denied("because")) is False


def test_denied_is_frozen():
    """A policy's verdict must not be mutable after the engine has it (same as Unsupported)."""
    with pytest.raises(Exception):
        Denied("x").reason = "y"


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


def test_subagent_factory_deny_gate_prevents_child_tool_execution(monkeypatch):
    """Behavioral: tool_approval passed to make_subagent_tool actually gates child tool calls.

    A real Agent is built by the factory (not _RecordingAgent); a MockModelClient drives the
    child's scripted turn so the child requests the gated tool. The approval gate denies it,
    so the side-effect list stays empty and the transcript carries the refusal message.
    """
    ran = []

    @tool
    def danger() -> str:
        """Risky child tool."""
        ran.append(1)
        return "ran"

    from aimu.tools.builtin import make_subagent_tool

    # Patch ModelClient so the child agent gets a scripted mock instead of making network calls.
    monkeypatch.setattr(
        "aimu.models.model_client.ModelClient",
        lambda model: MockModelClient(["tool", "done"]),
    )

    spawn = make_subagent_tool("anthropic:claude-sonnet-4-6", tools=[danger], tool_approval=_deny_all)
    result = spawn("do the risky thing")

    # The child ran and returned "done" (the second scripted response after the denied tool turn).
    assert result == "done"
    # The danger tool body never ran.
    assert ran == []


def test_denied_reason_survives_a_whole_agent_run():
    """End to end through Agent.run, with the host-allowlist policy the how-to documents."""
    from urllib.parse import urlparse

    calls = []

    @tool
    def submit_json(url: str, payload: dict) -> str:
        """Send JSON to a URL."""
        calls.append(url)
        return "ok"

    def allowlist_hosts(name, arguments):
        host = urlparse(arguments.get("url", "")).hostname or ""
        if host == "api.example.com":
            return True
        return Denied(f"{host!r} is not allowed; permitted hosts are ['api.example.com']")

    disallowed = {"tool": "submit_json", "arguments": {"url": "https://evil.example/x", "payload": {}}}
    client = MockModelClient([disallowed, "done"])
    agent = Agent(client, tools=[submit_json], tool_approval=allowlist_hosts)
    assert agent.run("post it") == "done"

    assert calls == []  # the tool never ran
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert tool_msgs[-1]["content"] == (
        "Tool 'submit_json' was not approved: 'evil.example' is not allowed; permitted hosts are ['api.example.com']"
    )

    # The same policy allows the listed host, so the reason path is not just denying everything.
    allowed = {"tool": "submit_json", "arguments": {"url": "https://api.example.com/x", "payload": {}}}
    client2 = MockModelClient([allowed, "done"])
    Agent(client2, tools=[submit_json], tool_approval=allowlist_hosts).run("post it")
    assert calls == ["https://api.example.com/x"]


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
