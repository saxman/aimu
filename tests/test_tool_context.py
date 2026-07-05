"""Tests for ToolContext dependency injection and Agent deps/schema support.

All deterministic via MockModelClient; no backend needed.
"""

from dataclasses import dataclass, field


from aimu.agents import Agent
from aimu.agents._tool_loop import _ToolLoop
from aimu.tools import ToolContext, tool
from helpers import MockModelClient


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
# Decorator: injected params are stripped from the model-facing schema
# ---------------------------------------------------------------------------


def test_tool_context_excluded_from_spec():
    @tool
    def remember(ctx: ToolContext, key: str) -> str:
        """Remember a value."""
        return key

    props = remember.__tool_spec__["function"]["parameters"]["properties"]
    assert "ctx" not in props
    assert "key" in props
    assert remember.__tool_spec__["function"]["parameters"]["required"] == ["key"]
    assert remember.__tool_injected__ == ["ctx"]


def test_tool_context_subscripted_excluded_from_spec():
    @dataclass
    class Deps:
        x: int = 0

    @tool
    def t(ctx: ToolContext[Deps], query: str) -> str:
        """Search."""
        return query

    props = t.__tool_spec__["function"]["parameters"]["properties"]
    assert "ctx" not in props
    assert "query" in props
    assert t.__tool_injected__ == ["ctx"]


def test_tool_without_context_has_no_injected():
    @tool
    def plain(x: str) -> str:
        """Plain."""
        return x

    assert plain.__tool_injected__ == []


# ---------------------------------------------------------------------------
# Dispatch: the agent fills ctx.deps at call time
# ---------------------------------------------------------------------------


def test_tool_context_injected_at_dispatch():
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
    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "remember", "arguments": {"key": "abc"}}])
    _ToolLoop(client, [remember], deps=deps)._dispatch()

    assert captured["deps"] is deps
    assert deps.seen == {"abc": True}
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "ok:abc"


def test_tool_context_deps_none_when_unset():
    captured = {}

    @tool
    def peek(ctx: ToolContext) -> str:
        """Peek at deps."""
        captured["deps"] = ctx.deps
        return "ok"

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "peek", "arguments": {}}])
    _ToolLoop(client, [peek])._dispatch()

    assert captured["deps"] is None


# ---------------------------------------------------------------------------
# Agent: deps field + per-run override reach the tool as ctx.deps
# ---------------------------------------------------------------------------


def test_agent_deps_field_reaches_tool_via_context():
    """Behavioral: the Agent's ``deps`` field is injected as ``ctx.deps`` when a tool runs during
    a run. (Replaces the old test asserting deps were published onto the client, which no longer
    holds them.)"""

    @dataclass
    class Deps:
        v: int = 1

    seen = {}

    @tool
    def capture(ctx: ToolContext) -> str:
        """Capture the injected deps."""
        seen["deps"] = ctx.deps
        return "ok"

    client = MockModelClient(["tool", "done"])
    deps = Deps()
    agent = Agent(client, tools=[capture], deps=deps)
    assert agent.run("task") == "done"
    assert seen["deps"] is deps


def test_agent_run_deps_overrides_field():
    """Behavioral: a per-run ``run(deps=)`` override wins over the ``deps`` field at the tool."""
    seen = {}

    @tool
    def capture(ctx: ToolContext) -> str:
        """Capture the injected deps."""
        seen["deps"] = ctx.deps
        return "ok"

    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[capture], deps="from-field")
    agent.run("task", deps="from-run")
    assert seen["deps"] == "from-run"


# ---------------------------------------------------------------------------
# Agent: single-pass structured output via schema=
# ---------------------------------------------------------------------------


def test_agent_run_schema_returns_typed_object():
    @dataclass
    class Verdict:
        passed: bool
        feedback: str = ""

    client = MockModelClient(['{"passed": true, "feedback": "good"}'])
    client.model.supports_structured_output = False  # exercise the parse-path
    agent = Agent(client)

    verdict = agent.run("judge this", schema=Verdict)
    assert isinstance(verdict, Verdict)
    assert verdict.passed is True
    assert verdict.feedback == "good"


def test_agent_run_schema_streams():
    @dataclass
    class Verdict:
        passed: bool

    client = MockModelClient(['{"passed": true}'])
    client.model.supports_structured_output = False  # parse-path
    agent = Agent(client)
    chunks = list(agent.run("t", schema=Verdict, stream=True))
    assert chunks[-1].is_done()
    assert chunks[-1].content["result"] == Verdict(passed=True)
