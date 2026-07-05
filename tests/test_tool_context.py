"""Tests for ToolContext dependency injection and Agent deps/schema support.

All deterministic via MockModelClient; no backend needed.
"""

from dataclasses import dataclass, field


from aimu.agents import Agent
from aimu.tools import ToolContext, tool
from helpers import MockModelClient


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
    client.tools = [remember]
    client.tool_context_deps = deps

    client._handle_tool_calls([{"name": "remember", "arguments": {"key": "abc"}}])

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
    client.tools = [peek]
    client._handle_tool_calls([{"name": "peek", "arguments": {}}])

    assert captured["deps"] is None


# ---------------------------------------------------------------------------
# Agent: deps field + per-run override are published to the model client
# ---------------------------------------------------------------------------


def test_agent_publishes_deps_field_to_client():
    @dataclass
    class Deps:
        v: int = 1

    client = MockModelClient(["done"])
    deps = Deps()
    agent = Agent(client, deps=deps)
    agent.run("task")
    assert client.tool_context_deps is deps


def test_agent_run_deps_overrides_field():
    client = MockModelClient(["done"])
    agent = Agent(client, deps="from-field")
    agent.run("task", deps="from-run")
    assert client.tool_context_deps == "from-run"


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
