"""Async Agent tests: async @tool dispatch, mixed sync/async tools, concurrent_tool_calls."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from aimu.aio import Agent
from aimu.aio._base import AsyncBaseModelClient
from aimu.models import StreamingContentType
from aimu.models._internal.message_meta import (
    PROVENANCE_CONTINUATION,
    PROVENANCE_FINAL_ANSWER,
    PROVENANCE_KEY,
)
from aimu.tools import tool
from aimu.tools.context import ToolContext
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


def _stage_tool_calls(client, calls):
    """Append the assistant(tool_calls) message a provider stores, so the engine can dispatch it.

    ``calls`` is a list of ``(name, arguments)`` tuples.
    """
    client.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": n, "arguments": a}, "id": f"id{i}"}
                for i, (n, a) in enumerate(calls)
            ],
        }
    )


async def test_sync_agent_rejects_async_tool():
    """The sync tool-loop engine should reject async @tool functions cleanly."""
    from aimu.agents._tool_loop import _ToolLoop
    from helpers import MockModelClient

    @tool
    async def async_tool(x: str) -> str:
        """Async tool."""
        return x

    client = MockModelClient([])
    _stage_tool_calls(client, [("async_tool", {"x": "hi"})])
    with pytest.raises(ValueError, match="async function"):
        _ToolLoop(client, [async_tool])._dispatch()


async def test_async_engine_awaits_async_tool():
    """The async tool-loop engine should await async tools."""
    from aimu.aio._tool_loop import _AsyncToolLoop

    @tool
    async def doubler(n: str) -> str:
        """Async doubler."""
        return str(int(n) * 2)

    client = MockAsyncModelClient(["unused"])
    _stage_tool_calls(client, [("doubler", {"n": "5"})])
    await _AsyncToolLoop(client, [doubler])._dispatch()
    tool_msg = client.messages[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "10"


async def test_concurrent_tool_calls_overlap():
    """concurrent_tool_calls=True under async should overlap via TaskGroup."""
    from aimu.aio._tool_loop import _AsyncToolLoop

    @tool
    async def slow_tool(n: str) -> str:
        """Sleeps half a second."""
        await asyncio.sleep(0.5)
        return f"done-{n}"

    client = MockAsyncModelClient(["unused"])
    _stage_tool_calls(client, [("slow_tool", {"n": "a"}), ("slow_tool", {"n": "b"})])

    t0 = time.perf_counter()
    await _AsyncToolLoop(client, [slow_tool], concurrent_tool_calls=True)._dispatch()
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

    async def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self.tools_per_call.append(list(self.tools))
        return await super()._chat(user_message, generate_kwargs, use_tools, stream, images=images)


@tool
def _base_tool() -> str:
    """A base tool."""
    return "base"


@tool
def _override_tool() -> str:
    """An override tool."""
    return "override"


async def test_async_agent_run_tools_override_applied_each_loop_call():
    client = _AsyncToolsRecordingClient(["after", "done"])  # no tool round → single chat() call
    agent = Agent(client, name="t", tools=[_base_tool])

    await agent.run("task", tools=[_override_tool])

    assert client.tools_per_call == [[_override_tool]]


async def test_async_agent_run_tools_none_uses_configured():
    client = _AsyncToolsRecordingClient(["done"])
    agent = Agent(client, name="t", tools=[_base_tool])

    await agent.run("task")

    assert client.tools_per_call == [[_base_tool]]


async def test_async_agent_run_streamed_tools_override_applied():
    client = _AsyncToolsRecordingClient(["after", "done"])
    agent = Agent(client, name="t", tools=[_base_tool])

    stream = await agent.run("task", stream=True, tools=[_override_tool])
    async for _ in stream:
        pass

    assert client.tools_per_call and all(seen == [_override_tool] for seen in client.tools_per_call)
    assert client.tools == []  # transient restored after


async def test_async_chat_is_single_turn_parses_tool_call_without_executing():
    # One chat() = one model turn. A tool-using turn parses and stores the tool call but does
    # NOT execute it; the loop and execution live in the Agent.
    client = MockAsyncModelClient(["tool", "answer"])

    first = await client.chat("q")
    assert first == ""
    assert client.messages[-1]["role"] == "assistant"
    assert client.messages[-1]["tool_calls"]
    assert not any(m["role"] == "tool" for m in client.messages)

    second = await client.chat()
    assert second == "answer"
    assert client.messages[-1] == {"role": "assistant", "content": "answer"}
    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["q"]


async def test_async_agent_run_produces_single_final_answer_no_duplication():
    client = MockAsyncModelClient(["tool", "the answer"])
    result = await Agent(client, name="a").run("do it")
    assert result == "the answer"
    assistant_texts = [m.get("content") for m in client.messages if m["role"] == "assistant" and m.get("content")]
    assert assistant_texts == ["the answer"]


# ---------------------------------------------------------------------------
# Async Agent final_answer_prompt (forced wrap-up at max_iterations)
# ---------------------------------------------------------------------------


@tool
def _dummy_async_tool(x: str) -> str:
    """A no-op tool used only to give an agent a non-empty tool list."""
    return x


class _AsyncLoopingToolClient(MockAsyncModelClient):
    """Async mirror of ``test_agents._LoopingToolClient``: calls a tool every turn while
    tools are available, and only produces a final answer once tools are disabled."""

    def __init__(self, final_text: str = "FORCED SUMMARY"):
        super().__init__([])
        self.final_text = final_text
        self.tools_seen: list[list] = []

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if user_message is not None:  # None = continuation turn (no new user message)
            self.messages.append({"role": "user", "content": user_message})
        self.tools_seen.append(list(self.tools))
        # Single turn: with tools available, call a tool and return (no follow-up answer).
        if self.tools:
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "t", "content": "result", "tool_call_id": "x"})
            return ""
        self.messages.append({"role": "assistant", "content": self.final_text})
        return self.final_text


async def test_async_agent_final_answer_prompt_forces_wrap_up_at_cap():
    client = _AsyncLoopingToolClient(final_text="FORCED SUMMARY")
    agent = Agent(
        client,
        name="capper",
        tools=[_dummy_async_tool],
        max_iterations=3,
        final_answer_prompt="Stop using tools and answer now.",
    )
    result = await agent.run("gather forever")

    assert result == "FORCED SUMMARY"
    assert client.tools_seen[-1] == []
    assert all(seen for seen in client.tools_seen[:-1])
    assert sum(1 for seen in client.tools_seen if seen == []) == 1


async def test_async_agent_final_answer_prompt_not_triggered_on_natural_finish():
    client = MockAsyncModelClient(["done"])
    agent = Agent(client, name="natural", final_answer_prompt="WRAP UP NOW")
    result = await agent.run("task")

    assert result == "done"
    assert client._call_count == 1
    assert "WRAP UP NOW" not in [m["content"] for m in client.messages if m["role"] == "user"]


async def test_async_agent_streamed_final_answer_prompt_forces_wrap_up():
    client = _AsyncLoopingToolClient(final_text="STREAMED SUMMARY")
    agent = Agent(
        client,
        name="scap",
        tools=[_dummy_async_tool],
        max_iterations=3,
        final_answer_prompt="Stop using tools and answer now.",
    )
    chunks = []
    stream = await agent.run("gather", stream=True)
    async for c in stream:
        chunks.append(c)

    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING and c.content]
    assert any(c.content == "STREAMED SUMMARY" for c in generating)
    assert client.tools_seen[-1] == []


# ---------------------------------------------------------------------------
# AsyncRunner.as_tool(): any async runner usable as a tool
# ---------------------------------------------------------------------------


async def test_async_runner_as_tool_dispatches_to_run():
    from aimu.aio import Agent as AsyncAgent

    agent = AsyncAgent(MockAsyncModelClient(["agent reply"]), system_message="Research.\nMore.", name="alpha")
    fn = agent.as_tool()
    assert fn.__name__ == "alpha"
    assert fn.__tool_spec__["function"]["description"] == "Research."
    assert fn.__tool_is_async__ is True
    assert await fn("anything") == "agent reply"


async def test_async_runner_as_tool_workflow_generic_description():
    from aimu.aio import Agent as AsyncAgent
    from aimu.aio import Chain as AsyncChain

    chain = AsyncChain(agents=[AsyncAgent(MockAsyncModelClient(["step out"]))], name="my chain")
    fn = chain.as_tool()
    assert fn.__name__ == "my_chain"
    assert fn.__tool_spec__["function"]["description"] == "Delegate a task to the my_chain runner."
    assert await fn("go") == "step out"


# ---------------------------------------------------------------------------
# Async SkillAgent.run() parity with Agent.run(): deps= and schema=
# ---------------------------------------------------------------------------


def _empty_skill_manager(tmp_path):
    """A SkillManager that discovers no skills, so async skill setup is a no-op."""
    from aimu.skills.manager import SkillManager

    return SkillManager(skill_dirs=[str(tmp_path / "no-skills-here")])


async def test_async_skill_agent_run_threads_deps(tmp_path):
    from aimu.aio import SkillAgent

    sentinel = object()
    seen = {}

    @tool
    async def capture(ctx: ToolContext) -> str:
        """Capture the injected deps."""
        seen["deps"] = ctx.deps
        return "ok"

    client = MockAsyncModelClient(["tool", "done"])
    agent = SkillAgent(client, name="s", tools=[capture], skill_manager=_empty_skill_manager(tmp_path))

    await agent.run("greet", deps=sentinel)

    assert seen["deps"] is sentinel


async def test_async_skill_agent_run_schema_returns_typed_instance(tmp_path):
    from dataclasses import dataclass

    from aimu.aio import SkillAgent

    @dataclass
    class Verdict:
        passed: bool
        feedback: str

    client = MockAsyncModelClient(['{"passed": true, "feedback": "ok"}'])
    client.model.supports_structured_output = False  # force parse-path
    agent = SkillAgent(client, name="s", skill_manager=_empty_skill_manager(tmp_path))

    result = await agent.run("judge", schema=Verdict)

    assert isinstance(result, Verdict)
    assert result.passed is True
    assert result.feedback == "ok"


async def test_async_agent_run_schema_streams():
    from dataclasses import dataclass

    from aimu.aio import Agent

    @dataclass
    class Out:
        x: int

    client = MockAsyncModelClient(['{"x": 9}'])
    client.model.supports_structured_output = False  # parse-path
    agent = Agent(client, name="a")

    chunks = [c async for c in await agent.run("t", stream=True, schema=Out)]
    assert chunks[-1].is_done()
    result = chunks[-1].content["result"]
    assert isinstance(result, Out) and result.x == 9
    assert all(c.agent == "a" for c in chunks)


async def test_async_skill_agent_run_schema_streams(tmp_path):
    from dataclasses import dataclass

    from aimu.aio import SkillAgent

    @dataclass
    class Out:
        x: int

    client = MockAsyncModelClient(['{"x": 7}'])
    client.model.supports_structured_output = False  # parse-path (mock _chat takes no response_format)
    agent = SkillAgent(client, name="s", skill_manager=_empty_skill_manager(tmp_path))

    chunks = [c async for c in await agent.run("t", stream=True, schema=Out)]
    assert chunks[-1].is_done()
    result = chunks[-1].content["result"]
    assert isinstance(result, Out) and result.x == 7
    assert all(c.agent == "s" for c in chunks)  # chunks tagged with the agent name


async def test_async_orchestrator_assemble_accepts_workflow_worker():
    from aimu.aio import Agent as AsyncAgent
    from aimu.aio import Chain as AsyncChain
    from aimu.aio import OrchestratorAgent as AsyncOrchestratorAgent

    client = MockAsyncModelClient(["done"])
    chain = AsyncChain(agents=[AsyncAgent(MockAsyncModelClient(["chain result"]))], name="researcher")
    agent_worker = AsyncAgent(MockAsyncModelClient(["agent result"]), system_message="Critique.", name="critic")

    orch = AsyncOrchestratorAgent.assemble(client, "Use the workers.", workers=[chain, agent_worker])
    tool_names = {t.__name__ for t in orch._orchestrator.tools}
    assert tool_names == {"researcher", "critic"}
    assert await orch.run("task") == "done"


# --------------------------------------------------------------------------------------
# Message provenance (async mirror of tests/test_provenance.py)
# --------------------------------------------------------------------------------------


class _AsyncLoopClient(AsyncBaseModelClient):
    """Async fake with precise control over when a turn ends in a pending tool result."""

    def __init__(self, tool_turns: int):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools = []
        self.last_thinking = ""
        self.last_usage = None
        self.tool_context_deps = None
        self.tool_approval = None
        self._tool_turns = tool_turns
        self._turn = 0

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        if user_message is not None:  # None = continuation turn (no new user message)
            self.messages.append({"role": "user", "content": user_message})
        self._turn += 1
        if use_tools and self._turn <= self._tool_turns:
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "t", "content": "r", "tool_call_id": "x"})
            return ""
        self.messages.append({"role": "assistant", "content": "done"})
        return "done"

    async def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        return await self._chat(prompt, generate_kwargs)


async def test_async_continuation_injects_no_user_turn():
    # The agent continues via chat() with no user message, so no synthetic user turn is
    # injected and nothing carries the (now-legacy) continuation provenance tag.
    client = _AsyncLoopClient(tool_turns=1)
    agent = Agent(client, tools=[])
    await agent.run("real question")

    user_turns = [m for m in client.messages if m["role"] == "user"]
    assert [m["content"] for m in user_turns] == ["real question"]
    assert PROVENANCE_CONTINUATION not in [m.get(PROVENANCE_KEY) for m in client.messages]


async def test_async_final_answer_turn_tagged():
    client = _AsyncLoopClient(tool_turns=99)
    agent = Agent(client, tools=[], max_iterations=2, final_answer_prompt="wrap up")
    await agent.run("real question")

    tags = [m.get(PROVENANCE_KEY) for m in client.messages]
    assert tags.count(PROVENANCE_FINAL_ANSWER) == 1
    assert PROVENANCE_CONTINUATION not in tags  # continuation turns are no longer injected/tagged
