"""Async mirror of tests/test_events.py: turn events from the async model client.

See tests/test_events.py for the sync surface; these tests exercise the same
ModelTurnStarted / ModelTurnFinished emission points on aimu.aio.
"""

import pytest

from tests.helpers_aio import MockAsyncModelClient


async def test_async_client_emits_turn_events_with_no_agent():
    """A bare await chat() is observable with no agent, same as the sync surface."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockAsyncModelClient(["hello"])
    client.events = seen.append
    await client.chat("hi")

    kinds = [type(e).__name__ for e in seen]
    assert "ModelTurnStarted" in kinds
    assert "ModelTurnFinished" in kinds
    started = next(e for e in seen if isinstance(e, ModelTurnStarted))
    finished = next(e for e in seen if isinstance(e, ModelTurnFinished))
    assert started.message_count >= 1
    assert finished.text == "hello"
    assert finished.duration_s >= 0.0


class _AsyncSeedingClient(MockAsyncModelClient):
    """Routes ``_chat`` through the shared ``_chat_setup`` seam, so the system message is
    seeded inside ``_chat`` exactly as a real provider seeds it."""

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        await self._chat_setup(user_message, generate_kwargs, use_tools, images, audio)
        reply = self._responses[self._call_count]
        self._call_count += 1
        self._append_message({"role": "assistant", "content": reply})
        return reply


async def test_async_failing_turn_still_reports_finished_with_the_error():
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    class Boom(MockAsyncModelClient):
        async def _chat(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    seen = []
    client = Boom([])
    client.events = seen.append
    with pytest.raises(RuntimeError):
        await client.chat("hi")

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


async def test_async_failing_generate_still_reports_finished_with_the_error():
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    class Boom(MockAsyncModelClient):
        async def _generate(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    seen = []
    client = Boom([])
    client.events = seen.append
    with pytest.raises(RuntimeError):
        await client.generate("hi")

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


async def test_async_message_count_counts_the_system_message_this_turn_seeds():
    from aimu.events import ModelTurnStarted

    seen = []
    client = _AsyncSeedingClient(["a", "b"])
    client.system_message = "You are helpful."
    client.events = seen.append

    await client.chat("first")
    assert next(e for e in seen if isinstance(e, ModelTurnStarted)).message_count == 2

    seen.clear()
    await client.chat("second")
    assert next(e for e in seen if isinstance(e, ModelTurnStarted)).message_count == 4


async def test_async_generate_reports_one_message_however_long_the_conversation():
    from aimu.events import ModelTurnStarted

    seen = []
    client = _AsyncSeedingClient(["a", "b", "c"])
    client.events = seen.append
    await client.chat("first")
    await client.chat("second")
    seen.clear()

    await client.generate("one-shot")
    assert next(e for e in seen if isinstance(e, ModelTurnStarted)).message_count == 1


class _Recording(MockAsyncModelClient):
    """Records its request payload, like every real provider's request path does."""

    async def _chat(self, *args, **kwargs):
        self._record_request({"messages": list(self.messages)})
        return await super()._chat(*args, **kwargs)


async def test_async_structured_run_emits_attributed_bracketed_events():
    """aio.Agent.run(schema=...) is still a run: bracketed, and the client's own events
    attributed to the agent rather than arriving orphaned."""
    from dataclasses import dataclass

    from aimu.aio import Agent
    from aimu.events import RunFinished

    @dataclass
    class Out:
        x: int

    client = _Recording(['{"x": 5}'])
    client.model.supports_structured_output = False
    seen = []
    agent = Agent(client, name="critic", events=seen.append)

    result = await agent.run("verdict?", schema=Out)
    assert isinstance(result, Out)

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    assert "RequestPrepared" in kinds, kinds
    assert all(e.agent == "critic" for e in seen), kinds
    assert next(e for e in seen if isinstance(e, RunFinished)).error is None


async def test_async_streamed_structured_run_emits_attributed_bracketed_events():
    from dataclasses import dataclass

    from aimu.aio import Agent

    @dataclass
    class Out:
        x: int

    client = _Recording(['{"x": 5}'])
    client.model.supports_structured_output = False
    seen = []
    agent = Agent(client, name="critic", events=seen.append)

    stream = await agent.run("verdict?", stream=True, schema=Out)
    async for _ in stream:
        pass

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    assert all(e.agent == "critic" for e in seen), kinds


async def test_async_skill_agent_structured_run_emits_bracketed_events(tmp_path):
    from dataclasses import dataclass

    from aimu.aio import SkillAgent
    from aimu.skills import SkillManager

    @dataclass
    class Out:
        x: int

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    client = _Recording(['{"x": 5}'])
    client.model.supports_structured_output = False
    seen = []
    agent = SkillAgent(
        client, name="critic", events=seen.append, skill_manager=SkillManager(skill_dirs=[str(skills_dir)])
    )

    result = await agent.run("verdict?", schema=Out)
    assert isinstance(result, Out)

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    assert all(e.agent == "critic" for e in seen), kinds


async def test_async_skill_agent_streamed_structured_run_emits_bracketed_events(tmp_path):
    from dataclasses import dataclass

    from aimu.aio import SkillAgent
    from aimu.skills import SkillManager

    @dataclass
    class Out:
        x: int

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    client = _Recording(['{"x": 5}'])
    client.model.supports_structured_output = False
    seen = []
    agent = SkillAgent(
        client, name="critic", events=seen.append, skill_manager=SkillManager(skill_dirs=[str(skills_dir)])
    )

    stream = await agent.run("verdict?", stream=True, schema=Out)
    async for _ in stream:
        pass

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    assert all(e.agent == "critic" for e in seen), kinds


async def test_async_hallucinated_tool_name_emits_ToolCalled_with_the_error():
    from aimu.aio import Agent
    from aimu.events import ToolCalled

    client = MockAsyncModelClient([{"tool": "no_such_tool", "arguments": {"a": 1}}, "sorry"])
    seen = []
    agent = Agent(client, name="a", tools=[], events=seen.append, max_iterations=3)
    await agent.run("go")

    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 1
    assert called[0].name == "no_such_tool"
    assert called[0].error == "Tool 'no_such_tool' not found."


async def test_async_hallucinated_tool_name_emits_ToolCalled_when_streaming():
    from aimu.aio import Agent
    from aimu.events import ToolCalled

    client = MockAsyncModelClient([{"tool": "no_such_tool", "arguments": {}}, "sorry"])
    seen = []
    agent = Agent(client, name="a", tools=[], events=seen.append, max_iterations=3)
    stream = await agent.run("go", stream=True)
    async for _ in stream:
        pass

    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 1
    assert called[0].error == "Tool 'no_such_tool' not found."


async def test_async_orchestrator_assemble_forwards_the_sink_to_the_inner_agent():
    from aimu.aio import Agent, OrchestratorAgent
    from aimu.events import ModelTurnStarted

    worker = Agent(MockAsyncModelClient(["worker output"]), "Do the work.", name="worker")
    client = MockAsyncModelClient(["orchestrated answer"])
    seen = []
    orch = OrchestratorAgent.assemble(client, "Use the worker.", workers=[worker], events=seen.append)

    assert await orch.run("task") == "orchestrated answer"
    kinds = [type(e).__name__ for e in seen]
    assert "RunStarted" in kinds and "RunFinished" in kinds
    assert any(isinstance(e, ModelTurnStarted) for e in seen)
    assert all(e.agent == "orchestrator" for e in seen), kinds


async def test_async_no_sink_means_no_behaviour_change():
    """The default path must be byte-identical to before."""
    client = MockAsyncModelClient(["hello"])
    assert client.events is None
    assert await client.chat("hi") == "hello"


async def test_async_generate_emits_turn_events_too():
    from aimu.events import ModelTurnFinished

    seen = []
    client = MockAsyncModelClient(["generated"])
    client.events = seen.append
    await client.generate("prompt")
    assert any(isinstance(e, ModelTurnFinished) for e in seen)


async def test_async_streamed_chat_emits_turn_finished_once_on_drain():
    """ModelTurnFinished must not fire until the async iterator is fully drained: usage
    only populates then, and emitting eagerly would report a turn that hasn't run yet."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockAsyncModelClient(["streamed hello"])
    client.events = seen.append

    stream = await client.chat("hi", stream=True)
    # ModelTurnStarted fires as soon as chat() is called, before any chunk is produced.
    assert any(isinstance(e, ModelTurnStarted) for e in seen)
    assert not any(isinstance(e, ModelTurnFinished) for e in seen)

    chunks = [chunk async for chunk in stream]
    assert chunks  # sanity: the mock actually streamed something

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].text == "streamed hello"


async def test_async_abandoned_stream_still_reports_partial_text():
    """A consumer that stops consuming part-way (triggering aclose()) still gets its turn
    reported, via the generator's finally block."""
    from aimu.events import ModelTurnFinished

    seen = []
    client = MockAsyncModelClient(["streamed hello"])
    client.events = seen.append

    stream = await client.chat("hi", stream=True)
    async for _chunk in stream:
        break  # abandon after the first chunk; triggers the async generator's aclose()
    await stream.aclose()

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1


def _a_tool() -> str:
    """Does nothing."""
    return ""


async def test_async_failing_tools_streamed_turn_emits_finished_before_any_chunk():
    """``_chat_with_tools_streamed`` is itself an async generator, so ``await chat()`` only
    builds it -- ``_emit_turn_started`` and the eager ``_chat()`` call happen on first
    iteration (mirroring ``_chat_setup`` running eagerly under a real provider). A raise
    there, before any chunk exists, must still close out the started/finished pair."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    class Boom(MockAsyncModelClient):
        async def _chat(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    seen = []
    client = Boom([])
    client.events = seen.append
    stream = await client.chat("hi", tools=[_a_tool], stream=True)
    assert not seen  # nothing fires until the async generator is iterated

    with pytest.raises(RuntimeError):
        async for _ in stream:
            pass

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


async def test_async_tools_streamed_turn_emits_exactly_one_finished_on_success():
    """A normal streamed turn with a ``tools=`` override must report exactly one
    ModelTurnFinished: not zero (the missed-guard bug) and not two (a double-emit from
    also firing on the successful eager call before ``_emit_when_drained`` runs)."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockAsyncModelClient(["streamed hello"])
    client.events = seen.append

    stream = await client.chat("hi", tools=[_a_tool], stream=True)
    chunks = [chunk async for chunk in stream]
    assert chunks  # sanity: the mock actually streamed something

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].text == "streamed hello"


# ---------------------------------------------------------------------------
# Wrapper double-emit regression tests (async mirror of tests/test_events.py).
# ---------------------------------------------------------------------------


async def test_async_agentic_view_single_turn_chat_emits_exactly_one_pair():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    assert await view.chat("question") == "final answer"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


async def test_async_agentic_view_tool_loop_chat_emits_one_pair_per_real_turn():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["tool", "after tool"])
    view = Agent(client, max_iterations=5).as_model_client()
    seen = []
    view.events = seen.append

    assert await view.chat("do something with tools") == "after tool"
    assert client._call_count == 2  # sanity: two real requests did happen

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 2
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 2


async def test_async_agentic_view_generate_emits_exactly_one_pair():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["generated"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    assert await view.generate("prompt") == "generated"
    assert client._call_count == 1

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


async def test_async_agentic_view_streamed_chat_emits_exactly_one_pair():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["stream result"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    stream = await view.chat("task", stream=True)
    chunks = [c async for c in stream]
    assert chunks

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


async def test_async_fallback_client_chat_emits_exactly_one_pair():
    """AsyncFallbackClient shares the same shape as the sync FallbackClient: it fully
    overrides chat()/generate() and delegates to the winning inner client's own public
    chat(), so it never had the wrapper double-emit defect. Confirmed here so a future
    refactor can't silently reintroduce it."""
    from aimu.aio.fallback import AsyncFallbackClient
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    primary = MockAsyncModelClient(["ok"])
    fc = AsyncFallbackClient([primary])
    seen = []
    fc.events = seen.append

    assert await fc.chat("hi") == "ok"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1
    # The sink is scoped onto the winning attempt for the call and restored afterwards, so
    # a sink the inner client owns is never destroyed (see test_fallback_api.py).
    assert primary.events is None


# ---------------------------------------------------------------------------
# Agent-loop events: async mirror of the sync RunStarted / RunFinished /
# ToolCalled / ToolDenied tests in tests/test_events.py.
# ---------------------------------------------------------------------------


async def test_async_agent_run_emits_run_and_tool_events():
    from aimu.aio.agent import Agent
    from aimu.events import RunFinished, RunStarted, ToolCalled
    from aimu.tools import tool

    seen = []

    @tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    client = MockAsyncModelClient([{"tool": "add", "arguments": {"a": 2, "b": 3}}, "5 is the answer"])
    agent = Agent(client, tools=[add], events=seen.append)
    await agent.run("add 2 and 3")

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    called = next(e for e in seen if isinstance(e, ToolCalled))
    assert called.name == "add" and called.arguments == {"a": 2, "b": 3}
    assert called.result == "5"
    assert isinstance(seen[0], RunStarted)
    assert isinstance(seen[-1], RunFinished)


async def test_async_events_is_a_per_run_override():
    """Mirrors deps / tool_approval / thinking: None uses the field."""
    from aimu.aio.agent import Agent

    field_seen = []
    override_seen = []
    client = MockAsyncModelClient(["answer"])
    agent = Agent(client, events=field_seen.append)

    await agent.run("q", events=override_seen.append)
    assert override_seen
    assert not field_seen

    client2 = MockAsyncModelClient(["answer"])
    agent2 = Agent(client2, events=field_seen.append)
    await agent2.run("q")
    assert field_seen


async def test_async_tool_events_carry_the_agent_name_and_iteration():
    from aimu.aio.agent import Agent
    from aimu.events import ToolCalled
    from aimu.tools import tool

    seen = []

    @tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    client = MockAsyncModelClient([{"tool": "add", "arguments": {"a": 2, "b": 3}}, "done"])
    agent = Agent(client, tools=[add], name="alpha", events=seen.append)
    await agent.run("add 2 and 3")

    called = next(e for e in seen if isinstance(e, ToolCalled))
    assert called.agent == "alpha"
    assert called.iteration == 0


async def test_async_turn_events_are_attributed_to_the_named_agent():
    """Async mirror of the sync attribution test: the client's own turn events must be
    stamped with the agent's name and the current round, not left at agent=None/iteration=0."""
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockAsyncModelClient(["tool", "after tool"])
    agent = Agent(client, name="my-agent", events=seen.append)
    await agent.run("go")

    assert len(seen) >= 4
    assert all(e.agent == "my-agent" for e in seen)

    turn_events = [e for e in seen if isinstance(e, (ModelTurnStarted, ModelTurnFinished))]
    assert len(turn_events) == 4
    iterations = [e.iteration for e in turn_events]
    assert iterations == [0, 0, 1, 1]


async def test_async_run_finished_iteration_matches_the_forced_wrap_up_turn():
    """Async mirror of the sync forced-wrap-up test. The two engines now share one
    ``max_iterations`` convention (v0.22 aligned the async ``run``/``run_streamed`` while
    conditions down to the sync ones, which were already correct -- see
    ``tests/test_loop_iteration_parity.py``), so the same ``max_iterations=2`` and the same
    scripted responses that force sync into a genuine extra wrap-up call do the same here:
    two bounded-loop tool turns exhaust the cap while still pending tools, so the wrap-up
    makes a genuine third call (see aimu/aio/_tool_loop.py's ``_forced_wrap_up``). Before
    that alignment, async's off-by-one let its bounded loop consume all three scripted turns
    and reach ``TERMINAL_HEALTHY`` *inside* the loop, so this needed ``max_iterations=1``
    instead to force the same branch -- a different shape from the sync test it mirrors."""
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, RunFinished

    seen = []
    client = MockAsyncModelClient(["tool", "tool", "final answer"])
    agent = Agent(client, max_iterations=2, events=seen.append)
    result = await agent.run("go")

    assert result == "final answer"
    turn_finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(turn_finished) == 3  # the initial tool turn, the second tool turn, the forced wrap-up turn
    finished = next(e for e in seen if isinstance(e, RunFinished))
    assert finished.iteration == turn_finished[-1].iteration == 2


async def test_async_scoped_override_wins_over_a_durable_events_attribute():
    """Async mirror of ``test_events.py::test_scoped_override_wins_over_a_durable_events_attribute``."""
    from aimu.aio.agent import Agent

    durable_seen = []
    scoped_seen = []
    client = MockAsyncModelClient(["reply"])
    client.events = durable_seen.append

    agent = Agent(client, events=scoped_seen.append)
    await agent.run("hi")

    assert durable_seen == []
    assert len(scoped_seen) > 0


async def test_async_a_client_given_its_own_explicit_sink_inside_a_tool_receives_its_events():
    """Async mirror of ``test_events.py
    ::test_a_client_given_its_own_explicit_sink_inside_a_tool_receives_its_events``."""
    from aimu.aio.agent import Agent
    from aimu.tools import tool

    inner_seen = []

    @tool
    async def spawn_sub() -> str:
        """Build a fresh client with its own explicit sink, as the docs recommend."""
        sub_client = MockAsyncModelClient(["sub reply"])
        sub_client.events = inner_seen.append
        return await sub_client.chat("sub task")

    outer_seen = []
    outer = MockAsyncModelClient([{"tool": "spawn_sub"}, "final answer"])
    agent = Agent(outer, tools=[spawn_sub], events=outer_seen.append)
    await agent.run("do it")

    assert any(type(e).__name__ == "ModelTurnStarted" for e in inner_seen)
    assert not any(type(e).__name__ in ("RunStarted", "RunFinished") for e in inner_seen)
    assert inner_seen and all(e not in outer_seen for e in inner_seen)


async def test_async_a_denied_tool_emits_ToolDenied_not_ToolCalled():
    from aimu.aio.agent import Agent
    from aimu.events import ToolCalled, ToolDenied
    from aimu.tools import tool

    ran = []

    @tool
    async def danger() -> str:
        """Risky."""
        ran.append(1)
        return "ran"

    seen = []
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[danger], tool_approval=lambda name, arguments: False, events=seen.append)

    assert await agent.run("go") == "done"

    assert ran == []
    assert not any(isinstance(e, ToolCalled) for e in seen)
    denied = next(e for e in seen if isinstance(e, ToolDenied))
    assert denied.name == "danger"


async def test_async_streaming_tool_emits_tool_called():
    """A streaming (async generator) @tool is dispatched by a different branch of
    _dispatch_streamed than a plain tool; it must still emit exactly one ToolCalled."""
    from aimu.aio.agent import Agent
    from aimu.events import ToolCalled
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.tools import tool

    @tool
    async def progress_tool(x: int) -> str:
        """A streaming tool that reports progress before returning."""
        yield StreamChunk(StreamingContentType.GENERATING, "working")
        # Async generators can't `return <value>` (SyntaxError); the final-chunk-result
        # convention (StreamChunk.content == {"result": ...}) is how they report one.
        yield StreamChunk(StreamingContentType.DONE, {"result": str(x * 2)})

    seen = []
    client = MockAsyncModelClient([{"tool": "progress_tool", "arguments": {"x": 3}}, "6 is the answer"])
    agent = Agent(client, tools=[progress_tool], events=seen.append)
    stream = await agent.run("double 3", stream=True)
    _ = [c async for c in stream]

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 1
    assert called[0].name == "progress_tool"
    assert called[0].result == "6"


async def test_async_streamed_run_via_agent_events_emits_turn_finished_with_usage():
    """Regression guard for ``_emit_when_drained`` specifically (the async streamed
    ``ModelTurnFinished`` emit site, ``aimu/aio/_base.py``). Its only other coverage
    (``test_async_agentic_view_streamed_chat_emits_exactly_one_pair`` above) sets
    ``client.events`` directly -- the durable attribute -- which ``_effective_sink`` and a
    bare ``getattr`` resolve identically when no scoped override is active, so a regression
    at that one line (reverting ``_effective_sink(self)`` back to
    ``getattr(self, "events", None)``) would leave the rest of the suite green. This drives
    the same emit through ``Agent.run(events=..., stream=True)`` instead -- the scoped
    ContextVar-override path -- and checks the reported ``usage``, so it also confirms
    ``last_usage`` threads through correctly on the streamed drain path.
    """
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished

    client = MockAsyncModelClient(["streamed answer"])
    client.model.supports_tools = False
    # Simulates a provider that has reported usage by the time the stream drains (the mock
    # itself never populates last_usage, unlike a real provider's _chat_streamed).
    client.last_usage = {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

    seen = []
    agent = Agent(client, events=seen.append)

    stream = await agent.run("hi", stream=True)
    chunks = [c async for c in stream]
    assert chunks

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].usage == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


async def test_async_run_finished_carries_the_error_when_a_run_raises():
    """A run that raises must still report -- emit from a finally."""
    from aimu.aio import DegenerateTurnError
    from aimu.aio.agent import Agent
    from aimu.events import RunFinished

    seen = []
    client = MockAsyncModelClient([""] * 6)
    agent = Agent(client, name="broken", max_iterations=3, events=seen.append)

    with pytest.raises(DegenerateTurnError):
        await agent.run("do something")

    finished = next(e for e in seen if isinstance(e, RunFinished))
    assert isinstance(finished.error, DegenerateTurnError)


async def test_async_concurrent_tool_dispatch_emits_exactly_one_tool_called_each():
    """The async engine dispatches concurrent tool calls under asyncio.TaskGroup; each must
    emit exactly one ToolCalled -- asserting the count (not mere presence) catches a sink
    firing twice for one dispatched call, which a presence check would miss."""
    from aimu.aio.agent import Agent
    from aimu.events import ToolCalled
    from aimu.tools import tool

    @tool
    async def slow_add(a: int, b: int) -> int:
        """Add two numbers slowly."""
        import asyncio

        await asyncio.sleep(0.01)
        return a + b

    seen = []
    client = MockAsyncModelClient(
        [
            {
                "tools": [
                    {"name": "slow_add", "arguments": {"a": 1, "b": 2}},
                    {"name": "slow_add", "arguments": {"a": 3, "b": 4}},
                ]
            },
            "done",
        ]
    )
    agent = Agent(client, tools=[slow_add], concurrent_tool_calls=True, events=seen.append)

    assert await agent.run("add two pairs") == "done"

    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 2
    results = sorted(c.result for c in called)
    assert results == ["3", "7"]


async def test_async_concurrent_tool_dispatch_does_not_leak_the_run_sink_to_a_different_client():
    """Async mirror of ``test_events.py
    ::test_concurrent_tool_dispatch_does_not_leak_the_run_sink_to_a_different_client`` -- and,
    after the ``_client_family`` fix, the outcome now *matches* sync rather than differing from
    it: a *different* client called from inside a tool never receives the outer run's scoped
    sink, regardless of ``asyncio.TaskGroup.create_task`` copying the current context, because
    ``_effective_sink`` checks family membership before returning the override, and this inner
    client was never part of the outer run's family. Two tool calls are required to actually
    exercise ``asyncio.TaskGroup``: a single call takes the sequential fallback. Contrast with
    ``test_async_concurrent_tool_dispatch_of_the_same_client_is_visible_under_taskgroup`` below,
    where the client *is* the same object and the two surfaces genuinely differ.
    """
    from aimu.aio.agent import Agent
    from aimu.tools import tool

    inner_client = MockAsyncModelClient(["inner reply a"])
    other_inner_client = MockAsyncModelClient(["inner reply b"])

    @tool
    async def call_inner_a() -> str:
        """Call a separate client's own chat(), relying on the ambient sink."""
        return await inner_client.chat("inner task a")

    @tool
    async def call_inner_b() -> str:
        """Call a separate client's own chat(), relying on the ambient sink."""
        return await other_inner_client.chat("inner task b")

    for concurrent in (False, True):
        inner_client._call_count = 0
        other_inner_client._call_count = 0
        seen = []
        outer_client = MockAsyncModelClient(
            [{"tools": [{"name": "call_inner_a"}, {"name": "call_inner_b"}]}, "final answer"]
        )
        agent = Agent(
            outer_client, tools=[call_inner_a, call_inner_b], concurrent_tool_calls=concurrent, events=seen.append
        )
        await agent.run("do it")

        # Only the outer client's own turns are reported: 1 RunStarted + 2 ModelTurnStarted +
        # 2 ModelTurnFinished + 2 ToolCalled + 1 RunFinished = 8, whether dispatch is
        # sequential or concurrent.
        assert len(seen) == 8, (concurrent, [type(e).__name__ for e in seen])
        assert all(getattr(e, "agent", None) in (None, agent.name) for e in seen)
        assert inner_client.events is None
        assert other_inner_client.events is None


async def test_async_concurrent_tool_dispatch_of_the_same_client_is_visible_under_taskgroup():
    """The one case that genuinely differs by surface: a tool that calls the *same* client
    the run's override was installed for. On async, ``asyncio.TaskGroup.create_task`` always
    copies the current context, so the reentrant call sees the override whether dispatch is
    sequential or concurrent -- unlike sync, where a concurrently dispatched
    ``ThreadPoolExecutor`` thread starts with an empty context. See the sync mirror,
    ``test_events.py::test_concurrent_tool_dispatch_of_the_same_client_is_invisible_under_threadpool``.
    """
    from aimu.aio.agent import Agent
    from aimu.tools import tool

    def make_agent(concurrent):
        outer = MockAsyncModelClient(
            [
                {"tools": [{"name": "reentrant_a"}, {"name": "reentrant_b"}]},
                "reentrant-a-reply",
                "reentrant-b-reply",
                "final answer",
            ]
        )

        @tool
        async def reentrant_a() -> str:
            return await outer.chat("extra turn a", use_tools=False)

        @tool
        async def reentrant_b() -> str:
            return await outer.chat("extra turn b", use_tools=False)

        seen = []
        agent = Agent(outer, tools=[reentrant_a, reentrant_b], concurrent_tool_calls=concurrent, events=seen.append)
        return agent, seen

    sequential_agent, sequential_seen = make_agent(concurrent=False)
    await sequential_agent.run("do it")
    assert len(sequential_seen) == 12, [type(e).__name__ for e in sequential_seen]

    concurrent_agent, concurrent_seen = make_agent(concurrent=True)
    await concurrent_agent.run("do it")
    # Unlike sync, the reentrant turns' events ARE present: still 12.
    assert len(concurrent_seen) == 12, [type(e).__name__ for e in concurrent_seen]


async def test_async_skill_agent_run_emits_run_and_tool_events(tmp_path):
    """aio.SkillAgent fully overrides run() (it needs async skill setup before the loop), so
    its events= threading is verified directly rather than assumed to fall out of inheritance
    (unlike the sync SkillAgent, which does inherit it). Uses the per-run override (field left
    unset) so the assertion actually exercises run()'s own `events` parameter passing into
    _make_tool_loop, not just the self.events fallback _make_tool_loop resolves on its own."""
    from aimu.aio import SkillAgent
    from aimu.skills.manager import SkillManager

    seen = []
    client = MockAsyncModelClient(["answer"])
    manager = SkillManager(skill_dirs=[str(tmp_path)])  # empty dir: no skills discovered
    agent = SkillAgent(client, skill_manager=manager)  # self.events left None
    await agent.run("q", events=seen.append)

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"


async def test_async_in_process_client_chat_emits_exactly_one_pair():
    """_AsyncInProcessClient (AsyncHuggingFaceClient/AsyncLlamaCppClient's shared base)
    wraps a sync client but calls its private _chat()/_generate() directly (via
    asyncio.to_thread), never the sync client's own public chat()/generate() -- so only
    the async wrapper's inherited turn-tracking fires, exactly once per real request. No
    fix was needed here; this test documents and locks in that it was already correct."""
    from aimu.aio.providers._inprocess import _AsyncInProcessClient
    from aimu.events import ModelTurnFinished, ModelTurnStarted
    from tests.helpers import MockModelClient

    class _Wrapper(_AsyncInProcessClient):
        _SYNC_CLASS = MockModelClient

    sync_client = MockModelClient(["hi back"])
    sync_client.model.supports_tools = False
    wrapper = _Wrapper(sync_client)
    seen = []
    wrapper.events = seen.append

    assert await wrapper.chat("hello") == "hi back"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


# ---------------------------------------------------------------------------
# Workflow factory forwarding (async mirror of the corresponding block in
# tests/test_events.py). See that file for why EvaluatorOptimizer is out of scope (no
# from_client()) and why Parallel is only exercised at construction time, not under a real
# concurrent run.
# ---------------------------------------------------------------------------


async def test_async_chain_forwards_the_sink_to_every_step():
    """One sink sees the whole pipeline, with each event attributed to its step."""
    from aimu.aio import Chain

    seen = []
    chain = Chain.from_client(
        MockAsyncModelClient(["step one output", "step two output"]), ["step one", "step two"], events=seen.append
    )
    await chain.run("go")
    agents = {e.agent for e in seen if e.agent}
    assert len(agents) == 2, f"expected both steps to be attributed, got {agents}"


async def test_async_router_from_client_forwards_events_to_the_classifier():
    """`Router.from_client()` only constructs the classifier Agent itself; handlers are
    supplied by the caller already built, so only the classifier is asserted here."""
    from aimu.aio import Agent, Router

    seen = []
    sink = seen.append
    client = MockAsyncModelClient(["classified-as-a"])
    router = Router.from_client(
        client,
        classifier_prompt="classify",
        handlers={"classified-as-a": Agent(MockAsyncModelClient(["handler output"]), name="handler")},
        events=sink,
    )
    assert router.routing_agent.events is sink


async def test_async_parallel_from_client_wires_events_onto_every_worker_and_the_aggregator():
    """Construction-time forwarding only -- see the sync module's note on why a real
    concurrent run is not exercised here (asyncio.TaskGroup has the identical shared-client
    sink-delivery hazard as the sync ThreadPoolExecutor path)."""
    from aimu.aio import Parallel

    seen = []
    sink = seen.append
    client = MockAsyncModelClient([])
    parallel = Parallel.from_client(
        client,
        worker_prompts=["A.", "B."],
        aggregator_prompt="Synthesize.",
        events=sink,
    )
    assert all(worker.events is sink for worker in parallel.workers)
    assert parallel.aggregator.events is sink


async def test_async_plan_execute_evaluator_from_client_forwards_events_to_planner_and_executor():
    """The planner and executor run sequentially (no shared-client concurrency hazard), so
    this exercises a real run, not just construction-time wiring. The scorer's judge_client
    is a sync MockModelClient (LLMJudgeScorer is sync; the factory docstring notes the judge
    client may be sync or async), so it never receives ``events`` -- it isn't a Runner."""
    from aimu.aio import PlanExecuteEvaluator
    from tests.helpers import MockModelClient

    seen = []
    client = MockAsyncModelClient(["plan it", "did it"])
    judge = MockModelClient(["8"])  # LLMJudgeScorer parses "8" -> 0.8 -> pass
    wf = PlanExecuteEvaluator.from_client(client, judge_client=judge, criteria="answer the task", events=seen.append)
    await wf.run("hi")

    agents = {e.agent for e in seen if e.agent}
    assert agents == {"planner", "executor"}
