"""The event vocabulary and its emission guard.

Events are AIMU's comprehension surface: a caller attaches one sink and sees what the
library and the model actually did. They are additive -- no sink means no behaviour change
-- and a sink that raises must never break the run it is observing.
"""

import logging

import pytest

from aimu.events import (
    ContextCompacted,
    EventSink,
    ModelTurnFinished,
    ModelTurnStarted,
    RequestPrepared,
    RunEvent,
    RunFinished,
    RunStarted,
    ToolCalled,
    ToolDenied,
    emit,
    log_events,
)
from tests.helpers import MockModelClient, RecordingModelClient


def test_events_are_frozen():
    """Immutable, so a sink cannot mutate an event another sink will see."""
    event = RunStarted(task="hi")
    with pytest.raises(Exception):
        event.task = "changed"


def test_every_event_carries_attribution():
    """agent and iteration mirror StreamChunk, so a sink can attribute events in a
    nested workflow."""
    for event in [
        RunStarted(task="t"),
        RunFinished(result="r", error=None),
        ModelTurnStarted(model="m", message_count=1, tool_names=()),
        RequestPrepared(provider="p", model="m", payload={}),
        ModelTurnFinished(model="m", text="t", usage=None, duration_s=0.0),
        ToolCalled(name="n", arguments={}, result="r", error=None, duration_s=0.0),
        ToolDenied(name="n", arguments={}),
        ContextCompacted(dropped=[], before_tokens=0, after_tokens=0),
    ]:
        assert isinstance(event, RunEvent)
        assert event.agent is None
        assert event.iteration == 0


def test_emit_with_no_sink_is_a_noop():
    emit(None, RunStarted(task="t"))  # must not raise


def test_emit_delivers():
    seen = []
    emit(seen.append, RunStarted(task="t"))
    assert len(seen) == 1 and seen[0].task == "t"


def test_a_raising_sink_is_swallowed_and_logged(caplog):
    """A display hook must never break the run it observes."""

    def broken(event):
        raise RuntimeError("sink is broken")

    with caplog.at_level(logging.WARNING):
        emit(broken, RunStarted(task="t"))  # must not raise
    assert "sink" in caplog.text.lower()


def test_log_events_sink_writes_one_line_per_event(caplog):
    logger = logging.getLogger("aimu.test.events")
    sink: EventSink = log_events(logger)
    with caplog.at_level(logging.INFO, logger="aimu.test.events"):
        sink(RunStarted(task="summarize the docs"))
        sink(ToolCalled(name="search", arguments={"q": "x"}, result="ok", error=None, duration_s=0.5))
    assert "RunStarted" in caplog.text
    assert "search" in caplog.text


def test_client_emits_turn_events_with_no_agent():
    """A bare chat() is observable: the person learning what a model does is exactly the
    person who has not reached for an Agent yet."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockModelClient(["hello"])
    client.events = seen.append
    client.chat("hi")

    kinds = [type(e).__name__ for e in seen]
    assert "ModelTurnStarted" in kinds
    assert "ModelTurnFinished" in kinds
    started = next(e for e in seen if isinstance(e, ModelTurnStarted))
    finished = next(e for e in seen if isinstance(e, ModelTurnFinished))
    assert started.message_count >= 1
    assert finished.text == "hello"
    assert finished.duration_s >= 0.0


def test_a_failing_turn_still_reports_finished_with_the_error():
    """A turn that raises still ended. Without the try/finally, a ContextOverflowError or a
    provider 4xx leaves a dangling ModelTurnStarted and every started/finished sink leaks."""

    class Boom(MockModelClient):
        def _chat(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    seen = []
    client = Boom([])
    client.events = seen.append
    with pytest.raises(RuntimeError):
        client.chat("hi")

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)
    assert finished[0].text is None


def test_a_failing_generate_still_reports_finished_with_the_error():
    class Boom(MockModelClient):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    seen = []
    client = Boom([])
    client.events = seen.append
    with pytest.raises(RuntimeError):
        client.generate("hi")

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


def test_a_failing_turn_with_a_tools_override_still_reports_finished():
    class Boom(MockModelClient):
        def _chat(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    def a_tool() -> str:
        """Does nothing."""
        return ""

    seen = []
    client = Boom([])
    client.events = seen.append
    with pytest.raises(RuntimeError):
        client.chat("hi", tools=[a_tool])

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


def test_a_stream_that_raises_mid_flight_reports_the_error():
    class Boom(MockModelClient):
        def _chat(self, *args, **kwargs):
            def gen():
                raise RuntimeError("dropped")
                yield  # pragma: no cover

            return gen()

    seen = []
    client = Boom([])
    client.events = seen.append
    with pytest.raises(RuntimeError):
        list(client.chat("hi", stream=True))

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


def _a_tool() -> str:
    """Does nothing."""
    return ""


def test_a_failing_tools_streamed_turn_emits_finished_before_any_chunk():
    """``_chat_with_tools_streamed`` is itself a generator, so calling ``chat()`` only
    builds it -- ``_emit_turn_started`` and the eager ``_chat()`` call happen on first
    iteration (mirroring ``_chat_setup`` running eagerly under a real provider). A raise
    there, before any chunk exists, must still close out the started/finished pair."""

    class Boom(MockModelClient):
        def _chat(self, *args, **kwargs):
            raise RuntimeError("provider said no")

    seen = []
    client = Boom([])
    client.events = seen.append
    stream = client.chat("hi", tools=[_a_tool], stream=True)
    assert not seen  # nothing fires until the generator is iterated

    with pytest.raises(RuntimeError):
        list(stream)

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


def test_tools_streamed_turn_emits_exactly_one_finished_on_success():
    """A normal streamed turn with a ``tools=`` override must report exactly one
    ModelTurnFinished: not zero (the missed-guard bug) and not two (a double-emit from
    also firing on the successful eager call before ``_emit_when_drained`` runs)."""
    seen = []
    client = MockModelClient(["streamed hello"])
    client.events = seen.append

    chunks = list(client.chat("hi", tools=[_a_tool], stream=True))
    assert chunks  # sanity: the mock actually streamed something

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].text == "streamed hello"


class _SeedingClient(MockModelClient):
    """A client whose ``_chat`` goes through the shared ``_chat_setup`` seam, so the system
    message is seeded exactly as a real provider seeds it: inside ``_chat``, after the
    ModelTurnStarted emit."""

    def _chat(self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self._chat_setup(user_message, generate_kwargs, use_tools, images, audio)
        reply = self._responses[self._call_count]
        self._call_count += 1
        self._append_message({"role": "assistant", "content": reply})
        return reply


def test_message_count_counts_the_system_message_this_turn_seeds():
    """The first turn sends system + user, so it must report 2 -- RequestPrepared, logged
    from the same call moments later, shows both."""
    seen = []
    client = _SeedingClient(["a", "b"])
    client.system_message = "You are helpful."
    client.events = seen.append

    client.chat("first")
    first = next(e for e in seen if isinstance(e, ModelTurnStarted))
    assert first.message_count == 2
    assert len(client.messages) == 3  # system + user + assistant

    seen.clear()
    client.chat("second")
    second = next(e for e in seen if isinstance(e, ModelTurnStarted))
    assert second.message_count == 4  # system, user, assistant, + the new user turn


def test_message_count_without_a_system_message():
    seen = []
    client = _SeedingClient(["a"])
    client.events = seen.append
    client.chat("first")
    assert next(e for e in seen if isinstance(e, ModelTurnStarted)).message_count == 1


def test_generate_reports_one_message_however_long_the_conversation():
    """generate() is stateless: it sends the prompt, not self.messages."""
    seen = []
    client = _SeedingClient(["a", "b", "c"])
    client.events = seen.append
    client.chat("first")
    client.chat("second")
    seen.clear()

    client.generate("one-shot")
    started = next(e for e in seen if isinstance(e, ModelTurnStarted))
    assert started.message_count == 1


def test_no_sink_means_no_behaviour_change():
    """The default path must be byte-identical to before."""
    client = MockModelClient(["hello"])
    assert client.events is None
    assert client.chat("hi") == "hello"


def test_generate_emits_turn_events_too():
    from aimu.events import ModelTurnFinished

    seen = []
    client = MockModelClient(["generated"])
    client.events = seen.append
    client.generate("prompt")
    assert any(isinstance(e, ModelTurnFinished) for e in seen)


def test_streamed_chat_emits_turn_finished_once_on_drain():
    """ModelTurnFinished must not fire until the iterator is fully drained: usage only
    populates then, and emitting eagerly would report a turn that hasn't run yet."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockModelClient(["streamed hello"])
    client.events = seen.append

    stream = client.chat("hi", stream=True)
    # ModelTurnStarted fires as soon as chat() is called, before any chunk is produced.
    assert any(isinstance(e, ModelTurnStarted) for e in seen)
    assert not any(isinstance(e, ModelTurnFinished) for e in seen)

    chunks = list(stream)
    assert chunks  # sanity: the mock actually streamed something

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].text == "streamed hello"


def test_abandoned_stream_still_reports_partial_text():
    """A consumer that stops consuming part-way (triggering generator close via GC or
    explicit .close()) still gets its turn reported, via the generator's finally block."""
    from aimu.events import ModelTurnFinished

    seen = []
    client = MockModelClient(["streamed hello"])
    client.events = seen.append

    stream = client.chat("hi", stream=True)
    next(stream)  # consume the first (and only) chunk
    stream.close()  # abandon explicitly; triggers the generator's finally block

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1


# ---------------------------------------------------------------------------
# Wrapper double-emit regression tests.
#
# A ModelTurnStarted/ModelTurnFinished pair must mean exactly one real request to a
# provider. A wrapper client (one whose chat()/generate() delegates to some other
# client rather than issuing a request itself) must never emit a phantom pair of its
# own on top of the real one(s) the delegate reports. Each test below asserts an exact
# count, not mere presence, since a duplicate-emission bug still passes a presence check.
# ---------------------------------------------------------------------------


def test_agentic_view_single_turn_chat_emits_exactly_one_pair():
    """Agent.as_model_client().chat() for a plain (no-tool-call) turn is one real request:
    the view itself must not add a second, phantom pair on top of the inner client's."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockModelClient(["final answer"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    assert view.chat("question") == "final answer"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


def test_agentic_view_tool_loop_chat_emits_one_pair_per_real_turn():
    """A tool-calling loop makes two real model requests (the tool-call turn and the
    follow-up); the view must report exactly two pairs, not four (two real + two phantom)."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockModelClient(["tool", "after tool"])
    view = Agent(client, max_iterations=5).as_model_client()
    seen = []
    view.events = seen.append

    assert view.chat("do something with tools") == "after tool"
    assert client._call_count == 2  # sanity: two real requests did happen

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 2
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 2


def test_agentic_view_generate_emits_exactly_one_pair():
    """generate() bypasses the agent loop and delegates straight to the inner client's
    real generate() -- one real request, so exactly one pair, not zero and not two."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockModelClient(["generated"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    assert view.generate("prompt") == "generated"
    assert client._call_count == 1

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


def test_agentic_view_streamed_chat_emits_exactly_one_pair():
    """The streaming path must not double-wrap either: the view's own _emit_when_drained
    is a no-op pass-through, so only the inner client's real streamed turn reports."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockModelClient(["stream result"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    chunks = list(view.chat("task", stream=True))
    assert chunks  # sanity: the mock actually streamed something

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


def test_fallback_client_chat_emits_exactly_one_pair():
    """FallbackClient fully overrides chat()/generate() (it never calls the inherited
    _chat()/_generate() turn-tracking on itself) and delegates to the winning inner
    client's own public chat(), so it never had the wrapper double-emit shape -- confirmed
    here so a future refactor that reintroduces delegation via _chat()/_generate() trips
    this test rather than silently regressing."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted
    from aimu.models.fallback import FallbackClient

    primary = MockModelClient(["ok"])
    fc = FallbackClient([primary])
    seen = []
    fc.events = seen.append

    assert fc.chat("hi") == "ok"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1
    # The sink is scoped onto the winning attempt for the call and restored afterwards, so
    # a sink the inner client owns is never destroyed (see test_fallback_api.py).
    assert primary.events is None


# ---------------------------------------------------------------------------
# The client family: which inner clients a scoped override reaches.
#
# A scoped per-run override is installed on whatever object the caller handed the Agent as
# `model_client`, but a wrapper's inherited `chat()` and the `_record_request()` seam run
# with `self` bound to two *different* objects, so `_effective_sink` matches against the
# whole family (`_client_family`, driven by the `_DELEGATE_ATTRS` / `_DELEGATE_LIST_ATTRS`
# name tables in `aimu.models._internal.chat_state`).
#
# Each wrapper shape below is pinned by asserting *both* halves of the run: the turn events
# emitted on the wrapper and the RequestPrepared emitted on the inner client. A missing
# family entry silences exactly one half, so a bare total-count assertion would let a
# regression through as "fewer events" without naming which emitter went dark. Every test
# here was verified to fail with its entry deleted from the table.
# ---------------------------------------------------------------------------


def _run_with_scoped_sink(wrapper) -> list:
    """One plain agent turn through ``wrapper``, collecting the run's scoped-override events."""
    from aimu.agents.agent import Agent

    seen: list = []
    Agent(wrapper, events=seen.append).run("task")
    return seen


def _assert_both_halves(seen: list) -> None:
    kinds = [type(e).__name__ for e in seen]
    assert any(isinstance(e, ModelTurnStarted) for e in seen), kinds
    assert any(isinstance(e, ModelTurnFinished) for e in seen), kinds
    assert any(isinstance(e, RequestPrepared) for e in seen), kinds


def test_family_reaches_the_inner_client_of_a_model_client():
    """`_client`: ModelClient is what `aimu.client()` returns, so it is the wrapper most runs
    go through. Its inherited chat() emits the turn events with self=the wrapper, while the
    request seam fires on the inner client -- drop `_client` from the family and the run keeps
    its turn bracket but loses RequestPrepared entirely."""
    from aimu.models.model_client import ModelClient

    inner = RecordingModelClient(["final"])
    inner.model.supports_tools = False
    wrapper = ModelClient.__new__(ModelClient)  # bypass provider dispatch; see test_tool_decorator.py
    wrapper._client = inner
    wrapper.model = inner.model
    wrapper.model_kwargs = None

    seen = _run_with_scoped_sink(wrapper)

    _assert_both_halves(seen)
    assert [type(e).__name__ for e in seen] == [
        "RunStarted",
        "ModelTurnStarted",
        "RequestPrepared",
        "ModelTurnFinished",
        "RunFinished",
    ]


def test_family_reaches_every_inner_client_of_a_fallback_client():
    """`clients`: FallbackClient overrides the *public* chat() and delegates to the winning
    inner client's own public chat(), so the inner client is the sole emitter of both halves.
    Drop `clients` and the run goes dark below the agent bracket, not merely quieter."""
    from aimu.models.fallback import FallbackClient

    primary = RecordingModelClient(["final"])
    primary.model.supports_tools = False
    wrapper = FallbackClient([primary])

    seen = _run_with_scoped_sink(wrapper)

    _assert_both_halves(seen)
    assert primary.events is None  # the sink arrived by scope, not by mutating the inner client


def test_family_reaches_the_inner_client_of_an_agentic_view():
    """`_inner_client`: _AgenticView suppresses its own turn events by design (they would be
    phantoms on top of the inner client's real request), so the inner client emits *both*
    halves here. Drop `_inner_client` and the run reports only its own RunStarted/RunFinished."""
    from aimu.agents.agent import Agent

    inner = RecordingModelClient(["final"])
    inner.model.supports_tools = False
    wrapper = Agent(inner).as_model_client()

    seen = _run_with_scoped_sink(wrapper)

    _assert_both_halves(seen)
    assert [type(e).__name__ for e in seen] == [
        "RunStarted",
        "ModelTurnStarted",
        "RequestPrepared",
        "ModelTurnFinished",
        "RunFinished",
    ]


# ---------------------------------------------------------------------------
# Agent-loop events: RunStarted / RunFinished / ToolCalled / ToolDenied.
#
# The client reports turns; only the agentic loop knows that a run started, which tools
# were dispatched with what, and which a policy refused. `events=` is an Agent field with
# a per-run `run(events=...)` override, following `deps` / `tool_approval` / `thinking`.
# ---------------------------------------------------------------------------


def test_agent_run_emits_run_and_tool_events():
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    seen = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    client = MockModelClient([{"tool": "add", "arguments": {"a": 2, "b": 3}}, "5 is the answer"])
    agent = Agent(client, tools=[add], events=seen.append)
    agent.run("add 2 and 3")

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    called = next(e for e in seen if isinstance(e, ToolCalled))
    assert called.name == "add" and called.arguments == {"a": 2, "b": 3}
    assert called.result == "5"


def test_structured_run_emits_attributed_bracketed_events():
    """Agent.run(schema=...) is still a run: it must open and close a span, and the client's
    own events must be attributed to the agent rather than arriving orphaned."""
    from dataclasses import dataclass

    from aimu.agents import Agent

    @dataclass
    class Out:
        x: int

    class Recording(MockModelClient):
        """A provider records its request payload; the structured path is no exception, and
        that RequestPrepared is what used to arrive orphaned."""

        def _chat(self, *args, **kwargs):
            self._record_request({"messages": list(self.messages)})
            return super()._chat(*args, **kwargs)

    client = Recording(['{"x": 5}'])
    client.model.supports_structured_output = False  # parse path
    seen = []
    agent = Agent(client, name="critic", events=seen.append)

    result = agent.run("verdict?", schema=Out)
    assert isinstance(result, Out)

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    assert "RequestPrepared" in kinds, kinds
    assert all(e.agent == "critic" for e in seen), kinds
    assert next(e for e in seen if isinstance(e, RunFinished)).error is None


def test_structured_run_reports_the_error_when_it_raises():
    from dataclasses import dataclass

    from aimu.agents import Agent

    @dataclass
    class Out:
        x: int

    class Boom(MockModelClient):
        def _chat(self, *args, **kwargs):
            raise RuntimeError("nope")

    seen = []
    agent = Agent(Boom([]), name="critic", events=seen.append)
    with pytest.raises(RuntimeError):
        agent.run("verdict?", schema=Out)

    finished = [e for e in seen if isinstance(e, RunFinished)]
    assert len(finished) == 1
    assert isinstance(finished[0].error, RuntimeError)


def test_streamed_structured_run_emits_attributed_bracketed_events():
    from dataclasses import dataclass

    from aimu.agents import Agent

    @dataclass
    class Out:
        x: int

    class Recording(MockModelClient):
        def _chat(self, *args, **kwargs):
            self._record_request({"messages": list(self.messages)})
            return super()._chat(*args, **kwargs)

    client = Recording(['{"x": 5}'])
    client.model.supports_structured_output = False
    seen = []
    agent = Agent(client, name="critic", events=seen.append)

    list(agent.run("verdict?", stream=True, schema=Out))

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    assert "RequestPrepared" in kinds, kinds
    assert all(e.agent == "critic" for e in seen), kinds


def test_events_is_a_per_run_override():
    """Mirrors deps / tool_approval / thinking: None uses the field."""
    from aimu.agents.agent import Agent

    field_seen = []
    override_seen = []
    client = MockModelClient(["answer"])
    agent = Agent(client, events=field_seen.append)

    agent.run("q", events=override_seen.append)
    assert override_seen  # the override sink saw the run
    assert not field_seen  # the field sink did not: the override replaced it

    client2 = MockModelClient(["answer"])
    agent2 = Agent(client2, events=field_seen.append)
    agent2.run("q")  # no override -> None -> falls back to self.events
    assert field_seen


def test_tool_events_carry_the_agent_name_and_iteration():
    """So a sink can attribute events inside a nested workflow."""
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    seen = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    client = MockModelClient([{"tool": "add", "arguments": {"a": 2, "b": 3}}, "done"])
    agent = Agent(client, tools=[add], name="alpha", events=seen.append)
    agent.run("add 2 and 3")

    called = next(e for e in seen if isinstance(e, ToolCalled))
    assert called.agent == "alpha"
    assert called.iteration == 0
    started = next(e for e in seen if isinstance(e, RunStarted))
    assert started.agent == "alpha"


def test_turn_events_are_attributed_to_the_named_agent():
    """The client emits its own turn events (ModelTurnStarted/ModelTurnFinished/
    RequestPrepared) with no idea which agent or which round it is serving. The loop
    must stamp both onto those events on the way through, not just onto the ones it
    emits itself -- otherwise a sink watching a nested workflow (e.g. three Parallel
    workers) cannot tell whose turn cost what."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockModelClient(["tool", "after tool"])
    agent = Agent(client, name="my-agent", events=seen.append)
    agent.run("go")

    assert len(seen) >= 4  # RunStarted, 2x(ModelTurnStarted+ModelTurnFinished), RunFinished
    assert all(e.agent == "my-agent" for e in seen)

    turn_events = [e for e in seen if isinstance(e, (ModelTurnStarted, ModelTurnFinished))]
    assert len(turn_events) == 4  # two full model turns in this tool-calling run
    iterations = [e.iteration for e in turn_events]
    # The first turn's pair is round 0, the follow-up turn's pair is round 1: iteration
    # must advance across the run, not stay pinned at the client's own default of 0.
    assert iterations == [0, 0, 1, 1]


def test_run_finished_iteration_matches_the_forced_wrap_up_turn():
    """RunFinished.iteration must land on the round the forced wrap-up actually ran in, not
    one behind it. The wrap-up is one more real model call issued after the round-cap while
    loop exits, so a naive iteration bookkeeping variable that only updates inside that loop
    is stale by exactly one turn once the wrap-up fires -- this pins the fix."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished, RunFinished

    seen = []
    client = MockModelClient(["tool", "tool", "final answer"])
    agent = Agent(client, max_iterations=2, events=seen.append)
    result = agent.run("go")

    assert result == "final answer"
    turn_finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(turn_finished) == 3  # initial tool turn, second tool turn, forced wrap-up turn
    finished = next(e for e in seen if isinstance(e, RunFinished))
    assert finished.iteration == turn_finished[-1].iteration == 2


def test_concurrent_tool_dispatch_emits_exactly_one_tool_called_each():
    """Sync mirror of the async ThreadPoolExecutor concurrency test: the sync engine
    dispatches concurrent tool calls under ThreadPoolExecutor too, and each must emit
    exactly one ToolCalled -- asserting the count (not mere presence) catches a sink
    firing twice for one dispatched call, which a presence check would miss."""
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    @tool
    def slow_add(a: int, b: int) -> int:
        """Add two numbers slowly."""
        import time

        time.sleep(0.01)
        return a + b

    seen = []
    client = MockModelClient(
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

    assert agent.run("add two pairs") == "done"

    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 2
    results = sorted(c.result for c in called)
    assert results == ["3", "7"]


def test_concurrent_tool_dispatch_does_not_leak_the_run_sink_to_a_different_client():
    """A *different* client called from inside a tool never receives the outer run's scoped
    sink, on either surface and regardless of ``concurrent_tool_calls`` -- the ContextVar
    override is scoped to the client family it was installed for (``_client_family`` /
    ``_effective_sink`` in ``aimu.models._internal.chat_state``), not to "any client called
    while the scope is open". This is the fix for a real regression an earlier version of this
    override shipped with: a bare ``ContextVar.get()`` with no family check reached *any*
    client invoked during the run, so a tool building a fresh client (the
    ``make_subagent_tool`` shape) had its own turns silently folded into the parent's sink and
    stamped with the parent's agent name, while its own ``RunStarted``/``RunFinished`` never
    appeared at all. See the async mirror,
    ``test_aio_events.py::test_async_concurrent_tool_dispatch_does_not_leak_the_run_sink_to_a_different_client``,
    and contrast with ``test_concurrent_tool_dispatch_of_the_same_client_is_invisible_under_threadpool``
    below, where the client *is* the same object and the two surfaces genuinely differ.
    Two tool calls are required to actually exercise ``ThreadPoolExecutor``: a single call
    takes the sequential fallback and runs on the same thread, which would not test this.
    """
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    inner_client = MockModelClient(["inner reply a"])
    other_inner_client = MockModelClient(["inner reply b"])

    @tool
    def call_inner_a() -> str:
        """Call a separate client's own chat(), relying on the ambient sink."""
        return inner_client.chat("inner task a")

    @tool
    def call_inner_b() -> str:
        """Call a separate client's own chat(), relying on the ambient sink."""
        return other_inner_client.chat("inner task b")

    for concurrent in (False, True):
        inner_client._call_count = 0
        other_inner_client._call_count = 0
        seen = []
        outer_client = MockModelClient(
            [{"tools": [{"name": "call_inner_a"}, {"name": "call_inner_b"}]}, "final answer"]
        )
        agent = Agent(
            outer_client, tools=[call_inner_a, call_inner_b], concurrent_tool_calls=concurrent, events=seen.append
        )
        agent.run("do it")

        # Only the outer client's own turns are reported: 1 RunStarted + 2 ModelTurnStarted +
        # 2 ModelTurnFinished + 2 ToolCalled + 1 RunFinished = 8, whether dispatch is
        # sequential or concurrent. If the inner clients' turns leaked in, this would be 12
        # (their own ModelTurnStarted/Finished pairs included).
        assert len(seen) == 8, (concurrent, [type(e).__name__ for e in seen])
        assert all(getattr(e, "agent", None) in (None, agent.name) for e in seen)
        # The inner clients' own `events` attribute (their durable setting) was never touched.
        assert inner_client.events is None
        assert other_inner_client.events is None


def test_concurrent_tool_dispatch_of_the_same_client_is_invisible_under_threadpool():
    """The one case that genuinely differs by surface: a tool that calls the *same* client
    the run's override was installed for (the ordinary case -- e.g. a tool that reuses
    ``ctx.deps`` holding that client). Sequentially dispatched (``concurrent_tool_calls=False``,
    the default), the reentrant call runs in the same execution context and sees the override.
    Concurrently dispatched, it runs in a fresh ``ThreadPoolExecutor`` thread with an empty
    ``Context`` -- the override does not reach it even though the client matches the family --
    so its turn events are invisible (not misattributed, just absent) from the run's sink. This
    is not a leak (nothing goes to the wrong place) and not new (the pre-fix implementation had
    the identical gap for the same reason: a plain thread has no shared attribute mutation to
    read either). See the async mirror,
    ``test_aio_events.py::test_async_concurrent_tool_dispatch_of_the_same_client_is_visible_under_taskgroup``,
    where ``asyncio.TaskGroup.create_task`` copies the context and both dispatch modes see it.
    """
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    def make_agent(concurrent):
        outer = MockModelClient(
            [
                {"tools": [{"name": "reentrant_a"}, {"name": "reentrant_b"}]},
                "reentrant-a-reply",
                "reentrant-b-reply",
                "final answer",
            ]
        )

        @tool
        def reentrant_a() -> str:
            return outer.chat("extra turn a", use_tools=False)

        @tool
        def reentrant_b() -> str:
            return outer.chat("extra turn b", use_tools=False)

        seen = []
        agent = Agent(outer, tools=[reentrant_a, reentrant_b], concurrent_tool_calls=concurrent, events=seen.append)
        return agent, seen

    sequential_agent, sequential_seen = make_agent(concurrent=False)
    sequential_agent.run("do it")
    # 1 RunStarted + 2 outer turns + 2 reentrant turns + 2 ToolCalled + 1 RunFinished = 12.
    assert len(sequential_seen) == 12, [type(e).__name__ for e in sequential_seen]

    concurrent_agent, concurrent_seen = make_agent(concurrent=True)
    concurrent_agent.run("do it")
    # The 2 reentrant turns' events are missing: 1 RunStarted + 2 outer turns + 2 ToolCalled +
    # 1 RunFinished = 8.
    assert len(concurrent_seen) == 8, [type(e).__name__ for e in concurrent_seen]


def test_scoped_override_wins_over_a_durable_events_attribute():
    """``Agent.run``'s scoped sink (``_events_override``) must win over whatever
    ``client.events`` durably holds, since the whole point of the scoped mechanism is that it
    is checked *before* falling back to the attribute (see ``_effective_sink``)."""
    from aimu.agents.agent import Agent

    durable_seen = []
    scoped_seen = []
    client = MockModelClient(["reply"])
    client.events = durable_seen.append  # a durable sink set directly on the client

    agent = Agent(client, events=scoped_seen.append)  # the scoped per-run override
    agent.run("hi")

    assert durable_seen == []
    assert len(scoped_seen) > 0


def test_a_client_given_its_own_explicit_sink_inside_a_tool_receives_its_events():
    """The documented remedy -- pass ``events=`` explicitly to a client built inside a tool --
    actually works: its own turns land in its own sink, not the outer run's, and the outer
    run's sink never sees them either. This is the positive twin of
    ``test_concurrent_tool_dispatch_does_not_leak_the_run_sink_to_a_different_client``, which
    only asserts the *absence* of a leak; this asserts the explicit sink is *not* itself
    starved by the outer override's ``or`` short-circuit (the exact way Critical 2 failed
    before ``_effective_sink`` checked family membership)."""
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    inner_seen = []

    @tool
    def spawn_sub() -> str:
        """Build a fresh client with its own explicit sink, as the docs recommend."""
        sub_client = MockModelClient(["sub reply"])
        sub_client.events = inner_seen.append
        return sub_client.chat("sub task")

    outer_seen = []
    outer = MockModelClient([{"tool": "spawn_sub"}, "final answer"])
    agent = Agent(outer, tools=[spawn_sub], events=outer_seen.append)
    agent.run("do it")

    assert any(type(e).__name__ == "ModelTurnStarted" for e in inner_seen)
    assert not any(type(e).__name__ in ("RunStarted", "RunFinished") for e in inner_seen)
    assert inner_seen and all(e not in outer_seen for e in inner_seen)


def test_a_hallucinated_tool_name_emits_ToolCalled_with_the_error():
    """A model inventing a tool name is exactly what a sink wants to see. The transcript
    records the "not found" tool message; telemetry must not stay silent about it."""
    from aimu.agents import Agent

    client = MockModelClient([{"tool": "no_such_tool", "arguments": {"a": 1}}, "sorry"])
    seen = []
    agent = Agent(client, name="a", tools=[], events=seen.append, max_iterations=3)
    agent.run("go")

    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 1
    assert called[0].name == "no_such_tool"
    assert called[0].arguments == {"a": 1}
    assert called[0].error == "Tool 'no_such_tool' not found."
    assert called[0].result == "Tool 'no_such_tool' not found."


def test_a_hallucinated_tool_name_emits_ToolCalled_when_streaming():
    from aimu.agents import Agent

    client = MockModelClient([{"tool": "no_such_tool", "arguments": {}}, "sorry"])
    seen = []
    agent = Agent(client, name="a", tools=[], events=seen.append, max_iterations=3)
    list(agent.run("go", stream=True))

    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 1
    assert called[0].error == "Tool 'no_such_tool' not found."


def test_a_denied_tool_emits_ToolDenied_not_ToolCalled():
    """Gate a tool with a refusing approval policy and assert the event pair."""
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    ran = []

    @tool
    def danger() -> str:
        """Risky."""
        ran.append(1)
        return "ran"

    seen = []
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[danger], tool_approval=lambda name, arguments: False, events=seen.append)

    assert agent.run("go") == "done"

    assert ran == []
    assert not any(isinstance(e, ToolCalled) for e in seen)
    denied = next(e for e in seen if isinstance(e, ToolDenied))
    assert denied.name == "danger"


def test_streaming_tool_emits_tool_called():
    """A generator (streaming) @tool is dispatched by a different code path than a plain
    tool (_dispatch_streamed's own branch, not _call_plain_tool); it must still emit
    exactly one ToolCalled."""
    from aimu.agents.agent import Agent
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.tools import tool

    @tool
    def progress_tool(x: int) -> str:
        """A streaming tool that reports progress before returning."""
        yield StreamChunk(StreamingContentType.GENERATING, "working")
        return str(x * 2)

    seen = []
    client = MockModelClient([{"tool": "progress_tool", "arguments": {"x": 3}}, "6 is the answer"])
    agent = Agent(client, tools=[progress_tool], events=seen.append)
    list(agent.run("double 3", stream=True))

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    called = [e for e in seen if isinstance(e, ToolCalled)]
    assert len(called) == 1
    assert called[0].name == "progress_tool"
    assert called[0].result == "6"


def test_streamed_run_via_agent_events_emits_turn_finished_with_usage():
    """Sync twin of ``test_aio_events.py
    ::test_async_streamed_run_via_agent_events_emits_turn_finished_with_usage`` -- a direct
    test of ``_emit_when_drained`` (``aimu/models/_base/text.py``) via the scoped-override
    path (``Agent.run(events=..., stream=True)``), rather than the durable ``client.events``
    attribute every other streamed-events test in this file uses (which cannot distinguish
    ``_effective_sink(self)`` from a bare ``getattr`` when no scoped override is active)."""
    from aimu.agents.agent import Agent
    from aimu.events import ModelTurnFinished

    client = MockModelClient(["streamed answer"])
    client.model.supports_tools = False
    client.last_usage = {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}

    seen = []
    agent = Agent(client, events=seen.append)

    stream = agent.run("hi", stream=True)
    chunks = list(stream)
    assert chunks

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].usage == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


def test_run_finished_carries_the_error_when_a_run_raises():
    """A run that raises must still report -- emit from a finally."""
    from aimu.agents.agent import Agent
    from aimu.tools import tool

    @tool
    async def bad_tool(x: int) -> int:
        """An async tool the sync Agent cannot dispatch."""
        return x

    seen = []
    client = MockModelClient([{"tool": "bad_tool", "arguments": {"x": 1}}])
    agent = Agent(client, tools=[bad_tool], events=seen.append)

    with pytest.raises(ValueError):
        agent.run("go")

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"
    finished = next(e for e in seen if isinstance(e, RunFinished))
    assert isinstance(finished.error, ValueError)


def test_skill_agent_run_emits_run_and_tool_events(tmp_path):
    """SkillAgent inherits Agent.run() / Agent._make_tool_loop() unmodified, so events=
    threading requires no SkillAgent-specific code -- confirmed here rather than assumed."""
    from aimu.agents.skill_agent import SkillAgent
    from aimu.skills.manager import SkillManager

    seen = []
    client = MockModelClient(["answer"])
    manager = SkillManager(skill_dirs=[str(tmp_path)])  # empty dir: no skills discovered
    agent = SkillAgent(client, skill_manager=manager, events=seen.append)
    agent.run("q")

    kinds = [type(e).__name__ for e in seen]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunFinished"


def test_fallback_client_streamed_chat_emits_exactly_one_pair():
    from aimu.events import ModelTurnFinished, ModelTurnStarted
    from aimu.models.fallback import FallbackClient

    primary = MockModelClient(["streamed"])
    fc = FallbackClient([primary])
    seen = []
    fc.events = seen.append

    chunks = list(fc.chat("hi", stream=True))
    assert chunks

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


# ---------------------------------------------------------------------------
# Workflow factory forwarding: `events=` on a `from_client()` factory reaches every
# Agent/SkillAgent it constructs, so one sink can see a whole pipeline attributed by step.
#
# `EvaluatorOptimizer` has no `from_client()` factory (its generator/evaluator are always
# supplied pre-built by the caller), so there is nothing for this workflow to forward: the
# caller already holds the Agent instances and can pass `events=` to them directly.
#
# `Parallel.from_client()`'s reliable-concurrent-delivery case has its own dedicated test
# exercising a real concurrent run: `test_workflow_parallel.py
# ::test_parallel_from_client_shared_events_sink_survives_concurrent_workers` (async mirror
# in `test_aio_workflow_parallel.py`). Those tests replaced
# `test_KNOWN_GAP_parallel_from_client_shared_events_sink_drops_events`, which pinned the
# pre-fix bug (a scoped sink delivered by mutating the client's `events` attribute, clobbered
# by overlapping concurrent workers). The test below only checks that the sink is wired onto
# each worker/aggregator Agent at construction time; it doesn't need a real concurrent run to
# prove that.
# ---------------------------------------------------------------------------


def test_orchestrator_assemble_forwards_the_sink_to_the_inner_agent():
    """The inner orchestrator Agent is private, so without events= on assemble() the
    flagship autonomous class (and every prebuilt orchestrator) was unobservable."""
    from aimu.agents import Agent, OrchestratorAgent

    worker = Agent(MockModelClient(["worker output"]), "Do the work.", name="worker")
    client = MockModelClient(["orchestrated answer"])
    seen = []
    orch = OrchestratorAgent.assemble(client, "Use the worker.", workers=[worker], events=seen.append)

    assert orch.run("task") == "orchestrated answer"
    kinds = [type(e).__name__ for e in seen]
    assert "RunStarted" in kinds and "RunFinished" in kinds
    assert any(isinstance(e, ModelTurnStarted) for e in seen)
    assert all(e.agent == "orchestrator" for e in seen), kinds


def test_prebuilt_orchestrator_forwards_the_sink(monkeypatch):
    """All three prebuilt orchestrators were unobservable for the same reason; one stands
    for the set, since they share _init_orchestrator."""
    from aimu.agents.prebuilt import CodeReviewAgent, _base

    monkeypatch.setattr(_base, "ModelClient", lambda model: MockModelClient(["worker output"]))

    seen = []
    agent = CodeReviewAgent(MockModelClient(["review"]), events=seen.append)
    agent.run("some code")

    assert any(isinstance(e, RunStarted) for e in seen)
    assert all(e.agent == "code-review-agent" for e in seen)


def test_chain_forwards_the_sink_to_every_step():
    """One sink sees the whole pipeline, with each event attributed to its step."""
    from aimu.agents import Chain

    seen = []
    chain = Chain.from_client(
        MockModelClient(["step one output", "step two output"]), ["step one", "step two"], events=seen.append
    )
    chain.run("go")
    agents = {e.agent for e in seen if e.agent}
    assert len(agents) == 2, f"expected both steps to be attributed, got {agents}"


def test_router_from_client_forwards_events_to_the_classifier():
    """`Router.from_client()` only constructs the classifier Agent itself; handlers are
    supplied by the caller already built, so only the classifier is asserted here."""
    from aimu.agents import Agent, Router

    seen = []
    sink = seen.append
    client = MockModelClient(["classified-as-a"])
    router = Router.from_client(
        client,
        classifier_prompt="classify",
        handlers={"classified-as-a": Agent(MockModelClient(["handler output"]), name="handler")},
        events=sink,
    )
    assert router.routing_agent.events is sink


def test_parallel_from_client_wires_events_onto_every_worker_and_the_aggregator():
    """Construction-time forwarding only -- see the module-level note above on why a real
    concurrent run is not exercised here."""
    from aimu.agents import Parallel

    seen = []
    sink = seen.append
    client = MockModelClient([])
    parallel = Parallel.from_client(
        client,
        worker_prompts=["A.", "B."],
        aggregator_prompt="Synthesize.",
        events=sink,
    )
    assert all(worker.events is sink for worker in parallel.workers)
    assert parallel.aggregator.events is sink


def test_plan_execute_evaluator_from_client_forwards_events_to_planner_and_executor():
    """The planner and executor run sequentially (no shared-client concurrency hazard), so
    this exercises a real run, not just construction-time wiring."""
    from aimu.agents import PlanExecuteEvaluator

    seen = []
    # planner round 1, executor round 1, judge round 1 ("8" -> LLMJudgeScorer parses 0.8 -> pass).
    client = MockModelClient(["plan it", "did it", "8"])
    wf = PlanExecuteEvaluator.from_client(client, criteria="answer the task", events=seen.append)
    wf.run("hi")

    agents = {e.agent for e in seen if e.agent}
    assert agents == {"planner", "executor"}


def test_two_streamed_runs_drained_out_of_lifo_order_leave_no_stale_sink():
    """Two streamed runs open on one execution context and drained out of LIFO order must
    each still report to their own sink *and* leave nothing behind afterwards.

    Independent generators may legally be drained in any order, so the ``_events_override``
    scopes close out of order. Restoring the ContextVar with a ``Token`` cannot survive that:
    ``reset()`` restores ``token.old_value``, so tearing down the *first* scope silently drops
    the second, and the var ends up stuck holding the first run's payload. A later,
    completely un-instrumented ``chat()`` on that run's client then reported to its
    abandoned sink. Teardown flips ``_EventScope.active`` and rebuilds the stack instead;
    see ``_ACTIVE_EVENT_SINK`` in ``aimu.models._internal.chat_state``."""
    from aimu.agents.agent import Agent
    from aimu.models._internal.chat_state import _ACTIVE_EVENT_SINK

    seen_a, seen_b = [], []
    client_a, client_b = MockModelClient(["answer A"]), MockModelClient(["answer B"])
    client_a.model.supports_tools = False
    client_b.model.supports_tools = False

    stream_a = Agent(client_a, "sys").run("task A", stream=True, events=seen_a.append)
    stream_b = Agent(client_b, "sys").run("task B", stream=True, events=seen_b.append)
    next(stream_a)  # opens A's scope
    next(stream_b)  # opens B's scope, nested inside A's
    list(stream_a)  # closes A's scope first -- out of LIFO order
    list(stream_b)

    assert "ModelTurnStarted" in [type(e).__name__ for e in seen_a]
    assert "ModelTurnStarted" in [type(e).__name__ for e in seen_b]

    seen_a.clear()
    seen_b.clear()
    client_a.reset()
    client_a._responses, client_a._call_count = ["later"], 0
    client_a.chat("un-instrumented")

    assert seen_a == []
    assert seen_b == []
    # Checked last, and separately from the behaviour above: no *active* scope survives either.
    assert [scope for scope in _ACTIVE_EVENT_SINK.get() if scope.active] == []


def test_abandoning_a_streamed_run_leaves_no_stale_sink():
    """A sync streamed run whose consumer stops early must tear its scope down cleanly.

    The generator is finalized on this same thread, so this shape was already correct; it is
    pinned as the sync twin of the async abandon test, which was not."""
    from aimu.agents.agent import Agent

    seen = []
    client = MockModelClient(["a longer streamed answer"])
    client.model.supports_tools = False

    stream = Agent(client, "sys").run("task", stream=True, events=seen.append)
    next(stream)
    stream.close()

    seen.clear()
    client.reset()
    client._responses, client._call_count = ["later"], 0
    client.chat("un-instrumented")

    assert seen == []

