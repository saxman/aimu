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
from tests.helpers import MockModelClient


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
    # events is pushed down by _load_state so the winning attempt is the one that emits.
    assert primary.events is not None


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
