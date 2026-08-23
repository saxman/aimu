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
