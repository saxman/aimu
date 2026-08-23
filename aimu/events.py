"""Run events: AIMU's comprehension surface.

A sink is one callable taking one event. Attach it to a client, an agent, or a workflow and
you see what happened -- which turns ran, what payload actually went to the provider, which
tools were called with what, what was dropped when the conversation was compacted.

Why a dataclass union and one callback, rather than a Protocol of named methods: adding an
event must never break an existing consumer. A Protocol grows a method and every
implementation is suddenly incomplete; a union grows a member and old sinks ignore it. The
same reasoning that keeps conversation state a ``list[dict]`` keeps telemetry plain data.

Events are the *telemetry* channel. ``StreamChunk`` remains the *content* channel -- what
the model produced, for display. Both can be active at once and neither replaces the other:
one says what the model said, the other says what the library did with it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ContextCompacted",
    "EventSink",
    "ModelTurnFinished",
    "ModelTurnStarted",
    "RequestPrepared",
    "RunEvent",
    "RunFinished",
    "RunStarted",
    "ToolCalled",
    "ToolDenied",
    "emit",
    "log_events",
]


@dataclass(frozen=True)
class RunEvent:
    """Base for every event.

    ``agent`` and ``iteration`` mirror :class:`~aimu.models.StreamChunk`'s fields of the same
    name, so one sink can attribute events correctly inside a nested workflow.
    """

    agent: Optional[str] = None
    iteration: int = 0


@dataclass(frozen=True)
class RunStarted(RunEvent):
    task: str = ""


@dataclass(frozen=True)
class RunFinished(RunEvent):
    result: Optional[str] = None
    error: Optional[BaseException] = None


@dataclass(frozen=True)
class ModelTurnStarted(RunEvent):
    model: str = ""
    message_count: int = 0
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestPrepared(RunEvent):
    """The payload as it goes to the provider, after every adaptation AIMU applies.

    This is the event that makes "when a model surprises you, the surprise should be the
    model's" checkable: between a caller's ``chat()`` and the wire sit the four-tier
    generate_kwargs merge, the GENERATE_KWARG_SUPPORT renames and drops, thinking
    resolution, ``strip_inert_keys``, and provider format adaptation. Without this, none of
    it is visible at runtime.

    The payload is the request, unredacted -- it contains whatever the caller put in the
    conversation. Redacting here would reintroduce the hiding this exists to remove; a sink
    that ships events off the machine is the right place to filter.
    """

    provider: str = ""
    model: str = ""
    payload: Any = None


@dataclass(frozen=True)
class ModelTurnFinished(RunEvent):
    model: str = ""
    text: Optional[str] = None
    usage: Optional[dict] = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class ToolCalled(RunEvent):
    name: str = ""
    arguments: dict = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class ToolDenied(RunEvent):
    """A ``tool_approval`` policy refused this call."""

    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContextCompacted(RunEvent):
    """Conversation history was rewritten. ``dropped`` is the removed messages, not a count,
    so a caller who wants the discarded turns still has them."""

    dropped: list = field(default_factory=list)
    before_tokens: int = 0
    after_tokens: int = 0


EventSink = Callable[[RunEvent], None]


def emit(sink: Optional[EventSink], event: RunEvent) -> None:
    """Deliver ``event`` to ``sink``, logging rather than propagating its failure.

    Observation must not change what it observes: a sink that raises would otherwise break
    a run that was working. Same contract as the ``SubagentObserver`` display hook.
    """
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        logger.warning("An event sink raised; continuing the run.", exc_info=True)


def log_events(target: logging.Logger, level: int = logging.INFO) -> EventSink:
    """A sink that writes one line per event.

    The shortest path to the comprehension payoff -- attach it and watch what actually
    happened -- and the sink the docs lead with.
    """

    def sink(event: RunEvent) -> None:
        target.log(level, "%s %s", type(event).__name__, event)

    return sink
