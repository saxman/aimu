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
    """A ``Runner`` (an agent or a workflow) began a run.

    Fires once at the top of ``run()``, before the first model turn, and is paired with
    exactly one :class:`RunFinished`. ``task`` is the task string as the caller passed it.
    """

    task: str = ""


@dataclass(frozen=True)
class RunFinished(RunEvent):
    """A ``Runner`` finished, successfully or not.

    Fires in a ``finally``, so it is emitted even when the run raised: ``error`` is the
    exception in that case and ``result`` is ``None``.

    ``result`` is also ``None`` on a **streamed** run (``run(stream=True)``): the chunks go
    to the caller as they arrive and the runner never assembles a final string, so there is
    no result to report. Read the text off the ``GENERATING`` chunks instead.
    """

    result: Optional[str] = None
    error: Optional[BaseException] = None


@dataclass(frozen=True)
class ModelTurnStarted(RunEvent):
    """One model request is about to be issued.

    Fires inside ``chat()`` / ``generate()``, before the provider is called, and is paired
    with exactly one :class:`ModelTurnFinished` even when the request raises.

    ``message_count`` is how many messages this request carries (including the user turn
    it is about to append, and the system message when this turn seeds one); it is ``1``
    for a stateless ``generate()``, which sends only the prompt. ``tool_names`` are the
    tools advertised on this turn, not the ones called.
    """

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
    """The model request that a :class:`ModelTurnStarted` announced has ended.

    Fires for every started turn, including one that failed: ``error`` is then the
    exception (``ContextOverflowError``, ``ModelConnectionError``, a provider 4xx, ...) and
    ``text`` / ``usage`` are ``None``. A sink pairing started/finished can therefore close
    its span unconditionally.

    On a streamed turn this fires when the stream is actually drained (or abandoned), not
    when it was created, so ``usage`` -- which only populates at the end -- is real and
    ``duration_s`` covers the whole stream. ``text`` is the concatenated ``GENERATING``
    content on a streamed turn, and the returned string on a non-streamed one; it is
    ``None`` for a structured (``schema=``) result, which is not a string.
    """

    model: str = ""
    text: Optional[str] = None
    usage: Optional[dict] = None
    duration_s: float = 0.0
    error: Optional[BaseException] = None


@dataclass(frozen=True)
class ToolCalled(RunEvent):
    """A tool the model asked for was dispatched.

    Fires after the call returns, for every outcome the model gets to see: a normal result,
    a tool that raised (``error`` set, ``result`` carrying the message handed back to the
    model), invalid arguments, and a name the model invented that matches no tool. It does
    *not* fire for a call a ``tool_approval`` policy refused -- that is :class:`ToolDenied`.

    ``arguments`` is what the model passed, before AIMU's coercion. With
    ``concurrent_tool_calls=True`` these are emitted from worker threads (sync) or
    concurrent tasks (async): see :func:`emit` on sink thread-safety and ordering.
    """

    name: str = ""
    arguments: dict = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class ToolDenied(RunEvent):
    """A ``tool_approval`` policy refused this call.

    Fires instead of :class:`ToolCalled`, before the tool runs: nothing was executed. The
    model still sees a tool message saying the call was not approved.
    """

    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContextCompacted(RunEvent):
    """Conversation history was rewritten by an ``Agent``'s ``compaction=`` callable.

    Fires once per compaction, after the rewrite, before the next model turn.

    ``dropped`` is the removed messages, not a count, so a caller who wants the discarded
    turns still has them. They are copies, so a later in-place edit of ``client.messages``
    (the provenance tag an agent loop adds, say) cannot mutate an already-emitted event.

    ``before_tokens`` / ``after_tokens`` are AIMU's own default token *estimate* (see
    :func:`aimu.context.count_tokens`), not a measurement of whatever the ``compaction``
    callable itself counted to decide what to drop. A callable that used a real tokenizer,
    a word count, or any other budget will disagree with these numbers -- that is stated
    rather than hidden, since AIMU cannot see inside an opaque callable to know what it
    actually counted. Treat them as rough orientation, not a claim about the number that
    drove the decision.
    """

    dropped: list = field(default_factory=list)
    before_tokens: int = 0
    after_tokens: int = 0


EventSink = Callable[[RunEvent], None]
"""One callable taking one :class:`RunEvent`. Its return value is ignored and an exception
it raises is logged, not propagated (see :func:`emit`). It must be thread-safe: with
``concurrent_tool_calls=True`` tool events arrive from concurrent workers in
nondeterministic order."""


def emit(sink: Optional[EventSink], event: RunEvent) -> None:
    """Deliver ``event`` to ``sink``, logging rather than propagating its failure.

    Observation must not change what it observes: a sink that raises would otherwise break
    a run that was working. Same contract as the ``SubagentObserver`` display hook.

    **A sink must be thread-safe.** With ``concurrent_tool_calls=True`` the tool-loop engine
    dispatches a turn's tool calls from a ``ThreadPoolExecutor`` (sync) or an
    ``asyncio.TaskGroup`` (async), so :class:`ToolCalled` / :class:`ToolDenied` are emitted
    concurrently and their **order is nondeterministic** -- a sink that appends to a plain
    list, writes a file, or accumulates per-tool state needs its own lock. Turn and run
    events are emitted from the calling thread and stay ordered.
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
