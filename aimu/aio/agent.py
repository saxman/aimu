"""Async ``AsyncRunner`` ABC and ``Agent`` class.

Mirrors :mod:`aimu.agents.base` and :mod:`aimu.agents.agent` but with ``async def run()``
and ``AsyncIterator[StreamChunk]`` streaming.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Union

from aimu.agents._loop import _AgentLoopMixin
from aimu.agents.base import MessageHistory
from aimu.events import EventSink
from aimu.models.base import StreamChunk

from ._base import AsyncBaseModelClient
from ._tool_loop import _AsyncToolLoop

logger = logging.getLogger(__name__)

DEFAULT_CONTINUATION_PROMPT = (
    "Continue working on the task using available tools as needed. If you have the answer "
    "and don't need to use any more tools, just provide the final response."
)


class AsyncRunner(ABC):
    """Abstract base for every concrete async agent and workflow."""

    @abstractmethod
    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Run asynchronously (``stream=False``) or streaming (``stream=True``)."""
        ...

    @property
    @abstractmethod
    def messages(self) -> MessageHistory:
        """Message histories of all sub-runners, keyed by runner name."""
        ...

    def as_tool(self, *, name: Optional[str] = None, description: Optional[str] = None) -> Callable:
        """Wrap this async runner as an async ``@tool``-style callable: ``await tool(task)``.

        Async mirror of :meth:`aimu.agents.base.Runner.as_tool`. The returned callable is an
        ``async def`` delegating to ``await self.run(task)``, so the ``@tool`` decorator marks
        it ``__tool_is_async__ = True`` and the async agent loop awaits it directly.
        """
        from aimu.agents._as_tool import build_as_tool

        async def _dispatch(task: str) -> str:
            return await self.run(task)

        return build_as_tool(self, _dispatch, name=name, description=description)


@dataclass
class Agent(_AgentLoopMixin, AsyncRunner):
    """Async equivalent of :class:`aimu.agents.Agent`.

    Calls ``await model_client.chat()`` repeatedly until the model produces a turn
    without invoking tools, or ``max_iterations`` is reached.

    Quick start::

        from aimu.tools import tool
        from aimu import aio

        @tool
        async def fetch(url: str) -> str:
            \"\"\"Fetch the contents of a URL.\"\"\"
            import httpx
            async with httpx.AsyncClient() as c:
                return (await c.get(url)).text[:500]

        client = aio.client("anthropic:claude-sonnet-4-6")
        agent = aio.Agent(client, "You are a helpful assistant.", tools=[fetch])
        print(await agent.run("Fetch example.com"))
    """

    model_client: AsyncBaseModelClient
    system_message: Optional[str] = None
    name: Optional[str] = None
    tools: list[Callable] = field(default_factory=list)
    max_iterations: int = 10
    continuation_prompt: str = field(default=DEFAULT_CONTINUATION_PROMPT)
    reset_messages_on_run: bool = False
    final_answer_prompt: Optional[str] = None
    deps: Optional[Any] = None
    tool_approval: Optional[Callable] = None
    thinking: Optional[Union[bool, str]] = None
    # Delivered by mutating model_client.events for the run's duration (see AsyncBaseModelClient.
    # events); not safe if this agent's model_client is shared with another concurrently running
    # agent (e.g. every worker Agent in a Parallel built via Parallel.from_client) -- events will
    # drop and misorder. Give each concurrently-running agent its own model_client if that matters.
    events: Optional[EventSink] = None
    # None (default): the loop never rewrites model_client.messages on its own -- an agent that
    # doesn't opt in behaves exactly as it did before this field existed. Set to a callable such
    # as ``lambda msgs: trim_messages(msgs, max_tokens=8000)`` or
    # ``lambda msgs: summarize_messages(client, msgs)`` (see aimu.context) to compact automatically
    # before every model turn. An *applied* compaction (one that actually dropped a message) is
    # never silent: it emits a ContextCompacted event for a caller with a sink attached, and logs
    # a WARNING unconditionally, so a caller without one still learns their conversation was
    # rewritten (see the sync aimu.agents.Agent.compaction docstring for the full rationale). A
    # compaction that returns the conversation unchanged is a no-op and announces nothing.
    compaction: Optional[Callable[[list[dict]], list[dict]]] = None
    concurrent_tool_calls: bool = False
    _last_messages: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"agent-{id(self) & 0xFFFFFF:06x}"

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
        deps: Optional[Any] = None,
        tool_approval: Optional[Callable] = None,
        schema: Optional[type] = None,
        thinking: Optional[Union[bool, str]] = None,
        events: Optional[EventSink] = None,
        compaction: Optional[Callable[[list[dict]], list[dict]]] = None,
    ) -> Union[str, Any, AsyncIterator[StreamChunk]]:
        """Run the async agentic loop. ``images`` attach only to the initial turn.

        ``tools`` is a per-run override of the agent's configured ``self.tools``; ``deps`` is a
        per-run override of the agent's ``self.deps`` (injected as ``ctx.deps`` into tools that
        declare a :class:`~aimu.tools.ToolContext` parameter); ``tool_approval`` is a per-run
        override of ``self.tool_approval`` (the gate run before each tool call, ``(name, arguments)
        -> bool``, which may be a coroutine; deny appends a refusal tool message); ``schema`` makes
        the run a single structured-output turn returning a validated instance; ``thinking`` is a
        per-run override of ``self.thinking`` (the portable reasoning control), applied to every
        model turn the run makes; ``events`` is a per-run override of ``self.events`` (a callable
        taking one :class:`~aimu.events.RunEvent`), attached to ``model_client.events`` for the
        run's duration. **Not safe across agents that share a ``model_client`` and run
        concurrently** (e.g. every worker ``Agent`` in a :class:`~aimu.agents.Parallel` built via
        ``Parallel.from_client``): attaching a sink mutates shared state rather than passing a
        per-call argument, so concurrent runs will drop and misorder events. Give each
        concurrently-running agent its own ``model_client`` if you need reliable delivery.
        ``compaction`` is a per-run override of ``self.compaction`` (a callable applied to the
        conversation before every model turn the run makes; see :mod:`aimu.context`), not used
        by the ``schema=`` structured-output path. See the sync :meth:`aimu.agents.Agent.run`
        for full semantics.
        """
        thinking = thinking if thinking is not None else self.thinking
        events = events if events is not None else self.events
        compaction = compaction if compaction is not None else self.compaction
        if schema is not None:
            if stream:
                return self._run_structured_streamed(
                    task, generate_kwargs, images, deps, tool_approval, schema, thinking, events
                )
            self._prepare_run(deps, tool_approval)
            try:
                # Raw events (unwrapped by _attributing_sink, unlike the tool-loop call
                # sites): this path emits no turn events today (schema= does not currently
                # call _emit_turn_started/_emit_turn_finished), so there is nothing for the
                # wrapper to attribute yet. Swap in self._attributing_sink() here too if that
                # ever changes.
                with self.model_client._events_override(events):
                    return await self.model_client.chat(
                        task, generate_kwargs=generate_kwargs, images=images, schema=schema, thinking=thinking
                    )
            finally:
                self._last_messages = list(self.model_client.messages)
        self._prepare_run(deps, tool_approval)
        loop = self._make_tool_loop(tools, deps, tool_approval, thinking, events, compaction)
        if stream:
            return self._run_loop_streamed(loop, task, generate_kwargs, images)
        return await self._run_loop(loop, task, generate_kwargs, images)

    def _effective_tools(self, tools: Optional[list[Callable]]) -> list[Callable]:
        """The tool callables for this run: the ``tools=`` override, else ``self.tools``.
        (``SkillAgent`` overrides this to add its discovered skill tools.)"""
        return list(tools) if tools is not None else list(self.tools)

    def _make_tool_loop(
        self,
        tools: Optional[list[Callable]],
        deps: Optional[Any],
        tool_approval: Optional[Callable],
        thinking: Optional[Union[bool, str]] = None,
        events: Optional[EventSink] = None,
        compaction: Optional[Callable[[list[dict]], list[dict]]] = None,
    ) -> _AsyncToolLoop:
        """Build the async iterative tool-calling engine with this run's effective tools + policy."""
        from aimu.tools.approval import approve_all

        return _AsyncToolLoop(
            self.model_client,
            lambda: self._effective_tools(tools),  # re-read each round (SkillAgent may add skill tools mid-run)
            deps=deps if deps is not None else self.deps,
            tool_approval=tool_approval or self.tool_approval or approve_all,
            concurrent_tool_calls=self.concurrent_tool_calls,
            max_rounds=self.max_iterations,
            final_answer_prompt=self.final_answer_prompt,
            continuation_prompt=self.continuation_prompt,
            thinking=thinking,
            events=events if events is not None else self.events,
            agent_name=self.name,
            compaction=compaction if compaction is not None else self.compaction,
        )

    async def _run_loop(
        self,
        loop: _AsyncToolLoop,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> str:
        """Drive the tool-loop engine (``_prepare_run`` + any skill setup already ran). Shared by
        :meth:`run` and :class:`aimu.aio.SkillAgent`. Snapshots ``messages`` in a ``finally`` so a
        cancelled run (e.g. via :class:`~aimu.aio.RunHandle`) still records its partial turn."""
        try:
            return await loop.run(task, generate_kwargs=generate_kwargs, images=images)
        finally:
            self._last_messages = list(self.model_client.messages)

    async def _run_structured_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        deps: Optional[Any],
        tool_approval: Optional[Callable],
        schema: type,
        thinking: Optional[Union[bool, str]] = None,
        events: Optional[EventSink] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Single structured-output turn, streamed (async). Forwards the client's chunks tagged
        with this agent's name; snapshots ``_last_messages`` in a ``finally`` for cancel-safe resume."""
        self._prepare_run(deps, tool_approval)
        try:
            # Raw events (unwrapped by _attributing_sink, unlike the tool-loop call sites):
            # this path emits no turn events today (schema= does not currently call
            # _emit_turn_started/_emit_turn_finished), so there is nothing for the wrapper to
            # attribute yet. Swap in self._attributing_sink() here too if that ever changes.
            with self.model_client._events_override(events):
                stream = await self.model_client.chat(
                    task,
                    generate_kwargs=generate_kwargs,
                    stream=True,
                    images=images,
                    schema=schema,
                    thinking=thinking,
                )
                async for chunk in stream:
                    yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=0)
        finally:
            self._last_messages = list(self.model_client.messages)

    async def _run_loop_streamed(
        self,
        loop: _AsyncToolLoop,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streamed twin of :meth:`_run_loop`. Shared by :meth:`run` and :class:`aimu.aio.SkillAgent`.
        The ``messages`` snapshot is taken in a ``finally`` (which runs when a cancelled consumer
        closes this generator), so a cancelled streamed run still records its partial turn."""
        try:
            async for chunk in loop.run_streamed(task, generate_kwargs=generate_kwargs, images=images):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=chunk.iteration)
        finally:
            self._last_messages = list(self.model_client.messages)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

    def as_model_client(self) -> AsyncBaseModelClient:
        """Return an :class:`AsyncBaseModelClient` view of this agent.

        Each ``await client.chat()`` runs the full agent loop.
        """
        from aimu.aio.agentic_client import _AsyncAgenticView

        return _AsyncAgenticView(self)

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: AsyncBaseModelClient) -> Agent:
        sm = config.get("system_message")
        return cls(
            model_client=model_client,
            system_message=sm,
            name=config.get("name"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
            final_answer_prompt=config.get("final_answer_prompt"),
            thinking=config.get("thinking"),
        )
