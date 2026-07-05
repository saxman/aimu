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
from aimu.models._internal.message_meta import PROVENANCE_FINAL_ANSWER
from aimu.models.base import StreamChunk

from ._base import AsyncBaseModelClient

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
    ) -> Union[str, Any, AsyncIterator[StreamChunk]]:
        """Run the async agentic loop. ``images`` attach only to the initial turn.

        ``tools`` is a per-run override of the agent's configured ``self.tools``; ``deps`` is a
        per-run override of the agent's ``self.deps`` (injected as ``ctx.deps`` into tools that
        declare a :class:`~aimu.tools.ToolContext` parameter); ``tool_approval`` is a per-run
        override of ``self.tool_approval`` (the gate run before each tool call, ``(name, arguments)
        -> bool``, which may be a coroutine; deny appends a refusal tool message); ``schema`` makes
        the run a single structured-output turn returning a validated instance. See the sync
        :meth:`aimu.agents.Agent.run` for full semantics.
        """
        if schema is not None:
            if stream:
                return self._run_structured_streamed(task, generate_kwargs, images, deps, tool_approval, schema)
            self._prepare_run(deps, tool_approval)
            try:
                return await self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, schema=schema)
            finally:
                self._last_messages = list(self.model_client.messages)
        if stream:
            return self._run_streamed(
                task, generate_kwargs, images=images, tools=tools, deps=deps, tool_approval=tool_approval
            )
        self._prepare_run(deps, tool_approval)
        return await self._run_loop(task, generate_kwargs, images=images, tools=tools)

    async def _run_loop(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
    ) -> str:
        """The non-streamed tool-calling loop, assuming ``_prepare_run`` already ran.

        Shared by :meth:`run` and :class:`aimu.aio.SkillAgent` (which must run its async
        skill setup between ``_prepare_run`` and the loop).

        The ``messages`` snapshot is taken in a ``finally`` so a cancelled run (an app cancelling
        the task, e.g. via :class:`~aimu.aio.RunHandle`) still records its partial turn for resume.
        """
        try:
            # Turn 1 executes any requested tools and returns. Continue (chat() with no user
            # message → a turn on the current messages) until a turn makes no tool calls.
            response = await self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, tools=tools)
            iteration = 1
            while self._last_turn_called_tools() and iteration < self.max_iterations:
                logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
                response = await self.model_client.chat(generate_kwargs=generate_kwargs, tools=tools)
                iteration += 1

            if self.final_answer_prompt is not None and self._last_turn_called_tools():
                logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
                injected_at = len(self.model_client.messages)
                response = await self.model_client.chat(
                    self.final_answer_prompt, generate_kwargs=generate_kwargs, tools=[]
                )
                self._tag_injected_turn(injected_at, PROVENANCE_FINAL_ANSWER)

            return response
        finally:
            self._last_messages = list(self.model_client.messages)

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
        deps: Optional[Any] = None,
        tool_approval: Optional[Callable] = None,
    ) -> AsyncIterator[StreamChunk]:
        self._prepare_run(deps, tool_approval)
        async for chunk in self._run_loop_streamed(task, generate_kwargs, images=images, tools=tools):
            yield chunk

    async def _run_structured_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        deps: Optional[Any],
        tool_approval: Optional[Callable],
        schema: type,
    ) -> AsyncIterator[StreamChunk]:
        """Single structured-output turn, streamed (async). Forwards the client's chunks tagged
        with this agent's name; snapshots ``_last_messages`` in a ``finally`` for cancel-safe resume."""
        self._prepare_run(deps, tool_approval)
        try:
            stream = await self.model_client.chat(
                task, generate_kwargs=generate_kwargs, stream=True, images=images, schema=schema
            )
            async for chunk in stream:
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=0)
        finally:
            self._last_messages = list(self.model_client.messages)

    async def _run_loop_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """The streamed tool-calling loop, assuming ``_prepare_run`` already ran.

        Shared by :meth:`_run_streamed` and :class:`aimu.aio.SkillAgent`. The ``messages`` snapshot
        is taken in a ``finally`` (which runs when a cancelled consumer closes this generator), so a
        cancelled streamed run still records its partial turn for resume.
        """
        try:
            iteration = 0
            first_stream = await self.model_client.chat(
                task, generate_kwargs=generate_kwargs, stream=True, images=images, tools=tools
            )
            async for chunk in first_stream:
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)

            iteration += 1
            while self._last_turn_called_tools() and iteration < self.max_iterations:
                logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
                # chat() with no user message → a turn on the current messages (tool results).
                next_stream = await self.model_client.chat(generate_kwargs=generate_kwargs, stream=True, tools=tools)
                async for chunk in next_stream:
                    yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
                iteration += 1

            if self.final_answer_prompt is not None and self._last_turn_called_tools():
                logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
                injected_at = len(self.model_client.messages)
                final_stream = await self.model_client.chat(
                    self.final_answer_prompt, generate_kwargs=generate_kwargs, stream=True, tools=[]
                )
                async for chunk in final_stream:
                    yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
                self._tag_injected_turn(injected_at, PROVENANCE_FINAL_ANSWER)
                iteration += 1
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
        )
