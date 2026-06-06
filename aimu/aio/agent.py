"""Async ``AsyncRunner`` ABC and ``Agent`` class.

Mirrors :mod:`aimu.agents.base` and :mod:`aimu.agents.agent` but with ``async def run()``
and ``AsyncIterator[StreamChunk]`` streaming.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Union

from aimu.agents.base import MessageHistory
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


@dataclass
class Agent(AsyncRunner):
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
    _last_messages: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"agent-{id(self) & 0xFFFFFF:06x}"

    def _prepare_run(self) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self.model_client.reset(system_message=self.system_message)
        if self.tools:
            self.model_client.tools = list(self.tools)

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Run the async agentic loop. ``images`` attach only to the initial turn.

        ``tools`` is a per-run override of the agent's configured ``self.tools`` — see the
        sync :meth:`aimu.agents.Agent.run` for semantics.
        """
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images, tools=tools)
        self._prepare_run()
        response = await self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, tools=tools)

        for _ in range(self.max_iterations - 1):
            if not self._last_turn_called_tools():
                break
            logger.debug("Agent '%s' continuing, tools were used in last turn.", self.name)
            response = await self.model_client.chat(
                self.continuation_prompt, generate_kwargs=generate_kwargs, tools=tools
            )

        if self.final_answer_prompt is not None and self._last_turn_called_tools():
            logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
            response = await self.model_client.chat(self.final_answer_prompt, generate_kwargs=generate_kwargs, tools=[])

        self._last_messages = list(self.model_client.messages)
        return response

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
    ) -> AsyncIterator[StreamChunk]:
        self._prepare_run()
        iteration = 0
        first_stream = await self.model_client.chat(
            task, generate_kwargs=generate_kwargs, stream=True, images=images, tools=tools
        )
        async for chunk in first_stream:
            yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)

        iteration += 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            next_stream = await self.model_client.chat(
                self.continuation_prompt, generate_kwargs=generate_kwargs, stream=True, tools=tools
            )
            async for chunk in next_stream:
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
            iteration += 1

        if self.final_answer_prompt is not None and self._last_turn_called_tools():
            logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
            final_stream = await self.model_client.chat(
                self.final_answer_prompt, generate_kwargs=generate_kwargs, stream=True, tools=[]
            )
            async for chunk in final_stream:
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
            iteration += 1

        self._last_messages = list(self.model_client.messages)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

    def _last_turn_called_tools(self) -> bool:
        for msg in reversed(self.model_client.messages):
            if msg.get("role") == "user":
                return False
            if msg.get("role") == "tool":
                return True
        return False

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
