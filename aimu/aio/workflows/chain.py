"""Async Chain workflow (prompt chaining pattern)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Union

from aimu.agents.base import MessageHistory
from aimu.models.base import StreamChunk, StreamingContentType

from .._base import AsyncBaseModelClient
from ..agent import AsyncRunner

logger = logging.getLogger(__name__)


@dataclass
class Chain(AsyncRunner):
    """Async prompt-chaining: each step's output feeds the next step's input.

    Steps run sequentially (the pattern requires it). Each step may be an
    :class:`aimu.aio.Agent` or a nested async workflow.
    """

    agents: list
    name: str = "chain"

    @classmethod
    def from_client(cls, client: AsyncBaseModelClient, prompts: list[str], *, name: str = "chain") -> Chain:
        """Build a Chain from a single client and a list of step system_messages."""
        from ..agent import Agent

        agents = [
            Agent(client, system_message=prompt, name=f"step-{i}", reset_messages_on_run=True)
            for i, prompt in enumerate(prompts)
        ]
        return cls(agents=agents, name=name)

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Chain step %d, agent '%s'.", step, agent.name)
            step_images = images if step == 0 else None
            result = await agent.run(result, generate_kwargs=generate_kwargs, images=step_images)
        return result

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Chain step %d, agent '%s'.", step, agent.name)
            step_images = images if step == 0 else None
            step_chunks: list[StreamChunk] = []
            step_stream = await agent.run(result, generate_kwargs=generate_kwargs, stream=True, images=step_images)
            async for chunk in step_stream:
                step_chunks.append(chunk)
                yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=step)
            result = "".join(c.content for c in step_chunks if c.phase == StreamingContentType.GENERATING)

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        for agent in self.agents:
            result.update(agent.messages)
        return result

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]], client: AsyncBaseModelClient) -> Chain:
        from ..agent import Agent

        agents = []
        for cfg in configs:
            agent = Agent.from_config(cfg, client)
            agent.reset_messages_on_run = True
            agents.append(agent)
        return cls(agents=agents)
