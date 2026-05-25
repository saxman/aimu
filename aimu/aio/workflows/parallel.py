"""Async Parallel workflow — the headline ``asyncio.TaskGroup`` win.

Replaces the sync ``ThreadPoolExecutor`` with structured concurrency: a worker
exception cancels its siblings cleanly, and all errors surface together as an
``ExceptionGroup``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Union

from aimu.agents.base import MessageHistory
from aimu.models.base import StreamChunk, StreamingContentType

from .._base import AsyncBaseModelClient
from ..agent import AsyncRunner

logger = logging.getLogger(__name__)


@dataclass
class Parallel(AsyncRunner):
    """Async parallelization: run workers concurrently via ``asyncio.TaskGroup``, aggregate.

    Each worker receives the same task. An optional aggregator receives all worker
    outputs joined by ``separator``. Without an aggregator, the joined output is returned.

    *Structured concurrency*: if one worker raises, in-flight siblings are cancelled
    and an ``ExceptionGroup`` surfaces with all errors.
    """

    workers: list
    name: str = "parallel"
    aggregator: Optional[AsyncRunner] = None
    separator: str = "\n\n---\n\n"

    @classmethod
    def from_client(
        cls,
        client: AsyncBaseModelClient,
        worker_prompts: list[str],
        *,
        aggregator_prompt: Optional[str] = None,
        separator: str = "\n\n---\n\n",
        name: str = "parallel",
    ) -> Parallel:
        from ..agent import Agent

        workers: list = [
            Agent(client, system_message=p, name=f"worker-{i}", reset_messages_on_run=True)
            for i, p in enumerate(worker_prompts)
        ]
        aggregator: Optional[AsyncRunner] = None
        if aggregator_prompt is not None:
            aggregator = Agent(
                client,
                system_message=aggregator_prompt,
                name="aggregator",
                reset_messages_on_run=True,
            )
        return cls(workers=workers, aggregator=aggregator, separator=separator, name=name)

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        for worker in self.workers:
            result.update(worker.messages)
        if self.aggregator:
            result.update(self.aggregator.messages)
        return result

    async def _run_workers(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list] = None,
    ) -> list[str]:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(w.run(task, generate_kwargs, False, images)) for w in self.workers
            ]
        return [t.result() for t in tasks]

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        results = await self._run_workers(task, generate_kwargs, images=images)
        combined = self.separator.join(results)
        if self.aggregator:
            return await self.aggregator.run(combined, generate_kwargs=generate_kwargs)
        return combined

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        results = await self._run_workers(task, generate_kwargs, images=images)
        combined = self.separator.join(results)
        if self.aggregator:
            agg_stream = await self.aggregator.run(
                combined, generate_kwargs=generate_kwargs, stream=True
            )
            async for chunk in agg_stream:
                yield chunk
        else:
            yield StreamChunk(StreamingContentType.GENERATING, combined, agent=self.name, iteration=0)
