from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.base import Workflow, Runner, AgentChunk, MessageHistory
from aimu.models.base import StreamingContentType

logger = logging.getLogger(__name__)


@dataclass
class Parallel(Workflow):
    """
    Workflow pattern: run multiple runners concurrently and aggregate results.

    Each worker runner receives the same task. The optional aggregator runner
    receives all worker outputs joined by ``separator`` and produces the final
    result. If no aggregator is provided, the joined worker outputs are returned
    directly.

    Workers run concurrently via ``ThreadPoolExecutor``; results are collected
    in submission order. Workers may be any Runner subclass (agents or workflows).

    Usage::

        parallel = Parallel(
            workers=[
                Agent(client_a, name="perspective-a"),
                Agent(client_b, name="perspective-b"),
                Agent(client_c, name="perspective-c"),
            ],
            aggregator=Agent(client, name="synthesizer",
                system_message="Synthesize the following perspectives into one answer."),
        )
        result = parallel.run("What are the risks of this approach?")
    """

    workers: list[Runner]
    name: str = "parallel"
    aggregator: Optional[Runner] = None
    separator: str = "\n\n---\n\n"
    max_workers: Optional[int] = None  # None = one thread per worker

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        for worker in self.workers:
            result.update(worker.messages)
        if self.aggregator:
            result.update(self.aggregator.messages)
        return result

    def _run_workers(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list] = None,
    ) -> list[str]:
        """Run all workers concurrently and return their results in submission order."""
        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.workers)) as ex:
            futures = [
                ex.submit(w.run, task, generate_kwargs, False, images) for w in self.workers
            ]
            return [f.result() for f in futures]

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[AgentChunk]]:
        """Run workers concurrently then aggregate. Returns str or AgentChunk iterator.

        ``images`` are forwarded to every worker; the aggregator runs on text only.
        """
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        results = self._run_workers(task, generate_kwargs, images=images)
        combined = self.separator.join(results)
        if self.aggregator:
            return self.aggregator.run(combined, generate_kwargs=generate_kwargs)
        return combined

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[AgentChunk]:
        results = self._run_workers(task, generate_kwargs, images=images)
        combined = self.separator.join(results)
        if self.aggregator:
            yield from self.aggregator.run(combined, generate_kwargs=generate_kwargs, stream=True)
        else:
            yield AgentChunk(self.name, 0, StreamingContentType.GENERATING, combined)
