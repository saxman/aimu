from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import BaseModelClient, StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


@dataclass
class Parallel(Runner):
    """**Parallelization** pattern: run workers concurrently, aggregate.

    Each worker receives the same task. An optional aggregator receives all worker
    outputs joined by ``separator`` and produces the final result. Without an
    aggregator, the joined worker outputs are returned directly.

    Workers run concurrently via ``ThreadPoolExecutor``; results are collected in
    submission order. Workers may be any :class:`Runner`.

    Quick start::

        parallel = Parallel.from_client(
            client,
            worker_prompts=[
                "Analyze this from a security perspective.",
                "Analyze this from a performance perspective.",
                "Analyze this from a readability perspective.",
            ],
            aggregator_prompt="Synthesize the perspectives into one concise review.",
        )
    """

    workers: list[Runner]
    name: str = "parallel"
    aggregator: Optional[Runner] = None
    separator: str = "\n\n---\n\n"
    max_workers: Optional[int] = None  # None = one thread per worker

    @classmethod
    def from_client(
        cls,
        client: BaseModelClient,
        worker_prompts: list[str],
        *,
        aggregator_prompt: Optional[str] = None,
        separator: str = "\n\n---\n\n",
        name: str = "parallel",
    ) -> Parallel:
        """Build a Parallel using ``client`` for all workers (and aggregator)."""
        from aimu.agents.agent import Agent

        workers: list[Runner] = [
            Agent(client, system_message=p, name=f"worker-{i}", reset_messages_on_run=True)
            for i, p in enumerate(worker_prompts)
        ]
        aggregator: Optional[Runner] = None
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

    def restore(self, messages: list[dict], *, worker: int = 0) -> None:
        """Restore one worker's state from a saved message list.

        *worker* (keyword-only) selects which worker by index (default 0), mirroring
        ``Chain.restore``'s ``step``. Other workers and the aggregator start fresh on the
        next ``run()``. Raises ``IndexError`` if *worker* is out of range. See
        :meth:`Agent.restore` for the full save/restore pattern.
        """
        if not 0 <= worker < len(self.workers):
            raise IndexError(f"worker {worker} out of range for {len(self.workers)} worker(s).")
        self.workers[worker].restore(messages)

    def _run_workers(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list] = None,
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.workers)) as ex:
            futures = [ex.submit(w.run, task, generate_kwargs, False, images) for w in self.workers]
            return [f.result() for f in futures]

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Run workers concurrently then aggregate.

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
    ) -> Iterator[StreamChunk]:
        results = self._run_workers(task, generate_kwargs, images=images)
        combined = self.separator.join(results)
        if self.aggregator:
            yield from self.aggregator.run(combined, generate_kwargs=generate_kwargs, stream=True)
        else:
            yield StreamChunk(StreamingContentType.GENERATING, combined, agent=self.name, iteration=0)
