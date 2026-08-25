"""Async Parallel workflow tests: verifies ``asyncio.TaskGroup`` semantics.

Two correctness goals:
1. Workers actually overlap (wall-clock time < serial baseline).
2. When one worker raises, siblings are cancelled and an ``ExceptionGroup`` surfaces.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aimu.aio import Agent, Parallel
from aimu.aio.agent import AsyncRunner
from helpers_aio import MockAsyncModelClient


class _SlowWorker(AsyncRunner):
    """A minimal worker that sleeps for ``delay`` seconds before returning ``result``."""

    def __init__(self, name: str, delay: float, result: str = "ok", raises: bool = False):
        self.name = name
        self.delay = delay
        self.result = result
        self.raises = raises

    async def run(self, task, generate_kwargs=None, stream=False, images=None):
        await asyncio.sleep(self.delay)
        if self.raises:
            raise RuntimeError(f"worker {self.name} boom")
        return self.result

    @property
    def messages(self):
        return {self.name: []}


async def test_parallel_overlaps_workers():
    """Two 0.5s workers should finish in ~0.5s, not 1.0s."""
    workers = [_SlowWorker(f"w{i}", 0.5, result=f"r{i}") for i in range(2)]
    parallel = Parallel(workers=workers)
    t0 = time.perf_counter()
    result = await parallel.run("task")
    elapsed = time.perf_counter() - t0
    assert "r0" in result and "r1" in result
    assert elapsed < 0.9, f"expected overlap (<0.9s), got {elapsed:.2f}s"


async def test_parallel_cancels_siblings_on_failure():
    """When one worker raises, TaskGroup cancels siblings and raises ExceptionGroup."""
    workers = [
        _SlowWorker("good", 5.0, result="ok"),
        _SlowWorker("bad", 0.05, raises=True),
    ]
    parallel = Parallel(workers=workers)
    t0 = time.perf_counter()
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await parallel.run("task")
    elapsed = time.perf_counter() - t0
    # Failed within ~0.1s (bad's delay), well before good would have finished.
    assert elapsed < 2.0, f"sibling not cancelled, took {elapsed:.2f}s"
    # The ExceptionGroup contains the RuntimeError from the bad worker.
    flat_errors = [e for e in exc_info.value.exceptions if isinstance(e, RuntimeError)]
    assert any("boom" in str(e) for e in flat_errors)


async def test_parallel_with_aggregator():
    """Aggregator receives joined worker outputs."""
    workers = [
        Agent(MockAsyncModelClient(["alpha"]), reset_messages_on_run=True, name="w0"),
        Agent(MockAsyncModelClient(["beta"]), reset_messages_on_run=True, name="w1"),
    ]
    agg = Agent(MockAsyncModelClient(["combined"]), reset_messages_on_run=True, name="agg")
    parallel = Parallel(workers=workers, aggregator=agg)
    result = await parallel.run("task")
    assert result == "combined"


class _RacingMockAsyncModelClient(MockAsyncModelClient):
    """Async twin of ``tests/test_workflow_parallel.py``'s ``_RacingMockModelClient``.

    Forces genuine cross-task overlap between concurrent ``_events_override`` scopes on one
    shared client: an ``asyncio.Lock``-protected counter assigns each call an arrival index
    the moment it reaches ``_chat()`` (before any sleep, reflecting the order calls entered
    their scope), an ``asyncio.Barrier`` holds every caller until all three have arrived, and
    each then sleeps a duration keyed by its arrival index so the scopes close in a different
    order than they opened -- the same crossing (non-nested) overlap the sync test forces via
    threads, driven here by ``asyncio.TaskGroup`` instead.
    """

    def __init__(self, responses: list, *, barrier: "asyncio.Barrier", delays: dict[int, float]):
        super().__init__(responses)
        self._barrier = barrier
        self._delays = delays
        self._arrival_lock = asyncio.Lock()
        self._next_arrival = 0

    async def _chat(self, *args, **kwargs):
        async with self._arrival_lock:
            arrival = self._next_arrival
            self._next_arrival += 1
        await self._barrier.wait()
        await asyncio.sleep(self._delays[arrival])
        return await super()._chat(*args, **kwargs)


async def test_parallel_from_client_shared_events_sink_survives_concurrent_workers():
    """Async mirror of the sync isolation test in ``tests/test_workflow_parallel.py``.

    That test replaced ``test_KNOWN_GAP_parallel_from_client_shared_events_sink_drops_events``
    (removed), which pinned the pre-fix bug: a scoped event sink delivered by mutating the
    client's own ``self.events`` attribute is visible to every concurrently-running worker on
    a shared client, so overlapping swap/restore sequences clobber each other's sink. Here the
    overlap is real ``asyncio.TaskGroup`` concurrency (``aimu.aio.workflows.parallel.Parallel``
    docstring calls out the identical hazard for the async surface) rather than
    ``ThreadPoolExecutor`` threads, which the fix (a ``contextvars.ContextVar`` scoped
    override) closes "by construction" for asyncio: each ``Task`` gets its own copy of the
    context it was created in, so one task's override can never leak into another's. Every
    worker's ``RunStarted``/``ModelTurnStarted``/``ModelTurnFinished``/``RunFinished`` must
    still arrive, attributed to the right worker, in causal order.
    """
    barrier = asyncio.Barrier(3)
    delays = {0: 0.15, 1: 0.05, 2: 0.10}
    client = _RacingMockAsyncModelClient(["A", "B", "C"], barrier=barrier, delays=delays)

    seen: list = []
    parallel = Parallel.from_client(
        client,
        worker_prompts=["Do A.", "Do B.", "Do C."],
        events=seen.append,
    )
    await parallel.run("task")

    by_agent: dict[str, list[str]] = {}
    for event in seen:
        by_agent.setdefault(event.agent, []).append(type(event).__name__)

    assert set(by_agent) == {"worker-0", "worker-1", "worker-2"}, by_agent
    expected_sequence = ["RunStarted", "ModelTurnStarted", "ModelTurnFinished", "RunFinished"]
    for name, sequence in by_agent.items():
        assert sequence == expected_sequence, f"{name}: {sequence}"
