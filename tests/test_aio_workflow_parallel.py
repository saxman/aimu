"""Async Parallel workflow tests — verifies ``asyncio.TaskGroup`` semantics.

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
