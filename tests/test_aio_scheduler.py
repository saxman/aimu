"""Tests for the async Scheduler (proactive agent triggers)."""

from __future__ import annotations

import asyncio
import logging

from aimu.aio import Scheduler


async def _run_until(scheduler: Scheduler, *, settle: float = 0.05) -> asyncio.Task:
    """Start scheduler.run() in a task and give it a moment to spin up its jobs."""
    task = asyncio.create_task(scheduler.run())
    await asyncio.sleep(settle)
    return task


async def _stop(scheduler: Scheduler, task: asyncio.Task) -> None:
    scheduler.stop()
    async with asyncio.timeout(2):
        await task


async def test_one_shot_fires_once():
    fired = asyncio.Event()
    count = 0

    async def cb():
        nonlocal count
        count += 1
        fired.set()

    scheduler = Scheduler()
    scheduler.at(0.0, cb)
    task = await _run_until(scheduler)
    async with asyncio.timeout(2):
        await fired.wait()
    await asyncio.sleep(0.05)  # ensure it does not fire again
    await _stop(scheduler, task)
    assert count == 1


async def test_interval_fires_repeatedly():
    count = 0
    enough = asyncio.Event()

    async def cb():
        nonlocal count
        count += 1
        if count >= 3:
            enough.set()

    scheduler = Scheduler()
    scheduler.every(0.01, cb, first_delay=0.0)
    task = await _run_until(scheduler)
    async with asyncio.timeout(2):
        await enough.wait()
    await _stop(scheduler, task)
    assert count >= 3


async def test_cancel_removes_job():
    count = 0

    async def cb():
        nonlocal count
        count += 1

    scheduler = Scheduler()
    name = scheduler.every(0.01, cb, first_delay=0.5)  # long initial delay, won't fire soon
    assert scheduler.cancel(name) is True
    assert scheduler.cancel("nonexistent") is False
    task = await _run_until(scheduler, settle=0.1)
    await _stop(scheduler, task)
    assert count == 0


async def test_job_exception_is_isolated(caplog):
    good_fired = asyncio.Event()

    async def bad():
        raise RuntimeError("boom")

    async def good():
        good_fired.set()

    scheduler = Scheduler()
    scheduler.every(0.01, bad, first_delay=0.0)
    scheduler.every(0.01, good, first_delay=0.0)
    task = await _run_until(scheduler)
    with caplog.at_level(logging.ERROR):
        async with asyncio.timeout(2):
            await good_fired.wait()  # sibling still fires despite bad raising
    await _stop(scheduler, task)
    assert any("boom" in r.message or "raised" in r.message for r in caplog.records)


async def test_stop_returns_cleanly():
    scheduler = Scheduler()
    scheduler.every(0.01, lambda: asyncio.sleep(0), first_delay=0.0)
    task = await _run_until(scheduler)
    await _stop(scheduler, task)
    assert task.done() and task.exception() is None


async def test_add_job_after_start():
    fired = asyncio.Event()

    async def cb():
        fired.set()

    scheduler = Scheduler()
    task = await _run_until(scheduler)
    scheduler.at(0.0, cb)  # registered while running
    async with asyncio.timeout(2):
        await fired.wait()
    await _stop(scheduler, task)
    assert fired.is_set()


async def test_stop_before_run_returns_immediately():
    # A stop signalled before run() (e.g. by a sibling task) must not be lost.
    scheduler = Scheduler()
    scheduler.stop()
    async with asyncio.timeout(2):
        await scheduler.run()
