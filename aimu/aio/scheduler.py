"""A minimal async scheduler for proactive agent triggers.

The defining trait of a personal assistant (vs. a request/response chatbot) is acting
*unprompted*: reminders, check-ins, periodic tasks. :class:`Scheduler` registers interval
and one-shot async jobs and runs them concurrently under a single
:class:`asyncio.TaskGroup`, mirroring the structured-concurrency idiom in
``aimu/aio/workflows/parallel.py``.

A job is a zero-argument coroutine factory; bind context (an agent, a channel) with a
closure or :func:`functools.partial`, keeping the scheduler decoupled from what it fires.

Persistence is intentionally out of scope: jobs are Python callables (not serializable),
the assistant is a long-lived process, and durable cron-like scheduling belongs in a
wrapper above the library, not in it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

Job = Callable[[], Awaitable[None]]


@dataclass
class _ScheduledJob:
    name: str
    callback: Job
    delay: float
    interval: Optional[float]  # None => one-shot
    task: Optional[asyncio.Task] = field(default=None, repr=False)


class Scheduler:
    """Run interval and one-shot async jobs concurrently until stopped.

    Usage::

        scheduler = Scheduler()
        scheduler.every(60, check_inbox, name="inbox")
        scheduler.at(5, lambda: channel.send("Welcome!"))
        await scheduler.run()   # blocks until scheduler.stop()

    A job that raises is logged and, for interval jobs, the loop continues on the next tick
    (one misbehaving reminder must not tear down the daemon). Only :meth:`stop` unwinds the
    run loop.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, _ScheduledJob] = {}
        self._tg: Optional[asyncio.TaskGroup] = None
        self._stop = asyncio.Event()
        self._counter = 0

    def _next_name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def every(
        self, seconds: float, callback: Job, *, name: Optional[str] = None, first_delay: Optional[float] = None
    ) -> str:
        """Register a recurring job firing every ``seconds``. Returns the job id.

        ``first_delay`` overrides the initial wait before the first fire (defaults to
        ``seconds``). If the scheduler is already running, the job starts immediately.
        """
        name = name or self._next_name("interval")
        job = _ScheduledJob(name, callback, first_delay if first_delay is not None else seconds, seconds)
        self._register(job)
        return name

    def at(self, delay_seconds: float, callback: Job, *, name: Optional[str] = None) -> str:
        """Register a one-shot job firing once after ``delay_seconds``. Returns the job id."""
        name = name or self._next_name("once")
        job = _ScheduledJob(name, callback, delay_seconds, None)
        self._register(job)
        return name

    def _register(self, job: _ScheduledJob) -> None:
        self._jobs[job.name] = job
        if self._tg is not None:  # running: launch now
            job.task = self._tg.create_task(self._run_job(job))

    def cancel(self, name: str) -> bool:
        """Cancel a registered job by id. Returns True if it existed."""
        job = self._jobs.pop(name, None)
        if job is None:
            return False
        if job.task is not None:
            job.task.cancel()
        return True

    def stop(self) -> None:
        """Signal :meth:`run` to cancel all jobs and return."""
        self._stop.set()

    async def _run_job(self, job: _ScheduledJob) -> None:
        await asyncio.sleep(job.delay)
        while True:
            try:
                await job.callback()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduled job '%s' raised; continuing", job.name)
            if job.interval is None:
                return
            await asyncio.sleep(job.interval)

    async def run(self) -> None:
        """Run all registered jobs concurrently, blocking until :meth:`stop` is called.

        A single-use run: if :meth:`stop` was already called (even before ``run``), the loop
        returns immediately. This avoids a lost-stop race when a sibling task signals stop
        before the run loop is scheduled. Use a fresh :class:`Scheduler` to run again.
        """
        try:
            async with asyncio.TaskGroup() as tg:
                self._tg = tg
                for job in list(self._jobs.values()):
                    job.task = tg.create_task(self._run_job(job))
                stop_task = tg.create_task(self._stop.wait())
                await stop_task
                # Stop requested: cancel every job loop so the group can exit cleanly.
                for job in self._jobs.values():
                    if job.task is not None:
                        job.task.cancel()
        finally:
            self._tg = None
            for job in self._jobs.values():
                job.task = None
