"""A cancellable handle for an in-flight async run.

`RunHandle` is a thin wrapper around the `asyncio.Task` driving a run (e.g. `aio.Agent.run(...)`),
so an application can cancel it from elsewhere, an app's ``/stop`` command, a timeout, a supervisor.
It is the cancellation seam for the async surface: cancellation rides on `asyncio` task cancellation
(no threaded token), so the natural `await` boundaries inside the agent loop are the cancel points.

A cancelled run still captures its partial messages (the agent snapshots in a ``finally``), so the
conversation can be resumed with :meth:`aimu.aio.Agent.restore`.

    handle = RunHandle.start(agent.run("do the thing"))
    ...
    handle.cancel()                  # from a /stop handler, timeout, etc.
    try:
        reply = await handle.result()
    except asyncio.CancelledError:
        ...                          # agent.model_client.messages holds the partial turn
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Generator


class RunHandle:
    """Wraps the `asyncio.Task` of an in-flight run with `cancel()` + `await result()`."""

    def __init__(self, task: asyncio.Task):
        self._task = task

    @classmethod
    def start(cls, coro: Awaitable[Any]) -> "RunHandle":
        """Schedule ``coro`` (e.g. ``agent.run(...)``) as a task and return a handle to it."""
        return cls(asyncio.ensure_future(coro))

    @property
    def task(self) -> asyncio.Task:
        return self._task

    @property
    def done(self) -> bool:
        return self._task.done()

    @property
    def cancelled(self) -> bool:
        return self._task.cancelled()

    def cancel(self) -> bool:
        """Request cancellation of the run. Returns False if it had already finished."""
        return self._task.cancel()

    async def result(self) -> Any:
        """Await the run and return its result; raises `asyncio.CancelledError` if cancelled."""
        return await self._task

    def __await__(self) -> Generator[Any, None, Any]:
        return self._task.__await__()
