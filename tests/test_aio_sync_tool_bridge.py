"""Mock-only tests for the async-tool -> sync-tool bridge.

The async wrappers around in-process sync clients (HuggingFace, LlamaCpp) run the
sync ``_chat`` loop in a worker thread, whose tool dispatcher can't await async
tools. ``bridge_async_tools`` wraps each async tool as a sync callable that runs the
coroutine back on the main loop. These tests exercise that bridge directly; no model
weights required.
"""

from __future__ import annotations

import asyncio

from aimu.aio.providers._sync_tool_bridge import bridge_async_tools
from aimu.tools import tool


def test_bridge_is_noop_without_async_tools():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert bridge_async_tools([add], asyncio.new_event_loop()) is None


async def test_bridge_runs_async_tool_from_worker_thread():
    """A bridged async tool is sync-callable and returns the awaited result."""

    @tool
    async def temperature(location: str) -> float:
        """Return the temperature for a location."""
        await asyncio.sleep(0)
        return 27.0

    loop = asyncio.get_running_loop()
    bridged = bridge_async_tools([temperature], loop)
    assert bridged is not None
    (shim,) = bridged

    # Spec, name, and async flag are set up for the sync dispatcher.
    assert shim.__name__ == "temperature"
    assert shim.__tool_spec__ == temperature.__tool_spec__
    assert shim.__tool_is_async__ is False
    assert shim.__tool_is_streaming__ is False

    # Calling the shim from a worker thread (as the sync dispatcher does) runs the
    # coroutine on this loop and blocks for the result.
    result = await asyncio.to_thread(lambda: shim(location="Paris"))
    assert result == 27.0


async def test_bridge_runs_async_generator_tool_from_worker_thread():
    """A bridged async-generator tool becomes a sync generator yielding the same chunks."""

    @tool
    async def counter(n: int):
        """Yield numbers up to n."""
        for i in range(n):
            await asyncio.sleep(0)
            yield i

    loop = asyncio.get_running_loop()
    (shim,) = bridge_async_tools([counter], loop)

    assert shim.__tool_is_async__ is False
    assert shim.__tool_is_streaming__ is True

    chunks = await asyncio.to_thread(lambda: list(shim(n=3)))
    assert chunks == [0, 1, 2]


async def test_bridge_passes_sync_tools_through_unchanged():
    @tool
    def echo(text: str) -> str:
        """Echo the text."""
        return text

    @tool
    async def fetch(url: str) -> str:
        """Fetch a url."""
        return url

    loop = asyncio.get_running_loop()
    bridged = bridge_async_tools([echo, fetch], loop)
    assert bridged is not None
    sync_passthrough, async_shim = bridged
    assert sync_passthrough is echo  # unchanged
    assert async_shim is not fetch  # wrapped
    assert async_shim.__name__ == "fetch"
