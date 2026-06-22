"""Bridge async tools onto the sync tool dispatcher used by wrapped in-process clients.

The async wrappers around in-process sync clients (:class:`AsyncHuggingFaceClient`,
:class:`AsyncLlamaCppClient`) run the sync client's ``_chat`` loop (including its tool
dispatch) inside a worker thread via ``asyncio.to_thread``. That sync dispatcher
(:meth:`BaseModelClient._handle_tool_calls`) deliberately refuses ``async def`` tools.
But the async surface routinely attaches async tools (e.g. ``await aio.MCPClient.as_tools()``).

The fix is to wrap each async tool as a *sync* callable that runs the underlying
coroutine back on the main event loop (``run_coroutine_threadsafe``) and blocks the
worker thread for the result. The main loop is free while the worker waits (it is
awaiting the ``to_thread`` future), so there is no deadlock. The sync dispatcher then
sees only sync callables and dispatches them through its normal path.

This keeps the sync surface untouched and avoids duplicating each provider's chat
orchestration in the async wrapper; only tool dispatch needs to cross back to the loop.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Callable, Optional


def _bridge_tool(fn: Callable, loop: asyncio.AbstractEventLoop) -> Callable:
    """Wrap a single async tool as a sync callable bound to *loop*.

    Sync tools are returned unchanged. A plain ``async def`` tool becomes a sync
    function; an async-generator (streaming) tool becomes a sync generator. The
    OpenAI tool spec and the function name (used for by-name dispatch) are preserved.
    """
    if not getattr(fn, "__tool_is_async__", False):
        return fn

    if getattr(fn, "__tool_is_streaming__", False):

        @functools.wraps(fn)
        def shim(**kwargs):
            agen = fn(**kwargs)
            while True:
                step = asyncio.run_coroutine_threadsafe(agen.__anext__(), loop)
                try:
                    yield step.result()
                except StopAsyncIteration:
                    break

        shim.__tool_is_streaming__ = True
    else:

        @functools.wraps(fn)
        def shim(**kwargs):
            return asyncio.run_coroutine_threadsafe(fn(**kwargs), loop).result()

        shim.__tool_is_streaming__ = False

    # functools.wraps copies __dict__ (incl. __tool_is_async__=True and __tool_spec__);
    # re-assert the flags so the sync dispatcher treats the shim as a sync tool.
    shim.__tool_spec__ = fn.__tool_spec__
    shim.__tool_is_async__ = False
    return shim


def bridge_async_tools(tools: list, loop: asyncio.AbstractEventLoop) -> Optional[list]:
    """Return *tools* with async tools replaced by sync bridges to *loop*.

    Returns ``None`` when there is nothing to bridge (no async tools present), so the
    caller can skip the temporary tools swap entirely.
    """
    if not any(getattr(fn, "__tool_is_async__", False) for fn in tools):
        return None
    return [_bridge_tool(fn, loop) for fn in tools]
