"""Async built-in tools.

Mirrors :mod:`aimu.tools.builtin` for the async surface. CPU-bound and I/O-bound
tools (weather, calculator, etc.) are reused from the sync module; the async
tool-loop engine (:class:`aimu.aio._tool_loop._AsyncToolLoop`) dispatches them
through ``asyncio.to_thread`` automatically. The image-generation tool is
async-native here so it awaits the wrapped :class:`AsyncDiffusionClient` directly.
"""

from . import builtin

__all__ = ["builtin"]
