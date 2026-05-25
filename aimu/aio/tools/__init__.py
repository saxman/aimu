"""Async built-in tools.

Mirrors :mod:`aimu.tools.builtin` for the async surface. CPU-bound and I/O-bound
tools (weather, calculator, etc.) are reused from the sync module — the async
``_handle_tool_calls`` dispatches them through ``asyncio.to_thread`` automatically.
The image-generation tool is async-native here so it awaits the wrapped
:class:`AsyncDiffusionClient` directly.
"""

from . import builtin

__all__ = ["builtin"]
