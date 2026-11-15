"""Async twin of :mod:`aimu.context`: only :mod:`aimu.aio.context.compaction` has anything to add.

:func:`aimu.context.count_tokens` and every compaction except summarizing are pure and synchronous,
so both surfaces use :mod:`aimu.context` directly.
"""

from . import compaction

__all__ = ["compaction"]
