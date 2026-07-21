"""Async LlamaCpp client: wraps an existing sync :class:`LlamaCppClient`.

Same pattern as :mod:`aimu.aio.providers.hf` (see Decision 7). GGUF weights load
once via the sync client; this class adds an awaitable interface on top. All
wrapping mechanics live in :class:`_AsyncInProcessClient`.
"""

from __future__ import annotations

from aimu.models.providers.llamacpp import LlamaCppClient, LlamaCppModel

from ._inprocess import _AsyncInProcessClient


class AsyncLlamaCppClient(_AsyncInProcessClient):
    """Async wrapper around a sync :class:`LlamaCppClient` (no weight reload)."""

    MODELS = LlamaCppModel
    _SYNC_CLASS = LlamaCppClient
