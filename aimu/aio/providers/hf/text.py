"""Async HuggingFace client: wraps an existing sync :class:`HuggingFaceClient`.

Per Decision 7 in the plan, this class does *not* load model weights independently.
The async surface exists for event-loop integration, not coroutine concurrency
(HF transformers is sync-only and GIL/CUDA-bound). Construct a sync client first
and pass it in::

    sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)
    async_client = aio.client(sync_client)

State (``messages``, ``system_message``, ``tools``) is shared with the wrapped sync
client (there's conceptually one client; the async version just adds an awaitable
interface). All wrapping mechanics live in :class:`_AsyncInProcessClient`.
"""

from __future__ import annotations

from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

from .._inprocess import _AsyncInProcessClient


class AsyncHuggingFaceClient(_AsyncInProcessClient):
    """Async wrapper around a sync :class:`HuggingFaceClient` (no weight reload)."""

    MODELS = HuggingFaceModel
    _SYNC_CLASS = HuggingFaceClient
