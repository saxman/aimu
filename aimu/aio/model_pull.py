"""Async twin of :func:`aimu.pull`: same resolution, same refusals, same downloads.

The Ollama pull awaits ``ollama.AsyncClient``; HuggingFace downloads run in a worker thread, since
``huggingface_hub`` has no async API and the event loop must stay free.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Optional, Union

from aimu.models import model_pull as _sync_pull


async def pull(model: Union[str, Enum], *, host: Optional[str] = None) -> None:
    """Download ``model`` so a client for it can be built without downloading anything.

    See :func:`aimu.pull` for what is downloaded per provider and what is refused.
    """
    backend, member = _sync_pull._resolve_pull_target(model, host)
    if backend == "ollama":
        await apull_ollama_model(_ollama_async_sdk_client(host), member.value)
    else:
        downloads = _sync_pull._hf_client_class(member)._hub_downloads(member.spec)
        await asyncio.to_thread(lambda: [_sync_pull.fetch(download) for download in downloads])


def _ollama_async_sdk_client(host: Optional[str]) -> Any:
    import ollama

    from aimu.models.providers.ollama import _ollama_client_kwargs

    return ollama.AsyncClient(**_ollama_client_kwargs(host, None))


async def apull_ollama_model(sdk_client: Any, model_id: str) -> None:
    from aimu.aio.providers.ollama import apull_ollama_model as pull_with_logging

    await pull_with_logging(sdk_client, model_id)
