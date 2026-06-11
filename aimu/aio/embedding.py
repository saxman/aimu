"""Async embedding surface mirroring :mod:`aimu.models.embedding_client`.

Embedding clients have no chat lifecycle, so (like audio / speech / transcription) the
async surface wraps an existing sync :class:`BaseEmbeddingClient` and routes
:meth:`embed` through ``asyncio.to_thread`` (Decision 7). One wrapper covers every
provider because they share the same ``embed()`` interface.
"""

from __future__ import annotations

import asyncio
from typing import Any, Union

try:
    from aimu.models.base import BaseEmbeddingClient

    _HAS_EMBEDDING = True
except ImportError:  # pragma: no cover - base import should always succeed
    _HAS_EMBEDDING = False
    BaseEmbeddingClient = None  # type: ignore[assignment,misc]


_WRAP_GUIDANCE = (
    "Build a sync embedding client first and pass it to aio.embedding_client():\n"
    "    sync_client = aimu.embedding_client({model})\n"
    "    async_client = aio.embedding_client(sync_client)\n"
    "(This also avoids loading weights twice for in-process providers.)"
)


def _is_embedding_client(obj: Any) -> bool:
    try:
        from aimu.models.base import BaseEmbeddingClient as _Cls

        return isinstance(obj, _Cls)
    except ImportError:  # pragma: no cover
        return False


class AsyncEmbeddingClient:
    """Async embedding client. Wraps an existing sync :class:`BaseEmbeddingClient`.

    Passing a spec, enum member, or string raises pointing at the sync-then-wrap pattern.
    """

    def __init__(self, sync_client: Any):
        if not _is_embedding_client(sync_client):
            if isinstance(sync_client, str):
                raise ValueError(_WRAP_GUIDANCE.format(model=repr(sync_client)))
            raise TypeError(
                f"AsyncEmbeddingClient expects a sync BaseEmbeddingClient. Got: {type(sync_client).__name__}. "
                + _WRAP_GUIDANCE.format(model=repr(sync_client))
            )
        self._sync = sync_client

    @property
    def model(self) -> Any:
        return self._sync.model

    @property
    def spec(self) -> Any:
        return self._sync.spec

    @property
    def dimensions(self) -> Any:
        return self._sync.dimensions

    async def embed(self, texts: Union[str, list], **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync.embed, texts, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncEmbeddingClient({self._sync!r})"


def embedding_client(sync_client: Any) -> AsyncEmbeddingClient:
    """Wrap an existing sync embedding client for async use."""
    return AsyncEmbeddingClient(sync_client)


async def embed(texts: Union[str, list], *, model: Any = None, **kwargs: Any) -> Any:
    """One-shot async embedding.

    ``model`` may be an existing sync embedding client (preferred), a model enum member,
    or a ``"provider:model_id"`` string. When omitted, the ``AIMU_EMBEDDING_MODEL`` env
    var is used; if unset a ``ValueError`` is raised.
    """
    if _is_embedding_client(model):
        sync_client: Any = model
    else:
        import aimu

        sync_client = aimu.embedding_client(model)
    return await embedding_client(sync_client).embed(texts, **kwargs)
