"""Mock-only async-embedding surface tests: wrap refusal, state sharing, async embed."""

from __future__ import annotations

import pytest

from aimu import aio
from aimu.models.base import BaseEmbeddingClient


class _SyncStub(BaseEmbeddingClient):
    def __init__(self):
        self.spec = type("S", (), {"id": "stub", "dimensions": 2})()
        self.model = "stub"
        self.model_kwargs = None

    def _embed(self, texts, **kwargs):
        return [[1.0, 2.0] for _ in texts]


def test_async_embedding_client_refuses_string():
    with pytest.raises(ValueError, match="sync embedding client"):
        aio.embedding_client("openai:text-embedding-3-small")


def test_async_embedding_client_refuses_spec_or_enum():
    with pytest.raises((TypeError, ValueError)):
        aio.embedding_client(object())


def test_async_wrapper_delegates_properties():
    wrapped = aio.embedding_client(_SyncStub())
    assert wrapped.spec.id == "stub"
    assert wrapped.dimensions == 2


async def test_async_embed_single_and_list():
    wrapped = aio.embedding_client(_SyncStub())
    assert await wrapped.embed("a") == [1.0, 2.0]
    assert await wrapped.embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]


async def test_async_top_level_embed_accepts_sync_client():
    out = await aio.embed(["x", "y"], model=_SyncStub())
    assert out == [[1.0, 2.0], [1.0, 2.0]]
