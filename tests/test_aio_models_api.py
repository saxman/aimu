"""Mock-only unit tests for the async model client surface.

Mirrors ``test_models_api.py`` against ``aimu.aio``.
"""

from __future__ import annotations

import pytest

from aimu.aio import AsyncModelClient
from aimu.models import StreamChunk, StreamingContentType
from helpers_aio import MockAsyncModelClient


# ---------------------------------------------------------------------------
# system_message lifecycle (sync mechanics shared via _ChatStateMixin)
# ---------------------------------------------------------------------------


async def test_system_message_locks_after_first_chat():
    client = MockAsyncModelClient(["hello"])
    client.system_message = "v1"
    await client.chat("hi")
    with pytest.raises(RuntimeError, match="immutable"):
        client.system_message = "v2"


async def test_reset_unlocks_system_message():
    client = MockAsyncModelClient(["hello", "world"])
    client.system_message = "v1"
    await client.chat("hi")
    client.reset()
    client.system_message = "v2"
    assert client.system_message == "v2"


async def test_reset_preserves_system_message_by_default():
    client = MockAsyncModelClient(["hello"])
    client.system_message = "preserved"
    await client.chat("hi")
    client.reset()
    assert client.system_message == "preserved"


# ---------------------------------------------------------------------------
# include= filter on async streams
# ---------------------------------------------------------------------------


async def test_chat_stream_include_filters_phases():
    """include=['generating'] drops THINKING chunks."""

    class _MultiPhaseClient(MockAsyncModelClient):
        async def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
            if not stream:
                return await super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            async def _gen():
                yield StreamChunk(StreamingContentType.THINKING, "thinking")
                yield StreamChunk(StreamingContentType.GENERATING, "text")
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": "text"})

            return _gen()

    client = _MultiPhaseClient(["text"])
    stream = await client.chat("hi", stream=True, include=["generating"])
    chunks = [c async for c in stream]
    assert all(c.phase == StreamingContentType.GENERATING for c in chunks)


# ---------------------------------------------------------------------------
# In-process provider guard: aio.client(HuggingFaceModel.X) raises clear error
# ---------------------------------------------------------------------------


def test_aio_client_rejects_hf_model_enum_with_guidance():
    """Constructing aio.client() with a HuggingFaceModel enum directly should raise
    pointing users to the wrap-existing-sync-client pattern."""
    pytest.importorskip("transformers")  # HF dep required
    from aimu.models import HuggingFaceModel

    with pytest.raises(ValueError, match="load model weights into memory"):
        AsyncModelClient(HuggingFaceModel.QWEN_3_8B)
