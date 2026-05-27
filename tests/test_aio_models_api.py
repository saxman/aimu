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


# ---------------------------------------------------------------------------
# Streaming tool dispatch — async generators
# ---------------------------------------------------------------------------


async def test_async_streaming_tool_yields_forward_through_dispatch():
    """An async generator @tool's chunks flow through _handle_tool_calls_streamed."""
    from aimu.tools.decorator import tool

    @tool
    async def slow_tool(x: str):
        """Async streaming tool with no return value — last chunk carries result."""
        yield StreamChunk(StreamingContentType.GENERATING, "working...")
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": f"done:{x}"},
        )

    client = MockAsyncModelClient(["dummy"])
    client.tools = [slow_tool]
    tc = [{"name": "slow_tool", "arguments": {"x": "input"}}]

    chunks = [c async for c in client._handle_tool_calls_streamed(tc, tools=[])]

    # 2 progress chunks + 1 final TOOL_CALLING chunk.
    progress = [c for c in chunks if c.phase != StreamingContentType.TOOL_CALLING]
    tool_calls = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert len(progress) == 2
    assert len(tool_calls) == 1
    # Response extracted from last chunk's content["result"] since async gen has no return.
    assert tool_calls[0].content["response"] == "done:input"


async def test_sync_streaming_tool_in_async_context():
    """A sync generator tool dispatched via the async surface should also work
    (sync next() pumped through asyncio.to_thread)."""
    from aimu.tools.decorator import tool

    @tool
    def sync_streamer(x: str):
        """Sync generator tool."""
        yield StreamChunk(StreamingContentType.GENERATING, "step1")
        yield StreamChunk(StreamingContentType.GENERATING, "step2")
        return f"sync:{x}"

    client = MockAsyncModelClient(["dummy"])
    client.tools = [sync_streamer]
    tc = [{"name": "sync_streamer", "arguments": {"x": "y"}}]

    chunks = [c async for c in client._handle_tool_calls_streamed(tc, tools=[])]
    tool_calls = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert tool_calls[0].content["response"] == "sync:y"


async def test_async_plain_tool_still_works_via_streamed_dispatch():
    """Plain async tools (no yield) dispatch through the streamed path unchanged."""
    from aimu.tools.decorator import tool

    @tool
    async def echo(x: str) -> str:
        """Plain async tool."""
        return x.upper()

    client = MockAsyncModelClient(["dummy"])
    client.tools = [echo]
    tc = [{"name": "echo", "arguments": {"x": "hi"}}]

    chunks = [c async for c in client._handle_tool_calls_streamed(tc, tools=[])]
    assert len(chunks) == 1
    assert chunks[0].phase == StreamingContentType.TOOL_CALLING
    assert chunks[0].content["response"] == "HI"
