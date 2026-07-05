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


async def test_system_message_settable_mid_conversation():
    client = MockAsyncModelClient(["hello"])
    client.system_message = "v1"
    await client.chat("hi")
    # Reassigning mid-conversation now swaps in place rather than raising.
    client.system_message = "v2"
    assert client.system_message == "v2"


async def test_system_message_swap_rewrites_history_entry_in_place():
    client = MockAsyncModelClient([])
    client.messages = [
        {"role": "system", "content": "old persona"},
        {"role": "user", "content": "hello"},
    ]
    client.system_message = "new persona"
    assert client.messages[0] == {"role": "system", "content": "new persona"}
    assert client.messages[1] == {"role": "user", "content": "hello"}


async def test_reset_then_system_message_settable():
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
        async def _chat(
            self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
        ):
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
# Streaming tool dispatch: async generators
# ---------------------------------------------------------------------------


async def test_async_streaming_tool_yields_forward_through_dispatch():
    """An async generator @tool's chunks flow through _handle_tool_calls_streamed."""
    from aimu.tools.decorator import tool

    @tool
    async def slow_tool(x: str):
        """Async streaming tool with no return value; last chunk carries result."""
        yield StreamChunk(StreamingContentType.GENERATING, "working...")
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": f"done:{x}"},
        )

    client = MockAsyncModelClient(["dummy"])
    client.tools = [slow_tool]
    tc = [{"name": "slow_tool", "arguments": {"x": "input"}}]

    chunks = [c async for c in client._handle_tool_calls_streamed(tc)]

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

    chunks = [c async for c in client._handle_tool_calls_streamed(tc)]
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

    chunks = [c async for c in client._handle_tool_calls_streamed(tc)]
    assert len(chunks) == 1
    assert chunks[0].phase == StreamingContentType.TOOL_CALLING
    assert chunks[0].content["response"] == "HI"


# ---------------------------------------------------------------------------
# generate(images=): stateless single-turn vision input (async mirror)
# ---------------------------------------------------------------------------

_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


async def test_async_generate_images_builds_content_blocks():
    client = MockAsyncModelClient(["a description"])
    client.model.supports_vision = True

    out = await client.generate("describe this", images=[_DATA_URL])

    assert out == "a description"
    user_content = client.messages[0]["content"]
    assert isinstance(user_content, list)
    assert [b["type"] for b in user_content] == ["text", "image_url"]
    assert user_content[1]["image_url"]["url"] == _DATA_URL


async def test_async_generate_images_rejected_for_non_vision_model():
    client = MockAsyncModelClient(["unused"])
    client.model.supports_vision = False

    with pytest.raises(ValueError, match="does not support vision input"):
        await client.generate("describe this", images=[_DATA_URL])


# ---------------------------------------------------------------------------
# Per-call tools= override on async chat()
# ---------------------------------------------------------------------------


class _AsyncToolsRecordingClient(MockAsyncModelClient):
    """Records ``self.tools`` seen inside ``_chat`` (proxy for what the override exposes)."""

    def __init__(self, responses):
        super().__init__(responses)
        self.tools_seen = None

    async def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self.tools_seen = list(self.tools)
        return await super()._chat(user_message, generate_kwargs, use_tools, stream, images=images)


async def test_async_chat_tools_override_applied_and_restored():
    base_tool, override_tool = object(), object()
    client = _AsyncToolsRecordingClient(["ok"])
    client.tools = [base_tool]

    await client.chat("hi", tools=[override_tool])

    assert client.tools_seen == [override_tool]
    assert client.tools == [base_tool]


async def test_async_chat_tools_none_is_noop():
    base_tool = object()
    client = _AsyncToolsRecordingClient(["ok"])
    client.tools = [base_tool]

    await client.chat("hi")

    assert client.tools_seen == [base_tool]
    assert client.tools == [base_tool]


async def test_async_chat_tools_override_applies_across_streamed_consumption():
    override_tool = object()
    client = _AsyncToolsRecordingClient(["ok"])
    client.tools = [object()]

    stream = await client.chat("hi", stream=True, tools=[override_tool])
    assert client.tools_seen is None  # lazy, not yet consumed
    async for _ in stream:
        pass
    assert client.tools_seen == [override_tool]
    assert len(client.tools) == 1 and client.tools[0] is not override_tool  # restored


# ---------------------------------------------------------------------------
# Thinking is stored in messages under a "thinking" key (async OpenAI-compat),
# matching the sync surface and the HuggingFace/Ollama convention.
# ---------------------------------------------------------------------------


async def test_aio_openai_compat_chat_stores_thinking_in_messages():
    import types

    from aimu.aio.providers.openai_compat import AsyncOpenAICompatClient

    async def _create(**kw):
        msg = types.SimpleNamespace(content="<think>deliberating</think>final", tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    async def _chat_setup(*a, **k):
        return ({}, [])

    fake = types.SimpleNamespace(
        _chat_setup=_chat_setup,
        _client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_usage=None,
        messages=[],
    )

    result = await AsyncOpenAICompatClient._chat(fake, "hi")

    assert result == "final"
    assert fake.messages[-1] == {"role": "assistant", "content": "final", "thinking": "deliberating"}


# ---------------------------------------------------------------------------
# Tool-call turn reasoning is preserved on the assistant tool-call message
# (async OpenAI-compat), matching the sync surface.
# ---------------------------------------------------------------------------


def _aio_seq(*items):
    it = iter(items)

    async def _create(**kw):
        return next(it)

    return _create


async def test_aio_openai_compat_tool_call_turn_thinking_preserved():
    import types

    from aimu.aio.providers.openai_compat import AsyncOpenAICompatClient

    tc = types.SimpleNamespace(function=types.SimpleNamespace(name="echo", arguments="{}"))
    first_msg = types.SimpleNamespace(content="<think>reasoning before tool</think>", tool_calls=[tc])
    first = types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_msg)])

    async def _chat_setup(*a, **k):
        return ({}, [])

    fake = types.SimpleNamespace(
        _chat_setup=_chat_setup,
        _client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_aio_seq(first)))
        ),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_usage=None,
        last_thinking="",
        messages=[],
    )

    async def _handle(tool_calls, content=""):
        msg = {"role": "assistant", "tool_calls": [{"id": "x"}]}
        if content:
            msg["content"] = content
        fake.messages.append(msg)
        fake.messages.append({"role": "tool", "content": "result", "tool_call_id": "x"})

    fake._handle_tool_calls = _handle

    # Single turn: the tool is executed and chat() returns (no follow-up answer). The tool-call
    # turn's reasoning is attached to the assistant(tool_calls) message; the answer comes next turn.
    result = await AsyncOpenAICompatClient._chat(fake, "hi")

    assert result == ""  # tool-only turn (its content was all thinking)
    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["thinking"] == "reasoning before tool"
    assert fake.messages[-1] == {"role": "tool", "content": "result", "tool_call_id": "x"}


async def test_aio_openai_compat_preserves_content_alongside_tool_call():
    """Async: a generation with both prose and a tool call keeps the prose as content."""
    import types

    from aimu.aio.providers.openai_compat import AsyncOpenAICompatClient

    tc = types.SimpleNamespace(function=types.SimpleNamespace(name="echo", arguments="{}"))
    first_msg = types.SimpleNamespace(content="Let me look that up.", tool_calls=[tc])
    first = types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_msg)])

    async def _chat_setup(*a, **k):
        return ({}, [])

    fake = types.SimpleNamespace(
        _chat_setup=_chat_setup,
        _client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_aio_seq(first)))
        ),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=False,
        last_usage=None,
        last_thinking="",
        messages=[],
    )

    async def _handle(tool_calls, content=""):
        msg = {"role": "assistant", "tool_calls": [{"id": "x"}]}
        if content:
            msg["content"] = content
        fake.messages.append(msg)
        fake.messages.append({"role": "tool", "content": "result", "tool_call_id": "x"})

    fake._handle_tool_calls = _handle

    result = await AsyncOpenAICompatClient._chat(fake, "hi")

    assert result == "Let me look that up."
    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["content"] == "Let me look that up."


async def test_aio_openai_compat_streamed_tool_call_turn_thinking_preserved():
    import types

    from aimu.aio.providers.openai_compat import AsyncOpenAICompatClient

    def _delta(content=None, tool_calls=None):
        return types.SimpleNamespace(
            usage=None,
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content, tool_calls=tool_calls))],
        )

    async def _astream(items):
        for it in items:
            yield it

    tc_delta = types.SimpleNamespace(index=0, function=types.SimpleNamespace(name="echo", arguments="{}"))
    first = [_delta(content="<think>reasoning before tool</think>"), _delta(tool_calls=[tc_delta])]
    streams = iter([_astream(first)])

    async def _create(**kw):
        return next(streams)

    fake = types.SimpleNamespace(
        _client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_thinking="",
        last_usage=None,
        messages=[],
    )
    fake._iter_stream = lambda stream: AsyncOpenAICompatClient._iter_stream(fake, stream)

    async def _handle(tool_calls, content=""):
        msg = {"role": "assistant", "tool_calls": [{"id": "x"}]}
        if content:
            msg["content"] = content
        fake.messages.append(msg)
        fake.messages.append({"role": "tool", "content": "result", "tool_call_id": "x"})
        return
        yield  # make this an async generator

    fake._handle_tool_calls_streamed = _handle

    # Single turn: dispatch the tools and stop (no second stream). Thinking from the tool-call
    # turn is attached to the assistant(tool_calls) message; the answer comes on the next turn.
    async for _ in AsyncOpenAICompatClient._chat_streamed(fake, {}, []):
        pass

    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["thinking"] == "reasoning before tool"
    assert fake.messages[-1] == {"role": "tool", "content": "result", "tool_call_id": "x"}
