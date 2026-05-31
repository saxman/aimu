"""Mock-only unit tests for the model client surface:

* ``resolve_model_string`` parses ``"provider:model_id"``.
* ``ModelSpec`` migration preserves capability flags + ``.value`` semantics.
* ``StreamChunk`` defaults + ``is_text()`` / ``is_tool_call()`` helpers.
* ``system_message`` is immutable after the first chat; ``reset()`` unlocks it.
* ``include=`` stream filter selects phases.
"""

from __future__ import annotations

import pytest

from aimu.models import (
    BaseImageClient,
    BaseModelClient,
    ModelSpec,
    StreamChunk,
    StreamingContentType,
    available_image_clients,
    available_text_clients,
    resolve_model_string,
)
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Model strings + ModelSpec
# ---------------------------------------------------------------------------


def test_resolve_model_string_known_id():
    """resolve_model_string finds an enum member by provider:id."""
    from aimu.models import HAS_ANTHROPIC

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    m = resolve_model_string("anthropic:claude-sonnet-4-6")
    assert m.value == "claude-sonnet-4-6"
    assert m.supports_tools


def test_resolve_model_string_unknown_provider_lists_options():
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_model_string("nonexistent:foo")


def test_resolve_model_string_requires_colon():
    with pytest.raises(ValueError, match="provider:model_id"):
        resolve_model_string("just-an-id")


def test_modelspec_holds_capabilities():
    spec = ModelSpec("x", tools=True, thinking=True, vision=False)
    assert spec.id == "x"
    assert spec.tools and spec.thinking
    assert not spec.vision


def test_modelspec_equality_by_id_only():
    """ModelSpec equality uses ``id`` so it stays hashable even with a dict field."""
    assert ModelSpec("a", tools=True) == ModelSpec("a")
    assert ModelSpec("a") != ModelSpec("b")


def test_model_enum_value_is_string():
    """``Model.X.value`` returns the id string."""
    from aimu.models import AnthropicModel

    assert AnthropicModel.CLAUDE_SONNET_4_6.value == "claude-sonnet-4-6"
    assert isinstance(AnthropicModel.CLAUDE_SONNET_4_6.value, str)


def test_model_enum_capability_attrs_accessible():
    """Each member exposes supports_tools / supports_thinking / supports_vision."""
    from aimu.models import AnthropicModel, OllamaModel

    assert AnthropicModel.CLAUDE_SONNET_4_6.supports_tools is True
    assert AnthropicModel.CLAUDE_SONNET_4_6.supports_vision is True
    assert OllamaModel.QWEN_3_8B.supports_thinking is True


# ---------------------------------------------------------------------------
# StreamChunk helpers
# ---------------------------------------------------------------------------


def test_streamchunk_has_agent_and_iteration_defaults():
    c = StreamChunk(StreamingContentType.GENERATING, "hi")
    assert c.agent is None
    assert c.iteration == 0


def test_streamchunk_is_text_helpers():
    text = StreamChunk(StreamingContentType.GENERATING, "hi")
    thinking = StreamChunk(StreamingContentType.THINKING, "hmm")
    tool_call = StreamChunk(
        StreamingContentType.TOOL_CALLING,
        {"name": "x", "arguments": {"k": "v"}, "response": "y"},
    )
    assert text.is_text() and not text.is_tool_call()
    assert thinking.is_text()
    assert tool_call.is_tool_call() and not tool_call.is_text()


def test_streamchunk_tool_calling_content_keys():
    """TOOL_CALLING chunks carry the model's argument dict alongside name and response."""
    chunk = StreamChunk(
        StreamingContentType.TOOL_CALLING,
        {"name": "calculate", "arguments": {"expression": "2+2"}, "response": "4"},
    )
    assert chunk.content["name"] == "calculate"
    assert chunk.content["arguments"] == {"expression": "2+2"}
    assert chunk.content["response"] == "4"


def test_streamchunk_image_progress_phase():
    """IMAGE_GENERATING phase + is_image_progress() helper."""
    chunk = StreamChunk(
        StreamingContentType.IMAGE_GENERATING,
        {"step": 5, "total_steps": 10, "image": None, "final": False, "result": None},
    )
    assert chunk.is_image_progress()
    assert not chunk.is_text()
    assert not chunk.is_tool_call()


# ---------------------------------------------------------------------------
# Streaming tool dispatch (generator tools through _handle_tool_calls_streamed)
# ---------------------------------------------------------------------------


def test_streaming_tool_yields_forward_through_dispatch():
    """A generator @tool's chunks should pass through _handle_tool_calls_streamed,
    and the generator's return value becomes the canonical tool response."""
    from aimu.tools.decorator import tool

    @tool
    def slow_tool(x: str):
        """Streaming tool that yields progress + returns a result."""
        yield StreamChunk(StreamingContentType.GENERATING, "working...")
        yield StreamChunk(StreamingContentType.GENERATING, "almost done...")
        return f"processed:{x}"

    client = MockModelClient(["dummy"])
    client.tools = [slow_tool]
    tc = [{"name": "slow_tool", "arguments": {"x": "input"}}]

    chunks = list(client._handle_tool_calls_streamed(tc, tools=[]))

    # Two progress chunks + one final TOOL_CALLING chunk.
    progress = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    tool_calls = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert len(progress) == 2
    assert progress[0].content == "working..."
    assert progress[1].content == "almost done..."
    assert len(tool_calls) == 1
    assert tool_calls[0].content["name"] == "slow_tool"
    assert tool_calls[0].content["response"] == "processed:input"


def test_streaming_tool_result_falls_back_to_last_chunk_result_key():
    """When the tool doesn't `return` a value, fall back to last chunk's content['result']."""
    from aimu.tools.decorator import tool

    @tool
    def streamer(prompt: str):
        """Tool with no return value — last chunk carries the result."""
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": "/tmp/x.png"},
        )
        # No return → return value is None → fall back to content["result"]

    client = MockModelClient(["dummy"])
    client.tools = [streamer]
    tc = [{"name": "streamer", "arguments": {"prompt": "hi"}}]

    chunks = list(client._handle_tool_calls_streamed(tc, tools=[]))
    tool_calls = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert len(tool_calls) == 1
    assert tool_calls[0].content["response"] == "/tmp/x.png"


def test_plain_tool_still_works_via_streamed_dispatch():
    """Non-generator @tool functions go through unchanged."""
    from aimu.tools.decorator import tool

    @tool
    def echo(x: str) -> str:
        """Echo the input."""
        return x.upper()

    client = MockModelClient(["dummy"])
    client.tools = [echo]
    tc = [{"name": "echo", "arguments": {"x": "hi"}}]

    chunks = list(client._handle_tool_calls_streamed(tc, tools=[]))
    assert len(chunks) == 1  # only TOOL_CALLING, no progress chunks
    assert chunks[0].phase == StreamingContentType.TOOL_CALLING
    assert chunks[0].content["response"] == "HI"


def test_streaming_tool_rejected_by_non_streaming_dispatch():
    """The non-streaming _handle_tool_calls path should refuse generator tools clearly."""
    from aimu.tools.decorator import tool

    @tool
    def slow_tool(x: str):
        """Generator tool."""
        yield StreamChunk(StreamingContentType.GENERATING, "hi")

    client = MockModelClient(["dummy"])
    client.tools = [slow_tool]
    tc = [{"name": "slow_tool", "arguments": {"x": "input"}}]

    # _handle_tool_calls dispatches per-tool via _call_plain_tool which raises
    # ValueError with a clear actionable message ("Failures are apparent" principle).
    with pytest.raises(ValueError, match=r"streaming.*stream=True"):
        client._handle_tool_calls(tc, tools=[])


# ---------------------------------------------------------------------------
# system_message immutability + reset()
# ---------------------------------------------------------------------------


def test_system_message_locks_after_first_chat():
    client = MockModelClient(["hello"])
    client.system_message = "v1"
    client.chat("hi")
    with pytest.raises(RuntimeError, match="immutable"):
        client.system_message = "v2"


def test_reset_unlocks_system_message():
    client = MockModelClient(["hello", "world"])
    client.system_message = "v1"
    client.chat("hi")
    client.reset()
    client.system_message = "v2"
    assert client.system_message == "v2"


def test_reset_preserves_system_message_by_default():
    client = MockModelClient(["hello"])
    client.system_message = "preserved"
    client.chat("hi")
    client.reset()
    assert client.system_message == "preserved"


def test_reset_can_clear_system_message():
    client = MockModelClient(["hello"])
    client.system_message = "v1"
    client.chat("hi")
    client.reset(system_message=None)
    assert client.system_message is None


# ---------------------------------------------------------------------------
# include= stream filter
# ---------------------------------------------------------------------------


def test_chat_stream_include_filters_phases():
    """include=['generating'] drops THINKING and TOOL_CALLING chunks."""

    class _MultiPhaseClient(MockModelClient):
        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
            if not stream:
                return super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            def _gen():
                yield StreamChunk(StreamingContentType.THINKING, "thinking")
                yield StreamChunk(StreamingContentType.GENERATING, "text")
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": "text"})

            return _gen()

    client = _MultiPhaseClient(["text"])
    chunks = list(client.chat("hi", stream=True, include=["generating"]))
    assert all(c.phase == StreamingContentType.GENERATING for c in chunks)


def test_chat_stream_include_thinking_only():
    class _MultiPhaseClient(MockModelClient):
        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
            if not stream:
                return super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            def _gen():
                yield StreamChunk(StreamingContentType.THINKING, "thinking")
                yield StreamChunk(StreamingContentType.GENERATING, "text")
                self.messages.append({"role": "user", "content": user_message})

            return _gen()

    client = _MultiPhaseClient(["x"])
    chunks = list(client.chat("hi", stream=True, include=["thinking"]))
    assert all(c.phase == StreamingContentType.THINKING for c in chunks)


# ---------------------------------------------------------------------------
# available_text_clients / available_image_clients
# ---------------------------------------------------------------------------


def test_available_text_clients_returns_list():
    clients = available_text_clients()
    assert isinstance(clients, list)


def test_available_text_clients_are_base_model_client_subclasses():
    for cls in available_text_clients():
        assert issubclass(cls, BaseModelClient), f"{cls} is not a BaseModelClient subclass"


def test_available_text_clients_have_tool_models():
    for cls in available_text_clients():
        assert hasattr(cls, "TOOL_MODELS"), f"{cls} missing TOOL_MODELS"


def test_available_text_clients_no_duplicates():
    clients = available_text_clients()
    assert len(clients) == len(set(clients))


def test_available_image_clients_returns_list():
    clients = available_image_clients()
    assert isinstance(clients, list)


def test_available_image_clients_are_base_image_client_subclasses():
    for cls in available_image_clients():
        assert issubclass(cls, BaseImageClient), f"{cls} is not a BaseImageClient subclass"


def test_available_image_clients_no_duplicates():
    clients = available_image_clients()
    assert len(clients) == len(set(clients))


# ---------------------------------------------------------------------------
# generate(images=) — stateless single-turn vision input
# ---------------------------------------------------------------------------

_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


def test_generate_images_builds_content_blocks():
    """generate(images=) threads the image into the request as OpenAI content blocks."""
    client = MockModelClient(["a description"])
    client.model.supports_vision = True

    out = client.generate("describe this", images=[_DATA_URL])

    assert out == "a description"
    # The user turn carries multimodal content blocks (text + image_url), not a bare string.
    user_content = client.messages[0]["content"]
    assert isinstance(user_content, list)
    assert [b["type"] for b in user_content] == ["text", "image_url"]
    assert user_content[1]["image_url"]["url"] == _DATA_URL


def test_generate_images_rejected_for_non_vision_model():
    """A non-vision model raises before any request is built (no file read, no API call)."""
    client = MockModelClient(["unused"])
    client.model.supports_vision = False

    with pytest.raises(ValueError, match="does not support vision input"):
        client.generate("describe this", images=[_DATA_URL])


def test_generate_without_images_stays_plain_string():
    """Omitting images keeps the single-turn content a plain string (unchanged behavior)."""
    client = MockModelClient(["plain"])
    client.model.supports_vision = False

    out = client.generate("hello")

    assert out == "plain"
    assert client.messages[0]["content"] == "hello"
