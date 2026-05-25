"""Mock-only unit tests for the model client surface:

* ``resolve_model_string`` parses ``"provider:model_id"``.
* ``ModelSpec`` migration preserves capability flags + ``.value`` semantics.
* ``StreamChunk`` defaults + ``is_text()`` / ``is_tool_call()`` helpers.
* ``system_message`` is immutable after the first chat; ``reset()`` unlocks it.
* ``include=`` stream filter selects phases.
"""

from __future__ import annotations

import pytest

from aimu.models import ModelSpec, StreamChunk, StreamingContentType, resolve_model_string
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
