"""Mock-only unit tests for the model client surface:

* ``resolve_model_string`` parses ``"provider:model_id"``.
* ``ModelSpec`` migration preserves capability flags + ``.value`` semantics.
* ``StreamChunk`` defaults + ``is_text()`` / ``is_tool_call()`` helpers.
* ``system_message`` swaps the in-history system entry mid-conversation; ``reset()`` clears history.
* ``include=`` stream filter selects phases.
* ``parse_json_response`` / ``generate_json`` / ``extract_tool_calls`` utilities.
* HuggingFace model weight caching (mock-only).
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


# ---------------------------------------------------------------------------
# resolve_model_enum: enum passthrough / string / bare name / ambiguity
# ---------------------------------------------------------------------------


def test_resolve_model_enum_passes_through_member():
    from aimu.models import HAS_ANTHROPIC, AnthropicModel, resolve_model_enum

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    member = AnthropicModel.CLAUDE_SONNET_4_6
    assert resolve_model_enum(member) is member


def test_resolve_model_enum_delegates_provider_id_string():
    from aimu.models import HAS_ANTHROPIC, AnthropicModel, resolve_model_enum

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    assert resolve_model_enum("anthropic:claude-sonnet-4-6") is AnthropicModel.CLAUDE_SONNET_4_6


def test_resolve_model_enum_bare_name_unique(monkeypatch):
    """A bare name present in exactly one provider enum resolves to that member."""
    from enum import Enum

    import aimu.models.model_client as mc

    class FakeEnum(Enum):
        ONLY = "only-id"

    monkeypatch.setattr(mc, "_provider_registry", lambda: {"fake": (FakeEnum, object())})
    assert mc.resolve_model_enum("ONLY") is FakeEnum.ONLY


def test_resolve_model_enum_bare_name_ambiguous_unavailable_raises(monkeypatch):
    """An ambiguous bare name that isn't available under any provider raises."""
    from enum import Enum

    import aimu.models.model_client as mc

    class EnumA(Enum):
        SHARED = "a-id"

    class EnumB(Enum):
        SHARED = "b-id"

    monkeypatch.setattr(mc, "_provider_registry", lambda: {"a": (EnumA, object()), "b": (EnumB, object())})
    monkeypatch.setattr(mc, "available_text_models", lambda: [])  # nothing loadable locally
    with pytest.raises(ValueError, match="ambiguous"):
        mc.resolve_model_enum("SHARED")


def test_resolve_model_enum_ambiguous_resolved_by_local_availability(monkeypatch):
    """An ambiguous bare name resolves to the provider where it's locally available."""
    from enum import Enum

    import aimu.models.model_client as mc

    class EnumA(Enum):
        SHARED = "a-id"

    class EnumB(Enum):
        SHARED = "b-id"

    monkeypatch.setattr(mc, "_provider_registry", lambda: {"a": (EnumA, object()), "b": (EnumB, object())})
    # Only EnumB.SHARED is locally available → it wins, no raise.
    monkeypatch.setattr(mc, "available_text_models", lambda: [EnumB.SHARED])
    assert mc.resolve_model_enum("SHARED") is EnumB.SHARED


def test_resolve_model_enum_ambiguous_prefers_tool_capable(monkeypatch):
    """Among locally-available ambiguous matches, tool-capable wins (default-resolver order)."""
    from enum import Enum

    import aimu.models.model_client as mc

    class EnumA(Enum):
        SHARED = "a-id"

    class EnumB(Enum):
        SHARED = "b-id"

    class _FakeAvail:
        def __init__(self, tools):
            self.name = "SHARED"
            self.supports_tools = tools

    plain, tooly = _FakeAvail(False), _FakeAvail(True)
    monkeypatch.setattr(mc, "_provider_registry", lambda: {"a": (EnumA, object()), "b": (EnumB, object())})
    # Plain availability order lists the non-tool member first; tool-preference picks ``tooly``.
    monkeypatch.setattr(mc, "available_text_models", lambda: [plain, tooly])
    assert mc.resolve_model_enum("SHARED") is tooly


def test_resolve_model_enum_unknown_name_raises():
    from aimu.models import resolve_model_enum

    with pytest.raises(ValueError, match="Unknown model name"):
        resolve_model_enum("definitely-not-a-model")


def test_resolve_model_enum_rejects_bad_type():
    from aimu.models import resolve_model_enum

    with pytest.raises(TypeError):
        resolve_model_enum(123)


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

    chunks = list(client._handle_tool_calls_streamed(tc))

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
        """Tool with no return value; last chunk carries the result."""
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": "/tmp/x.png"},
        )
        # No return → return value is None → fall back to content["result"]

    client = MockModelClient(["dummy"])
    client.tools = [streamer]
    tc = [{"name": "streamer", "arguments": {"prompt": "hi"}}]

    chunks = list(client._handle_tool_calls_streamed(tc))
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

    chunks = list(client._handle_tool_calls_streamed(tc))
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
        client._handle_tool_calls(tc)


# ---------------------------------------------------------------------------
# Per-call tools= override on chat()
# ---------------------------------------------------------------------------


class _ToolsRecordingClient(MockModelClient):
    """MockModelClient that records the value of ``self.tools`` seen inside ``_chat``.

    Both spec-building (``_collect_python_tool_specs``) and dispatch (``_call_plain_tool``)
    read ``self.tools``, so observing it inside ``_chat`` is a faithful proxy for what the
    override exposes to the whole call.
    """

    def __init__(self, responses):
        super().__init__(responses)
        self.tools_seen = None

    def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self.tools_seen = list(self.tools)
        return super()._chat(user_message, generate_kwargs, use_tools, stream, images=images, audio=audio)


def test_chat_tools_override_applied_during_call():
    base_tool = object()
    override_tool = object()
    client = _ToolsRecordingClient(["ok"])
    client.tools = [base_tool]

    client.chat("hi", tools=[override_tool])

    assert client.tools_seen == [override_tool]


def test_chat_tools_override_restored_after_call():
    base_tool = object()
    client = _ToolsRecordingClient(["ok"])
    client.tools = [base_tool]

    client.chat("hi", tools=[object()])

    assert client.tools == [base_tool]


def test_chat_tools_none_is_noop_uses_client_tools():
    base_tool = object()
    client = _ToolsRecordingClient(["ok"])
    client.tools = [base_tool]

    client.chat("hi")  # tools omitted

    assert client.tools_seen == [base_tool]
    assert client.tools == [base_tool]


def test_chat_tools_empty_list_disables_python_tools_for_call():
    base_tool = object()
    client = _ToolsRecordingClient(["ok"])
    client.tools = [base_tool]

    client.chat("hi", tools=[])

    assert client.tools_seen == []  # override active...
    assert client.tools == [base_tool]  # ...and restored


def test_chat_tools_override_restored_on_exception():
    base_tool = object()

    class _Boom(_ToolsRecordingClient):
        def _chat(self, *a, **k):
            raise RuntimeError("boom")

    client = _Boom(["ok"])
    client.tools = [base_tool]

    with pytest.raises(RuntimeError, match="boom"):
        client.chat("hi", tools=[object()])

    assert client.tools == [base_tool]


def test_chat_tools_override_applies_across_streamed_consumption():
    """The override must stay live while the stream is consumed, not just at call time."""
    override_tool = object()
    client = _ToolsRecordingClient(["ok"])
    client.tools = [object()]

    stream = client.chat("hi", stream=True, tools=[override_tool])
    # _chat (where tools_seen is recorded) runs lazily on iteration.
    assert client.tools_seen is None
    list(stream)
    assert client.tools_seen == [override_tool]
    assert len(client.tools) == 1 and client.tools[0] is not override_tool  # restored


# ---------------------------------------------------------------------------
# system_message swap mid-conversation + reset()
# ---------------------------------------------------------------------------


def test_system_message_settable_mid_conversation():
    client = MockModelClient(["hello"])
    client.system_message = "v1"
    client.chat("hi")
    # Reassigning mid-conversation now swaps in place rather than raising.
    client.system_message = "v2"
    assert client.system_message == "v2"


def test_system_message_swap_rewrites_history_entry_in_place():
    client = MockModelClient([])
    client.messages = [
        {"role": "system", "content": "old persona"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    client.system_message = "new persona"
    assert client.messages[0] == {"role": "system", "content": "new persona"}
    # Conversation history is preserved; only the system entry changed.
    assert len(client.messages) == 3
    assert client.messages[1:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_system_message_swap_inserts_when_history_has_no_system_entry():
    client = MockModelClient([])
    client.messages = [{"role": "user", "content": "hello"}]
    client.system_message = "added persona"
    assert client.messages[0] == {"role": "system", "content": "added persona"}
    assert len(client.messages) == 2


def test_system_message_set_none_removes_history_entry():
    client = MockModelClient([])
    client.messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hello"},
    ]
    client.system_message = None
    assert client.system_message is None
    assert client.messages == [{"role": "user", "content": "hello"}]


def test_reset_then_system_message_settable():
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
        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
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
        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
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
# generate(images=): stateless single-turn vision input
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


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_raw():
    from aimu.models._internal.json import parse_json_response

    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_fenced_block():
    from aimu.models._internal.json import parse_json_response

    text = 'Here is the result:\n```json\n{"x": 42}\n```'
    assert parse_json_response(text) == {"x": 42}


def test_parse_json_brace_substring():
    from aimu.models._internal.json import parse_json_response

    text = 'The answer is {"key": "value"} as shown.'
    assert parse_json_response(text) == {"key": "value"}


def test_parse_json_raises_on_failure():
    from aimu.models._internal.json import parse_json_response

    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_json_response("no json here at all")


def test_parse_json_dataclass_schema():
    import dataclasses
    from aimu.models._internal.json import parse_json_response

    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    result = parse_json_response('{"x": 3, "y": 7}', schema=Point)
    assert isinstance(result, Point)
    assert result.x == 3
    assert result.y == 7


# ---------------------------------------------------------------------------
# generate_json
# ---------------------------------------------------------------------------


def test_generate_json_parses_on_first_attempt():
    from aimu.models._internal.json import generate_json

    client = MockModelClient(['{"score": 9}'])
    result = generate_json(client, "Rate this on a scale of 1-10 as JSON.")
    assert result == {"score": 9}


def test_generate_json_retries_on_bad_output():
    from aimu.models._internal.json import generate_json

    client = MockModelClient(["not json", '{"ok": true}'])
    result = generate_json(client, "Return JSON.", retries=2)
    assert result == {"ok": True}


def test_generate_json_raises_after_exhausted_retries():
    from aimu.models._internal.json import generate_json

    client = MockModelClient(["bad", "also bad", "still bad"])
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        generate_json(client, "Return JSON.", retries=2)


# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------


def test_extract_tool_calls_empty():
    from aimu.models._internal.json import extract_tool_calls

    assert extract_tool_calls([]) == []


def test_extract_tool_calls_single():
    from aimu.models._internal.json import extract_tool_calls

    messages = [
        {"role": "user", "content": "Search for cats"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "search", "arguments": '{"query": "cats"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "Found 42 cats"},
    ]
    calls = extract_tool_calls(messages)
    assert len(calls) == 1
    assert calls[0]["tool"] == "search"
    assert calls[0]["arguments"] == {"query": "cats"}
    assert calls[0]["result"] == "Found 42 cats"
    assert calls[0]["iteration"] == 1


def test_extract_tool_calls_multiple_iterations():
    from aimu.models._internal.json import extract_tool_calls

    messages = [
        {"role": "user", "content": "Start"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "a1", "function": {"name": "tool_a", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "a1", "content": "result_a"},
        {"role": "user", "content": "Continue"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "b1", "function": {"name": "tool_b", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "b1", "content": "result_b"},
    ]
    calls = extract_tool_calls(messages)
    assert len(calls) == 2
    assert calls[0]["iteration"] == 1
    assert calls[0]["tool"] == "tool_a"
    assert calls[1]["iteration"] == 2
    assert calls[1]["tool"] == "tool_b"


def test_extract_tool_calls_parameters_key_normalized():
    """Some models use 'parameters' instead of 'arguments'."""
    from aimu.models._internal.json import extract_tool_calls

    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "t1", "function": {"name": "fn", "parameters": '{"n": 5}'}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "done"},
    ]
    calls = extract_tool_calls(messages)
    assert calls[0]["arguments"] == {"n": 5}


def test_extract_tool_calls_missing_result_returns_empty_string():
    from aimu.models._internal.json import extract_tool_calls

    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "orphan", "function": {"name": "x", "arguments": "{}"}}],
        }
        # No matching tool-role message
    ]
    calls = extract_tool_calls(messages)
    assert calls[0]["result"] == ""


# ---------------------------------------------------------------------------
# HuggingFace model weight caching (mock-only)
# ---------------------------------------------------------------------------


def test_hf_client_cache_key_includes_model_kwargs():
    from aimu.models.providers.hf.text import _make_cache_key

    k1 = _make_cache_key("model/id", "causal-lm", {"device_map": "auto"})
    k2 = _make_cache_key("model/id", "causal-lm", {"device_map": "cuda:0"})
    assert k1 != k2


def test_hf_client_cache_key_same_kwargs_same_key():
    from aimu.models.providers.hf.text import _make_cache_key

    k1 = _make_cache_key("model/id", "causal-lm", {"device_map": "auto", "torch_dtype": "auto"})
    k2 = _make_cache_key("model/id", "causal-lm", {"torch_dtype": "auto", "device_map": "auto"})
    assert k1 == k2  # order of kwargs doesn't matter


def test_hf_client_cache_key_includes_load_profile():
    # Two enum members can share a repo id + kwargs but load via different classes
    # (AutoModelForCausalLM vs AutoModelForImageTextToText); the load-profile tag keeps
    # them from colliding in the weight cache.
    from aimu.models.providers.hf.text import _make_cache_key

    k1 = _make_cache_key("repo/id", "causal-lm", {"device_map": "auto"})
    k2 = _make_cache_key("repo/id", "qwen-multimodal", {"device_map": "auto"})
    assert k1 != k2


def test_hf_image_cache_key_includes_pipeline_class():
    from aimu.models.providers.hf.image import _make_cache_key

    k1 = _make_cache_key("repo/id", "FluxPipeline", {})
    k2 = _make_cache_key("repo/id", "DiffusionPipeline", {})
    assert k1 != k2


def test_clear_hf_cache_clears_registry():
    import aimu
    from aimu.models.providers.hf.text import _model_registry, _registry_lock

    fake_key = ("fake-model-id",)
    with _registry_lock:
        _model_registry[fake_key] = ("fake_model", "fake_tokenizer", None, False)
    assert fake_key in _model_registry

    aimu.clear_hf_cache()

    assert fake_key not in _model_registry


def test_clear_llamacpp_cache_clears_registry():
    import aimu
    from aimu.models.providers.llamacpp import _model_registry, _registry_lock

    fake_key = ("/path/to/model.gguf", 4096, -1, "None")
    with _registry_lock:
        _model_registry[fake_key] = object()
    assert fake_key in _model_registry

    aimu.clear_llamacpp_cache()

    assert fake_key not in _model_registry


# ---------------------------------------------------------------------------
# HuggingFace streaming: thinking-only turn (regression for the unguarded next())
# ---------------------------------------------------------------------------


def test_hf_chat_streamed_thinking_only_turn_does_not_crash(monkeypatch):
    """A thinking-only streamed turn must not raise ``RuntimeError``.

    When a thinking model opens a ``<think>`` block but generation ends before
    ``</think>`` (e.g. token budget exhausted with ``do_sample=True``),
    ``_generate_streaming`` returns an empty iterator. ``_chat_streamed`` used to
    do an unguarded ``next(it)`` on it, and a ``StopIteration`` raised inside the
    generator surfaced as ``RuntimeError: generator raised StopIteration`` (PEP 479).
    It must instead surface the buffered thinking and finish with empty content,
    mirroring the non-streaming ``_generate_sync`` path.
    """
    import types

    pytest.importorskip("transformers")
    import aimu.models.providers.hf.text as hf_text

    # No real tokenizer/model needed: the streamer is stubbed and _generate_streaming
    # is replaced with one that returns the empty iterator the bug hinges on.
    monkeypatch.setattr(hf_text, "TextIteratorStreamer", lambda *a, **k: iter([]))

    fake = types.SimpleNamespace(
        _uses_processor_parse_response=False,
        _hf_tokenizer=types.SimpleNamespace(decode=lambda *_a, **_k: "<eos>"),
        _hf_model=types.SimpleNamespace(config=types.SimpleNamespace(eos_token_id=0)),
        model=types.SimpleNamespace(tool_call_format=None),
        messages=[],
        last_thinking="reasoning that never closed",
        _pending_thinking_tokens=["reasoning that never closed"],
        _generate_streaming=lambda *a, **k: iter([]),
    )

    chunks = list(hf_text.HuggingFaceClient._chat_streamed(fake, {}, []))

    assert [c.phase for c in chunks] == [StreamingContentType.THINKING]
    assert chunks[0].content == "reasoning that never closed"
    assert fake.messages == [{"role": "assistant", "content": "", "thinking": "reasoning that never closed"}]


# ---------------------------------------------------------------------------
# Thinking is stored in messages under a "thinking" key, consistently across
# providers (HuggingFace and Ollama already do this; llama-cpp and OpenAI-compat
# must match so the UI / persistence sees per-turn reasoning uniformly).
# ---------------------------------------------------------------------------


def _llamacpp_fake(content: str):
    import types

    return types.SimpleNamespace(
        _chat_setup=lambda *a, **k: ({}, []),
        _llm=types.SimpleNamespace(
            create_chat_completion=lambda **kw: {"choices": [{"message": {"content": content}}]}
        ),
        is_thinking_model=True,
        messages=[],
    )


def test_llamacpp_chat_stores_thinking_in_messages():
    from aimu.models.providers.llamacpp import LlamaCppClient

    fake = _llamacpp_fake("<think>step by step</think>the answer")
    result = LlamaCppClient._chat(fake, "hi")

    assert result == "the answer"
    assert fake.messages[-1] == {"role": "assistant", "content": "the answer", "thinking": "step by step"}


def test_llamacpp_chat_omits_thinking_key_when_no_thinking():
    from aimu.models.providers.llamacpp import LlamaCppClient

    fake = _llamacpp_fake("just an answer, no think block")
    LlamaCppClient._chat(fake, "hi")

    assert fake.messages[-1] == {"role": "assistant", "content": "just an answer, no think block"}
    assert "thinking" not in fake.messages[-1]


def test_llamacpp_chat_streamed_stores_thinking_in_messages():
    import types

    from aimu.models.providers.llamacpp import LlamaCppClient

    deltas = [
        {"choices": [{"delta": {"content": "<think>reasoning</think>"}}]},
        {"choices": [{"delta": {"content": "the answer"}}]},
    ]
    fake = types.SimpleNamespace(
        _llm=types.SimpleNamespace(create_chat_completion=lambda **kw: iter(deltas)),
        is_thinking_model=True,
        messages=[],
    )

    chunks = list(LlamaCppClient._chat_streamed(fake, {}, []))

    assert "".join(c.content for c in chunks if c.phase == StreamingContentType.GENERATING) == "the answer"
    assert fake.messages[-1] == {"role": "assistant", "content": "the answer", "thinking": "reasoning"}


def _openai_compat_fake(content: str):
    import types

    msg = types.SimpleNamespace(content=content, tool_calls=None)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    completions = types.SimpleNamespace(create=lambda **kw: response)
    return types.SimpleNamespace(
        _chat_setup=lambda *a, **k: ({}, []),
        _with_response_format=lambda gk, rf: gk,
        _client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions)),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_usage=None,
        messages=[],
    )


def test_openai_compat_chat_stores_thinking_in_messages():
    from aimu.models.providers.openai_compat import OpenAICompatClient

    fake = _openai_compat_fake("<think>deliberating</think>final")
    result = OpenAICompatClient._chat(fake, "hi")

    assert result == "final"
    assert fake.messages[-1] == {"role": "assistant", "content": "final", "thinking": "deliberating"}


def test_openai_compat_chat_omits_thinking_key_when_no_thinking():
    from aimu.models.providers.openai_compat import OpenAICompatClient

    fake = _openai_compat_fake("plain answer")
    OpenAICompatClient._chat(fake, "hi")

    assert fake.messages[-1] == {"role": "assistant", "content": "plain answer"}
    assert "thinking" not in fake.messages[-1]


def test_openai_compat_chat_streamed_stores_thinking_in_messages():
    import types

    from aimu.models.providers.openai_compat import OpenAICompatClient

    def _delta(content):
        return types.SimpleNamespace(
            usage=None,
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(tool_calls=None, content=content))],
        )

    deltas = [_delta("<think>reasoning</think>"), _delta("the answer")]
    fake = types.SimpleNamespace(
        _client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: iter(deltas)))
        ),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_thinking="",
        last_usage=None,
        messages=[],
    )

    chunks = list(OpenAICompatClient._chat_streamed(fake, {}, []))

    assert "".join(c.content for c in chunks if c.phase == StreamingContentType.GENERATING) == "the answer"
    assert fake.messages[-1] == {"role": "assistant", "content": "the answer", "thinking": "reasoning"}


# ---------------------------------------------------------------------------
# Tool-call turn reasoning is preserved on the assistant tool-call message
# (llama-cpp + OpenAI-compat; HuggingFace and Ollama already did this).
# ---------------------------------------------------------------------------


def _seq(*items):
    it = iter(items)
    return lambda **kw: next(it)


def _stub_tool_dispatch(fake):
    """Make _handle_tool_calls append a tool-call assistant message + a tool result."""

    def _handle(tool_calls, content=""):
        msg = {"role": "assistant", "tool_calls": [{"id": "x"}]}
        if content:
            msg["content"] = content
        fake.messages.append(msg)
        fake.messages.append({"role": "tool", "content": "result", "tool_call_id": "x"})

    return _handle


def test_llamacpp_tool_call_turn_thinking_preserved():
    import types

    from aimu.models.providers.llamacpp import LlamaCppClient

    first = {
        "choices": [
            {
                "message": {
                    "content": "<think>reasoning before tool</think>",
                    "tool_calls": [{"function": {"name": "echo", "arguments": "{}"}}],
                }
            }
        ]
    }
    fake = types.SimpleNamespace(
        _chat_setup=lambda *a, **k: ({}, []),
        _llm=types.SimpleNamespace(create_chat_completion=_seq(first)),
        is_thinking_model=True,
        last_thinking="",
        messages=[],
    )
    fake._handle_tool_calls = _stub_tool_dispatch(fake)

    # Single turn: the tool executes and chat() returns (no follow-up). Reasoning from the
    # tool-call turn is attached to the assistant(tool_calls) message; the answer comes next turn.
    result = LlamaCppClient._chat(fake, "hi")

    assert result == ""  # tool-only turn (its content was all thinking)
    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["thinking"] == "reasoning before tool"
    assert fake.messages[-1] == {"role": "tool", "content": "result", "tool_call_id": "x"}


def test_openai_compat_tool_call_turn_thinking_preserved():
    import types

    from aimu.models.providers.openai_compat import OpenAICompatClient

    tc = types.SimpleNamespace(function=types.SimpleNamespace(name="echo", arguments="{}"))
    first_msg = types.SimpleNamespace(content="<think>reasoning before tool</think>", tool_calls=[tc])
    first = types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_msg)])
    fake = types.SimpleNamespace(
        _chat_setup=lambda *a, **k: ({}, []),
        _with_response_format=lambda gk, rf: gk,
        _client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_seq(first)))
        ),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_usage=None,
        last_thinking="",
        messages=[],
    )
    fake._handle_tool_calls = _stub_tool_dispatch(fake)

    # Single turn: the tool executes and chat() returns (no follow-up answer).
    result = OpenAICompatClient._chat(fake, "hi")

    assert result == ""  # tool-only turn (its content was all thinking)
    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["thinking"] == "reasoning before tool"
    assert fake.messages[-1] == {"role": "tool", "content": "result", "tool_call_id": "x"}


def test_openai_compat_preserves_content_alongside_tool_call():
    """A single generation carrying BOTH prose and a tool call keeps the prose as the
    assistant message's content, and chat() returns it (matches HuggingFace behavior)."""
    import types

    from aimu.models.providers.openai_compat import OpenAICompatClient

    tc = types.SimpleNamespace(function=types.SimpleNamespace(name="echo", arguments="{}"))
    first_msg = types.SimpleNamespace(content="Let me look that up for you.", tool_calls=[tc])
    first = types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_msg)])
    fake = types.SimpleNamespace(
        _chat_setup=lambda *a, **k: ({}, []),
        _with_response_format=lambda gk, rf: gk,
        _client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_seq(first)))
        ),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=False,
        last_usage=None,
        last_thinking="",
        messages=[],
    )
    fake._handle_tool_calls = _stub_tool_dispatch(fake)

    result = OpenAICompatClient._chat(fake, "hi")

    assert result == "Let me look that up for you."  # prose returned even on a tool turn
    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["content"] == "Let me look that up for you."  # prose stored with the tool call


def _stub_streamed_tool_dispatch(fake):
    def _handle(tool_calls, content=""):
        msg = {"role": "assistant", "tool_calls": [{"id": "x"}]}
        if content:
            msg["content"] = content
        fake.messages.append(msg)
        fake.messages.append({"role": "tool", "content": "result", "tool_call_id": "x"})
        return iter(())

    return _handle


def test_llamacpp_streamed_tool_call_turn_thinking_preserved():
    import types

    from aimu.models.providers.llamacpp import LlamaCppClient

    first_stream = [
        {"choices": [{"delta": {"content": "<think>reasoning before tool</think>"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "echo", "arguments": "{}"}}]}}]},
    ]
    fake = types.SimpleNamespace(
        _llm=types.SimpleNamespace(create_chat_completion=_seq(iter(first_stream))),
        is_thinking_model=True,
        last_thinking="",
        messages=[],
    )
    fake._handle_tool_calls_streamed = _stub_streamed_tool_dispatch(fake)
    fake._iter_stream = lambda stream: LlamaCppClient._iter_stream(fake, stream)

    # Single turn: dispatch the tools and stop (no second stream); answer comes next turn.
    list(LlamaCppClient._chat_streamed(fake, {}, []))

    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["thinking"] == "reasoning before tool"
    assert fake.messages[-1] == {"role": "tool", "content": "result", "tool_call_id": "x"}


def test_openai_compat_streamed_tool_call_turn_thinking_preserved():
    import types

    from aimu.models.providers.openai_compat import OpenAICompatClient

    def _delta(content=None, tool_calls=None):
        return types.SimpleNamespace(
            usage=None,
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content, tool_calls=tool_calls))],
        )

    tc_delta = types.SimpleNamespace(index=0, function=types.SimpleNamespace(name="echo", arguments="{}"))
    first_stream = [_delta(content="<think>reasoning before tool</think>"), _delta(tool_calls=[tc_delta])]
    fake = types.SimpleNamespace(
        _client=types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_seq(iter(first_stream))))
        ),
        model=types.SimpleNamespace(value="fake-model"),
        is_thinking_model=True,
        last_thinking="",
        last_usage=None,
        messages=[],
    )
    fake._handle_tool_calls_streamed = _stub_streamed_tool_dispatch(fake)
    fake._iter_stream = lambda stream: OpenAICompatClient._iter_stream(fake, stream)

    # Single turn: dispatch the tools and stop (no second stream); answer comes next turn.
    list(OpenAICompatClient._chat_streamed(fake, {}, []))

    tool_call_msg = next(m for m in fake.messages if "tool_calls" in m)
    assert tool_call_msg["thinking"] == "reasoning before tool"
    assert fake.messages[-1] == {"role": "tool", "content": "result", "tool_call_id": "x"}


def test_toolcall_format_strip_calls_recovers_prose():
    """ToolCallFormat.strip_calls returns the prose around a tool call (HF single-generation
    content+tool_calls support). JSON-only formats carry no accompanying prose."""
    pytest.importorskip("transformers")
    from aimu.models.providers.hf.text import ToolCallFormat

    xml = 'Let me check.\n<tool_call>{"name": "w", "arguments": {}}</tool_call>'
    assert ToolCallFormat.XML.strip_calls(xml) == "Let me check."

    bracketed = 'Sure thing. [TOOL_CALLS][{"name": "w"}]'
    assert ToolCallFormat.BRACKETED.strip_calls(bracketed) == "Sure thing."

    # JSON-only formats are the whole tool call -> no prose.
    assert ToolCallFormat.JSON_OBJECT.strip_calls('{"name": "w"}') == ""
