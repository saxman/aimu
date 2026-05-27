"""Mock-only unit tests for the built-in ``generate_image`` tool and ``make_image_tool``.

Verifies the lazy singleton wiring, the tool spec shape, env-var override, and
the per-agent factory escape hatch. No diffusers install required.
"""

from __future__ import annotations

# Install the diffusers stub before importing aimu.tools.builtin's image bits.
from test_images_api import _install_diffusers_stub  # noqa: F401 — side effect

_install_diffusers_stub()

import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_IMAGE:
    aimu.models.hf_image = importlib.import_module("aimu.models.hf_image")
    aimu.models.HAS_HF_IMAGE = True
    aimu.models.HuggingFaceImageClient = aimu.models.hf_image.HuggingFaceImageClient
    aimu.models.HuggingFaceImageModel = aimu.models.hf_image.HuggingFaceImageModel

import aimu  # noqa: E402

aimu.HAS_HF_IMAGE = True
aimu.HuggingFaceImageClient = aimu.models.HuggingFaceImageClient
aimu.HuggingFaceImageModel = aimu.models.HuggingFaceImageModel

from aimu.tools import builtin  # noqa: E402


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_generate_image_has_tool_spec():
    spec = builtin.generate_image.__tool_spec__
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "generate_image"
    assert "prompt" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["properties"]["prompt"]["type"] == "string"
    assert spec["function"]["parameters"]["required"] == ["prompt"]


def test_generate_image_is_sync():
    """The sync built-in must be a regular def, not async — async lives under aimu.aio.tools."""
    assert builtin.generate_image.__tool_is_async__ is False


def test_generate_image_is_streaming():
    """The sync built-in is now a generator (streaming) tool."""
    assert builtin.generate_image.__tool_is_streaming__ is True


def test_generate_image_in_image_subgroup():
    assert builtin.generate_image in builtin.image
    assert builtin.generate_image in builtin.ALL_TOOLS


# ---------------------------------------------------------------------------
# Singleton + env var
# ---------------------------------------------------------------------------


def test_lazy_singleton_constructed_once(monkeypatch):
    """_get_image_client should cache a single image client and reuse it."""
    monkeypatch.setattr(builtin, "_image_client", None)
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:test/repo")

    constructed: list[str] = []

    def fake_image_client(model_str):
        constructed.append(model_str)
        from aimu.models.hf_image import HuggingFaceImageClient

        return HuggingFaceImageClient(model_str)

    monkeypatch.setattr("aimu.image_client", fake_image_client)

    c1 = builtin._get_image_client()
    c2 = builtin._get_image_client()
    assert c1 is c2
    assert constructed == ["hf:test/repo"]


def test_singleton_honours_env_var(monkeypatch, tmp_path):
    """AIMU_IMAGE_MODEL env var should pick the singleton's model."""
    monkeypatch.setattr(builtin, "_image_client", None)
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:my/custom-repo")

    c = builtin._get_image_client()
    assert c.spec.id == "my/custom-repo"


def test_tool_drains_generator_and_returns_path(monkeypatch, tmp_path):
    """Calling generate_image() yields IMAGE_GENERATING chunks and `return`s the final path."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    final_path = f"{tmp_path}/fake.png"

    def fake_stream(prompt, format, stream):  # noqa: ARG001
        # Two progress chunks, then a final chunk carrying the result.
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 2, "image": None, "final": False, "result": None},
        )
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 2, "total_steps": 2, "image": None, "final": True, "result": final_path},
        )

    monkeypatch.setattr(builtin, "_image_client", HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5))
    monkeypatch.setattr(builtin._get_image_client(), "generate", fake_stream)

    # The tool is a generator — drain it, then read the return value via StopIteration.value.
    gen = builtin.generate_image("a cat")
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        result = stop.value

    assert len(chunks) == 2
    assert all(c.phase == StreamingContentType.IMAGE_GENERATING for c in chunks)
    assert result == final_path


# ---------------------------------------------------------------------------
# make_image_tool — per-agent factory escape hatch
# ---------------------------------------------------------------------------


def test_make_image_tool_returns_new_streaming_tool_bound_to_supplied_client():
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_SCHNELL)
    bound_tool = builtin.make_image_tool(client)

    assert bound_tool is not builtin.generate_image
    assert bound_tool.__tool_spec__["function"]["name"] == "generate_image"
    assert bound_tool.__tool_is_async__ is False
    assert bound_tool.__tool_is_streaming__ is True


def test_make_image_tool_threads_preview_every_through(monkeypatch, tmp_path):
    """make_image_tool(preview_every=N) should pass N to client.generate(stream=True)."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    captured: dict = {}

    def fake_stream(prompt, format, stream, preview_every):  # noqa: ARG001
        captured["preview_every"] = preview_every
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": f"{tmp_path}/x.png"},
        )

    custom = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_SCHNELL)
    monkeypatch.setattr(custom, "generate", fake_stream)
    bound = builtin.make_image_tool(custom, preview_every=7)

    # Drain the generator.
    list(bound("a fox"))
    assert captured["preview_every"] == 7


def test_make_image_tool_uses_its_client_not_singleton(monkeypatch, tmp_path):
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    # Singleton should remain untouched; the bound tool should call its own client.
    sentinel = "singleton-should-not-be-touched"
    monkeypatch.setattr(builtin, "_image_client", sentinel)

    custom = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_SCHNELL)
    bound_tool = builtin.make_image_tool(custom)

    final_path = f"{tmp_path}/custom.png"

    def fake_stream(prompt, format, stream, preview_every):  # noqa: ARG001
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": final_path},
        )

    monkeypatch.setattr(custom, "generate", fake_stream)

    gen = bound_tool("a fox")
    list(gen)  # drain
    # Singleton wasn't constructed.
    assert builtin._image_client is sentinel


# ---------------------------------------------------------------------------
# make_describe_image_tool — vision-capable chat client binding
# ---------------------------------------------------------------------------


def _fake_vision_chat_client(supports_vision=True):
    """A minimal stub matching the BaseModelClient interface used by describe_image."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.model = MagicMock()
    client.model.supports_vision = supports_vision
    client.model.value = "stub:vision" if supports_vision else "stub:text-only"
    client.messages = []
    # chat() echoes the (instruction, images) so tests can assert plumbing.
    client.chat = MagicMock(side_effect=lambda inst, images, use_tools=True: f"saw {images} :: {inst}")
    return client


def test_make_describe_image_tool_rejects_non_vision_client():
    """ValueError when the bound client's model lacks vision support."""
    import pytest

    client = _fake_vision_chat_client(supports_vision=False)
    with pytest.raises(ValueError, match="does not support vision input"):
        builtin.make_describe_image_tool(client)


def test_describe_image_tool_spec_and_flags():
    client = _fake_vision_chat_client()
    tool_fn = builtin.make_describe_image_tool(client)

    spec = tool_fn.__tool_spec__
    assert spec["function"]["name"] == "describe_image"
    # Both args present; instruction is optional (has default), image_path is required.
    assert "image_path" in spec["function"]["parameters"]["properties"]
    assert "instruction" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["required"] == ["image_path"]
    # Not a streaming tool — it's a plain function returning a string.
    assert tool_fn.__tool_is_streaming__ is False
    assert tool_fn.__tool_is_async__ is False


def test_describe_image_calls_chat_with_images_and_use_tools_false():
    client = _fake_vision_chat_client()
    tool_fn = builtin.make_describe_image_tool(client)

    result = tool_fn("/tmp/cat.png")
    assert result == "saw ['/tmp/cat.png'] :: Describe this image in detail."
    # use_tools=False was passed (prevents recursive tool calls during vision).
    _, kwargs = client.chat.call_args
    assert kwargs["use_tools"] is False
    assert kwargs["images"] == ["/tmp/cat.png"]


def test_describe_image_preserves_message_history():
    """The vision call must not pollute the agent's conversation log."""
    client = _fake_vision_chat_client()
    # Simulate an in-progress conversation.
    original_messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    client.messages = list(original_messages)

    # Stub chat() to mutate messages the way a real client would, so we can verify restoration.
    def _chat(instruction, images, use_tools=True):  # noqa: ARG001
        client.messages.append({"role": "user", "content": instruction})
        client.messages.append({"role": "assistant", "content": "looks like a cat"})
        return "looks like a cat"

    client.chat = _chat

    tool_fn = builtin.make_describe_image_tool(client)
    result = tool_fn("/tmp/cat.png")
    assert result == "looks like a cat"
    # History restored exactly — including identity ordering.
    assert client.messages == original_messages


def test_describe_image_custom_instruction():
    client = _fake_vision_chat_client()
    tool_fn = builtin.make_describe_image_tool(client)
    out = tool_fn("/tmp/x.png", instruction="What text appears in this image?")
    _, kwargs = client.chat.call_args
    assert "What text appears" in out  # the stub echoes the instruction
    # No assertion on use_tools — already covered above.
    del kwargs  # silence linter


def test_describe_image_factory_default_instruction_override():
    """default_instruction= on the factory propagates to the bound tool."""
    client = _fake_vision_chat_client()
    tool_fn = builtin.make_describe_image_tool(client, default_instruction="Identify objects.")
    out = tool_fn("/tmp/x.png")
    assert "Identify objects." in out
