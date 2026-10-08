"""Mock-only unit tests for the built-in ``generate_image`` tool and ``make_image_tool``.

Verifies the lazy singleton wiring, the tool spec shape, env-var override, and
the per-agent factory escape hatch. No diffusers install required.
"""

from __future__ import annotations

# Install the diffusers stub before importing aimu.tools.builtin's image bits.
from test_images_api import _force_diffusers_stub, _install_diffusers_stub  # noqa: F401 (side effect + autouse fixture)

_install_diffusers_stub()

import importlib  # noqa: E402

import pytest  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_IMAGE:
    aimu.models.providers.hf.image = importlib.import_module("aimu.models.providers.hf.image")
    aimu.models.HAS_HF_IMAGE = True
    aimu.models.HuggingFaceImageClient = aimu.models.providers.hf.image.HuggingFaceImageClient
    aimu.models.HuggingFaceImageModel = aimu.models.providers.hf.image.HuggingFaceImageModel

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
    """The sync built-in must be a regular def, not async; async lives under aimu.aio.tools."""
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
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:runwayml/stable-diffusion-v1-5")

    constructed: list[str] = []

    def fake_image_client(model=None):
        import os

        model_str = model or os.environ.get("AIMU_IMAGE_MODEL")
        constructed.append(model_str)
        from aimu.models.providers.hf.image import HuggingFaceImageClient

        return HuggingFaceImageClient(model_str)

    monkeypatch.setattr("aimu.image_client", fake_image_client)

    c1 = builtin._get_image_client()
    c2 = builtin._get_image_client()
    assert c1 is c2
    assert constructed == ["hf:runwayml/stable-diffusion-v1-5"]


def test_singleton_honours_env_var(monkeypatch, tmp_path):
    """AIMU_IMAGE_MODEL env var should pick the singleton's model."""
    monkeypatch.setattr(builtin, "_image_client", None)
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:stabilityai/stable-diffusion-xl-base-1.0")

    c = builtin._get_image_client()
    assert c.spec.id == "stabilityai/stable-diffusion-xl-base-1.0"


def test_singleton_raises_when_env_unset(monkeypatch):
    """With AIMU_IMAGE_MODEL unset, the tool raises and never downloads a default."""
    import pytest

    monkeypatch.setattr(builtin, "_image_client", None)
    monkeypatch.delenv("AIMU_IMAGE_MODEL", raising=False)

    with pytest.raises(ValueError, match="AIMU_IMAGE_MODEL"):
        builtin._get_image_client()
    assert builtin._image_client is None  # nothing constructed/cached


def test_tool_drains_generator_and_returns_path(monkeypatch, tmp_path):
    """Calling generate_image() yields IMAGE_GENERATING chunks and `return`s the final path."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

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

    # The tool is a generator: drain it, then read the return value via StopIteration.value.
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
# make_image_tool: per-agent factory escape hatch
# ---------------------------------------------------------------------------


def test_make_image_tool_returns_new_streaming_tool_bound_to_supplied_client():
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_1_SCHNELL)
    bound_tool = builtin.make_image_tool(client)

    assert bound_tool is not builtin.generate_image
    assert bound_tool.__tool_spec__["function"]["name"] == "generate_image"
    assert bound_tool.__tool_is_async__ is False
    assert bound_tool.__tool_is_streaming__ is True


def test_make_image_tool_threads_preview_every_through(monkeypatch, tmp_path):
    """make_image_tool(preview_every=N) should pass N to client.generate(stream=True)."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    captured: dict = {}

    def fake_stream(prompt, format, stream, preview_every):  # noqa: ARG001
        captured["preview_every"] = preview_every
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": f"{tmp_path}/x.png"},
        )

    custom = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_1_SCHNELL)
    monkeypatch.setattr(custom, "generate", fake_stream)
    bound = builtin.make_image_tool(custom, preview_every=7)

    # Drain the generator.
    list(bound("a fox"))
    assert captured["preview_every"] == 7


def test_make_image_tool_uses_its_client_not_singleton(monkeypatch, tmp_path):
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    # Singleton should remain untouched; the bound tool should call its own client.
    sentinel = "singleton-should-not-be-touched"
    monkeypatch.setattr(builtin, "_image_client", sentinel)

    custom = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_1_SCHNELL)
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
# make_describe_image_tool: vision-capable chat client binding
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
    # Not a streaming tool; it's a plain function returning a string.
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
    # History restored exactly, including identity ordering.
    assert client.messages == original_messages


def test_describe_image_custom_instruction():
    client = _fake_vision_chat_client()
    tool_fn = builtin.make_describe_image_tool(client)
    out = tool_fn("/tmp/x.png", instruction="What text appears in this image?")
    _, kwargs = client.chat.call_args
    assert "What text appears" in out  # the stub echoes the instruction
    # No assertion on use_tools; already covered above.
    del kwargs  # silence linter


def test_describe_image_factory_default_instruction_override():
    """default_instruction= on the factory propagates to the bound tool."""
    client = _fake_vision_chat_client()
    tool_fn = builtin.make_describe_image_tool(client, default_instruction="Identify objects.")
    out = tool_fn("/tmp/x.png")
    assert "Identify objects." in out


# ---------------------------------------------------------------------------
# make_tools: standard tool-list assembly
# ---------------------------------------------------------------------------


def test_make_tools_no_image_client_no_vision():
    """With no image client and no vision, make_tools returns a copy of ALL_TOOLS."""
    client = _fake_vision_chat_client(supports_vision=False)
    tools = builtin.make_tools(client)
    assert tools == list(builtin.ALL_TOOLS)
    assert tools is not builtin.ALL_TOOLS


def test_make_tools_with_image_client_replaces_generate_image():
    """When an image client is supplied, the default generate_image singleton is replaced."""
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    client = _fake_vision_chat_client(supports_vision=False)
    image_client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_1_SCHNELL)
    tools = builtin.make_tools(client, image_client=image_client)

    assert len(tools) == len(builtin.ALL_TOOLS)
    assert builtin.generate_image not in tools
    names = [t.__tool_spec__["function"]["name"] for t in tools]
    assert "generate_image" in names


def test_make_tools_with_vision_client_appends_describe_image():
    """When the base client supports vision, describe_image is appended."""
    client = _fake_vision_chat_client(supports_vision=True)
    tools = builtin.make_tools(client)

    assert len(tools) == len(builtin.ALL_TOOLS) + 1
    assert tools[-1].__tool_spec__["function"]["name"] == "describe_image"


def test_make_tools_with_both_applies_both_transformations():
    """Image client + vision client: generate_image replaced AND describe_image appended."""
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    client = _fake_vision_chat_client(supports_vision=True)
    image_client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_1_SCHNELL)
    tools = builtin.make_tools(client, image_client=image_client)

    assert len(tools) == len(builtin.ALL_TOOLS) + 1
    assert builtin.generate_image not in tools
    names = [t.__tool_spec__["function"]["name"] for t in tools]
    assert "generate_image" in names
    assert "describe_image" in names


# ---------------------------------------------------------------------------
# reference_image: capability gating and the paths.output restriction
# ---------------------------------------------------------------------------


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _capturing_client(monkeypatch, spec_or_model):
    """An HF image client whose generate() records its kwargs and yields one final chunk."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.hf.image import HuggingFaceImageClient

    client = HuggingFaceImageClient(spec_or_model)
    captured: dict = {}

    def fake_stream(prompt, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        yield StreamChunk(
            StreamingContentType.IMAGE_GENERATING,
            {"step": 1, "total_steps": 1, "image": None, "final": True, "result": "/out/new.png"},
        )

    monkeypatch.setattr(client, "generate", fake_stream)
    return client, captured


@pytest.fixture
def output_dir(monkeypatch, tmp_path):
    """Point aimu.paths.output at a fresh directory, beside (not containing) tmp_path's other files."""
    from aimu import paths

    output = tmp_path / "output"
    (output / "images").mkdir(parents=True)
    monkeypatch.setattr(paths, "output", output)
    return output


def test_reference_support_is_declared_per_spec():
    from aimu.models.base import GeminiImageSpec, HuggingFaceImageSpec, ImageSpec
    from aimu.models.providers.hf.image import HuggingFaceImageModel

    assert HuggingFaceImageModel.SD_1_5.spec.supports_reference_image is True
    assert HuggingFaceImageModel.FLUX_2_KLEIN_4B.spec.supports_reference_image is True
    # An ad-hoc HF spec has no img2img pipeline, so it cannot take a reference.
    assert HuggingFaceImageSpec("org/some-model").supports_reference_image is False
    assert GeminiImageSpec("gemini-2.5-flash-image").supports_reference_image is True
    assert ImageSpec("anything").supports_reference_image is False


def test_make_image_tool_advertises_reference_image_only_when_supported(monkeypatch):
    from aimu.models.base import HuggingFaceImageSpec
    from aimu.models.providers.hf.image import HuggingFaceImageModel

    capable, _ = _capturing_client(monkeypatch, HuggingFaceImageModel.SD_1_5)
    incapable, _ = _capturing_client(monkeypatch, HuggingFaceImageSpec("org/some-model"))

    capable_params = builtin.make_image_tool(capable).__tool_spec__["function"]["parameters"]
    incapable_params = builtin.make_image_tool(incapable).__tool_spec__["function"]["parameters"]

    assert "reference_image" in capable_params["properties"]
    assert capable_params["required"] == ["prompt"]
    assert "reference_image" not in incapable_params["properties"]


def test_singleton_advertises_optional_reference_image():
    params = builtin.generate_image.__tool_spec__["function"]["parameters"]
    assert "reference_image" in params["properties"]
    assert params["required"] == ["prompt"]


def test_reference_image_under_output_reaches_the_client(monkeypatch, output_dir):
    from aimu.models.providers.hf.image import HuggingFaceImageModel

    reference = output_dir / "images" / "earlier.png"
    reference.write_bytes(b"png")

    client, captured = _capturing_client(monkeypatch, HuggingFaceImageModel.SD_1_5)
    result = _drain(builtin.make_image_tool(client)("make it blue", reference_image=str(reference)))

    assert result == "/out/new.png"
    assert captured["reference_image"] == reference.resolve()


def test_relative_reference_image_resolves_against_output(monkeypatch, output_dir):
    from aimu.models.providers.hf.image import HuggingFaceImageModel

    (output_dir / "images" / "earlier.png").write_bytes(b"png")

    client, captured = _capturing_client(monkeypatch, HuggingFaceImageModel.SD_1_5)
    _drain(builtin.make_image_tool(client)("make it blue", reference_image="images/earlier.png"))

    assert captured["reference_image"] == (output_dir / "images" / "earlier.png").resolve()


def test_no_reference_image_sends_none_to_the_client(monkeypatch):
    from aimu.models.providers.hf.image import HuggingFaceImageModel

    client, captured = _capturing_client(monkeypatch, HuggingFaceImageModel.SD_1_5)
    _drain(builtin.make_image_tool(client)("a cat"))

    assert "reference_image" not in captured


@pytest.mark.parametrize(
    ("reference_image", "complaint"),
    [
        ("{tmp}/secret.png", "must be under"),
        ("../secret.png", "must be under"),
        ("images/link.png", "must be under"),  # a symlink pointing out of the output directory
        ("https://example.com/cat.png", "not a URL"),
        ("data:image/png;base64,AAAA", "not a URL"),
        ("images/never-made.png", "does not exist"),
    ],
)
def test_reference_image_outside_output_is_refused(monkeypatch, tmp_path, output_dir, reference_image, complaint):
    from aimu.models.providers.hf.image import HuggingFaceImageModel
    from aimu.tools import ToolArgumentError

    secret = tmp_path / "secret.png"
    secret.write_bytes(b"png")
    (output_dir / "images" / "link.png").symlink_to(secret)

    client, captured = _capturing_client(monkeypatch, HuggingFaceImageModel.SD_1_5)
    tool_fn = builtin.make_image_tool(client)

    with pytest.raises(ToolArgumentError, match=complaint):
        _drain(tool_fn("a cat", reference_image=reference_image.format(tmp=tmp_path)))
    assert captured == {}  # refused before the image model was called


def test_singleton_refuses_reference_on_a_model_without_support(monkeypatch, output_dir):
    """The singleton advertises reference_image before its model is known, so it must refuse it where unusable."""
    from aimu.models.base import HuggingFaceImageSpec
    from aimu.tools import ToolArgumentError

    (output_dir / "earlier.png").write_bytes(b"png")
    client, captured = _capturing_client(monkeypatch, HuggingFaceImageSpec("org/some-model"))
    monkeypatch.setattr(builtin, "_image_client", client)

    with pytest.raises(ToolArgumentError, match="does not accept a reference image"):
        _drain(builtin.generate_image("a cat", reference_image="earlier.png"))
    assert captured == {}


def test_refusal_reaches_the_model_as_a_tool_result(monkeypatch, output_dir):  # noqa: ARG001
    """Through the tool loop, a refused reference is a tool message the model can correct from."""
    from aimu.agents._tool_loop import _ToolLoop
    from aimu.models.base import StreamingContentType
    from aimu.models.providers.hf.image import HuggingFaceImageModel
    from helpers import MockModelClient

    client, _ = _capturing_client(monkeypatch, HuggingFaceImageModel.SD_1_5)
    model = MockModelClient(["dummy"])
    model.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "arguments": {"prompt": "a cat", "reference_image": "/etc/hosts"},
                    },
                    "id": "id0",
                }
            ],
        }
    )

    chunks = list(_ToolLoop(model, [builtin.make_image_tool(client)])._dispatch_streamed(0))

    tool_chunks = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert len(tool_chunks) == 1
    assert "must be under" in tool_chunks[0].content["response"]
    assert "raised an error" not in tool_chunks[0].content["response"]
