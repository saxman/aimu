"""Mock-only unit tests for the image-client surface (both providers).

Parallel to :mod:`tests/test_models_api.py` for the image modality. Covers:

* Shared infrastructure: :class:`ImageSpec` equality (via subclasses),
  :func:`aimu.models._image_output.encode_image` format matrix.
* HuggingFace provider: ``HuggingFaceImageClient`` construction, ``"hf:..."``
  string parsing, lazy pipeline load, per-call kwargs threading,
  ``seed`` plumbing, unknown-pipeline error path. ``diffusers`` is stubbed
  in ``sys.modules`` so these tests run without real weights or a GPU.
* Gemini provider: ``GeminiImageClient`` construction, ``"gemini:..."`` string
  parsing (incl. alias resolution), API-key resolution (env vs ``model_kwargs``),
  ``aspect_ratio`` threading into the underlying ``ImageConfig``, no-image-data
  failure path. ``genai.Client`` is patched per-test.
* Top-level dispatch: ``aimu.image_client()`` and ``aimu.generate_image()``
  route to the right concrete client based on prefix / enum type.

Live integration tests live in :mod:`tests/test_images.py` (opt-in via
``--image-client``). Tool-wrapping tests live in :mod:`tests/test_image_tools.py`.
Async tests live in :mod:`tests/test_aio_image_api.py`.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level diffusers stub (used by HuggingFace section + downstream test
# files that import `_install_diffusers_stub` from here).
# ---------------------------------------------------------------------------


def _make_stub_image(width: int = 8, height: int = 8, color: str = "red") -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _install_diffusers_stub():
    """Install a fake ``diffusers`` module so HF image tests don't need real weights.

    Also rebinds the ``diffusers`` name inside any already-loaded
    ``aimu.models.hf_image.hf_image_client`` module — when the real ``diffusers``
    is installed and gets imported first (e.g. by an earlier test importing
    ``aimu``), the stub in ``sys.modules`` doesn't reach the module's local binding.
    """
    import types

    if (
        "diffusers" in sys.modules
        and isinstance(sys.modules["diffusers"], types.ModuleType)
        and getattr(sys.modules["diffusers"], "_aimu_stub", False)
    ):
        return

    class _FakePipeline:
        device = MagicMock(type="cpu")
        # VAE + processor stubs so _decode_latents_to_pil works under preview_every.
        vae = MagicMock()
        vae.config.scaling_factor = 1.0
        vae.decode = MagicMock(return_value=(MagicMock(),))
        image_processor = MagicMock()
        image_processor.postprocess = MagicMock(return_value=[_make_stub_image()])

        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            inst = cls()
            inst.repo_id = repo_id
            inst.load_kwargs = kwargs
            return inst

        def to(self, device):  # noqa: ARG002
            return self

        def __call__(self, **kwargs):
            num = kwargs.get("num_images_per_prompt", 1)
            self._last_call_kwargs = kwargs
            # If the caller installed a step-end callback (streaming path), fire it
            # once per `num_inference_steps` to mimic real diffusers behaviour.
            callback = kwargs.get("callback_on_step_end")
            if callback is not None:
                total_steps = kwargs.get("num_inference_steps", 1)
                # Pretend we have a latents tensor when the caller asked for it.
                wants_latents = "latents" in (kwargs.get("callback_on_step_end_tensor_inputs") or [])
                for step_index in range(total_steps):
                    callback_kwargs = {"latents": MagicMock()} if wants_latents else {}
                    callback(self, step_index, step_index, callback_kwargs)
            result = MagicMock()
            result.images = [_make_stub_image() for _ in range(num)]
            return result

    stub = types.ModuleType("diffusers")
    stub._aimu_stub = True
    stub.DiffusionPipeline = _FakePipeline
    stub.StableDiffusionPipeline = _FakePipeline
    stub.StableDiffusionXLPipeline = _FakePipeline
    stub.StableDiffusion3Pipeline = _FakePipeline
    stub.FluxPipeline = _FakePipeline
    sys.modules["diffusers"] = stub

    # Rebind the name inside the HF image client module if it's already loaded.
    hf_mod = sys.modules.get("aimu.models.hf_image.hf_image_client")
    if hf_mod is not None:
        hf_mod.diffusers = stub


_install_diffusers_stub()


import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_IMAGE:
    aimu.models.hf_image = importlib.import_module("aimu.models.hf_image")
    aimu.models.HAS_HF_IMAGE = True
    aimu.models.HuggingFaceImageClient = aimu.models.hf_image.HuggingFaceImageClient
    aimu.models.HuggingFaceImageModel = aimu.models.hf_image.HuggingFaceImageModel


# ---------------------------------------------------------------------------
# Shared imports (post-stub so HF picks up the fake diffusers)
# ---------------------------------------------------------------------------

import aimu  # noqa: E402
from aimu.models import (  # noqa: E402
    GeminiImageClient,
    GeminiImageModel,
    HuggingFaceImageClient,
    HuggingFaceImageModel,
    HuggingFaceImageSpec,
)
from aimu.models._image_output import encode_image  # noqa: E402
from aimu.models.base import GeminiImageSpec  # noqa: E402
from aimu.models.gemini_image import gemini_image_client as _gic_mod  # noqa: E402


# ===========================================================================
# Shared infrastructure
# ===========================================================================


def test_hf_spec_equality_by_id():
    """HuggingFaceImageSpec equality uses id only — dict fields stay hashable."""
    a = HuggingFaceImageSpec("repo/x", default_steps=10)
    b = HuggingFaceImageSpec("repo/x", default_steps=999)
    c = HuggingFaceImageSpec("repo/y")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_gemini_spec_equality_by_id():
    """GeminiImageSpec equality uses id only."""
    a = GeminiImageSpec("gemini-2.5-flash-image", default_aspect_ratio="1:1")
    b = GeminiImageSpec("gemini-2.5-flash-image", default_aspect_ratio="16:9")
    c = GeminiImageSpec("other")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_max_prompt_tokens_capability():
    """max_prompt_tokens defaults to CLIP's 77; T5 catalog models carry more; Gemini is uncapped."""
    assert HuggingFaceImageModel.SD_1_5.spec.max_prompt_tokens == 77
    assert HuggingFaceImageModel.SDXL_BASE.spec.max_prompt_tokens == 77
    assert HuggingFaceImageModel.SD_3_5_MEDIUM.spec.max_prompt_tokens == 256
    assert HuggingFaceImageModel.FLUX_DEV.spec.max_prompt_tokens == 512
    assert GeminiImageModel.NANO_BANANA.spec.max_prompt_tokens is None
    # Ad-hoc string specs have no catalog knowledge → conservative CLIP default.
    assert HuggingFaceImageSpec("custom/repo").max_prompt_tokens == 77


def test_supports_long_prompts_property():
    """The client property derives long-prompt support from the spec's token budget."""
    assert HuggingFaceImageClient(HuggingFaceImageModel.SDXL_BASE).supports_long_prompts is False
    assert HuggingFaceImageClient(HuggingFaceImageModel.SD_3_5_MEDIUM).supports_long_prompts is True


def test_auto_place_pipeline_tiers(monkeypatch):
    """Memory-aware placement picks pin / model-offload / sequential-offload by free VRAM."""
    import torch
    import torch.nn as nn

    import aimu.models._hf_device as hd

    class _Comp(nn.Module):
        def __init__(self, nbytes):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(nbytes // 4))  # float32 → 4 bytes/elem

    class _Pipe:
        def __init__(self, comps):
            self._c = comps
            self.placed = None

        @property
        def components(self):
            return self._c

        def to(self, dev):
            self.placed = ("to", dev)
            return self

        def enable_model_cpu_offload(self, gpu_id=None):
            self.placed = ("model", gpu_id)

        def enable_sequential_cpu_offload(self, gpu_id=None):
            self.placed = ("seq", gpu_id)

    gib = 1024**3

    def place(free_gib, comp_gib, accel=True):
        monkeypatch.setattr(hd, "cuda_free_memory", lambda: [(0, int(free_gib * gib))])
        monkeypatch.setattr(hd, "has_accelerate", lambda: accel)
        pipe = _Pipe({f"c{i}": _Comp(int(g * gib)) for i, g in enumerate(comp_gib)})
        hd.auto_place_pipeline(pipe)
        return pipe.placed

    assert place(20, [1.0, 1.0]) == ("to", "cuda:0")  # fits one GPU → pin
    assert place(12, [10.0, 6.0]) == ("model", 0)  # too big, largest fits → model offload
    assert place(8, [10.0, 12.0]) == ("seq", 0)  # largest doesn't fit → sequential offload
    assert place(8, [10.0, 12.0], accel=False) == ("to", "cuda:0")  # no accelerate → pin (warns)


def test_hf_model_value_is_repo_id():
    assert HuggingFaceImageModel.SD_1_5.value == "runwayml/stable-diffusion-v1-5"
    assert HuggingFaceImageModel.SDXL_BASE.spec.pipeline_class == "StableDiffusionXLPipeline"


def test_gemini_model_value_is_api_id():
    assert GeminiImageModel.NANO_BANANA.value == "gemini-2.5-flash-image"
    assert GeminiImageModel.NANO_BANANA_PREVIEW.value == "gemini-2.5-flash-image-preview"


# ---------------------------------------------------------------------------
# encode_image — shared format conversion helper
# ---------------------------------------------------------------------------


def test_encode_image_pil_returns_image_unchanged():
    img = _make_stub_image()
    assert encode_image(img, format="pil") is img


def test_encode_image_bytes_returns_png_bytes():
    img = _make_stub_image()
    out = encode_image(img, format="bytes")
    assert isinstance(out, bytes)
    decoded = Image.open(BytesIO(out))
    assert decoded.size == img.size


def test_encode_image_data_url_returns_base64_data_url():
    img = _make_stub_image()
    out = encode_image(img, format="data_url")
    assert isinstance(out, str)
    assert out.startswith("data:image/png;base64,")


def test_encode_image_path_writes_file_and_returns_string(tmp_path):
    img = _make_stub_image()
    out = encode_image(img, format="path", prompt="hello", output_dir=tmp_path)
    saved = Path(out)
    assert saved.exists() and saved.suffix == ".png" and saved.parent == tmp_path


def test_encode_image_path_creates_output_dir(tmp_path):
    target = tmp_path / "nested" / "images"
    img = _make_stub_image()
    out = encode_image(img, format="path", output_dir=target)
    assert Path(out).parent == target


def test_encode_image_invalid_format_raises():
    img = _make_stub_image()
    with pytest.raises(ValueError, match="Unknown format"):
        encode_image(img, format="webp")


# ===========================================================================
# HuggingFaceImageClient — construction
# ===========================================================================


def test_hf_client_accepts_enum_member():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    assert client.spec.id == "runwayml/stable-diffusion-v1-5"
    assert client.spec.pipeline_class == "StableDiffusionPipeline"


def test_hf_client_accepts_spec():
    spec = HuggingFaceImageSpec("custom/repo", default_steps=7)
    client = HuggingFaceImageClient(spec)
    assert client.spec is spec


def test_hf_client_accepts_hf_string():
    client = HuggingFaceImageClient("hf:custom/repo-id")
    assert client.spec.id == "custom/repo-id"
    assert client.spec.pipeline_class == "DiffusionPipeline"


def test_hf_client_rejects_bare_repo_id():
    with pytest.raises(ValueError, match="provider:repo_id"):
        HuggingFaceImageClient("runwayml/stable-diffusion-v1-5")


def test_hf_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Only 'hf:' provider"):
        HuggingFaceImageClient("replicate:foo/bar")


def test_hf_client_rejects_empty_repo_id():
    with pytest.raises(ValueError, match="Empty model id"):
        HuggingFaceImageClient("hf:")


def test_hf_client_rejects_wrong_type():
    with pytest.raises(TypeError, match="HuggingFaceImageModel member"):
        HuggingFaceImageClient(42)


def test_hf_client_lazy_pipeline_not_loaded_at_init():
    """Constructing a client must not load weights."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    assert client._pipe is None
    assert "unloaded" in repr(client)


# ---------------------------------------------------------------------------
# HuggingFaceImageClient — generate()
# ---------------------------------------------------------------------------


def test_hf_generate_returns_pil_by_default():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    img = client.generate("a cat")
    assert isinstance(img, Image.Image)


def test_hf_generate_applies_spec_defaults_when_unset():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat")
    call_kwargs = client.pipeline._last_call_kwargs
    assert call_kwargs["num_inference_steps"] == 25  # SD_1_5 default
    assert call_kwargs["width"] == 512
    assert call_kwargs["height"] == 512


def test_hf_generate_overrides_propagate_to_pipeline():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat", width=256, height=256, num_inference_steps=10, guidance_scale=3.0)
    call_kwargs = client.pipeline._last_call_kwargs
    assert call_kwargs["width"] == 256
    assert call_kwargs["num_inference_steps"] == 10
    assert call_kwargs["guidance_scale"] == 3.0


def test_hf_generate_negative_prompt_passed_through():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat", negative_prompt="blurry")
    assert client.pipeline._last_call_kwargs["negative_prompt"] == "blurry"


def test_hf_generate_omits_negative_prompt_when_unset():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat")
    assert "negative_prompt" not in client.pipeline._last_call_kwargs


def test_hf_generate_num_images_returns_list():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    imgs = client.generate("a cat", num_images=3)
    assert isinstance(imgs, list) and len(imgs) == 3
    assert all(isinstance(i, Image.Image) for i in imgs)


def test_hf_generate_with_format_path(tmp_path):
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    path = client.generate("a cat", format="path", output_dir=tmp_path)
    assert isinstance(path, str) and Path(path).exists()


def test_hf_generate_with_format_bytes_returns_bytes():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    out = client.generate("a cat", format="bytes")
    assert isinstance(out, bytes)


def test_hf_generate_seed_plumbed_through(monkeypatch):
    """seed= should produce a torch.Generator with manual_seed called."""
    import torch

    captured = {}
    real_generator = torch.Generator

    class TrackingGenerator(real_generator):
        def manual_seed(self, seed):
            captured["seed"] = seed
            return super().manual_seed(seed)

    monkeypatch.setattr(torch, "Generator", TrackingGenerator)

    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat", seed=42)
    assert captured["seed"] == 42
    assert "generator" in client.pipeline._last_call_kwargs


def test_hf_unknown_pipeline_class_raises_clear_error():
    """If spec.pipeline_class isn't in the diffusers namespace, raise a helpful error."""
    bad_spec = HuggingFaceImageSpec("custom/repo", pipeline_class="NotARealPipeline")
    client = HuggingFaceImageClient(bad_spec)
    with pytest.raises(ValueError, match="NotARealPipeline"):
        client.generate("a cat")


# ===========================================================================
# GeminiImageClient — mock plumbing
# ===========================================================================


def _png_bytes(color: str = "red", size=(8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class _FakePart:
    def __init__(self, data: bytes | None):
        if data is None:
            self.inline_data = None
        else:
            inline = MagicMock()
            inline.data = data
            inline.mime_type = "image/png"
            self.inline_data = inline


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _FakeContent(parts)


class _FakeResponse:
    def __init__(self, candidates, prompt_feedback=None):
        self.candidates = candidates
        self.prompt_feedback = prompt_feedback


class _FakeModels:
    """Records the last call so tests can assert kwargs."""

    def __init__(self, response):
        self._response = response
        self.last_call_kwargs: dict | None = None

    def generate_content(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


class _FakeGenaiClient:
    def __init__(self, response, api_key=None):
        self.api_key = api_key
        self.models = _FakeModels(response)


@pytest.fixture
def fake_genai(monkeypatch):
    """Patch ``genai.Client`` to a fake that returns one image per call."""
    response = _FakeResponse(candidates=[_FakeCandidate([_FakePart(_png_bytes())])])
    constructed: list[str | None] = []

    def factory(*, api_key=None):
        constructed.append(api_key)
        return _FakeGenaiClient(response, api_key=api_key)

    monkeypatch.setattr(_gic_mod.genai, "Client", factory)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    return {"constructed": constructed, "response": response}


# ---------------------------------------------------------------------------
# GeminiImageClient — model-string parsing
# ---------------------------------------------------------------------------


def test_gemini_model_string_alias_resolution(fake_genai):
    c = GeminiImageClient("gemini:nano-banana")
    assert c.spec.id == "gemini-2.5-flash-image"


def test_gemini_model_string_full_id(fake_genai):
    c = GeminiImageClient("gemini:gemini-2.5-flash-image")
    assert c.spec.id == "gemini-2.5-flash-image"


def test_gemini_model_string_bare_id_rejected(fake_genai):
    with pytest.raises(ValueError, match="provider:model_id"):
        GeminiImageClient("nano-banana")


def test_gemini_model_string_wrong_provider_rejected(fake_genai):
    with pytest.raises(ValueError, match="gemini:"):
        GeminiImageClient("openai:nano-banana")


def test_gemini_model_string_empty_id_rejected(fake_genai):
    with pytest.raises(ValueError, match="Empty model id"):
        GeminiImageClient("gemini:")


# ---------------------------------------------------------------------------
# GeminiImageClient — API-key resolution
# ---------------------------------------------------------------------------


def test_gemini_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GeminiImageClient(GeminiImageModel.NANO_BANANA)


def test_gemini_api_key_via_model_kwargs_overrides_env(monkeypatch, fake_genai):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA, model_kwargs={"api_key": "explicit"})
    _ = c.client  # trigger lazy construction
    assert fake_genai["constructed"][-1] == "explicit"


# ---------------------------------------------------------------------------
# GeminiImageClient — generate()
# ---------------------------------------------------------------------------


def test_gemini_generate_returns_pil_by_default(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    img = c.generate("a cat")
    assert isinstance(img, Image.Image)


def test_gemini_generate_format_bytes(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    data = c.generate("a cat", format="bytes")
    assert isinstance(data, bytes) and data.startswith(b"\x89PNG")


def test_gemini_generate_format_path(fake_genai, tmp_path):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    out = c.generate("a cat", format="path", output_dir=tmp_path)
    p = Path(out)
    assert p.exists() and p.suffix == ".png" and p.parent == tmp_path


def test_gemini_generate_num_images_returns_list(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    imgs = c.generate("a cat", num_images=3)
    assert isinstance(imgs, list) and len(imgs) == 3
    assert all(isinstance(i, Image.Image) for i in imgs)


def test_gemini_generate_aspect_ratio_threaded_into_image_config(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    c.generate("a cat", aspect_ratio="16:9")
    kwargs = c.client.models.last_call_kwargs
    assert kwargs is not None and kwargs["config"].image_config.aspect_ratio == "16:9"


def test_gemini_generate_uses_spec_default_aspect_ratio_when_unset(fake_genai):
    spec = GeminiImageSpec("gemini-2.5-flash-image", default_aspect_ratio="4:3")
    c = GeminiImageClient(spec)
    c.generate("a cat")
    config = c.client.models.last_call_kwargs["config"]
    assert config.image_config.aspect_ratio == "4:3"


def test_gemini_generate_call_args_forwarded(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    c.generate("a cat")
    kwargs = c.client.models.last_call_kwargs
    assert kwargs["model"] == "gemini-2.5-flash-image"
    assert kwargs["contents"] == ["a cat"]


def test_gemini_no_inline_image_raises_runtime_error(monkeypatch):
    """When the API returns no image parts, raise a clear error."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    response = _FakeResponse(
        candidates=[_FakeCandidate([_FakePart(None)])],
        prompt_feedback="blocked: safety filter",
    )

    monkeypatch.setattr(_gic_mod.genai, "Client", lambda **kw: _FakeGenaiClient(response, api_key=kw.get("api_key")))
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    with pytest.raises(RuntimeError, match="no image data"):
        c.generate("a cat")


def test_gemini_client_constructed_lazily(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    assert c._client is None
    _ = c.client
    assert c._client is not None


def test_gemini_num_images_zero_or_negative_rejected(fake_genai):
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    with pytest.raises(ValueError, match="num_images must be"):
        c.generate("x", num_images=0)


# ===========================================================================
# Top-level dispatch (aimu.image_client / aimu.generate_image)
# ===========================================================================


def test_top_level_image_client_dispatches_hf_string():
    c = aimu.image_client("hf:runwayml/stable-diffusion-v1-5")
    assert isinstance(c, aimu.ImageClient)
    assert isinstance(c._client, HuggingFaceImageClient)


def test_top_level_image_client_dispatches_hf_enum():
    c = aimu.image_client(HuggingFaceImageModel.SD_1_5)
    assert isinstance(c, aimu.ImageClient)
    assert isinstance(c._client, HuggingFaceImageClient)


def test_top_level_image_client_dispatches_gemini_string(fake_genai):
    c = aimu.image_client("gemini:nano-banana")
    assert isinstance(c, aimu.ImageClient)
    assert isinstance(c._client, GeminiImageClient)


def test_top_level_image_client_dispatches_gemini_enum(fake_genai):
    c = aimu.image_client(GeminiImageModel.NANO_BANANA)
    assert isinstance(c, aimu.ImageClient)
    assert isinstance(c._client, GeminiImageClient)


def test_top_level_generate_image_one_shot_gemini(fake_genai):
    img = aimu.generate_image("a cat", model="gemini:nano-banana")
    assert isinstance(img, Image.Image)


def test_top_level_generate_image_one_shot_hf():
    img = aimu.generate_image("a cat", model="hf:runwayml/stable-diffusion-v1-5")
    assert isinstance(img, Image.Image)


# ===========================================================================
# Streaming (IMAGE_GENERATING) — both providers
# ===========================================================================


def test_hf_stream_emits_one_chunk_per_step_then_final():
    """HF streaming: N progress chunks + 1 final chunk (num_images=1, N steps)."""
    from aimu.models.base import StreamingContentType

    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    chunks = list(client.generate("a cat", stream=True, num_inference_steps=5))

    # 5 progress + 1 final = 6 chunks.
    assert len(chunks) == 6
    assert all(c.phase == StreamingContentType.IMAGE_GENERATING for c in chunks)
    # First 5 are non-final, step 1..5.
    for i, c in enumerate(chunks[:5]):
        assert c.content["step"] == i + 1
        assert c.content["total_steps"] == 5
        assert c.content["final"] is False
        assert c.content["image"] is None  # no preview by default
    # Final chunk has final=True + result.
    assert chunks[-1].content["final"] is True
    assert chunks[-1].content["result"] is not None


def test_hf_stream_preview_every_decodes_at_matching_steps():
    """preview_every=2 should populate `image` at steps 2, 4 (and the final)."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    chunks = list(client.generate("a cat", stream=True, num_inference_steps=4, preview_every=2))
    # 4 progress + 1 final = 5 chunks.
    assert len(chunks) == 5
    # Step 1: no preview; step 2: preview; step 3: no; step 4: preview; final: yes.
    assert chunks[0].content["image"] is None
    assert chunks[1].content["image"] is not None
    assert chunks[2].content["image"] is None
    assert chunks[3].content["image"] is not None
    assert chunks[4].content["final"] is True
    assert chunks[4].content["image"] is not None


def test_hf_stream_num_images_emits_one_final_chunk_per_image():
    """num_images=3 should emit final chunks for each generated image."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    chunks = list(client.generate("a cat", stream=True, num_inference_steps=2, num_images=3))
    finals = [c for c in chunks if c.content["final"]]
    assert len(finals) == 3
    for i, c in enumerate(finals):
        assert c.content["image_index"] == i
        assert c.content["num_images"] == 3


def test_gemini_stream_coarse_start_then_done(fake_genai):
    """Gemini streaming: one start chunk + one final chunk per image."""
    from aimu.models.base import StreamingContentType

    client = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    chunks = list(client.generate("a cat", stream=True))

    assert len(chunks) == 2
    assert all(c.phase == StreamingContentType.IMAGE_GENERATING for c in chunks)
    assert chunks[0].content["final"] is False
    assert chunks[0].content["image"] is None
    assert chunks[1].content["final"] is True
    assert chunks[1].content["image"] is not None


def test_gemini_stream_num_images_two_pairs(fake_genai):
    """num_images=2 emits two start/done pairs."""
    client = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    chunks = list(client.generate("a cat", stream=True, num_images=2))
    starts = [c for c in chunks if not c.content["final"]]
    finals = [c for c in chunks if c.content["final"]]
    assert len(starts) == 2
    assert len(finals) == 2
