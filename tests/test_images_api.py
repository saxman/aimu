"""Mock-only unit tests for the image-client surface (both providers).

Parallel to :mod:`tests/test_models_api.py` for the image modality. Covers:

* Shared infrastructure: :class:`ImageSpec` equality (via subclasses),
  :func:`aimu.models._internal.image_output.encode_image` format matrix.
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


def _real_present(name: str) -> bool:
    """True if module ``name`` is the real (non-stub) package, or importable as one.

    Keeps the permanent collection-time install from displacing a real ``diffusers``:
    a bare stub breaks the real package's own relative imports for the rest of the
    session, which silently kills live image/audio-diffusion model tests.
    """
    mod = sys.modules.get(name)
    if mod is not None:
        return not getattr(mod, "_aimu_stub", False)
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _install_diffusers_stub(monkeypatch=None, force=False):
    """Install a fake ``diffusers`` module so HF image tests don't need real weights.

    Also rebinds the ``diffusers`` name inside any already-loaded
    ``aimu.models.providers.hf.image`` module, when the real ``diffusers``
    is installed and gets imported first (e.g. by an earlier test importing
    ``aimu``), the stub in ``sys.modules`` doesn't reach the module's local binding.

    Pass ``monkeypatch`` (from the autouse fixture below) to scope the stub to one
    test and restore the real module + binding afterwards. Pass ``force=True`` to
    install regardless of the guard. The permanent (collection-time) call short-
    circuits when the real ``diffusers`` is installed, so live model tests keep it.
    """
    import types

    # Permanent (collection-time) path: never displace a real diffusers, and don't
    # re-install if a stub is already in place. The autouse fixture re-installs the
    # fake per-test via monkeypatch when the real dep is present.
    if monkeypatch is None and not force:
        if _real_present("diffusers"):
            return
        if isinstance(sys.modules.get("diffusers"), types.ModuleType) and getattr(
            sys.modules.get("diffusers"), "_aimu_stub", False
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

        @classmethod
        def from_pipe(cls, pipe):
            inst = cls()
            inst._source_pipe = pipe
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
    stub.Flux2KleinPipeline = _FakePipeline
    for _img2img_name in [
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusion3Img2ImgPipeline",
        "FluxImg2ImgPipeline",
    ]:
        setattr(stub, _img2img_name, _FakePipeline)
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, "diffusers", stub)
    else:
        sys.modules["diffusers"] = stub

    # Rebind the name inside the HF image client module if it's already loaded
    # (it does ``import diffusers`` at module level and resolves via getattr).
    hf_mod = sys.modules.get("aimu.models.providers.hf.image")
    if hf_mod is not None:
        if monkeypatch is not None:
            monkeypatch.setattr(hf_mod, "diffusers", stub, raising=False)
        else:
            hf_mod.diffusers = stub


_install_diffusers_stub()


@pytest.fixture(autouse=True)
def _force_diffusers_stub(monkeypatch):
    """Re-assert the diffusers stub before each test, scoped + restored via monkeypatch.

    In a full dev/CI environment the permanent install above is a no-op (it must not
    displace the real ``diffusers``, which would break live image/audio-diffusion
    tests). This installs the fake for each mock test and restores the real module
    and the HF image client's module-level binding afterwards. Imported by the
    sibling image mock files (test_image_tools, test_aio_image_api).
    """
    _install_diffusers_stub(monkeypatch=monkeypatch, force=True)


import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_IMAGE:
    aimu.models.providers.hf.image = importlib.import_module("aimu.models.providers.hf.image")
    aimu.models.HAS_HF_IMAGE = True
    aimu.models.HuggingFaceImageClient = aimu.models.providers.hf.image.HuggingFaceImageClient
    aimu.models.HuggingFaceImageModel = aimu.models.providers.hf.image.HuggingFaceImageModel


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
    resolve_image_model_enum,
)
from aimu.models._internal.image_output import encode_image  # noqa: E402
from aimu.models.base import GeminiImageSpec  # noqa: E402
from aimu.models.providers.gemini import image as _gic_mod  # noqa: E402


# ===========================================================================
# Shared infrastructure
# ===========================================================================


def test_resolve_image_model_enum_passes_through_member():
    member = HuggingFaceImageModel.SD_1_5
    assert resolve_image_model_enum(member) is member


def test_resolve_image_model_enum_bare_name_unique():
    """A bare member name in exactly one image-provider enum resolves to it."""
    assert resolve_image_model_enum("FLUX_2_KLEIN_4B") is HuggingFaceImageModel.FLUX_2_KLEIN_4B


def test_resolve_image_model_enum_delegates_provider_id_string():
    assert resolve_image_model_enum("hf:stabilityai/stable-diffusion-3.5-medium") is HuggingFaceImageModel.SD_3_5_MEDIUM


def test_resolve_image_model_enum_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown image model name"):
        resolve_image_model_enum("NOT_A_REAL_IMAGE_MODEL")


def test_resolve_image_model_enum_rejects_bad_type():
    with pytest.raises(TypeError):
        resolve_image_model_enum(123)


def test_hf_spec_equality_by_id():
    """HuggingFaceImageSpec equality uses id only; dict fields stay hashable."""
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
    assert HuggingFaceImageModel.FLUX_1_DEV.spec.max_prompt_tokens == 512
    assert GeminiImageModel.NANO_BANANA.spec.max_prompt_tokens is None
    # Ad-hoc string specs have no catalog knowledge → conservative CLIP default.
    assert HuggingFaceImageSpec("custom/repo").max_prompt_tokens == 77


def test_default_torch_dtype_injected_and_overridable():
    """torch_dtype defaults to a real device dtype (not 'auto'), but an explicit value wins."""
    import torch

    # No torch_dtype given → a concrete torch.dtype is passed to from_pretrained.
    c = HuggingFaceImageClient(HuggingFaceImageModel.SD_3_5_MEDIUM)
    assert isinstance(c.pipeline.load_kwargs["torch_dtype"], torch.dtype)

    # Explicit value is respected (e.g. opting back into "auto").
    c2 = HuggingFaceImageClient(HuggingFaceImageModel.SD_3_5_MEDIUM, model_kwargs={"torch_dtype": "auto"})
    assert c2.pipeline.load_kwargs["torch_dtype"] == "auto"


def test_auto_place_pipeline_tiers(monkeypatch):
    """Memory-aware placement picks pin / model-offload / sequential-offload by free VRAM."""
    import torch
    import torch.nn as nn

    import aimu.models.providers.hf._device as hd

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
# encode_image: shared format conversion helper
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
# HuggingFaceImageClient: construction
# ===========================================================================


def test_hf_client_accepts_enum_member():
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    assert client.spec.id == "runwayml/stable-diffusion-v1-5"
    assert client.spec.pipeline_class == "StableDiffusionPipeline"


def test_hf_client_accepts_spec():
    spec = HuggingFaceImageSpec("custom/repo", default_steps=7)
    client = HuggingFaceImageClient(spec)
    assert client.spec is spec


def test_hf_client_accepts_known_hf_string():
    # A known id string resolves to the same spec object as the enum member.
    member = HuggingFaceImageModel.SD_1_5
    client = HuggingFaceImageClient(f"hf:{member.value}")
    assert client.spec is member.spec


def test_hf_client_rejects_unknown_repo_id():
    # Arbitrary repos are not supported via string; curated enum models only.
    with pytest.raises(ValueError, match="curated models only"):
        HuggingFaceImageClient("hf:custom/repo-id")


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
# HuggingFaceImageClient: generate()
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
# GeminiImageClient: mock plumbing
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
# GeminiImageClient: model-string parsing
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
# GeminiImageClient: API-key resolution
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
# GeminiImageClient: generate()
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
# Streaming (IMAGE_GENERATING): both providers
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


# ===========================================================================
# reference_image: HuggingFace img2img
# ===========================================================================


def test_hf_img2img_uses_img2img_pipeline_and_pil_image():
    """reference_image routes to the img2img pipeline with image + strength in kwargs."""
    from PIL import Image as _PIL

    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    ref = _make_stub_image()
    client.generate("a cat", reference_image=ref)

    # The img2img pipeline was created and called (not the txt2img pipeline).
    assert client._img2img_pipe is not None
    kw = client._img2img_pipe._last_call_kwargs
    assert isinstance(kw["image"], _PIL.Image)
    assert kw["strength"] == 0.75
    # width + height must not be passed; img2img derives size from the input image.
    assert "width" not in kw
    assert "height" not in kw


def test_hf_img2img_strength_override():
    """Explicit strength kwarg overrides the 0.75 default."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat", reference_image=_make_stub_image(), strength=0.4)
    assert client._img2img_pipe._last_call_kwargs["strength"] == 0.4


def test_hf_img2img_no_reference_uses_txt2img_pipeline():
    """Without reference_image the original txt2img pipeline is used and has width/height."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat")
    assert client._img2img_pipe is None
    assert "width" in client.pipeline._last_call_kwargs


def test_hf_img2img_accepts_path_input(tmp_path):
    """A file-path string for reference_image is decoded to a PIL Image."""
    from PIL import Image as _PIL

    path = tmp_path / "ref.png"
    _make_stub_image().save(path, format="PNG")

    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat", reference_image=str(path))
    assert isinstance(client._img2img_pipe._last_call_kwargs["image"], _PIL.Image)


def test_hf_img2img_accepts_bytes_input():
    """Raw PNG bytes for reference_image are decoded to a PIL Image."""
    from io import BytesIO

    from PIL import Image as _PIL

    buf = BytesIO()
    _make_stub_image().save(buf, format="PNG")

    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    client.generate("a cat", reference_image=buf.getvalue())
    assert isinstance(client._img2img_pipe._last_call_kwargs["image"], _PIL.Image)


def test_hf_img2img_no_img2img_class_raises():
    """A spec without img2img_pipeline_class raises ValueError when reference_image is given."""
    spec = HuggingFaceImageSpec("custom/repo")  # img2img_pipeline_class=None
    client = HuggingFaceImageClient(spec)
    with pytest.raises(ValueError, match="img2img_pipeline_class"):
        client.generate("a cat", reference_image=_make_stub_image())


def test_hf_img2img_from_pipe_called_only_once():
    """_img2img_pipe is cached; from_pipe() is not called on the second generate."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    ref = _make_stub_image()
    client.generate("a cat", reference_image=ref)
    pipe_after_first = client._img2img_pipe
    assert pipe_after_first is not None

    client.generate("a dog", reference_image=ref)
    assert client._img2img_pipe is pipe_after_first  # same cached instance


def test_hf_img2img_streamed_uses_img2img_pipeline():
    """stream=True + reference_image routes to the img2img pipeline and emits a final chunk."""
    from PIL import Image as _PIL

    from aimu.models.base import StreamingContentType

    client = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    ref = _make_stub_image()
    chunks = list(client.generate("a cat", reference_image=ref, stream=True, num_inference_steps=2))

    # img2img pipeline was used
    assert client._img2img_pipe is not None
    kw = client._img2img_pipe._last_call_kwargs
    assert isinstance(kw["image"], _PIL.Image)
    assert kw["strength"] == 0.75

    # At least one final chunk
    finals = [c for c in chunks if c.content.get("final")]
    assert len(finals) >= 1
    assert finals[-1].phase == StreamingContentType.IMAGE_GENERATING


# ===========================================================================
# reference_image: Gemini image editing
# ===========================================================================


def test_gemini_without_reference_image_uses_plain_string_contents(fake_genai):
    """Baseline: no reference_image → contents is a bare string list (backward compat)."""
    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    c.generate("a cat")
    contents = c.client.models.last_call_kwargs["contents"]
    assert contents == ["a cat"]


def test_gemini_with_reference_image_builds_multipart_content(fake_genai):
    """reference_image → contents is a Content object with text + inlineData parts."""
    from google.genai import types as _gt

    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    c.generate("a cat", reference_image=_make_stub_image())

    contents = c.client.models.last_call_kwargs["contents"]
    assert len(contents) == 1
    content = contents[0]
    assert isinstance(content, _gt.Content)
    assert len(content.parts) == 2
    assert content.parts[0].text == "a cat"
    assert content.parts[1].inline_data is not None
    assert content.parts[1].inline_data.mime_type == "image/png"


def test_gemini_with_reference_image_path_input(fake_genai, tmp_path):
    """A file path for reference_image is accepted and encoded as PNG inline data."""
    from google.genai import types as _gt

    path = tmp_path / "ref.png"
    _make_stub_image().save(path, format="PNG")

    c = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    c.generate("a cat", reference_image=str(path))
    contents = c.client.models.last_call_kwargs["contents"]
    assert isinstance(contents[0], _gt.Content)
    assert contents[0].parts[1].inline_data.mime_type == "image/png"


# ===========================================================================
# FLUX.2 Klein: catalog and img2img (no strength)
# ===========================================================================


def test_flux2_klein_4b_catalog_entry():
    """FLUX.2 Klein 4B is in the catalog with the right pipeline class and defaults."""
    spec = HuggingFaceImageModel.FLUX_2_KLEIN_4B.spec
    assert spec.id == "black-forest-labs/FLUX.2-klein-4B"
    assert spec.pipeline_class == "Flux2KleinPipeline"
    assert spec.img2img_pipeline_class == "Flux2KleinPipeline"
    assert spec.img2img_uses_strength is False
    assert spec.default_steps == 4
    assert spec.default_guidance == 4.0
    assert spec.max_prompt_tokens == 512


def test_flux2_klein_9b_catalog_entry():
    """FLUX.2 Klein 9B is in the catalog with the right pipeline class and defaults."""
    spec = HuggingFaceImageModel.FLUX_2_KLEIN_9B.spec
    assert spec.id == "black-forest-labs/FLUX.2-klein-9B"
    assert spec.pipeline_class == "Flux2KleinPipeline"
    assert spec.img2img_uses_strength is False


def test_flux2_klein_txt2img_generates():
    """FLUX.2 Klein 4B can generate without a reference image."""
    client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_2_KLEIN_4B)
    img = client.generate("a cat")
    assert isinstance(img, Image.Image)
    kw = client.pipeline._last_call_kwargs
    assert kw["num_inference_steps"] == 4
    assert kw["guidance_scale"] == 4.0


def test_flux2_klein_img2img_no_strength_in_call_kwargs():
    """FLUX.2 Klein img2img passes image= but NOT strength= (unified pipeline)."""
    from PIL import Image as _PIL

    client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_2_KLEIN_4B)
    client.generate("a cat", reference_image=_make_stub_image())

    assert client._img2img_pipe is not None
    kw = client._img2img_pipe._last_call_kwargs
    assert isinstance(kw["image"], _PIL.Image)
    assert "strength" not in kw
    assert "width" not in kw
    assert "height" not in kw
