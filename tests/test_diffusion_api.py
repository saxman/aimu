"""Mock-only unit tests for the diffusion client surface.

Stubs the ``diffusers`` module in ``sys.modules`` so these tests run on machines
without the optional ``[diffusion]`` extra installed. Real pipeline loading is
exercised by ``tests/test_diffusion.py`` behind the ``--diffusion-model`` flag.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level diffusers stub
# ---------------------------------------------------------------------------


def _make_stub_image(width: int = 8, height: int = 8, color: str = "red") -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _install_diffusers_stub():
    """Install a fake ``diffusers`` module so ``aimu.models.diffusion`` imports cleanly."""
    import types

    if "diffusers" in sys.modules and isinstance(sys.modules["diffusers"], types.ModuleType) and getattr(
        sys.modules["diffusers"], "_aimu_stub", False
    ):
        return

    class _FakePipeline:
        device = MagicMock(type="cpu")

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


_install_diffusers_stub()


# Re-import after stub install so HAS_DIFFUSION picks up the stub.
# We have to reload the diffusion submodule because aimu was already imported
# (without diffusers) earlier in the test session.
import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_DIFFUSION:
    aimu.models.diffusion = importlib.import_module("aimu.models.diffusion")
    aimu.models.HAS_DIFFUSION = True
    aimu.models.DiffusionClient = aimu.models.diffusion.DiffusionClient
    aimu.models.DiffusionModel = aimu.models.diffusion.DiffusionModel

from aimu.models import DiffusionClient, DiffusionModel, DiffusionSpec  # noqa: E402
from aimu.models._image_output import encode_image  # noqa: E402


# ---------------------------------------------------------------------------
# DiffusionSpec
# ---------------------------------------------------------------------------


def test_diffusion_spec_equality_by_id():
    """DiffusionSpec equality uses id only — dict fields stay hashable."""
    a = DiffusionSpec("repo/x", default_steps=10)
    b = DiffusionSpec("repo/x", default_steps=999)
    c = DiffusionSpec("repo/y")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_diffusion_model_value_is_repo_id():
    """DiffusionModel.X.value returns the HF repo id string."""
    assert DiffusionModel.SD_1_5.value == "runwayml/stable-diffusion-v1-5"
    assert DiffusionModel.SDXL_BASE.spec.pipeline_class == "StableDiffusionXLPipeline"


# ---------------------------------------------------------------------------
# encode_image: format matrix
# ---------------------------------------------------------------------------


def test_encode_image_pil_returns_image_unchanged():
    img = _make_stub_image()
    assert encode_image(img, format="pil") is img


def test_encode_image_bytes_returns_png_bytes():
    img = _make_stub_image()
    out = encode_image(img, format="bytes")
    assert isinstance(out, bytes)
    # Round-trip: bytes should be a valid PNG
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
    assert isinstance(out, str)
    saved = Path(out)
    assert saved.exists()
    assert saved.suffix == ".png"
    assert saved.parent == tmp_path
    decoded = Image.open(saved)
    assert decoded.size == img.size


def test_encode_image_path_creates_output_dir(tmp_path):
    target = tmp_path / "nested" / "images"
    img = _make_stub_image()
    out = encode_image(img, format="path", output_dir=target)
    assert Path(out).parent == target


def test_encode_image_invalid_format_raises():
    img = _make_stub_image()
    with pytest.raises(ValueError, match="Unknown format"):
        encode_image(img, format="webp")


# ---------------------------------------------------------------------------
# DiffusionClient — construction
# ---------------------------------------------------------------------------


def test_diffusion_client_accepts_enum_member():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    assert client.spec.id == "runwayml/stable-diffusion-v1-5"
    assert client.spec.pipeline_class == "StableDiffusionPipeline"


def test_diffusion_client_accepts_diffusion_spec():
    spec = DiffusionSpec("custom/repo", default_steps=7)
    client = DiffusionClient(spec)
    assert client.spec is spec


def test_diffusion_client_accepts_hf_string():
    client = DiffusionClient("hf:custom/repo-id")
    assert client.spec.id == "custom/repo-id"
    assert client.spec.pipeline_class == "DiffusionPipeline"  # default


def test_diffusion_client_rejects_bare_repo_id():
    with pytest.raises(ValueError, match="provider:repo_id"):
        DiffusionClient("runwayml/stable-diffusion-v1-5")


def test_diffusion_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Only 'hf:' provider"):
        DiffusionClient("replicate:foo/bar")


def test_diffusion_client_rejects_empty_repo_id():
    with pytest.raises(ValueError, match="Empty model id"):
        DiffusionClient("hf:")


def test_diffusion_client_rejects_wrong_type():
    with pytest.raises(TypeError, match="DiffusionModel member"):
        DiffusionClient(42)


def test_diffusion_client_lazy_pipeline_not_loaded_at_init():
    """Constructing a DiffusionClient must not load weights."""
    client = DiffusionClient(DiffusionModel.SD_1_5)
    assert client._pipe is None
    assert "unloaded" in repr(client)


# ---------------------------------------------------------------------------
# DiffusionClient — generate()
# ---------------------------------------------------------------------------


def test_generate_returns_pil_by_default():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    img = client.generate("a cat")
    assert isinstance(img, Image.Image)


def test_generate_applies_spec_defaults_when_unset():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    client.generate("a cat")
    call_kwargs = client.pipeline._last_call_kwargs
    assert call_kwargs["num_inference_steps"] == 25  # SD_1_5 default
    assert call_kwargs["width"] == 512
    assert call_kwargs["height"] == 512


def test_generate_overrides_propagate_to_pipeline():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    client.generate("a cat", width=256, height=256, num_inference_steps=10, guidance_scale=3.0)
    call_kwargs = client.pipeline._last_call_kwargs
    assert call_kwargs["width"] == 256
    assert call_kwargs["num_inference_steps"] == 10
    assert call_kwargs["guidance_scale"] == 3.0


def test_generate_negative_prompt_passed_through():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    client.generate("a cat", negative_prompt="blurry")
    assert client.pipeline._last_call_kwargs["negative_prompt"] == "blurry"


def test_generate_omits_negative_prompt_when_unset():
    """No negative_prompt should be passed if neither the call nor the spec sets one."""
    client = DiffusionClient(DiffusionModel.SD_1_5)
    client.generate("a cat")
    assert "negative_prompt" not in client.pipeline._last_call_kwargs


def test_generate_num_images_returns_list():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    imgs = client.generate("a cat", num_images=3)
    assert isinstance(imgs, list)
    assert len(imgs) == 3
    assert all(isinstance(i, Image.Image) for i in imgs)


def test_generate_with_format_path(tmp_path):
    client = DiffusionClient(DiffusionModel.SD_1_5)
    path = client.generate("a cat", format="path", output_dir=tmp_path)
    assert isinstance(path, str)
    assert Path(path).exists()


def test_generate_with_format_bytes_returns_bytes():
    client = DiffusionClient(DiffusionModel.SD_1_5)
    out = client.generate("a cat", format="bytes")
    assert isinstance(out, bytes)


def test_generate_seed_plumbed_through(monkeypatch):
    """seed= should produce a torch.Generator with manual_seed called."""
    import torch

    captured = {}
    real_generator = torch.Generator

    class TrackingGenerator(real_generator):
        def manual_seed(self, seed):
            captured["seed"] = seed
            return super().manual_seed(seed)

    monkeypatch.setattr(torch, "Generator", TrackingGenerator)

    client = DiffusionClient(DiffusionModel.SD_1_5)
    client.generate("a cat", seed=42)
    assert captured["seed"] == 42
    assert "generator" in client.pipeline._last_call_kwargs


def test_unknown_pipeline_class_raises_clear_error():
    """If spec.pipeline_class isn't in the diffusers namespace, raise a helpful error."""
    bad_spec = DiffusionSpec("custom/repo", pipeline_class="NotARealPipeline")
    client = DiffusionClient(bad_spec)
    with pytest.raises(ValueError, match="NotARealPipeline"):
        client.generate("a cat")
