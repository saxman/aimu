"""Mock-only tests for the async image surface (``aimu.aio.image``).

Covers:
* Direct enum/string construction is refused with a helpful pointer at the
  sync-then-wrap pattern (mirrors HF/LlamaCpp refusal in ``aio/_model_client.py``).
* ``AsyncImageClient`` / ``AsyncHuggingFaceImageClient`` / ``AsyncGeminiImageClient``
  wrap their sync siblings and share state.
* ``aio.tools.builtin.generate_image`` is detected as async by ``@tool``.
"""

from __future__ import annotations

import pytest

from test_images_api import _install_diffusers_stub  # noqa: F401 — side effect

_install_diffusers_stub()

import importlib  # noqa: E402

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

# The aio submodule may have been imported earlier with _HAS_HF_IMAGE=False;
# reload it so it picks up the stubbed diffusers + HuggingFaceImageClient.
import aimu.aio.image as _aio_image  # noqa: E402

importlib.reload(_aio_image)

from aimu.aio.providers.hf.image import AsyncHuggingFaceImageClient  # noqa: E402
from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel  # noqa: E402


# ---------------------------------------------------------------------------
# Construction & refusal
# ---------------------------------------------------------------------------


def test_async_hf_client_wraps_sync():
    sync = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    async_client = AsyncHuggingFaceImageClient(sync)
    assert async_client._sync is sync
    assert async_client.spec.id == "runwayml/stable-diffusion-v1-5"


def test_async_hf_client_rejects_non_sync():
    with pytest.raises(TypeError, match="existing sync HuggingFaceImageClient"):
        AsyncHuggingFaceImageClient(object())


def test_aio_image_client_wraps_sync():
    sync = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    async_client = _aio_image.image_client(sync)
    # The top-level AsyncImageClient wraps the per-provider async client.
    assert isinstance(async_client._client, AsyncHuggingFaceImageClient)
    assert async_client._client._sync is sync


def test_aio_image_client_refuses_enum_with_wrap_guidance():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.image_client"):
        _aio_image.image_client(HuggingFaceImageModel.SD_1_5)


def test_aio_image_client_refuses_string_with_wrap_guidance():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.image_client"):
        _aio_image.image_client("hf:runwayml/stable-diffusion-v1-5")


def test_aio_image_client_refuses_random_type():
    with pytest.raises(TypeError, match="sync HuggingFaceImageClient or GeminiImageClient"):
        _aio_image.image_client(42)


# ---------------------------------------------------------------------------
# Async generate() delegates to sync via to_thread
# ---------------------------------------------------------------------------


async def test_async_generate_returns_image():
    from PIL import Image

    sync = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    async_client = AsyncHuggingFaceImageClient(sync)
    img = await async_client.generate("a cat")
    assert isinstance(img, Image.Image)


async def test_async_generate_with_format_bytes():
    sync = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    async_client = AsyncHuggingFaceImageClient(sync)
    data = await async_client.generate("a cat", format="bytes")
    assert isinstance(data, bytes)
    assert data.startswith(b"\x89PNG")


async def test_async_generate_preserves_kwargs():
    sync = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    async_client = AsyncHuggingFaceImageClient(sync)
    await async_client.generate("a cat", width=256, height=256, num_inference_steps=5)
    call = sync.pipeline._last_call_kwargs
    assert call["width"] == 256
    assert call["num_inference_steps"] == 5


# ---------------------------------------------------------------------------
# Async tool detection
# ---------------------------------------------------------------------------


def test_async_generate_image_tool_is_async():
    import aimu.aio.tools.builtin as aio_builtin

    assert aio_builtin.generate_image.__tool_is_async__ is True
    spec = aio_builtin.generate_image.__tool_spec__
    assert spec["function"]["name"] == "generate_image"


def test_make_async_image_tool_wraps_sync_client():
    import aimu.aio.tools.builtin as aio_builtin

    sync = HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5)
    bound = aio_builtin.make_async_image_tool(sync)
    assert bound.__tool_is_async__ is True
    assert bound is not aio_builtin.generate_image


# ---------------------------------------------------------------------------
# Gemini image async — sync wrap + refusal of direct enum
# ---------------------------------------------------------------------------


def test_aio_image_client_wraps_sync_gemini(monkeypatch):
    """aio.image_client should accept a sync GeminiImageClient and wrap it."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from aimu.aio.providers.gemini.image import AsyncGeminiImageClient
    from aimu.models.providers.gemini.image import GeminiImageClient, GeminiImageModel

    sync = GeminiImageClient(GeminiImageModel.NANO_BANANA)
    async_client = _aio_image.image_client(sync)
    assert isinstance(async_client._client, AsyncGeminiImageClient)
    assert async_client._client._sync is sync


def test_aio_image_client_refuses_gemini_enum(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from aimu.models.providers.gemini.image import GeminiImageModel

    with pytest.raises(ValueError, match=r"sync_client = aimu\.image_client"):
        _aio_image.image_client(GeminiImageModel.NANO_BANANA)


def test_aio_image_client_refuses_gemini_string(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with pytest.raises(ValueError, match=r"sync_client = aimu\.image_client"):
        _aio_image.image_client("gemini:nano-banana")
