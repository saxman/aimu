"""Mock-only tests for the async diffusion surface (``aimu.aio.diffusion``).

Covers:
* Direct enum/string construction is refused with a helpful pointer at the
  sync-then-wrap pattern (mirrors HF/LlamaCpp refusal in ``aio/_model_client.py``).
* ``AsyncDiffusionClient`` wraps a sync client and shares state.
* ``aio.tools.builtin.generate_image`` is detected as async by ``@tool``.
"""

from __future__ import annotations

import pytest

from test_diffusion_api import _install_diffusers_stub  # noqa: F401 — side effect

_install_diffusers_stub()

import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_DIFFUSION:
    aimu.models.diffusion = importlib.import_module("aimu.models.diffusion")
    aimu.models.HAS_DIFFUSION = True
    aimu.models.DiffusionClient = aimu.models.diffusion.DiffusionClient
    aimu.models.DiffusionModel = aimu.models.diffusion.DiffusionModel

import aimu  # noqa: E402

aimu.HAS_DIFFUSION = True
aimu.DiffusionClient = aimu.models.DiffusionClient
aimu.DiffusionModel = aimu.models.DiffusionModel

# The aio submodule may have been imported earlier with _HAS_DIFFUSION=False;
# reload it so it picks up the stubbed diffusers + DiffusionClient.
import aimu.aio.diffusion as _aio_diffusion  # noqa: E402

importlib.reload(_aio_diffusion)

from aimu.aio.providers.diffusion import AsyncDiffusionClient  # noqa: E402
from aimu.models.diffusion import DiffusionClient, DiffusionModel  # noqa: E402


# ---------------------------------------------------------------------------
# Construction & refusal
# ---------------------------------------------------------------------------


def test_async_diffusion_client_wraps_sync():
    sync = DiffusionClient(DiffusionModel.SD_1_5)
    async_client = AsyncDiffusionClient(sync)
    assert async_client._sync is sync
    assert async_client.spec.id == "runwayml/stable-diffusion-v1-5"


def test_async_diffusion_client_rejects_non_sync():
    with pytest.raises(TypeError, match="existing sync DiffusionClient"):
        AsyncDiffusionClient(object())


def test_aio_image_client_wraps_sync():
    sync = DiffusionClient(DiffusionModel.SD_1_5)
    async_client = _aio_diffusion.image_client(sync)
    assert isinstance(async_client, AsyncDiffusionClient)
    assert async_client._sync is sync


def test_aio_image_client_refuses_enum_with_wrap_guidance():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.image_client"):
        _aio_diffusion.image_client(DiffusionModel.SD_1_5)


def test_aio_image_client_refuses_string_with_wrap_guidance():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.image_client"):
        _aio_diffusion.image_client("hf:runwayml/stable-diffusion-v1-5")


def test_aio_image_client_refuses_random_type():
    with pytest.raises(TypeError, match="sync DiffusionClient"):
        _aio_diffusion.image_client(42)


# ---------------------------------------------------------------------------
# Async generate() delegates to sync via to_thread
# ---------------------------------------------------------------------------


async def test_async_generate_returns_image():
    from PIL import Image

    sync = DiffusionClient(DiffusionModel.SD_1_5)
    async_client = AsyncDiffusionClient(sync)
    img = await async_client.generate("a cat")
    assert isinstance(img, Image.Image)


async def test_async_generate_with_format_bytes():
    sync = DiffusionClient(DiffusionModel.SD_1_5)
    async_client = AsyncDiffusionClient(sync)
    data = await async_client.generate("a cat", format="bytes")
    assert isinstance(data, bytes)
    assert data.startswith(b"\x89PNG")


async def test_async_generate_preserves_kwargs():
    sync = DiffusionClient(DiffusionModel.SD_1_5)
    async_client = AsyncDiffusionClient(sync)
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

    sync = DiffusionClient(DiffusionModel.SD_1_5)
    bound = aio_builtin.make_async_image_tool(sync)
    assert bound.__tool_is_async__ is True
    assert bound is not aio_builtin.generate_image
