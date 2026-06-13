"""Mock-only tests for the async audio surface (``aimu.aio.audio``).

Covers:
* Direct enum/string construction is refused with a helpful pointer at the
  sync-then-wrap pattern (mirrors HF/LlamaCpp refusal in ``aio/_model_client.py``).
* ``AsyncAudioClient`` / ``AsyncHuggingFaceAudioClient`` wrap their sync
  siblings and share state.
* ``AsyncHuggingFaceAudioClient.generate()`` delegates to the sync client
  via ``asyncio.to_thread``.
"""

from __future__ import annotations

import pytest

from test_audio_api import _force_audio_stubs, _install_audio_stubs  # noqa: F401 — side effect + autouse fixture

_install_audio_stubs()

import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_AUDIO:
    aimu.models.providers.hf.audio = importlib.import_module("aimu.models.providers.hf.audio")
    aimu.models.HAS_HF_AUDIO = True
    aimu.models.HuggingFaceAudioClient = aimu.models.providers.hf.audio.HuggingFaceAudioClient
    aimu.models.HuggingFaceAudioModel = aimu.models.providers.hf.audio.HuggingFaceAudioModel

import aimu  # noqa: E402

aimu.HAS_HF_AUDIO = True
aimu.HuggingFaceAudioClient = aimu.models.HuggingFaceAudioClient
aimu.HuggingFaceAudioModel = aimu.models.HuggingFaceAudioModel

# The aio submodule may have been imported earlier with _HAS_HF_AUDIO=False;
# reload it so it picks up the stubbed soundfile + HuggingFaceAudioClient.
import aimu.aio.audio as _aio_audio  # noqa: E402

importlib.reload(_aio_audio)

from aimu.aio.providers.hf.audio import AsyncHuggingFaceAudioClient  # noqa: E402
from aimu.models.providers.hf.audio import HuggingFaceAudioClient, HuggingFaceAudioModel  # noqa: E402


# ---------------------------------------------------------------------------
# Construction & refusal
# ---------------------------------------------------------------------------


def test_async_hf_audio_client_wraps_sync():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = AsyncHuggingFaceAudioClient(sync)
    assert async_client._sync is sync
    assert async_client.spec.id == "facebook/musicgen-small"


def test_async_hf_audio_client_rejects_non_sync():
    with pytest.raises(TypeError, match="existing sync HuggingFaceAudioClient"):
        AsyncHuggingFaceAudioClient(object())


def test_aio_audio_client_wraps_sync():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = _aio_audio.audio_client(sync)
    assert isinstance(async_client._client, AsyncHuggingFaceAudioClient)
    assert async_client._client._sync is sync


def test_aio_audio_client_refuses_enum_with_wrap_guidance():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.audio_client"):
        _aio_audio.audio_client(HuggingFaceAudioModel.MUSICGEN_SMALL)


def test_aio_audio_client_refuses_string_with_wrap_guidance():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.audio_client"):
        _aio_audio.audio_client("hf:facebook/musicgen-small")


def test_aio_audio_client_refuses_random_type():
    with pytest.raises(TypeError, match="sync HuggingFaceAudioClient"):
        _aio_audio.audio_client(42)


# ---------------------------------------------------------------------------
# State sharing between sync and async wrappers
# ---------------------------------------------------------------------------


def test_async_hf_audio_client_shares_model_property():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = AsyncHuggingFaceAudioClient(sync)
    assert async_client.model is sync.model


def test_async_hf_audio_client_shares_spec():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = AsyncHuggingFaceAudioClient(sync)
    assert async_client.spec is sync.spec


def test_async_hf_audio_client_shares_model_kwargs():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL, model_kwargs={"torch_dtype": "auto"})
    async_client = AsyncHuggingFaceAudioClient(sync)
    assert async_client.model_kwargs == {"torch_dtype": "auto"}


# ---------------------------------------------------------------------------
# Async generate() delegates to sync via to_thread
# ---------------------------------------------------------------------------


async def test_async_generate_returns_numpy_tuple():
    import numpy as np

    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = AsyncHuggingFaceAudioClient(sync)
    result = await async_client.generate("test", format="numpy")
    assert isinstance(result, tuple) and len(result) == 2
    sr, audio = result
    assert sr == 32_000
    assert isinstance(audio, np.ndarray)


async def test_async_generate_with_format_bytes():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = AsyncHuggingFaceAudioClient(sync)
    data = await async_client.generate("test", format="bytes")
    assert isinstance(data, bytes)
    assert data.startswith(b"RIFF")


async def test_async_generate_propagates_kwargs():
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = AsyncHuggingFaceAudioClient(sync)
    await async_client.generate("test", format="numpy", duration_s=5.0)
    # After generate, the pipe is loaded and _last_call_kwargs shows max_new_tokens.
    kwargs = sync._pipe._last_call_kwargs
    assert kwargs.get("max_new_tokens") == 250  # int(5.0 * 50)


async def test_async_aio_audio_client_generate():
    """aio.audio_client() wraps sync and its generate() returns correctly."""
    sync = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    ac = _aio_audio.audio_client(sync)
    result = await ac.generate("test", format="numpy")
    assert isinstance(result, tuple)
