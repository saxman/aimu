"""Mock-only tests for the async speech surface (``aimu.aio.speech``).

Covers:
* Direct enum/string construction is refused with a helpful pointer at the
  sync-then-wrap pattern.
* ``AsyncSpeechClient`` / ``AsyncHuggingFaceSpeechClient`` / ``AsyncOpenAISpeechClient``
  wrap their sync siblings and share state.
* ``AsyncHuggingFaceSpeechClient.generate()`` and ``AsyncOpenAISpeechClient.generate()``
  delegate to the sync client via ``asyncio.to_thread``.
"""

from __future__ import annotations

import importlib

import pytest

from test_speech_api import _install_speech_stubs  # noqa: F401 — side effect

_install_speech_stubs()

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_SPEECH:
    aimu.models.hf_speech = importlib.import_module("aimu.models.hf_speech")
    aimu.models.HAS_HF_SPEECH = True
    aimu.models.HuggingFaceSpeechClient = aimu.models.hf_speech.HuggingFaceSpeechClient
    aimu.models.HuggingFaceSpeechModel = aimu.models.hf_speech.HuggingFaceSpeechModel

if not aimu.models.HAS_OPENAI_SPEECH:
    aimu.models.openai_speech = importlib.import_module("aimu.models.openai_speech")
    aimu.models.HAS_OPENAI_SPEECH = True
    aimu.models.OpenAISpeechClient = aimu.models.openai_speech.OpenAISpeechClient
    aimu.models.OpenAISpeechModel = aimu.models.openai_speech.OpenAISpeechModel

import aimu  # noqa: E402

aimu.HAS_HF_SPEECH = True
aimu.HAS_OPENAI_SPEECH = True
aimu.HuggingFaceSpeechClient = aimu.models.HuggingFaceSpeechClient
aimu.HuggingFaceSpeechModel = aimu.models.HuggingFaceSpeechModel
aimu.OpenAISpeechClient = aimu.models.OpenAISpeechClient
aimu.OpenAISpeechModel = aimu.models.OpenAISpeechModel

import aimu.aio.speech as _aio_speech  # noqa: E402

importlib.reload(_aio_speech)

from aimu.aio.providers.hf_speech import AsyncHuggingFaceSpeechClient  # noqa: E402
from aimu.aio.providers.openai_speech import AsyncOpenAISpeechClient  # noqa: E402
from aimu.models.hf_speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel  # noqa: E402
from aimu.models.openai_speech import OpenAISpeechClient, OpenAISpeechModel  # noqa: E402


# ---------------------------------------------------------------------------
# Wrap refusal — direct enum / string construction
# ---------------------------------------------------------------------------


def test_aio_speech_client_refuses_hf_enum():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.speech_client"):
        _aio_speech.speech_client(HuggingFaceSpeechModel.MMS_TTS_ENG)


def test_aio_speech_client_refuses_openai_enum():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.speech_client"):
        _aio_speech.speech_client(OpenAISpeechModel.TTS_1)


def test_aio_speech_client_refuses_string():
    with pytest.raises(ValueError, match=r"sync_client = aimu\.speech_client"):
        _aio_speech.speech_client("openai:tts-1")


def test_aio_speech_client_refuses_random_type():
    with pytest.raises(TypeError, match="sync HuggingFaceSpeechClient or OpenAISpeechClient"):
        _aio_speech.speech_client(42)


# ---------------------------------------------------------------------------
# AsyncHuggingFaceSpeechClient — construction and state sharing
# ---------------------------------------------------------------------------


def test_async_hf_speech_client_wraps_sync():
    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    async_client = AsyncHuggingFaceSpeechClient(sync)
    assert async_client._sync is sync
    assert async_client.spec.id == "facebook/mms-tts-eng"


def test_async_hf_speech_client_rejects_non_sync():
    with pytest.raises(TypeError, match="existing sync HuggingFaceSpeechClient"):
        AsyncHuggingFaceSpeechClient(object())


def test_async_hf_speech_client_shares_model_property():
    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    async_client = AsyncHuggingFaceSpeechClient(sync)
    assert async_client.model is sync.model


def test_async_hf_speech_client_shares_spec():
    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    async_client = AsyncHuggingFaceSpeechClient(sync)
    assert async_client.spec is sync.spec


def test_async_hf_speech_client_shares_model_kwargs():
    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG, model_kwargs={"device": "cpu"})
    async_client = AsyncHuggingFaceSpeechClient(sync)
    assert async_client.model_kwargs == {"device": "cpu"}


# ---------------------------------------------------------------------------
# AsyncOpenAISpeechClient — construction and state sharing
# ---------------------------------------------------------------------------


def test_async_openai_speech_client_wraps_sync():
    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    async_client = AsyncOpenAISpeechClient(sync)
    assert async_client._sync is sync
    assert async_client.spec.id == "tts-1"


def test_async_openai_speech_client_rejects_non_sync():
    with pytest.raises(TypeError, match="existing sync OpenAISpeechClient"):
        AsyncOpenAISpeechClient(object())


def test_async_openai_speech_client_shares_model_property():
    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    async_client = AsyncOpenAISpeechClient(sync)
    assert async_client.model is sync.model


def test_async_openai_speech_client_shares_spec():
    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    async_client = AsyncOpenAISpeechClient(sync)
    assert async_client.spec is sync.spec


# ---------------------------------------------------------------------------
# AsyncSpeechClient factory — wraps correct inner client
# ---------------------------------------------------------------------------


def test_aio_speech_client_wraps_hf_sync():
    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    async_client = _aio_speech.speech_client(sync)
    assert isinstance(async_client._client, AsyncHuggingFaceSpeechClient)
    assert async_client._client._sync is sync


def test_aio_speech_client_wraps_openai_sync():
    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    async_client = _aio_speech.speech_client(sync)
    assert isinstance(async_client._client, AsyncOpenAISpeechClient)
    assert async_client._client._sync is sync


# ---------------------------------------------------------------------------
# Async generate() delegates to sync via to_thread
# ---------------------------------------------------------------------------


async def test_async_hf_speech_generate_returns_bytes(monkeypatch):
    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    expected = b"RIFF" + b"\x00" * 40

    def fake_generate(text, **kwargs):
        return expected

    monkeypatch.setattr(sync, "generate", fake_generate)
    async_client = AsyncHuggingFaceSpeechClient(sync)
    result = await async_client.generate("Hello", format="bytes")
    assert result == expected


async def test_async_openai_speech_generate_returns_bytes(monkeypatch):
    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)

    def fake_generate(text, **kwargs):
        return b"RIFF" + b"\x00" * 40

    monkeypatch.setattr(sync, "generate", fake_generate)
    async_client = AsyncOpenAISpeechClient(sync)
    result = await async_client.generate("Hello", format="bytes")
    assert result == b"RIFF" + b"\x00" * 40


async def test_async_aio_speech_client_generate(monkeypatch):
    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)

    def fake_generate(text, **kwargs):
        return b"RIFF" + b"\x00" * 40

    monkeypatch.setattr(sync, "generate", fake_generate)
    ac = _aio_speech.speech_client(sync)
    result = await ac.generate("Hello", format="bytes")
    assert result == b"RIFF" + b"\x00" * 40


# ---------------------------------------------------------------------------
# Streaming path — stream=True returns AsyncIterator
# ---------------------------------------------------------------------------


async def test_async_hf_speech_generate_stream(monkeypatch):
    from aimu.models.base import StreamChunk, StreamingContentType

    sync = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    expected_chunks = [
        StreamChunk(StreamingContentType.SPEECH_GENERATING, {"chunk_index": 1, "total_chunks": 1, "final": True, "result": "/tmp/out.wav"}),
    ]

    def fake_generate_stream(text, **kwargs):
        yield from expected_chunks

    monkeypatch.setattr(sync, "generate", fake_generate_stream)
    async_client = AsyncHuggingFaceSpeechClient(sync)
    chunks = []
    async for chunk in await async_client.generate("Hello", stream=True):
        chunks.append(chunk)
    assert chunks == expected_chunks


async def test_async_openai_speech_generate_stream(monkeypatch):
    from aimu.models.base import StreamChunk, StreamingContentType

    sync = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    expected_chunks = [
        StreamChunk(StreamingContentType.SPEECH_GENERATING, {"chunk_index": 1, "total_chunks": None, "final": True, "result": "/tmp/out.wav"}),
    ]

    def fake_generate_stream(text, **kwargs):
        yield from expected_chunks

    monkeypatch.setattr(sync, "generate", fake_generate_stream)
    async_client = AsyncOpenAISpeechClient(sync)
    chunks = []
    async for chunk in await async_client.generate("Hello", stream=True):
        chunks.append(chunk)
    assert chunks == expected_chunks
