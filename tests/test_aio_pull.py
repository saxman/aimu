"""``aimu.aio.pull()``: the async twin of ``aimu.pull()``, sharing its resolution and refusals."""

from __future__ import annotations

import pytest

pytest.importorskip("ollama")

import aimu.aio.model_pull as aio_pull_module  # noqa: E402
import aimu.models.model_pull as pull_module  # noqa: E402
from aimu import aio  # noqa: E402
from aimu.models import OllamaModel  # noqa: E402


async def test_aio_pull_ollama_uses_the_async_helper_on_host(monkeypatch):
    calls = []

    async def record(sdk, model_id):
        calls.append((model_id, sdk))

    monkeypatch.setattr(aio_pull_module, "_ollama_async_sdk_client", lambda host: {"host": host})
    monkeypatch.setattr(aio_pull_module, "apull_ollama_model", record)
    await aio.pull(OllamaModel.QWEN_3_8B, host="http://gpu-box:11434")
    await aio.pull("ollama:nomic-embed-text")
    assert calls == [("qwen3:8b", {"host": "http://gpu-box:11434"}), ("nomic-embed-text", {"host": None})]


async def test_aio_pull_hf_runs_every_declared_download(monkeypatch):
    pytest.importorskip("transformers")
    from aimu.models.providers.hf.speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    fetched = []
    monkeypatch.setattr(pull_module, "fetch", fetched.append)
    await aio.pull("hf:microsoft/speecht5_tts")
    assert fetched == HuggingFaceSpeechClient._hub_downloads(HuggingFaceSpeechModel.SPEECHT5.spec)


async def test_aio_pull_refuses_bare_name():
    with pytest.raises(ValueError, match='"provider:model_id"'):
        await aio.pull("QWEN_3_8B")
