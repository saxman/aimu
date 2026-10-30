"""Pulling an Ollama model: what gets logged, and when a client pulls at all.

No server. The SDK's ``pull(stream=True)`` is replaced with a list of ``ProgressResponse`` events in
the shape Ollama streams them: status-only events (manifest, verification, success) carry no
``total``, and each layer reports ``completed`` against ``total``. A layer already on disk arrives
complete; only one that is actually downloading has ``completed < total``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import ollama
import pytest

import aimu
from aimu.aio.providers.ollama import apull_ollama_model
from aimu.models import ModelNotInstalledError
from aimu.models.providers.ollama import (
    OllamaClient,
    OllamaEmbeddingClient,
    OllamaEmbeddingModel,
    OllamaModel,
    pull_ollama_model,
)

_GIB = 1024**3


def _event(status="pulling", digest=None, total=None, completed=None) -> ollama.ProgressResponse:
    return ollama.ProgressResponse(status=status, digest=digest, total=total, completed=completed)


_UP_TO_DATE = [
    _event("pulling manifest"),
    _event(digest="sha256:a", total=100, completed=100),
    _event("verifying sha256 digest"),
    _event("success"),
]

_DOWNLOADING = [
    _event("pulling manifest"),
    _event(digest="sha256:b", total=10 * 10**9, completed=0),
    _event(digest="sha256:b", total=10 * 10**9, completed=5 * 10**9),
    _event(digest="sha256:b", total=10 * 10**9, completed=10 * 10**9),
    _event("success"),
]


class _FakeSdk:
    def __init__(self, events=None, error=None):
        self.events = events or []
        self.error = error
        self.pulls: list[str] = []

    def pull(self, model, *, stream=False):
        self.pulls.append(model)
        assert stream, "the helper must stream the pull to see progress"
        if self.error is not None:
            raise self.error
        return iter(self.events)


class _AsyncFakeSdk(_FakeSdk):
    async def pull(self, model, *, stream=False):
        events = super().pull(model, stream=stream)

        async def stream_events():
            for event in events:
                yield event

        return stream_events()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_up_to_date_pull_logs_no_warning(caplog):
    caplog.set_level(logging.DEBUG)
    pull_ollama_model(_FakeSdk(_UP_TO_DATE), "qwen3:8b")
    assert _warnings(caplog) == []
    assert any(r.levelno == logging.DEBUG and "up to date" in r.getMessage() for r in caplog.records)


def test_partial_layer_logs_one_warning(caplog):
    caplog.set_level(logging.DEBUG)
    pull_ollama_model(_FakeSdk(_DOWNLOADING), "qwen3:8b")
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "qwen3:8b" in warnings[0].getMessage()
    assert "9.3 GiB" in warnings[0].getMessage()
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("9.3 GiB downloaded" in message for message in infos)


def test_events_without_total_are_ignored(caplog):
    caplog.set_level(logging.DEBUG)
    events = [_event("pulling manifest"), _event("verifying sha256 digest"), _event("success")]
    pull_ollama_model(_FakeSdk(events), "qwen3:8b")
    assert _warnings(caplog) == []


def test_pull_error_propagates_unwrapped():
    error = ollama.ResponseError("pull model manifest: file does not exist", status_code=404)
    with pytest.raises(ollama.ResponseError) as info:
        pull_ollama_model(_FakeSdk(error=error), "nope:1b")
    assert info.value is error
    assert not isinstance(info.value, ModelNotInstalledError)


async def test_async_up_to_date_pull_logs_no_warning(caplog):
    caplog.set_level(logging.DEBUG)
    await apull_ollama_model(_AsyncFakeSdk(_UP_TO_DATE), "qwen3:8b")
    assert _warnings(caplog) == []


async def test_async_partial_layer_logs_one_warning(caplog):
    caplog.set_level(logging.DEBUG)
    await apull_ollama_model(_AsyncFakeSdk(_DOWNLOADING), "qwen3:8b")
    assert len(_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# pull= on the clients
# ---------------------------------------------------------------------------


class _RecordingSdk:
    """Stands in for ``ollama.Client`` / ``ollama.AsyncClient``: records pulls and requests."""

    def __init__(self, pull_errors=(), **kwargs):
        self.kwargs = kwargs
        self.pulls: list[str] = []
        self.calls: list[str] = []
        self._pull_errors = list(pull_errors)

    def _record_pull(self, model):
        self.pulls.append(model)
        self.calls.append("pull")
        if self._pull_errors:
            raise self._pull_errors.pop(0)


class _SyncRecordingSdk(_RecordingSdk):
    def pull(self, model, *, stream=False):
        self._record_pull(model)
        return iter(())


class _AsyncRecordingSdk(_RecordingSdk):
    async def pull(self, model, *, stream=False):
        await asyncio.sleep(0)
        self._record_pull(model)

        async def no_events():
            return
            yield  # pragma: no cover

        return no_events()

    async def chat(self, **kwargs):
        self.calls.append("chat")
        return ollama.ChatResponse(message=ollama.Message(role="assistant", content="hi"), done=True)

    async def generate(self, **kwargs):
        self.calls.append("generate")
        return ollama.GenerateResponse(response="hi", done=True)


@pytest.fixture
def sync_sdks(monkeypatch):
    built: list[_SyncRecordingSdk] = []

    def factory(**kwargs):
        built.append(_SyncRecordingSdk(**kwargs))
        return built[-1]

    monkeypatch.setattr(ollama, "Client", factory)
    return built


def _async_client(sdk, **kwargs):
    from aimu.aio.providers.ollama import AsyncOllamaClient

    client = AsyncOllamaClient(OllamaModel.QWEN_3_8B, **kwargs)
    client._client = sdk
    return client


def test_default_construction_does_not_pull(sync_sdks):
    OllamaClient(OllamaModel.QWEN_3_8B)
    OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT)
    assert [sdk.pulls for sdk in sync_sdks] == [[], []]


async def test_async_default_does_not_pull():
    sdk = _AsyncRecordingSdk()
    client = _async_client(sdk)
    await client.chat("hello")
    assert sdk.pulls == []


def test_pull_true_pulls_once_on_host(sync_sdks):
    OllamaClient(OllamaModel.QWEN_3_8B, host="http://gpu-box:11434", pull=True)
    OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT, host="http://gpu-box:11434", pull=True)
    assert [sdk.pulls for sdk in sync_sdks] == [["qwen3:8b"], ["nomic-embed-text"]]
    assert all(sdk.kwargs["host"] == "http://gpu-box:11434" for sdk in sync_sdks)


def test_factories_forward_pull(sync_sdks):
    aimu.client("ollama:qwen3:8b", pull=True)
    aimu.embedding_client("ollama:nomic-embed-text", pull=True)
    assert [sdk.pulls for sdk in sync_sdks] == [["qwen3:8b"], ["nomic-embed-text"]]


async def test_async_pull_true_pulls_once_across_concurrent_first_requests():
    sdk = _AsyncRecordingSdk()
    client = _async_client(sdk, pull=True)
    assert sdk.pulls == []
    await asyncio.gather(*(client.generate("hello") for _ in range(3)))
    assert sdk.pulls == ["qwen3:8b"]


async def test_async_pull_happens_before_the_first_request():
    sdk = _AsyncRecordingSdk()
    client = _async_client(sdk, pull=True)
    await client.chat("hello")
    assert sdk.calls == ["pull", "chat"]


async def test_async_failed_pull_is_retried_next_request():
    sdk = _AsyncRecordingSdk(pull_errors=[httpx.ConnectError("refused")])
    client = _async_client(sdk, pull=True)
    with pytest.raises(httpx.ConnectError):
        await client.chat("hello")
    await client.chat("hello again")
    assert sdk.pulls == ["qwen3:8b", "qwen3:8b"]


def test_pull_true_unreachable_server_raises_transport_error(monkeypatch):
    sdk = _SyncRecordingSdk(pull_errors=[httpx.ConnectError("refused")])
    monkeypatch.setattr(ollama, "Client", lambda **kw: sdk)
    with pytest.raises(httpx.ConnectError):
        OllamaClient(OllamaModel.QWEN_3_8B, pull=True)
