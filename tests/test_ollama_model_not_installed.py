"""A model the Ollama server does not have surfaces as ModelNotInstalledError, not a raw 404.

No server: the SDK's ``chat`` / ``generate`` / ``embed`` are replaced with fakes raising the
``ResponseError`` Ollama returns for a missing model. Streamed fakes raise when first advanced, as the
SDK's do, so the translation has to cover stream consumption and not only the call.

The last tests pin what is *not* translated: an unrelated 404 (a proxy's "page not found") and a
"not found" carrying another status are reported as the server sent them.
"""

from __future__ import annotations

import types

import ollama
import pytest

import aimu
from aimu import aio
from aimu.models import ModelNotInstalledError
from aimu.models.providers.ollama import OllamaClient, OllamaEmbeddingClient, OllamaModel

_MISSING = 'model "qwen3:8b" not found, try pulling it first'


def _error(message: str = _MISSING, status: int = 404) -> ollama.ResponseError:
    return ollama.ResponseError(message, status_code=status)


def _sync_fake(error: ollama.ResponseError):
    def call(**kwargs):
        if kwargs.get("stream"):

            def stream():
                raise error
                yield  # pragma: no cover

            return stream()
        raise error

    return types.SimpleNamespace(pull=lambda *a, **k: None, chat=call, generate=call, embed=call)


def _async_fake(error: ollama.ResponseError):
    async def call(**kwargs):
        if kwargs.get("stream"):

            async def stream():
                raise error
                yield  # pragma: no cover

            return stream()
        raise error

    return types.SimpleNamespace(pull=call, chat=call, generate=call)


def _sync_client(monkeypatch, error=None) -> OllamaClient:
    monkeypatch.setattr(ollama, "Client", lambda **kw: _sync_fake(error or _error()))
    return OllamaClient(OllamaModel.QWEN_3_8B)


def _async_client(error=None):
    client = aio.client("ollama:qwen3:8b")
    client._client._client = _async_fake(error or _error())
    return client


def _assert_translated(exc: ModelNotInstalledError) -> None:
    assert exc.model == "qwen3:8b"
    assert isinstance(exc.__cause__, ollama.ResponseError)
    message = str(exc)
    assert 'aimu.pull("ollama:qwen3:8b")' in message
    assert "pull=True" in message
    assert "ollama pull qwen3:8b" in message


def _sync_call(client, method: str, stream: bool):
    result = getattr(client, method)("hello", stream=stream)
    if stream:
        list(result)


async def _async_call(client, method: str, stream: bool):
    result = await getattr(client, method)("hello", stream=stream)
    if stream:
        async for _ in result:
            pass


@pytest.mark.parametrize("method", ["chat", "generate"])
@pytest.mark.parametrize("stream", [False, True])
def test_sync_missing_model_raises_named_error(monkeypatch, method, stream):
    client = _sync_client(monkeypatch)
    with pytest.raises(ModelNotInstalledError) as info:
        _sync_call(client, method, stream)
    _assert_translated(info.value)


@pytest.mark.parametrize("method", ["chat", "generate"])
@pytest.mark.parametrize("stream", [False, True])
async def test_async_missing_model_raises_named_error(method, stream):
    client = _async_client()
    with pytest.raises(ModelNotInstalledError) as info:
        await _async_call(client, method, stream)
    _assert_translated(info.value)


def test_embedding_missing_model_raises_named_error(monkeypatch):
    monkeypatch.setattr(ollama, "Client", lambda **kw: _sync_fake(_error('model "nomic-embed-text" not found')))
    client = OllamaEmbeddingClient("ollama:nomic-embed-text")
    with pytest.raises(ModelNotInstalledError) as info:
        client.embed("hello")
    assert info.value.model == "nomic-embed-text"
    assert 'aimu.pull("ollama:nomic-embed-text")' in str(info.value)


def test_unrelated_404_is_not_translated(monkeypatch):
    client = _sync_client(monkeypatch, _error("404 page not found", 404))
    with pytest.raises(ollama.ResponseError) as info:
        client.chat("hello")
    assert not isinstance(info.value, ModelNotInstalledError)


def test_not_found_with_other_status_is_not_translated(monkeypatch):
    client = _sync_client(monkeypatch, _error(_MISSING, 500))
    with pytest.raises(ollama.ResponseError):
        client.generate("hello")


def test_exported_from_all_three_namespaces():
    assert aimu.ModelNotInstalledError is ModelNotInstalledError
    assert aio.ModelNotInstalledError is ModelNotInstalledError
