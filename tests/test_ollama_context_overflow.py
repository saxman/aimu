"""An over-long chat request surfaces as ContextOverflowError rather than Ollama's raw 500.

No server: the Ollama SDK's ``chat`` is replaced with one that raises the same ``ResponseError``
the server returns when it trims a request to fit the runner's context window and the trim reaches
the user turn, which the qwen3.5-family prompt renderer requires.

The last test pins the other half of the contract: a request that genuinely carries no user
message is a caller error, and the server's own wording already says so, so it is not translated.
"""

from __future__ import annotations

import types

import ollama
import pytest

import aimu.models.providers.ollama as sync_ollama
from aimu import aio
from aimu.models import ContextOverflowError
from aimu.models.providers.ollama import OllamaClient, OllamaModel

_SERVER_MESSAGE = "no user query found in messages"


def _raise_response_error(**kwargs):
    raise ollama.ResponseError(_SERVER_MESSAGE, status_code=500)


async def _araise_response_error(**kwargs):
    _raise_response_error()


def _sync_client(monkeypatch) -> OllamaClient:
    monkeypatch.setattr(sync_ollama, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(
        ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=_raise_response_error)
    )
    return OllamaClient(OllamaModel.QWEN_3_8B)


def _async_client():
    client = aio.client("ollama:qwen3:8b")
    client._client._client = types.SimpleNamespace(chat=_araise_response_error)
    return client


def _assert_translated(exc: ContextOverflowError) -> None:
    assert isinstance(exc.__cause__, ollama.ResponseError)
    assert _SERVER_MESSAGE in str(exc.__cause__)
    # The message has to name the actual problem and the knob that fixes it, because this is what
    # a delegating agent reads back as its tool result.
    assert "context window" in str(exc)
    assert "context_length" in str(exc)


def test_chat_translates_dropped_user_turn(monkeypatch):
    client = _sync_client(monkeypatch)
    with pytest.raises(ContextOverflowError) as info:
        client.chat("research NVIDIA earnings")
    _assert_translated(info.value)


def test_streamed_chat_translates_dropped_user_turn(monkeypatch):
    client = _sync_client(monkeypatch)
    with pytest.raises(ContextOverflowError) as info:
        list(client.chat("research NVIDIA earnings", stream=True))
    _assert_translated(info.value)


async def test_async_chat_translates_dropped_user_turn():
    client = _async_client()
    with pytest.raises(ContextOverflowError) as info:
        await client.chat("research NVIDIA earnings")
    _assert_translated(info.value)


async def test_async_streamed_chat_translates_dropped_user_turn():
    client = _async_client()
    with pytest.raises(ContextOverflowError) as info:
        stream = await client.chat("research NVIDIA earnings", stream=True)
        async for _ in stream:
            pass
    _assert_translated(info.value)


async def test_request_without_a_user_message_is_not_translated():
    client = _async_client()
    client.messages = [{"role": "system", "content": "You are a research worker."}]
    with pytest.raises(ollama.ResponseError):
        await client._client._chat(None)
