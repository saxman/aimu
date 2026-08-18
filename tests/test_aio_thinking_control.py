"""Async mirror of tests/test_thinking_control.py. Resolution logic is shared through
_ChatStateMixin, so this file covers only the async surface and async provider paths."""

from __future__ import annotations

import types

import ollama
import pytest

from aimu.models._internal.thinking import THINKING_KWARG, ResolvedThinking


def _fake_async_client(model, recorder):
    from aimu.aio._base import AsyncBaseModelClient

    class _Fake(AsyncBaseModelClient):
        MODELS = None

        def __init__(self):
            self.model = model
            self.model_kwargs = None
            self._system_message = None
            self.default_generate_kwargs = {}
            self.messages = []
            self.tools = []
            self.last_thinking = ""
            self.last_usage = None
            self.last_output_truncated = False
            self.last_structured = None

        def _resolve_generate_kwargs(self, generate_kwargs=None):
            return dict(generate_kwargs or {})

        async def _chat(self, user_message=None, generate_kwargs=None, **kw):
            recorder.append(generate_kwargs)
            return "ok"

        async def _generate(self, prompt, generate_kwargs=None, **kw):
            recorder.append(generate_kwargs)
            return "ok"

    return _Fake()


class _Model:
    def __init__(self, value="m", thinking=True, levels=True, optional=True):
        self.value = value
        self.supports_thinking = thinking
        self.thinking_levels = levels
        self.thinking_optional = optional
        self.generation_kwargs = {"temperature": 1.0}
        self.nonthinking_generation_kwargs = {}


async def test_async_chat_forwards_a_resolved_request():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    await client.chat("hi", thinking="medium")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="medium")


async def test_async_generate_forwards_a_resolved_request():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    await client.generate("hi", thinking=False)

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=False, level=None)


async def test_async_omitting_thinking_injects_nothing():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    await client.chat("hi")

    assert THINKING_KWARG not in (seen[0] or {})


async def test_async_invalid_value_raises():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    with pytest.raises(ValueError):
        await client.chat("hi", thinking="xhigh")

    assert seen == []


async def test_aio_ollama_maps_thinking_to_the_think_parameter(monkeypatch):
    import aimu.aio.providers.ollama as aio_ollama
    from aimu import aio

    calls: list[dict] = []

    async def record(**kw):
        calls.append(kw)
        # `response["message"]` is accessed with dot notation in `_chat` (it is a pydantic
        # object on the real SDK), so the stand-in needs attribute access too.
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None, thinking=None)
        return {"message": message}

    monkeypatch.setattr(aio_ollama, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(aio_ollama, "truncated_from_ollama", lambda *a, **k: False)
    monkeypatch.setattr(ollama, "AsyncClient", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    client = aio.client("ollama:qwen3.8:27b")
    client._client._client = types.SimpleNamespace(chat=record, generate=record)

    await client.chat("hi", thinking="low")

    assert calls[0]["think"] == "low"
    assert THINKING_KWARG not in calls[0]["options"]


async def test_top_level_chat_forwards_thinking(monkeypatch):
    import aimu.aio._model_client as aio_model_client
    from aimu import aio

    seen: list = []

    def fake_client(model=None, **kw):
        return _fake_async_client(_Model(), seen)

    monkeypatch.setattr(aio_model_client, "client", fake_client)

    await aio.chat("hi", model="ollama:qwen3.8:27b", thinking="low")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="low")
