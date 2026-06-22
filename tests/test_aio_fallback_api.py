"""Mock-only async tests for AsyncFallbackClient (P0-B). Mirrors test_fallback_api.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aimu.aio import AsyncFallbackClient
from aimu.aio._base import AsyncBaseModelClient
from aimu.models.fallback import FallbackExhaustedError


def _fake_model():
    model = MagicMock()
    model.supports_tools = True
    model.supports_thinking = False
    model.supports_vision = False
    model.supports_audio = False
    model.supports_structured_output = False
    return model


class AsyncStubClient:
    def __init__(self, *, replies=None, error=None):
        self.model = _fake_model()
        self.messages: list[dict] = []
        self.tools: list = []
        self._system_message = None
        self.tool_context_deps = None
        self.last_thinking = ""
        self.last_usage = None
        self.concurrent_tool_calls = False
        self._replies = list(replies or [])
        self._error = error
        self.chat_calls = 0
        self.generate_calls = 0

    @property
    def system_message(self):
        return self._system_message

    @system_message.setter
    def system_message(self, value):
        self._system_message = value

    def reset(self, system_message="__keep__"):
        self.messages = []
        if system_message != "__keep__":
            self._system_message = system_message
        self.last_thinking = ""
        self.last_usage = None

    def _reply(self):
        return self._replies.pop(0) if self._replies else "ok"

    async def chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, **kwargs):
        self.chat_calls += 1
        if stream:
            return self._chat_stream(user_message)
        if self._error is not None:
            raise self._error
        if not self.messages and self._system_message:
            self.messages.append({"role": "system", "content": self._system_message})
        self.messages.append({"role": "user", "content": user_message})
        reply = self._reply()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    async def _chat_stream(self, user_message):
        if self._error is not None:
            raise self._error
        self.messages.append({"role": "user", "content": user_message})
        reply = self._reply()
        for token in reply.split():
            yield token
        self.messages.append({"role": "assistant", "content": reply})

    async def generate(self, prompt, generate_kwargs=None, stream=False, **kwargs):
        self.generate_calls += 1
        if stream:
            return self._generate_stream()
        if self._error is not None:
            raise self._error
        return self._reply()

    async def _generate_stream(self):
        if self._error is not None:
            raise self._error
        for token in self._reply().split():
            yield token


async def test_async_fallback_is_base():
    fb = AsyncFallbackClient([AsyncStubClient(), AsyncStubClient()])
    assert isinstance(fb, AsyncBaseModelClient)


async def test_async_falls_back_on_error():
    a = AsyncStubClient(error=ConnectionError("down"))
    b = AsyncStubClient(replies=["B"])
    fb = AsyncFallbackClient([a, b])
    assert await fb.chat("hi") == "B"
    assert a.chat_calls == 1
    assert b.chat_calls == 1


async def test_async_all_fail_raises_chained():
    last = TimeoutError("second")
    fb = AsyncFallbackClient([AsyncStubClient(error=ConnectionError("first")), AsyncStubClient(error=last)])
    with pytest.raises(FallbackExhaustedError) as excinfo:
        await fb.chat("hi")
    assert excinfo.value.__cause__ is last


async def test_async_history_preserved_across_failover():
    a = AsyncStubClient(replies=["A1"])
    b = AsyncStubClient(replies=["B2"])
    fb = AsyncFallbackClient([a, b])
    assert await fb.chat("turn one") == "A1"
    a._error = ConnectionError("down")
    assert await fb.chat("turn two") == "B2"
    user_msgs = [m["content"] for m in b.messages if m["role"] == "user"]
    assert user_msgs == ["turn one", "turn two"]


async def test_async_retry_on_narrows():
    a = AsyncStubClient(error=ValueError("permanent"))
    b = AsyncStubClient(replies=["B"])
    fb = AsyncFallbackClient([a, b], retry_on=(TimeoutError,))
    with pytest.raises(ValueError, match="permanent"):
        await fb.chat("hi")
    assert b.chat_calls == 0


async def test_async_generate_falls_back():
    a = AsyncStubClient(error=ConnectionError("down"))
    b = AsyncStubClient(replies=["G"])
    fb = AsyncFallbackClient([a, b])
    assert await fb.generate("p") == "G"


async def test_async_streaming_falls_back_before_first_chunk():
    a = AsyncStubClient(error=ConnectionError("down"))
    b = AsyncStubClient(replies=["hello world"])
    fb = AsyncFallbackClient([a, b])
    chunks = [c async for c in await fb.chat("hi", stream=True)]
    assert chunks == ["hello", "world"]
    assert a.chat_calls == 1
    assert b.chat_calls == 1


async def test_async_streaming_all_fail_raises():
    fb = AsyncFallbackClient([AsyncStubClient(error=ConnectionError("a")), AsyncStubClient(error=ConnectionError("b"))])
    with pytest.raises(FallbackExhaustedError):
        [c async for c in await fb.chat("hi", stream=True)]
