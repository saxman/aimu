"""Mock-only regression tests for Ollama streaming tool-call detection (no server).

Reproduces the bug where an empty transitional stream part (content="" and no tool_calls) emitted
between the end of thinking and the part carrying tool_calls caused the tool call to be dropped and
an empty response to be streamed instead. Feeds scripted Ollama-shaped stream parts into the sync
and async `_chat_streamed`.
"""

from __future__ import annotations

import types

import ollama

import aimu
import aimu.aio.providers.ollama as aio_ollama
import aimu.models.providers.ollama as sync_ollama
from aimu import aio
from aimu.models import StreamingContentType
from aimu.models.providers.ollama import OllamaClient, OllamaModel


class _Fn:
    def __init__(self, n, a):
        self.name, self.arguments = n, a


class _TC:
    def __init__(self, n, a):
        self.function = _Fn(n, a)


class _Msg:
    def __init__(self, thinking="", content="", tool_calls=None, role="assistant"):
        self.thinking, self.content, self.tool_calls, self.role = thinking, content, tool_calls, role


class _Part(dict):
    def __init__(self, msg):
        super().__init__(message=msg)


@aimu.tool
def _weather(city: str) -> str:
    """Get the weather."""
    return f"{city}: sunny"


def _scenario_b():
    # thinking, then an empty transitional part, then the tool_calls part -- one turn. chat() is
    # single-turn: it parses and stores the tool call and returns (the Agent's engine executes it).
    tool_calls = [_TC("_weather", {"city": "Paris"})]
    first = [
        _Part(_Msg(thinking="thinking...")),
        _Part(_Msg(content="")),
        _Part(_Msg(content="", tool_calls=tool_calls)),
    ]
    return [first]


async def _aiter(parts):
    for p in parts:
        yield p


def _assert_tool_call_stored(chunks, messages):
    # The streaming parser assembled the tool call into the stored assistant message despite the
    # empty transitional part (it was not dropped). chat() is single-turn: it stores the call but
    # does NOT execute it (the Agent's tool-loop engine does) — so we assert on the stored message,
    # not a tool result or a TOOL_CALLING chunk.
    assistant = next((m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")), None)
    assert assistant is not None, "tool call was dropped"
    assert assistant["tool_calls"][0]["function"]["name"] == "_weather"
    assert not any(c.phase == StreamingContentType.GENERATING and c.content == "" for c in chunks), "stray empty token"


async def test_aio_tool_call_survives_empty_transitional_part(monkeypatch):
    monkeypatch.setattr(aio_ollama, "usage_from_ollama", lambda *a, **k: None)
    scen, st = _scenario_b(), {"i": 0}

    async def fake_chat(**kw):
        p = scen[st["i"]]
        st["i"] += 1
        return _aiter(p)

    client = aio.client("ollama:qwen3:8b")
    client.tools = [_weather]
    client._client._client = types.SimpleNamespace(chat=fake_chat)

    chunks = [c async for c in await client.chat("weather in Paris?", stream=True)]
    _assert_tool_call_stored(chunks, client.messages)


def test_sync_tool_call_survives_empty_transitional_part(monkeypatch):
    monkeypatch.setattr(sync_ollama, "usage_from_ollama", lambda *a, **k: None)
    scen, st = _scenario_b(), {"i": 0}

    def fake_chat(**kw):
        p = scen[st["i"]]
        st["i"] += 1
        return iter(p)

    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=fake_chat))
    client = OllamaClient(OllamaModel.QWEN_3_8B)
    client.tools = [_weather]

    chunks = list(client.chat("weather in Paris?", stream=True))
    _assert_tool_call_stored(chunks, client.messages)


async def test_aio_plain_answer_emits_no_empty_trailing_token(monkeypatch):
    monkeypatch.setattr(aio_ollama, "usage_from_ollama", lambda *a, **k: None)
    # thinking, content, then an empty `done` part -- no tool call.
    parts = [_Part(_Msg(thinking="t")), _Part(_Msg(content="Hi there.")), _Part(_Msg(content=""))]

    async def fake_chat(**kw):
        return _aiter(parts)

    client = aio.client("ollama:qwen3:8b")
    client._client._client = types.SimpleNamespace(chat=fake_chat)

    chunks = [c async for c in await client.chat("hi", stream=True)]
    gen = [c.content for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert gen == ["Hi there."]  # the empty done part is not yielded


def _multi_tool_call_response():
    # A thinking model's first turn: reasoning plus two tool calls, no user-facing content. This is
    # the shape that triggered `list index out of range` in non-streaming _chat.
    tool_calls = [_TC("_weather", {"city": "Paris"}), _TC("_weather", {"city": "Berlin"})]
    return _Part(_Msg(thinking="let me check both", content="", tool_calls=tool_calls))


def _assert_thinking_on_tool_call_message(messages):
    assistant = next(m for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
    assert assistant.get("thinking") == "let me check both"


async def test_aio_thinking_recorded_on_multi_tool_call_turn(monkeypatch):
    monkeypatch.setattr(aio_ollama, "usage_from_ollama", lambda *a, **k: None)
    response = _multi_tool_call_response()

    async def fake_chat(**kw):
        return response

    client = aio.client("ollama:qwen3:8b")
    client.tools = [_weather]
    client._client._client = types.SimpleNamespace(chat=fake_chat)

    await client.chat("weather in Paris and Berlin?")
    _assert_thinking_on_tool_call_message(client.messages)


def test_sync_thinking_recorded_on_multi_tool_call_turn(monkeypatch):
    monkeypatch.setattr(sync_ollama, "usage_from_ollama", lambda *a, **k: None)
    response = _multi_tool_call_response()

    def fake_chat(**kw):
        return response

    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=fake_chat))
    client = OllamaClient(OllamaModel.QWEN_3_8B)
    client.tools = [_weather]

    client.chat("weather in Paris and Berlin?")
    _assert_thinking_on_tool_call_message(client.messages)
