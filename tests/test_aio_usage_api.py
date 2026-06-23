"""Mock-only async tests for streaming token-usage capture (P1-B). Mirrors test_usage_api.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _content_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])


def _usage_chunk(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )


def _make_async_openai_compat_client():
    from aimu.models import HAS_OPENAI_COMPAT

    if not HAS_OPENAI_COMPAT:
        pytest.skip("openai not installed")
    from aimu.aio.providers.openai_compat import AsyncLMStudioOpenAIClient
    from aimu.models.providers.openai_compat import LMStudioOpenAIModel

    client = AsyncLMStudioOpenAIClient(LMStudioOpenAIModel.MISTRAL_7B)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    return client


async def test_async_openai_compat_streaming_captures_usage(monkeypatch):
    client = _make_async_openai_compat_client()

    async def fake_create(**kwargs):
        async def agen():
            yield _content_chunk("he")
            yield _content_chunk("llo")
            yield _usage_chunk(10, 5)

        return agen()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    chunks = [c async for c in await client.generate("hi", stream=True)]
    assert "".join(c.content for c in chunks) == "hello"
    assert client.last_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


async def test_async_openai_compat_streaming_no_usage_chunk_stays_none(monkeypatch):
    client = _make_async_openai_compat_client()

    async def fake_create(**kwargs):
        async def agen():
            yield _content_chunk("hi")

        return agen()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    _ = [c async for c in await client.generate("hi", stream=True)]
    assert client.last_usage is None


async def test_async_ollama_streaming_captures_usage(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.aio.providers import ollama as aio_ollama_mod
    from aimu.aio.providers.ollama import AsyncOllamaClient
    from aimu.models.providers.ollama import OllamaModel

    class _Part(dict):
        def __getattr__(self, key):
            return self.get(key)

    async def _aiter():
        yield _Part(response="he")
        yield _Part(response="llo", prompt_eval_count=10, eval_count=5, done=True)

    class FakeAsyncClient:
        def __init__(self, **kw):
            pass

        async def generate(self, **kw):
            return _aiter()

    monkeypatch.setattr(aio_ollama_mod.ollama, "AsyncClient", FakeAsyncClient)
    client = AsyncOllamaClient(OllamaModel.LLAMA_3_2_3B)

    chunks = [c async for c in await client.generate("hi", stream=True)]
    assert "".join(c.content for c in chunks) == "hello"
    assert client.last_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


async def test_async_anthropic_streaming_captures_usage():
    pytest.importorskip("anthropic")
    from aimu.aio.providers.anthropic import AsyncAnthropicClient
    from aimu.models import AnthropicModel

    client = object.__new__(AsyncAnthropicClient)
    client.model = AnthropicModel.CLAUDE_SONNET_4_6
    client.model_kwargs = None
    client._system_message = None
    client.default_generate_kwargs = {}
    client.messages = []
    client.tools = []
    client.last_thinking = ""
    client.last_usage = None
    client.concurrent_tool_calls = False
    client.cache_prompt = False

    events = [SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="hello"))]
    final = SimpleNamespace(usage=SimpleNamespace(input_tokens=12, output_tokens=4))

    class FakeAsyncStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def __aiter__(self):
            for event in events:
                yield event

        async def get_final_message(self):
            return final

    client._client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: FakeAsyncStream()))

    chunks = [c async for c in client._generate_streamed("hi", {})]
    assert "".join(c.content for c in chunks) == "hello"
    assert client.last_usage == {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
