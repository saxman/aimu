"""Mock-only async tests for P0-C: timeout + max_retries passthrough to async SDK clients.

Async provider clients build their SDK client synchronously in __init__, so these tests
are plain functions that monkeypatch the SDK constructor and assert forwarded kwargs.
Mirrors tests/test_resilience_api.py on the aimu.aio surface.
"""

from __future__ import annotations

import pytest


def test_async_anthropic_forwards_resilience_kwargs(monkeypatch):
    pytest.importorskip("anthropic")
    from aimu.aio.providers import anthropic as anthropic_mod
    from aimu.aio.providers.anthropic import AsyncAnthropicClient
    from aimu.models import AnthropicModel

    captured = {}
    monkeypatch.setattr(anthropic_mod.anthropic, "AsyncAnthropic", lambda **kw: captured.update(kw) or object())

    AsyncAnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6, timeout=30, max_retries=5)
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 5


def test_async_anthropic_omits_unset_resilience_kwargs(monkeypatch):
    pytest.importorskip("anthropic")
    from aimu.aio.providers import anthropic as anthropic_mod
    from aimu.aio.providers.anthropic import AsyncAnthropicClient
    from aimu.models import AnthropicModel

    captured = {}
    monkeypatch.setattr(anthropic_mod.anthropic, "AsyncAnthropic", lambda **kw: captured.update(kw) or object())

    AsyncAnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)
    assert "timeout" not in captured
    assert "max_retries" not in captured


def test_async_openai_compat_subclass_forwards_resilience_kwargs(monkeypatch):
    pytest.importorskip("openai")
    from aimu.aio.providers import openai_compat as compat_mod
    from aimu.aio.providers.openai_compat import AsyncLMStudioOpenAIClient, LMStudioOpenAIModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())

    AsyncLMStudioOpenAIClient(next(iter(LMStudioOpenAIModel)), timeout=12.5, max_retries=4)
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 4


def test_async_openai_client_forwards_resilience_kwargs(monkeypatch):
    pytest.importorskip("openai")
    from aimu.aio.providers import openai_compat as compat_mod
    from aimu.aio.providers.openai.text import AsyncOpenAIClient
    from aimu.models.providers.openai.text import OpenAIModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())

    AsyncOpenAIClient(next(iter(OpenAIModel)), timeout=20, max_retries=2)
    assert captured["timeout"] == 20
    assert captured["max_retries"] == 2


def test_async_gemini_client_forwards_resilience_kwargs(monkeypatch):
    pytest.importorskip("openai")
    from aimu.aio.providers import openai_compat as compat_mod
    from aimu.aio.providers.gemini.text import AsyncGeminiClient
    from aimu.models.providers.gemini.text import GeminiModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "AsyncOpenAI", lambda **kw: captured.update(kw) or object())

    AsyncGeminiClient(next(iter(GeminiModel)), timeout=20, max_retries=2)
    assert captured["timeout"] == 20
    assert captured["max_retries"] == 2


def test_async_ollama_forwards_timeout(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.aio.providers import ollama as ollama_mod
    from aimu.aio.providers.ollama import AsyncOllamaClient
    from aimu.models import OllamaModel

    captured = {}
    monkeypatch.setattr(ollama_mod.ollama, "AsyncClient", lambda **kw: captured.update(kw) or object())

    AsyncOllamaClient(OllamaModel.LLAMA_3_2_3B, timeout=15)
    assert captured["timeout"] == 15


def test_async_ollama_rejects_max_retries(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.aio.providers import ollama as ollama_mod
    from aimu.aio.providers.ollama import AsyncOllamaClient
    from aimu.models import OllamaModel

    monkeypatch.setattr(ollama_mod.ollama, "AsyncClient", lambda **kw: object())

    with pytest.raises(ValueError, match="max_retries"):
        AsyncOllamaClient(OllamaModel.LLAMA_3_2_3B, max_retries=3)
