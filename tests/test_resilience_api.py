"""Mock-only tests for P0-C: timeout + max_retries passthrough to provider SDK clients.

Each test monkeypatches the underlying SDK client constructor (anthropic.Anthropic,
openai.OpenAI, ollama.Client) to capture the kwargs AIMU forwards, so no live backend or
API key is needed. Mirrors the monkeypatch style in test_structured_api.py.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Anthropic (native SDK timeout/max_retries)
# ---------------------------------------------------------------------------


def test_anthropic_forwards_timeout_and_max_retries(monkeypatch):
    pytest.importorskip("anthropic")
    from aimu.models import AnthropicModel
    from aimu.models.providers import anthropic as anthropic_mod
    from aimu.models.providers.anthropic import AnthropicClient

    captured = {}
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", lambda **kw: captured.update(kw) or object())

    AnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6, timeout=30, max_retries=5)
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 5


def test_anthropic_omits_unset_resilience_kwargs(monkeypatch):
    pytest.importorskip("anthropic")
    from aimu.models import AnthropicModel
    from aimu.models.providers import anthropic as anthropic_mod
    from aimu.models.providers.anthropic import AnthropicClient

    captured = {}
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", lambda **kw: captured.update(kw) or object())

    AnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)
    assert "timeout" not in captured
    assert "max_retries" not in captured


# ---------------------------------------------------------------------------
# OpenAI-compat base + local-server subclass inheritance path
# ---------------------------------------------------------------------------


def test_openai_compat_forwards_timeout_and_max_retries(monkeypatch):
    pytest.importorskip("openai")
    from aimu.models.providers import openai_compat as compat_mod
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "OpenAI", lambda **kw: captured.update(kw) or object())

    # Subclass that forwards **kwargs to the base; exercises the inheritance path.
    LMStudioOpenAIClient(next(iter(LMStudioOpenAIModel)), timeout=12.5, max_retries=4)
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 4


def test_openai_compat_omits_unset_resilience_kwargs(monkeypatch):
    pytest.importorskip("openai")
    from aimu.models.providers import openai_compat as compat_mod
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "OpenAI", lambda **kw: captured.update(kw) or object())

    LMStudioOpenAIClient(next(iter(LMStudioOpenAIModel)))
    assert "timeout" not in captured
    assert "max_retries" not in captured


def test_openai_client_forwards_resilience_kwargs(monkeypatch):
    pytest.importorskip("openai")
    from aimu.models.providers import openai_compat as compat_mod
    from aimu.models.providers.openai.text import OpenAIClient, OpenAIModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "OpenAI", lambda **kw: captured.update(kw) or object())

    OpenAIClient(next(iter(OpenAIModel)), timeout=20, max_retries=2)
    assert captured["timeout"] == 20
    assert captured["max_retries"] == 2


def test_gemini_client_forwards_resilience_kwargs(monkeypatch):
    pytest.importorskip("openai")
    from aimu.models.providers import openai_compat as compat_mod
    from aimu.models.providers.gemini.text import GeminiClient, GeminiModel

    captured = {}
    monkeypatch.setattr(compat_mod.openai, "OpenAI", lambda **kw: captured.update(kw) or object())

    GeminiClient(next(iter(GeminiModel)), timeout=20, max_retries=2)
    assert captured["timeout"] == 20
    assert captured["max_retries"] == 2


# ---------------------------------------------------------------------------
# Ollama (timeout via httpx; max_retries unsupported)
# ---------------------------------------------------------------------------


def test_ollama_forwards_timeout(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.models.providers import ollama as ollama_mod
    from aimu.models.providers.ollama import OllamaClient, OllamaModel

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)

        def pull(self, *a, **k):
            return None

    monkeypatch.setattr(ollama_mod.ollama, "Client", FakeClient)

    OllamaClient(OllamaModel.LLAMA_3_2_3B, timeout=15)
    assert captured["timeout"] == 15


def test_ollama_rejects_max_retries(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.models.providers import ollama as ollama_mod
    from aimu.models.providers.ollama import OllamaClient, OllamaModel

    class FakeClient:
        def __init__(self, **kw):
            pass

        def pull(self, *a, **k):
            return None

    monkeypatch.setattr(ollama_mod.ollama, "Client", FakeClient)

    with pytest.raises(ValueError, match="max_retries"):
        OllamaClient(OllamaModel.LLAMA_3_2_3B, max_retries=3)


# ---------------------------------------------------------------------------
# In-process providers: timeout/max_retries not applicable -> TypeError at binding
# ---------------------------------------------------------------------------


def test_huggingface_rejects_resilience_kwargs():
    pytest.importorskip("transformers")
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    # Unexpected-keyword TypeError happens at call binding, before any weight load.
    with pytest.raises(TypeError):
        HuggingFaceClient(next(iter(HuggingFaceModel)), timeout=30)


def test_llamacpp_rejects_resilience_kwargs():
    pytest.importorskip("llama_cpp")
    from aimu.models.providers.llamacpp import LlamaCppClient, LlamaCppModel

    with pytest.raises(TypeError):
        LlamaCppClient(next(iter(LlamaCppModel)), model_path="/nonexistent.gguf", timeout=30)
