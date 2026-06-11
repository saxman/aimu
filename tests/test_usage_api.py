"""Mock-only unit tests for token-usage surfacing (``client.last_usage``).

Covers the provider-agnostic normalizers in ``aimu.models._internal.usage`` and the
non-streaming capture / streaming-reset behaviour on a real ``OpenAICompatClient``
with a stubbed SDK (no network).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aimu.models._internal.usage import usage_from_anthropic, usage_from_ollama, usage_from_openai


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def test_usage_from_openai_normalizes_fields():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    assert usage_from_openai(response) == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_usage_from_anthropic_normalizes_fields():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=7))
    assert usage_from_anthropic(response) == {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}


def test_usage_from_ollama_reads_attribute_or_mapping():
    attr_response = SimpleNamespace(prompt_eval_count=2, eval_count=4)
    assert usage_from_ollama(attr_response) == {"input_tokens": 2, "output_tokens": 4, "total_tokens": 6}
    dict_response = {"prompt_eval_count": 8, "eval_count": 1}
    assert usage_from_ollama(dict_response) == {"input_tokens": 8, "output_tokens": 1, "total_tokens": 9}


def test_usage_none_when_provider_omits_it():
    assert usage_from_openai(SimpleNamespace(usage=None)) is None
    assert usage_from_anthropic(SimpleNamespace()) is None
    assert usage_from_ollama(SimpleNamespace()) is None


def test_usage_partial_counts_default_missing_side_to_zero():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=None))
    assert usage_from_openai(response) == {"input_tokens": 12, "output_tokens": 0, "total_tokens": 12}


# ---------------------------------------------------------------------------
# last_usage lifecycle on the base client
# ---------------------------------------------------------------------------


def test_reset_clears_last_usage():
    from helpers import MockModelClient

    client = MockModelClient(["hi"])
    client.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    client.reset()
    assert client.last_usage is None


# ---------------------------------------------------------------------------
# Provider capture (OpenAI-compat with a stubbed SDK — no network)
# ---------------------------------------------------------------------------


def _make_openai_compat_client():
    from aimu.models import HAS_OPENAI_COMPAT

    if not HAS_OPENAI_COMPAT:
        pytest.skip("openai not installed")
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    client = LMStudioOpenAIClient(LMStudioOpenAIModel.MISTRAL_7B)
    # Replace the real SDK object with a fully-fake namespace. Touching the real client's
    # lazy `.chat`/`.embeddings` attrs imports `openai.resources.*`, which sibling mock-test
    # modules stub in sys.modules — so we never touch the real SDK here.
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    return client


def test_openai_compat_generate_captures_usage(monkeypatch):
    client = _make_openai_compat_client()

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8),
    )
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: fake_response)

    out = client.generate("hi")
    assert out == "hello"
    assert client.last_usage == {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}


def test_openai_compat_chat_captures_usage(monkeypatch):
    client = _make_openai_compat_client()

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: fake_response)

    client.chat("hi")
    assert client.last_usage == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}


def test_streaming_resets_last_usage(monkeypatch):
    client = _make_openai_compat_client()
    client.last_usage = {"input_tokens": 99, "output_tokens": 99, "total_tokens": 198}

    def fake_stream(**_):
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", tool_calls=None))])

    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: fake_stream())

    list(client.generate("hi", stream=True))
    assert client.last_usage is None
