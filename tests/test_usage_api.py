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
# Provider capture (OpenAI-compat with a stubbed SDK; no network)
# ---------------------------------------------------------------------------


def _make_openai_compat_client():
    from aimu.models import HAS_OPENAI_COMPAT

    if not HAS_OPENAI_COMPAT:
        pytest.skip("openai not installed")
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    client = LMStudioOpenAIClient(LMStudioOpenAIModel.MISTRAL_7B)
    # Replace the real SDK object with a fully-fake namespace. Touching the real client's
    # lazy `.chat`/`.embeddings` attrs imports `openai.resources.*`, which sibling mock-test
    # modules stub in sys.modules, so we never touch the real SDK here.
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


# ---------------------------------------------------------------------------
# Streaming usage capture (P1-B)
# ---------------------------------------------------------------------------


def _content_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])


def _usage_chunk(prompt_tokens, completion_tokens):
    # OpenAI's terminal stream_options usage chunk: empty choices + populated usage.
    return SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )


def test_openai_compat_streaming_generate_captures_usage(monkeypatch):
    client = _make_openai_compat_client()

    def fake_stream(**_):
        yield _content_chunk("he")
        yield _content_chunk("llo")
        yield _usage_chunk(10, 5)

    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: fake_stream())

    chunks = list(client.generate("hi", stream=True))
    assert "".join(c.content for c in chunks) == "hello"
    assert client.last_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_openai_compat_streaming_chat_captures_usage(monkeypatch):
    client = _make_openai_compat_client()

    def fake_stream(**_):
        yield _content_chunk("hi")
        yield _usage_chunk(7, 3)

    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: fake_stream())

    list(client.chat("hello", stream=True))
    assert client.last_usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


def test_openai_compat_streaming_requests_usage_option(monkeypatch):
    client = _make_openai_compat_client()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)

        def gen():
            yield _content_chunk("x")

        return gen()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    list(client.generate("hi", stream=True))
    assert captured.get("stream_options") == {"include_usage": True}


def test_ollama_streaming_captures_usage(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.models.providers import ollama as ollama_mod
    from aimu.models.providers.ollama import OllamaClient, OllamaModel

    class _Part(dict):
        def __getattr__(self, key):
            return self.get(key)

    parts = [
        _Part(response="he"),
        _Part(response="llo", prompt_eval_count=10, eval_count=5, done=True),
    ]

    class FakeClient:
        def __init__(self, **kw):
            pass

        def pull(self, *a, **k):
            return None

        def generate(self, **kw):
            return iter(parts)

    monkeypatch.setattr(ollama_mod.ollama, "Client", FakeClient)
    client = OllamaClient(OllamaModel.LLAMA_3_2_3B)  # non-thinking

    chunks = list(client.generate("hi", stream=True))
    assert "".join(c.content for c in chunks) == "hello"
    assert client.last_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_anthropic_streaming_captures_usage(monkeypatch):
    pytest.importorskip("anthropic")
    from aimu.models import AnthropicModel
    from aimu.models.providers.anthropic import AnthropicClient

    client = object.__new__(AnthropicClient)
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

    events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="hello")),
    ]
    final = SimpleNamespace(usage=SimpleNamespace(input_tokens=12, output_tokens=4))

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter(events)

        def get_final_message(self):
            return final

    client._client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: FakeStream()))

    chunks = list(client._generate_streamed("hi", {}))
    assert "".join(c.content for c in chunks) == "hello"
    assert client.last_usage == {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
