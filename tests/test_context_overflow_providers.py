"""A request that no longer fits the model's context window surfaces as the one portable
``ContextOverflowError`` on every provider, not each backend's own SDK exception.

Ollama already had this mapping (see ``tests/test_ollama_context_overflow.py``); this file
covers the rest: OpenAI-compatible (OpenAI/Gemini/local servers share one module),
Anthropic, and the two in-process backends (HuggingFace, llama.cpp), which have no server
error to translate and instead run a pre-flight token count against the model's own known
window.

No network I/O and no real model weights: each provider's SDK call (or, for the in-process
backends, the tokenizer/model load) is monkeypatched to produce the same failure shape the
real backend produces, so the mapping is exercised without a live server or a GPU.

Each section also pins the false-positive direction: an unrelated 400 (or an in-bounds
prompt) must propagate/pass through unchanged, not be swallowed into a misleading
``ContextOverflowError`` a catch-compact-retry loop could never fix.
"""

from __future__ import annotations

import httpx
import openai
import pytest
import torch

from aimu import aio
from aimu.models import HAS_ANTHROPIC, HAS_HF, HAS_LLAMACPP, HAS_OPENAI_COMPAT, ContextOverflowError, ModelClient

pytestmark_openai_compat = pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")
pytestmark_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
pytestmark_hf = pytest.mark.skipif(not HAS_HF, reason="transformers not installed")
pytestmark_llamacpp = pytest.mark.skipif(not HAS_LLAMACPP, reason="llama-cpp-python not installed")


# --------------------------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Gemini, and every local server share this one module)          #
# --------------------------------------------------------------------------------------- #

_OPENAI_COMPAT_URL = "http://gpu-box:8080/v1"


def _openai_bad_request(*, code, message="boom") -> openai.BadRequestError:
    request = httpx.Request("POST", _OPENAI_COMPAT_URL + "/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={"error": {"message": message, "type": "invalid_request_error", "param": "messages", "code": code}},
    )
    body = response.json()["error"]
    return openai.BadRequestError(message, response=response, body=body)


def _raise_context_length_exceeded(*args, **kwargs):
    raise _openai_bad_request(
        code="context_length_exceeded", message="This model's maximum context length is 4097 tokens."
    )


def _raise_unrelated_bad_request(*args, **kwargs):
    raise _openai_bad_request(code="invalid_value", message="'foo' is not a valid value for 'role'")


async def _araise_context_length_exceeded(*args, **kwargs):
    _raise_context_length_exceeded()


async def _araise_unrelated_bad_request(*args, **kwargs):
    _raise_unrelated_bad_request()


def _openai_compat_client(create_fn) -> ModelClient:
    c = ModelClient(f"llamaserver:custom.gguf@{_OPENAI_COMPAT_URL};tools")
    c._client._client.chat.completions.create = create_fn
    return c


def _async_openai_compat_client(create_fn) -> aio.AsyncModelClient:
    c = aio.AsyncModelClient(f"llamaserver:custom.gguf@{_OPENAI_COMPAT_URL};tools")
    c._client._client.chat.completions.create = create_fn
    return c


@pytestmark_openai_compat
def test_openai_compat_chat_translates_context_length_exceeded():
    client = _openai_compat_client(_raise_context_length_exceeded)
    with pytest.raises(ContextOverflowError) as info:
        client._client._chat("hi")
    assert isinstance(info.value.__cause__, openai.BadRequestError)
    assert info.value.__cause__.code == "context_length_exceeded"
    assert "context window" in str(info.value)


@pytestmark_openai_compat
def test_openai_compat_generate_translates_context_length_exceeded():
    client = _openai_compat_client(_raise_context_length_exceeded)
    with pytest.raises(ContextOverflowError):
        client._client._generate("hi")


@pytestmark_openai_compat
def test_openai_compat_streamed_chat_translates_context_length_exceeded():
    client = _openai_compat_client(_raise_context_length_exceeded)
    with pytest.raises(ContextOverflowError):
        list(client._client._chat("hi", stream=True))


@pytestmark_openai_compat
def test_openai_compat_streamed_generate_translates_context_length_exceeded():
    client = _openai_compat_client(_raise_context_length_exceeded)
    with pytest.raises(ContextOverflowError):
        list(client._client._generate("hi", stream=True))


@pytestmark_openai_compat
def test_openai_compat_unrelated_bad_request_is_not_translated():
    """The false-positive guard: a 400 that isn't context_length_exceeded stays itself."""
    client = _openai_compat_client(_raise_unrelated_bad_request)
    with pytest.raises(openai.BadRequestError) as info:
        client._client._chat("hi")
    assert info.value.code == "invalid_value"


@pytestmark_openai_compat
async def test_async_openai_compat_chat_translates_context_length_exceeded():
    client = _async_openai_compat_client(_araise_context_length_exceeded)
    with pytest.raises(ContextOverflowError) as info:
        await client._client._chat("hi")
    assert isinstance(info.value.__cause__, openai.BadRequestError)


@pytestmark_openai_compat
async def test_async_openai_compat_streamed_chat_translates_context_length_exceeded():
    client = _async_openai_compat_client(_araise_context_length_exceeded)
    with pytest.raises(ContextOverflowError):
        stream = await client._client._chat("hi", stream=True)
        async for _ in stream:
            pass


@pytestmark_openai_compat
async def test_async_openai_compat_unrelated_bad_request_is_not_translated():
    client = _async_openai_compat_client(_araise_unrelated_bad_request)
    with pytest.raises(openai.BadRequestError) as info:
        await client._client._chat("hi")
    assert info.value.code == "invalid_value"


# --------------------------------------------------------------------------------------- #
# Anthropic                                                                                  #
# --------------------------------------------------------------------------------------- #

_ANTHROPIC_URL = "http://x/v1/messages"


def _anthropic_bad_request(*, message: str):
    import anthropic

    request = httpx.Request("POST", _ANTHROPIC_URL)
    response = httpx.Response(
        400, request=request, json={"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    )
    return anthropic.BadRequestError(message, response=response, body=response.json())


def _anthropic_request_too_large(*, message: str):
    import anthropic

    request = httpx.Request("POST", _ANTHROPIC_URL)
    response = httpx.Response(
        413, request=request, json={"type": "error", "error": {"type": "request_too_large", "message": message}}
    )
    return anthropic.RequestTooLargeError(message, response=response, body=response.json())


def _raise_prompt_too_long(*args, **kwargs):
    raise _anthropic_bad_request(message="prompt is too long: 220000 tokens > 200000 maximum")


def _raise_unrelated_anthropic_bad_request(*args, **kwargs):
    raise _anthropic_bad_request(message='messages: roles must alternate between "user" and "assistant"')


def _raise_request_too_large(*args, **kwargs):
    raise _anthropic_request_too_large(message="request too large, please try again with fewer tokens")


async def _araise_prompt_too_long(*args, **kwargs):
    _raise_prompt_too_long()


async def _araise_unrelated_anthropic_bad_request(*args, **kwargs):
    _raise_unrelated_anthropic_bad_request()


async def _araise_request_too_large(*args, **kwargs):
    _raise_request_too_large()


def _anthropic_client(create_fn, stream_fn=None):
    from aimu.models.providers.anthropic import AnthropicClient, AnthropicModel

    client = AnthropicClient(AnthropicModel.CLAUDE_HAIKU_4_5)
    client._client.messages.create = create_fn
    if stream_fn is not None:
        client._client.messages.stream = stream_fn
    return client


def _async_anthropic_client(create_fn, stream_fn=None):
    from aimu.aio.providers.anthropic import AsyncAnthropicClient
    from aimu.models.providers.anthropic import AnthropicModel

    client = AsyncAnthropicClient(AnthropicModel.CLAUDE_HAIKU_4_5)
    client._client.messages.create = create_fn
    if stream_fn is not None:
        client._client.messages.stream = stream_fn
    return client


@pytestmark_anthropic
def test_anthropic_chat_translates_prompt_too_long():
    client = _anthropic_client(_raise_prompt_too_long)
    with pytest.raises(ContextOverflowError) as info:
        client._chat("hi")
    import anthropic

    assert isinstance(info.value.__cause__, anthropic.BadRequestError)
    assert "prompt is too long" in str(info.value.__cause__).lower()
    assert "context window" in str(info.value)


@pytestmark_anthropic
def test_anthropic_generate_translates_prompt_too_long():
    client = _anthropic_client(_raise_prompt_too_long)
    with pytest.raises(ContextOverflowError):
        client._generate("hi")


@pytestmark_anthropic
def test_anthropic_streamed_chat_translates_prompt_too_long():
    client = _anthropic_client(_raise_prompt_too_long, stream_fn=_raise_prompt_too_long)
    with pytest.raises(ContextOverflowError):
        list(client._chat("hi", stream=True))


@pytestmark_anthropic
def test_anthropic_streamed_generate_translates_prompt_too_long():
    client = _anthropic_client(_raise_prompt_too_long, stream_fn=_raise_prompt_too_long)
    with pytest.raises(ContextOverflowError):
        list(client._generate("hi", stream=True))


@pytestmark_anthropic
def test_anthropic_unrelated_bad_request_is_not_translated():
    """The false-positive guard: a 400 that isn't the prompt-too-long wording stays itself."""
    import anthropic

    client = _anthropic_client(_raise_unrelated_anthropic_bad_request)
    with pytest.raises(anthropic.BadRequestError) as info:
        client._chat("hi")
    assert "roles must alternate" in str(info.value)


@pytestmark_anthropic
def test_anthropic_chat_translates_request_too_large():
    """A 413 (RequestTooLargeError) is a distinct exception class from BadRequestError -- a bare
    ``except anthropic.BadRequestError`` cannot see it, so this needs (and has) its own clause."""
    import anthropic

    client = _anthropic_client(_raise_request_too_large)
    with pytest.raises(ContextOverflowError) as info:
        client._chat("hi")
    assert isinstance(info.value.__cause__, anthropic.RequestTooLargeError)
    assert "context window" in str(info.value)


@pytestmark_anthropic
def test_anthropic_streamed_chat_translates_request_too_large():
    client = _anthropic_client(_raise_request_too_large, stream_fn=_raise_request_too_large)
    with pytest.raises(ContextOverflowError):
        list(client._chat("hi", stream=True))


@pytestmark_anthropic
async def test_async_anthropic_chat_translates_prompt_too_long():
    client = _async_anthropic_client(_araise_prompt_too_long)
    with pytest.raises(ContextOverflowError) as info:
        await client._chat("hi")
    import anthropic

    assert isinstance(info.value.__cause__, anthropic.BadRequestError)


@pytestmark_anthropic
async def test_async_anthropic_streamed_chat_translates_prompt_too_long():
    client = _async_anthropic_client(_raise_prompt_too_long, stream_fn=_raise_prompt_too_long)
    with pytest.raises(ContextOverflowError):
        stream = await client._chat("hi", stream=True)
        async for _ in stream:
            pass


@pytestmark_anthropic
async def test_async_anthropic_unrelated_bad_request_is_not_translated():
    import anthropic

    client = _async_anthropic_client(_araise_unrelated_anthropic_bad_request)
    with pytest.raises(anthropic.BadRequestError):
        await client._chat("hi")


@pytestmark_anthropic
async def test_async_anthropic_chat_translates_request_too_large():
    import anthropic

    client = _async_anthropic_client(_araise_request_too_large)
    with pytest.raises(ContextOverflowError) as info:
        await client._chat("hi")
    assert isinstance(info.value.__cause__, anthropic.RequestTooLargeError)


@pytestmark_anthropic
async def test_async_anthropic_streamed_chat_translates_request_too_large():
    client = _async_anthropic_client(_raise_request_too_large, stream_fn=_raise_request_too_large)
    with pytest.raises(ContextOverflowError):
        stream = await client._chat("hi", stream=True)
        async for _ in stream:
            pass


# --------------------------------------------------------------------------------------- #
# HuggingFace (in-process; pre-flight token count against config.max_position_embeddings)   #
# --------------------------------------------------------------------------------------- #


class _FakeHFModelInputs:
    """Stands in for the tokenizer/processor output: only ``.input_ids`` and ``.to()`` are read."""

    def __init__(self, n_tokens: int):
        self.input_ids = torch.zeros((1, n_tokens), dtype=torch.long)

    def to(self, device):
        return self


class _FakeHFTokenizer:
    """A tokenizer stub whose rendered prompt is always exactly ``n_tokens`` tokens long,
    regardless of the real message content -- the point is to control the token count
    precisely, not to render a realistic prompt."""

    def __init__(self, n_tokens: int):
        self._n_tokens = n_tokens

    def apply_chat_template(self, messages, **kwargs):
        return "RENDERED PROMPT"

    def __call__(self, texts, return_tensors="pt"):
        return _FakeHFModelInputs(self._n_tokens)


class _FakeHFModel:
    def __init__(self, max_position_embeddings):
        import types as _types

        self.config = _types.SimpleNamespace(max_position_embeddings=max_position_embeddings)
        self.device = "cpu"

    def generate(self, **kwargs):
        raise AssertionError("generate() must not run once the pre-flight guard has already raised")


def _hf_client(monkeypatch, *, n_tokens: int, max_position_embeddings, cache_marker: str):
    import aimu.models.providers.hf.text as hf_text_mod

    fake_tokenizer = _FakeHFTokenizer(n_tokens)
    fake_model = _FakeHFModel(max_position_embeddings)
    monkeypatch.setattr(hf_text_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tokenizer)
    monkeypatch.setattr(hf_text_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: fake_model)
    return hf_text_mod.HuggingFaceClient(
        hf_text_mod.HuggingFaceModel.LLAMA_3_2_3B, model_kwargs={"_test_marker": cache_marker}
    )


@pytestmark_hf
def test_hf_generate_raises_when_prompt_overflows(monkeypatch):
    client = _hf_client(monkeypatch, n_tokens=200, max_position_embeddings=100, cache_marker="hf-overflow-generate")
    with pytest.raises(ContextOverflowError) as info:
        client.generate("hi")
    assert info.value.__cause__ is None  # pre-flight guard, not a caught SDK error
    assert "context window" in str(info.value)
    assert "200" in str(info.value) and "100" in str(info.value)


@pytestmark_hf
def test_hf_generate_overflow_still_records_last_request(monkeypatch):
    """Regression: the pre-flight raise must not leave last_request stale (or None). The moment a
    caller most wants to see the request AIMU built is the moment it was too big to send."""
    client = _hf_client(monkeypatch, n_tokens=200, max_position_embeddings=100, cache_marker="hf-overflow-last-request")
    assert client.last_request is None  # nothing sent yet
    with pytest.raises(ContextOverflowError):
        client.generate("hi")
    assert client.last_request is not None
    assert client.last_request["prompt"] == "RENDERED PROMPT"


@pytestmark_hf
def test_hf_generate_streamed_raises_when_prompt_overflows(monkeypatch):
    client = _hf_client(
        monkeypatch, n_tokens=200, max_position_embeddings=100, cache_marker="hf-overflow-generate-stream"
    )
    with pytest.raises(ContextOverflowError):
        list(client.generate("hi", stream=True))


@pytestmark_hf
def test_hf_chat_raises_when_prompt_overflows(monkeypatch):
    client = _hf_client(monkeypatch, n_tokens=200, max_position_embeddings=100, cache_marker="hf-overflow-chat")
    with pytest.raises(ContextOverflowError):
        client.chat("hi")


@pytestmark_hf
def test_hf_within_window_does_not_raise(monkeypatch):
    """The false-positive guard: a prompt within the declared window is not flagged."""
    client = _hf_client(monkeypatch, n_tokens=50, max_position_embeddings=100, cache_marker="hf-within-window")
    model_inputs = client._apply_chat_template([{"role": "user", "content": "hi"}])
    assert model_inputs.input_ids.shape[-1] == 50


@pytestmark_hf
def test_hf_unknown_window_is_not_guessed(monkeypatch):
    """If config.max_position_embeddings isn't declared, the check is skipped rather than
    inventing a number -- even an absurdly long prompt is not flagged."""
    import types as _types

    import aimu.models.providers.hf.text as hf_text_mod

    fake_tokenizer = _FakeHFTokenizer(999_999)
    fake_model = _types.SimpleNamespace(config=_types.SimpleNamespace(), device="cpu")
    monkeypatch.setattr(hf_text_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tokenizer)
    monkeypatch.setattr(hf_text_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: fake_model)
    client = hf_text_mod.HuggingFaceClient(
        hf_text_mod.HuggingFaceModel.LLAMA_3_2_3B, model_kwargs={"_test_marker": "hf-unknown-window"}
    )
    model_inputs = client._apply_chat_template([{"role": "user", "content": "hi"}])
    assert model_inputs.input_ids.shape[-1] == 999_999


# --------------------------------------------------------------------------------------- #
# llama.cpp (in-process; pre-flight token count against the constructed n_ctx)              #
# --------------------------------------------------------------------------------------- #


class _FakeLlama:
    def __init__(self, n_ctx: int, tokenize_count: int):
        self._n_ctx = n_ctx
        self._tokenize_count = tokenize_count

    def n_ctx(self):
        return self._n_ctx

    def tokenize(self, data: bytes, add_bos=True, special=True):
        return list(range(self._tokenize_count))

    def create_chat_completion(self, **kwargs):
        raise AssertionError("create_chat_completion() must not run once the pre-flight guard has already raised")


def _llamacpp_client(monkeypatch, *, n_ctx: int, tokenize_count: int, model_path: str):
    import aimu.models.providers.llamacpp as llamacpp_mod

    fake_llm = _FakeLlama(n_ctx=n_ctx, tokenize_count=tokenize_count)
    monkeypatch.setattr(llamacpp_mod.llama_cpp, "Llama", lambda **kwargs: fake_llm)
    return llamacpp_mod.LlamaCppClient(llamacpp_mod.LlamaCppModel.LLAMA_3_2_3B, model_path=model_path, n_ctx=n_ctx)


@pytestmark_llamacpp
def test_llamacpp_generate_raises_when_prompt_overflows(monkeypatch):
    client = _llamacpp_client(monkeypatch, n_ctx=50, tokenize_count=200, model_path="overflow-generate.gguf")
    with pytest.raises(ContextOverflowError) as info:
        client.generate("hi")
    assert info.value.__cause__ is None
    assert "context window" in str(info.value)
    assert "50" in str(info.value)


@pytestmark_llamacpp
def test_llamacpp_generate_overflow_still_records_last_request(monkeypatch):
    """Regression: the pre-flight raise must not leave last_request stale (or None) -- same
    defect class as the HuggingFace one above, on the other in-process provider."""
    client = _llamacpp_client(monkeypatch, n_ctx=50, tokenize_count=200, model_path="overflow-last-request.gguf")
    assert client.last_request is None  # nothing sent yet
    with pytest.raises(ContextOverflowError):
        client.generate("hi")
    assert client.last_request is not None
    assert client.last_request["messages"] == [{"role": "user", "content": "hi"}]


@pytestmark_llamacpp
def test_llamacpp_generate_streamed_raises_when_prompt_overflows(monkeypatch):
    client = _llamacpp_client(monkeypatch, n_ctx=50, tokenize_count=200, model_path="overflow-generate-stream.gguf")
    with pytest.raises(ContextOverflowError):
        list(client.generate("hi", stream=True))


@pytestmark_llamacpp
def test_llamacpp_chat_raises_when_prompt_overflows(monkeypatch):
    client = _llamacpp_client(monkeypatch, n_ctx=50, tokenize_count=200, model_path="overflow-chat.gguf")
    with pytest.raises(ContextOverflowError):
        client.chat("hi")


@pytestmark_llamacpp
def test_llamacpp_chat_streamed_raises_when_prompt_overflows(monkeypatch):
    client = _llamacpp_client(monkeypatch, n_ctx=50, tokenize_count=200, model_path="overflow-chat-stream.gguf")
    with pytest.raises(ContextOverflowError):
        list(client.chat("hi", stream=True))


@pytestmark_llamacpp
def test_llamacpp_within_window_does_not_raise(monkeypatch):
    """The false-positive guard: a conversation within n_ctx is not flagged."""
    import aimu.models.providers.llamacpp as llamacpp_mod

    client = _llamacpp_client(monkeypatch, n_ctx=50, tokenize_count=10, model_path="within-window.gguf")
    llamacpp_mod._raise_if_prompt_overflows(client._llm, [{"role": "user", "content": "hi"}])  # must not raise
