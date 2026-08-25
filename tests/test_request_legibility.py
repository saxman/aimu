"""What AIMU actually sent.

Between a caller's chat() and the wire sit the four-tier generate_kwargs merge, the
GENERATE_KWARG_SUPPORT renames and drops, thinking resolution, strip_inert_keys, and
provider format adaptation. Every step is principled and documented, and none of it was
visible at runtime before last_request. A last_request that showed a *pre*-adaptation
payload would be worse than none, so each transformation gets its own assertion.
"""

from __future__ import annotations

import json
import types

import pytest


def _fake_openai_compat_client(monkeypatch):
    """Build a real OpenAICompatClient (via LMStudioOpenAIClient) against a stubbed SDK.

    Mirrors ``tests/test_generate_kwargs_merge.py``'s ``_build_openai_compat`` /
    ``test_client_defaults_reach_the_wire``: patch ``openai.OpenAI`` so the client's own
    ``chat.completions.create`` records exactly the kwargs AIMU built, rather than inventing
    a second stub shape.
    """
    import openai

    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    def _fake_create(**kwargs):
        message = types.SimpleNamespace(content="hi", tool_calls=None, reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    fake_sdk = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_fake_create)))
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: fake_sdk)
    return LMStudioOpenAIClient(LMStudioOpenAIModel.QWEN_3_8B)


def test_last_request_starts_none_and_is_cleared_by_reset():
    from helpers import MockModelClient

    client = MockModelClient(["hi"])
    assert client.last_request is None
    client.chat("q")
    client.reset()
    assert client.last_request is None


def test_last_request_reflects_the_generate_kwargs_merge(monkeypatch):
    """The recorded payload must show the merged result, not the caller's fragment."""
    client = _fake_openai_compat_client(monkeypatch)
    client.default_generate_kwargs = {"temperature": 0.3}
    client.chat("q", generate_kwargs={"max_tokens": 99})
    assert client.last_request["temperature"] == 0.3
    assert client.last_request["max_tokens"] == 99


def test_last_request_reflects_a_kwarg_rename(monkeypatch):
    """top_k has no top-level place in the OpenAI schema; it is routed to extra_body."""
    client = _fake_openai_compat_client(monkeypatch)
    client.chat("q", generate_kwargs={"top_k": 20})
    assert "top_k" not in client.last_request
    assert client.last_request["extra_body"]["top_k"] == 20


def test_last_request_reflects_strip_inert_keys(monkeypatch):
    """timestamp / thinking / provenance are stamped on self.messages and must never
    appear in the payload."""
    client = _fake_openai_compat_client(monkeypatch)
    client.chat("q")
    for message in client.last_request["messages"]:
        assert "timestamp" not in message
        assert "thinking" not in message
        assert "provenance" not in message


def test_request_prepared_event_carries_the_same_payload(monkeypatch):
    from aimu.events import RequestPrepared

    seen = []
    client = _fake_openai_compat_client(monkeypatch)
    client.events = seen.append
    client.chat("q")
    prepared = next(e for e in seen if isinstance(e, RequestPrepared))
    assert prepared.payload == client.last_request
    assert prepared.provider


# ---------------------------------------------------------------------------
# In-process providers (HuggingFace, llama.cpp): direct exercises of their
# _record_request call sites with the *real* BaseModelClient._record_request bound, not
# a no-op. The existing fakes in test_models_api.py / test_thinking_control.py stub
# _record_request as a no-op (`lambda *a, **k: None`) purely so those pre-existing tests
# of unrelated logic don't raise AttributeError now that every provider calls it; they
# assert nothing about last_request. These are the tests that actually do.
# ---------------------------------------------------------------------------


def test_hf_generate_sync_records_the_rendered_prompt_and_kwargs():
    """HuggingFaceClient has no wire payload; _record_request's equivalent is the resolved
    generate_kwargs plus the exact prompt string handed to the tokenizer/processor."""
    from aimu.models.base import BaseModelClient
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    class _Inputs(dict):
        input_ids = [[0, 1, 2]]

        def to(self, device):
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return "rendered prompt text"

        def __call__(self, *a, **kw):
            return _Inputs()

        def decode(self, *a, **kw):
            return "answer"

    client = types.SimpleNamespace(
        model=HuggingFaceModel.QWEN_3_8_27B,
        MODELS=HuggingFaceModel,
        _hf_processor=None,
        _hf_tokenizer=_Tokenizer(),
        _hf_model=types.SimpleNamespace(device="cpu", generate=lambda **kw: [[0, 1, 2, 3]]),
        _uses_processor_parse_response=False,
        last_thinking=None,
        _pending_thinking_tokens=[],
        _parsed_tool_calls=None,
        events=None,
    )
    client._apply_chat_template = HuggingFaceClient._apply_chat_template.__get__(client, type(client))
    client._record_request = BaseModelClient._record_request.__get__(client)

    HuggingFaceClient._generate_sync(client, [{"role": "user", "content": "hi"}], {"max_new_tokens": 8}, None)

    assert client.last_request == {
        "generate_kwargs": {"max_new_tokens": 8},
        "prompt": "rendered prompt text",
        "images": None,
        "audio": None,
    }


def test_hf_generate_streaming_records_the_rendered_prompt_and_kwargs():
    """The streaming twin of the test above: _generate_streaming is HuggingFaceClient's other
    (and only other) _record_request call site, and _chat_streamed funnels through it too, so
    covering it covers every HF streaming path as well."""
    from aimu.models.base import BaseModelClient
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    class _Inputs(dict):
        input_ids = [[0, 1, 2]]

        def to(self, device):
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return "rendered prompt text"

        def __call__(self, *a, **kw):
            return _Inputs()

    client = types.SimpleNamespace(
        model=HuggingFaceModel.QWEN_3_8_27B,
        MODELS=HuggingFaceModel,
        _hf_processor=None,
        _hf_tokenizer=_Tokenizer(),
        _hf_model=types.SimpleNamespace(device="cpu", generate=lambda **kw: None),
        last_thinking=None,
        _pending_thinking_tokens=[],
        events=None,
    )
    client._apply_chat_template = HuggingFaceClient._apply_chat_template.__get__(client, type(client))
    client._record_request = BaseModelClient._record_request.__get__(client)

    # Transformers' real TextIteratorStreamer yields an empty first part, then tokens; a plain
    # iterator with that shape is enough since generate() is stubbed to not populate it itself.
    streamer = iter(["", "answer"])
    HuggingFaceClient._generate_streaming(
        client, [{"role": "user", "content": "hi"}], {"max_new_tokens": 8}, None, streamer
    )

    assert client.last_request == {
        "generate_kwargs": {"max_new_tokens": 8},
        "prompt": "rendered prompt text",
        "images": None,
        "audio": None,
    }


def test_hf_apply_chat_template_records_an_image_count(monkeypatch):
    """The processor branch is the only one that can carry images; a caller comparing a local
    vision run against a cloud one needs to see that media was sent, so the count (not the
    pixels, already discarded by request time) reaches the recorded payload."""
    import aimu.models.providers.hf.text as hf_text

    monkeypatch.setattr(hf_text, "_extract_pil_images", lambda messages: ["img1", "img2"])

    class _Processor:
        def apply_chat_template(self, messages, **kw):
            return "rendered"

        def __call__(self, **kw):
            return types.SimpleNamespace(to=lambda device: {})

    client = types.SimpleNamespace(
        model=hf_text.HuggingFaceModel.GEMMA_4_E4B,  # vision+audio+tools capable, processor branch
        _hf_processor=_Processor(),
        _hf_model=types.SimpleNamespace(device="cpu"),
    )
    hf_text.HuggingFaceClient._apply_chat_template(
        client, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    )

    assert client._last_rendered_images == 2
    assert client._last_rendered_audio is None


def test_llamacpp_chat_records_a_copy_of_messages_not_a_live_alias():
    """Regression: last_request['messages'] must be a snapshot, not self.messages itself.

    _chat appends the assistant reply to self.messages later in the same call, so a live
    alias would make last_request show the model's own answer as part of the request that
    produced it -- authoritative-looking and wrong, compounding on every subsequent turn.
    """
    from aimu.models._internal.chat_state import _ChatStateMixin
    from aimu.models.base import BaseModelClient
    from aimu.models.providers.llamacpp import LlamaCppClient

    response = {"choices": [{"message": {"content": "the answer", "tool_calls": None}}]}
    fake = types.SimpleNamespace(
        _chat_setup=lambda *a, **k: ({}, []),
        _llm=types.SimpleNamespace(create_chat_completion=lambda **kw: response),
        is_thinking_model=False,
        messages=[{"role": "user", "content": "hi"}],
        model=types.SimpleNamespace(value="fake-model"),
        events=None,
    )
    fake._append_message = _ChatStateMixin._append_message.__get__(fake)
    fake._record_request = BaseModelClient._record_request.__get__(fake)

    LlamaCppClient._chat(fake, "hi")

    assert fake.last_request["messages"] is not fake.messages
    assert fake.last_request["messages"] == [{"role": "user", "content": "hi"}]
    # The live history grew (the assistant turn was appended); the recorded snapshot did not.
    assert len(fake.messages) == 2
    assert len(fake.last_request["messages"]) == 1


def test_llamacpp_generate_records_kwargs_and_messages():
    """llama.cpp's second of four _record_request call sites: _generate (stateless, single-turn)."""
    from aimu.models.base import BaseModelClient
    from aimu.models.providers.llamacpp import LlamaCppClient

    response = {"choices": [{"message": {"content": "the answer", "reasoning_content": None}}]}
    fake = types.SimpleNamespace(
        model=types.SimpleNamespace(value="fake-model"),
        _resolve_generate_kwargs=lambda gk: gk or {},
        _llm=types.SimpleNamespace(create_chat_completion=lambda **kw: response),
        is_thinking_model=False,
        events=None,
    )
    fake._record_request = BaseModelClient._record_request.__get__(fake)

    LlamaCppClient._generate(fake, "hi", {"max_tokens": 5})

    assert fake.last_request == {"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}


def test_llamacpp_generate_streamed_records_kwargs_and_messages():
    """llama.cpp's third of four _record_request call sites: _generate_streamed."""
    from aimu.models.base import BaseModelClient
    from aimu.models.providers.llamacpp import LlamaCppClient

    deltas = [{"choices": [{"delta": {"content": "ok"}}]}]
    fake = types.SimpleNamespace(
        model=types.SimpleNamespace(value="fake-model"),
        _llm=types.SimpleNamespace(create_chat_completion=lambda **kw: iter(deltas)),
        is_thinking_model=False,
        events=None,
    )
    fake._record_request = BaseModelClient._record_request.__get__(fake)
    fake._iter_stream = lambda stream: LlamaCppClient._iter_stream(fake, stream)

    list(LlamaCppClient._generate_streamed(fake, "hi", {"max_tokens": 5}))

    assert fake.last_request == {
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def test_llamacpp_chat_streamed_records_a_copy_of_messages_not_a_live_alias():
    """llama.cpp's fourth of four _record_request call sites: _chat_streamed. Also a streaming
    twin of the aliasing regression above -- _chat_streamed copies self.messages the same way."""
    from aimu.models._internal.chat_state import _ChatStateMixin
    from aimu.models.base import BaseModelClient
    from aimu.models.providers.llamacpp import LlamaCppClient

    deltas = [{"choices": [{"delta": {"content": "ok"}}]}]
    fake = types.SimpleNamespace(
        model=types.SimpleNamespace(value="fake-model"),
        _llm=types.SimpleNamespace(create_chat_completion=lambda **kw: iter(deltas)),
        is_thinking_model=False,
        messages=[{"role": "user", "content": "hi"}],
        events=None,
    )
    fake._append_message = _ChatStateMixin._append_message.__get__(fake)
    fake._record_request = BaseModelClient._record_request.__get__(fake)

    list(LlamaCppClient._chat_streamed(fake, {"max_tokens": 5}, []))

    assert fake.last_request["messages"] is not fake.messages
    assert fake.last_request == {
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "tools": None,
    }


# ---------------------------------------------------------------------------
# The guard: a shipped client whose request path records nothing.
#
# Parametrized over both concrete client classes (sync + async twin, so a newly added
# provider's async side is covered automatically too) and over request path (chat/generate
# x stream/non-stream). A provider that only wired one of the four paths' calls into
# _record_request -- e.g. recording in _chat but not _chat_streamed -- would otherwise pass
# a client-level-only guard silently; per-path is the granularity the seam actually operates
# at (openai_compat alone has 4 call sites, anthropic 6).
# ---------------------------------------------------------------------------

_PATHS = [("chat", False), ("chat", True), ("generate", False), ("generate", True)]


class _AsyncIter:
    """Minimal async iterable/iterator over a fixed sequence of items."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


class _OllamaObj(dict):
    """Ollama's SDK objects support both attribute and subscript access; AIMU's own code
    mixes both styles (``response["response"]`` and ``response.thinking``), so the fake
    needs to answer both."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _ollama_part():
    message = _OllamaObj(role="assistant", thinking=None, tool_calls=None, content="ok")
    return _OllamaObj(message=message, response="ok", thinking=None)


class _FakeAnthropicStream:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])

    def get_final_message(self):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="ok")],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )


class _FakeAsyncAnthropicStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="ok")],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def _openai_compat_create(**kw):
    if kw.get("stream"):
        delta = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
        chunk = types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(delta=delta)])
        return iter([chunk])
    message = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)


async def _async_openai_compat_create(**kw):
    if kw.get("stream"):
        delta = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
        chunk = types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(delta=delta)])
        return _AsyncIter([chunk])
    message = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)


def _drive_ollama(client_cls, monkeypatch, method, stream, prepare=None):
    import ollama as ollama_sdk

    import aimu.models.providers.ollama as ollama_mod

    def _call(**kw):
        part = _ollama_part()
        return iter([part]) if kw.get("stream") else part

    monkeypatch.setattr(
        ollama_sdk,
        "Client",
        lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=_call, generate=_call),
    )
    monkeypatch.setattr(ollama_mod, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(ollama_mod, "truncated_from_ollama", lambda *a, **k: False)
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    if prepare:
        prepare(client)
    result = getattr(client, method)("hi", stream=stream)
    if stream:
        list(result)
    return client


async def _drive_async_ollama(client_cls, monkeypatch, method, stream):
    import ollama as ollama_sdk

    import aimu.aio.providers.ollama as ollama_mod

    async def _call(**kw):
        part = _ollama_part()
        return _AsyncIter([part]) if kw.get("stream") else part

    monkeypatch.setattr(
        ollama_sdk,
        "AsyncClient",
        lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=_call, generate=_call),
    )
    monkeypatch.setattr(ollama_mod, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(ollama_mod, "truncated_from_ollama", lambda *a, **k: False)
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    result = await getattr(client, method)("hi", stream=stream)
    if stream:
        async for _ in result:
            pass
    return client


def _drive_anthropic(client_cls, monkeypatch, method, stream):
    import anthropic as anthropic_sdk

    def fake_create(**kw):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="ok")],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(
        anthropic_sdk,
        "Anthropic",
        lambda **kw: types.SimpleNamespace(
            messages=types.SimpleNamespace(create=fake_create, stream=lambda **kw: _FakeAnthropicStream())
        ),
    )
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    result = getattr(client, method)("hi", stream=stream)
    if stream:
        list(result)
    return client


async def _drive_async_anthropic(client_cls, monkeypatch, method, stream):
    import anthropic as anthropic_sdk

    async def fake_create(**kw):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="ok")],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(
        anthropic_sdk,
        "AsyncAnthropic",
        lambda **kw: types.SimpleNamespace(
            messages=types.SimpleNamespace(create=fake_create, stream=lambda **kw: _FakeAsyncAnthropicStream())
        ),
    )
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    result = await getattr(client, method)("hi", stream=stream)
    if stream:
        async for _ in result:
            pass
    return client


def _drive_openai_compat(client_cls, monkeypatch, method, stream, prepare=None):
    import openai as openai_sdk

    monkeypatch.setattr(
        openai_sdk,
        "OpenAI",
        lambda **kw: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_openai_compat_create))
        ),
    )
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    if prepare:
        prepare(client)
    result = getattr(client, method)("hi", stream=stream)
    if stream:
        list(result)
    return client


async def _drive_async_openai_compat(client_cls, monkeypatch, method, stream, prepare=None):
    import openai as openai_sdk

    monkeypatch.setattr(
        openai_sdk,
        "AsyncOpenAI",
        lambda **kw: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_async_openai_compat_create))
        ),
    )
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    if prepare:
        prepare(client)
    result = await getattr(client, method)("hi", stream=stream)
    if stream:
        async for _ in result:
            pass
    return client


_IN_PROCESS_SKIP = {
    "HuggingFaceClient": (
        "loads real model weights in __init__. Both of its _record_request call sites are "
        "exercised directly instead, with the real BaseModelClient._record_request bound (not "
        "a no-op): _generate_sync by test_hf_generate_sync_records_the_rendered_prompt_and_kwargs, "
        "_generate_streaming (which _chat_streamed also funnels through) by "
        "test_hf_generate_streaming_records_the_rendered_prompt_and_kwargs. "
        "test_hf_apply_chat_template_records_an_image_count covers the images/audio marker but "
        "does not itself touch _record_request."
    ),
    "LlamaCppClient": (
        "loads a real GGUF file in __init__. All four of its _record_request call sites are "
        "exercised directly instead, with the real BaseModelClient._record_request bound (not "
        "a no-op): _chat by test_llamacpp_chat_records_a_copy_of_messages_not_a_live_alias, "
        "_generate by test_llamacpp_generate_records_kwargs_and_messages, _generate_streamed by "
        "test_llamacpp_generate_streamed_records_kwargs_and_messages, and _chat_streamed by "
        "test_llamacpp_chat_streamed_records_a_copy_of_messages_not_a_live_alias."
    ),
    "AsyncHuggingFaceClient": (
        "wraps a sync HuggingFaceClient via asyncio.to_thread and shares its _record_request "
        "call sites -- see the HuggingFaceClient skip above; there is no separate async "
        "record path to miss."
    ),
    "AsyncLlamaCppClient": (
        "wraps a sync LlamaCppClient via asyncio.to_thread and shares its _record_request "
        "call sites -- see the LlamaCppClient skip above; there is no separate async record "
        "path to miss."
    ),
}


def _clients_to_check():
    """Every installed sync client plus its async twin (mechanical name/module mapping:
    ``aimu.models.providers.X`` -> ``aimu.aio.providers.X``, ``FooClient`` -> ``AsyncFooClient``),
    so a newly added provider is covered on both surfaces without a list to remember to update."""
    import importlib

    from aimu.models import available_text_clients

    sync_clients = available_text_clients()
    async_clients = []
    for cls in sync_clients:
        async_module_name = cls.__module__.replace("aimu.models.providers", "aimu.aio.providers", 1)
        try:
            async_module = importlib.import_module(async_module_name)
        except ImportError:
            continue
        async_cls = getattr(async_module, "Async" + cls.__name__, None)
        if async_cls is not None:
            async_clients.append(async_cls)
    return sync_clients + async_clients


@pytest.mark.parametrize("method,stream", _PATHS, ids=[f"{m}-{'stream' if s else 'sync'}" for m, s in _PATHS])
@pytest.mark.parametrize("client_cls", _clients_to_check(), ids=lambda c: c.__name__)
async def test_every_client_records_its_request(client_cls, method, stream, monkeypatch):
    """A shipped client whose request path records nothing leaves last_request stale --
    silently showing the *previous* call's payload as if it were current.

    Mirrors test_every_client_declares_a_verdict_for_every_portable_key: the rule is
    cross-cutting, so a test enforces it rather than a convention. Parametrized over both
    the concrete client (sync + async twin, via _clients_to_check()) and the request path
    (chat/generate x stream/non-stream), since the seam is per-call-site, not per-class --
    openai_compat alone has 4 call sites and anthropic has 6, and a provider that wired
    _record_request into _chat but not _chat_streamed would pass a client-level-only check.

    HuggingFace and llama.cpp (sync and async) load real model weights / a real GGUF file
    and wrap via asyncio.to_thread, so they can't be driven this way; see _IN_PROCESS_SKIP
    for exactly which tests exercise their _record_request call sites instead. Named and
    skipped rather than silently passed, per the rule this test exists to enforce: a
    coverage gap must be visible, not quiet.
    """
    from aimu.aio.providers.openai_compat import AsyncOpenAICompatClient
    from aimu.models.providers.openai_compat import OpenAICompatClient

    name = client_cls.__name__
    if name in _IN_PROCESS_SKIP:
        pytest.skip(f"{name}: {_IN_PROCESS_SKIP[name]}")

    if name == "OllamaClient":
        client = _drive_ollama(client_cls, monkeypatch, method, stream)
    elif name == "AnthropicClient":
        client = _drive_anthropic(client_cls, monkeypatch, method, stream)
    elif issubclass(client_cls, OpenAICompatClient):
        client = _drive_openai_compat(client_cls, monkeypatch, method, stream)
    elif name == "AsyncOllamaClient":
        client = await _drive_async_ollama(client_cls, monkeypatch, method, stream)
    elif name == "AsyncAnthropicClient":
        client = await _drive_async_anthropic(client_cls, monkeypatch, method, stream)
    elif issubclass(client_cls, AsyncOpenAICompatClient):
        client = await _drive_async_openai_compat(client_cls, monkeypatch, method, stream)
    else:
        raise AssertionError(
            f"No driver wired for {name} in this guard test -- add one rather than letting "
            "it silently skip the check this test exists to enforce."
        )

    assert client.last_request is not None, f"{name}.{method}(stream={stream}) recorded nothing on last_request"


# ---------------------------------------------------------------------------
# Tool-call arguments: a dict in self.messages, a JSON string on the wire.
#
# Placed after the drivers above because it reuses them: the seam is the same four
# openai-compat request paths, and a second copy of the SDK-stubbing wiring would be one
# more thing to keep in step.
# ---------------------------------------------------------------------------

_TOOL_ARGUMENTS = {"query": "aimu", "num_results": 5}


def _seed_completed_tool_round(client) -> None:
    """Put a finished tool round in history the way a provider's tool-call path does.

    Goes through the real ``_record_tool_calls`` rather than hand-writing the assistant
    message, so the test pins the shape the store actually holds rather than one it invented.
    """
    client._record_tool_calls([{"name": "web_search", "arguments": dict(_TOOL_ARGUMENTS)}])
    tool_call = client.messages[-1]["tool_calls"][0]
    assert isinstance(tool_call["function"]["arguments"], dict), "the store no longer holds the parsed form"
    client.messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": "1. a result"})


def _assert_arguments_left_as_json_text(client) -> None:
    sent = [tc for message in client.last_request["messages"] for tc in message.get("tool_calls", ())]
    assert sent, "the seeded tool round never reached the payload"
    for tool_call in sent:
        arguments = tool_call["function"]["arguments"]
        assert isinstance(arguments, str), f"OpenAI's schema types this as a string, got {type(arguments).__name__}"
        assert json.loads(arguments) == _TOOL_ARGUMENTS
    stored = next(message for message in client.messages if message.get("tool_calls"))
    assert stored["tool_calls"][0]["function"]["arguments"] == _TOOL_ARGUMENTS, "the adaptation leaked into the store"


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_tool_call_arguments_reach_the_wire_as_json_text(monkeypatch, stream):
    """OpenAI's schema types ``tool_calls[].function.arguments`` as a string, and a server
    rendering its chat template calls ``json.loads`` on it, so sending the parsed dict does
    not merely deviate from the schema, it raises server-side: mlx-lm answers
    ``404 {'error': 'the JSON object must be str, bytes or bytearray, not dict'}``. That
    failed every turn which had called a tool, on the request *after* the tool result, so
    the first round of any tool-using conversation looked fine and the second died.

    The store keeps the parsed dict either way, which is what Ollama's and Anthropic's
    request paths want and what a UI or transcript reads; this is adaptation at request
    time, per the plain-data principle.
    """
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient

    client = _drive_openai_compat(LMStudioOpenAIClient, monkeypatch, "chat", stream, prepare=_seed_completed_tool_round)
    _assert_arguments_left_as_json_text(client)


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
async def test_async_tool_call_arguments_reach_the_wire_as_json_text(monkeypatch, stream):
    """The async twin of the test above. Four request paths carry this payload (``_chat`` and
    ``_chat_streamed`` on each surface) and each builds its own ``messages`` argument, so a
    fix applied to one is not applied to the others."""
    from aimu.aio.providers.openai_compat import AsyncLMStudioOpenAIClient

    client = await _drive_async_openai_compat(
        AsyncLMStudioOpenAIClient, monkeypatch, "chat", stream, prepare=_seed_completed_tool_round
    )
    _assert_arguments_left_as_json_text(client)


def test_ollama_keeps_tool_call_arguments_parsed_on_the_wire(monkeypatch):
    """The other half of the reason this adaptation cannot live in ``strip_inert_keys``:
    Ollama's request path calls that too, and its API wants an object here, not a string.
    Encoding for everyone would move the bug rather than fix it."""
    from aimu.models.providers.ollama import OllamaClient

    client = _drive_ollama(OllamaClient, monkeypatch, "chat", False, prepare=_seed_completed_tool_round)
    sent = [tc for message in client.last_request["messages"] for tc in message.get("tool_calls", ())]
    assert sent, "the seeded tool round never reached the payload"
    assert sent[0]["function"]["arguments"] == _TOOL_ARGUMENTS
