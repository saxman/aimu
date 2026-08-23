"""What AIMU actually sent.

Between a caller's chat() and the wire sit the four-tier generate_kwargs merge, the
GENERATE_KWARG_SUPPORT renames and drops, thinking resolution, strip_inert_keys, and
provider format adaptation. Every step is principled and documented, and none of it was
visible at runtime before last_request. A last_request that showed a *pre*-adaptation
payload would be worse than none, so each transformation gets its own assertion.
"""

from __future__ import annotations

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
# The guard: a shipped client whose request path records nothing.
# ---------------------------------------------------------------------------


def _drive_ollama(client_cls, monkeypatch):
    import ollama as ollama_sdk

    import aimu.models.providers.ollama as ollama_mod

    def _call(**kw):
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None, thinking=None)
        return {"message": message}

    monkeypatch.setattr(
        ollama_sdk,
        "Client",
        lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=_call, generate=_call),
    )
    monkeypatch.setattr(ollama_mod, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(ollama_mod, "truncated_from_ollama", lambda *a, **k: False)
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    client.chat("hi")
    return client


def _drive_anthropic(client_cls, monkeypatch):
    import anthropic as anthropic_sdk

    def fake_create(**kw):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="ok")],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(
        anthropic_sdk,
        "Anthropic",
        lambda **kw: types.SimpleNamespace(messages=types.SimpleNamespace(create=fake_create)),
    )
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    client.chat("hi")
    return client


def _drive_openai_compat(client_cls, monkeypatch):
    import openai as openai_sdk

    def create(**kw):
        message = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(
        openai_sdk,
        "OpenAI",
        lambda **kw: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        ),
    )
    model = next(iter(client_cls.MODELS))
    client = client_cls(model)
    client.chat("hi")
    return client


def _clients_to_check():
    from aimu.models import available_text_clients

    return available_text_clients()


@pytest.mark.parametrize("client_cls", _clients_to_check(), ids=lambda c: c.__name__)
def test_every_client_records_its_request(client_cls, monkeypatch):
    """A shipped client whose request path records nothing leaves last_request stale --
    silently showing the *previous* call's payload as if it were current.

    Mirrors test_every_client_declares_a_verdict_for_every_portable_key: the rule is
    cross-cutting, so a test enforces it rather than a convention. Drives one mocked
    request per concrete client (enumerated via available_text_clients(), so a newly added
    provider is covered automatically) and asserts last_request is populated afterward.

    HuggingFace and llama.cpp load real model weights / a real GGUF file in their
    constructors, so they can't be driven this way without an expensive or unfaithful stub;
    their _record_request call sites (in _generate_sync / _generate_streaming for HF, and in
    every create_chat_completion call for llama.cpp) are exercised directly by the
    hand-rolled fakes in tests/test_models_api.py and tests/test_thinking_control.py instead.
    Named and skipped here rather than silently passed, per the rule this test exists to
    enforce: a coverage gap must be visible, not quiet.
    """
    from aimu.models.providers.openai_compat import OpenAICompatClient

    name = client_cls.__name__
    if name == "HuggingFaceClient":
        pytest.skip(
            "HuggingFaceClient loads real model weights in __init__; its _record_request "
            "call sites are covered directly by fakes in test_models_api.py / "
            "test_thinking_control.py, not by an end-to-end chat() here."
        )
    if name == "LlamaCppClient":
        pytest.skip(
            "LlamaCppClient loads a real GGUF file in __init__; its _record_request call "
            "sites are covered directly by fakes in test_models_api.py, not by an "
            "end-to-end chat() here."
        )

    if name == "OllamaClient":
        client = _drive_ollama(client_cls, monkeypatch)
    elif name == "AnthropicClient":
        client = _drive_anthropic(client_cls, monkeypatch)
    elif issubclass(client_cls, OpenAICompatClient):
        client = _drive_openai_compat(client_cls, monkeypatch)
    else:
        raise AssertionError(
            f"No driver wired for {name} in this guard test -- add one rather than letting "
            "it silently skip the check this test exists to enforce."
        )

    assert client.last_request is not None, f"{name}.chat() recorded nothing on last_request"
