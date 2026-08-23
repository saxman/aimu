"""Mock-only tests for the portable `thinking=` control surface.

Covers the capability declaration on ModelSpec, argument resolution, warning behavior, the
per-provider wire translation, and the Agent-level threading of the argument through every turn
of a run. No server and no model weights required.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import anthropic as anthropic_sdk
import ollama
import pytest

from helpers import MockModelClient, client_stand_in

import aimu.models.providers.ollama as sync_ollama
from aimu.models._internal.thinking import (
    THINKING_KWARG,
    THINKING_LEVELS,
    ResolvedThinking,
    resolve_thinking,
)
from aimu.agents import Agent
from aimu.models.base import AdHocModel, Model, ModelSpec
from aimu.models.providers.ollama import OllamaModel
from aimu.tools import tool


def test_defaults_are_backward_compatible():
    spec = ModelSpec("m", thinking=True)

    assert spec.thinking_levels is False
    assert spec.thinking_optional is True
    assert spec.nonthinking_generation_kwargs is None


def test_flags_are_mirrored_onto_enum_members():
    class _Catalog(Model):
        LEVELLED = ModelSpec(
            "levelled",
            thinking=True,
            thinking_levels=True,
            generation_kwargs={"temperature": 1.0},
            nonthinking_generation_kwargs={"temperature": 0.7},
        )

    member = _Catalog.LEVELLED

    assert member.thinking_levels is True
    assert member.thinking_optional is True
    assert member.nonthinking_generation_kwargs == {"temperature": 0.7}


def test_flags_are_mirrored_onto_adhoc_models():
    model = AdHocModel(ModelSpec("srv/mystery", thinking=True, thinking_levels=True))

    assert model.thinking_levels is True
    assert model.thinking_optional is True
    assert model.nonthinking_generation_kwargs == {}


def test_levels_without_thinking_is_rejected():
    with pytest.raises(ValueError, match="thinking_levels"):
        ModelSpec("m", thinking=False, thinking_levels=True)


def test_non_optional_without_thinking_is_rejected():
    with pytest.raises(ValueError, match="thinking_optional"):
        ModelSpec("m", thinking=False, thinking_optional=False)


def test_qwen_3_6_thinking_profile_matches_its_card():
    """Regression guard: the card specifies presence_penalty=0.0 in thinking mode. Ollama
    carried 0.9 and HuggingFace carried 1.5 (the instruct value) before this fix."""
    from aimu.models.providers.hf.text import HuggingFaceModel
    from aimu.models.providers.ollama import OllamaModel

    assert OllamaModel.QWEN_3_6_27B.generation_kwargs["presence_penalty"] == 0.0
    assert HuggingFaceModel.QWEN_3_6_27B.generation_kwargs["presence_penalty"] == 0.0


@pytest.mark.parametrize(
    "member",
    ["QWEN_3_5_9B", "QWEN_3_6_27B", "QWEN_3_8_27B"],
)
def test_qwen_models_carry_an_instruct_profile(member):
    from aimu.models.providers.ollama import OllamaModel

    profile = getattr(OllamaModel, member).nonthinking_generation_kwargs

    assert profile["temperature"] == 0.7
    assert profile["top_p"] == 0.80
    assert profile["presence_penalty"] == 1.5


def test_only_qwen_3_8_declares_effort_levels():
    from aimu.models.providers.ollama import OllamaModel

    assert OllamaModel.QWEN_3_8_27B.thinking_levels is True
    assert OllamaModel.QWEN_3_6_27B.thinking_levels is False
    assert OllamaModel.QWEN_3_5_9B.thinking_levels is False


def test_gemini_2_5_pro_cannot_disable_reasoning():
    from aimu.models.providers.gemini.text import GeminiModel

    assert GeminiModel.GEMINI_2_5_PRO.thinking_optional is False
    assert GeminiModel.GEMINI_2_5_FLASH.thinking_optional is True


class _Model:
    """Minimal stand-in for a Model member, so these tests need no provider."""

    def __init__(self, value="m", thinking=True, levels=False, optional=True, nonthinking=None):
        self.value = value
        self.supports_thinking = thinking
        self.thinking_levels = levels
        self.thinking_optional = optional
        self.generation_kwargs = {"temperature": 1.0, "presence_penalty": 0.0}
        self.nonthinking_generation_kwargs = nonthinking or {}


def _collect():
    """Return a (warn callable, list of messages) pair."""
    seen: list[str] = []
    return seen.append, seen


def test_none_resolves_to_nothing():
    warn, seen = _collect()

    assert resolve_thinking(_Model(), None, warn=warn) is None
    assert seen == []


@pytest.mark.parametrize("value", [True, False])
def test_bools_resolve_on_a_toggle_model(value):
    warn, seen = _collect()

    resolved = resolve_thinking(_Model(), value, warn=warn)

    assert resolved == ResolvedThinking(enabled=value, level=None)
    assert seen == []


def test_level_resolves_on_a_levelled_model():
    warn, seen = _collect()

    resolved = resolve_thinking(_Model(levels=True), "high", warn=warn)

    assert resolved == ResolvedThinking(enabled=True, level="high")
    assert seen == []


def test_level_is_ignored_with_a_warning_on_a_toggle_model():
    warn, seen = _collect()

    resolved = resolve_thinking(_Model(value="gemma4:12b"), "high", warn=warn)

    assert resolved == ResolvedThinking(enabled=True, level=None)
    assert len(seen) == 1
    assert "gemma4:12b" in seen[0]


def test_enabling_a_non_thinking_model_warns_and_yields_nothing():
    warn, seen = _collect()

    assert resolve_thinking(_Model(value="llama3.2:3b", thinking=False), True, warn=warn) is None
    assert len(seen) == 1
    assert "llama3.2:3b" in seen[0]


def test_disabling_a_non_thinking_model_is_a_silent_no_op():
    warn, seen = _collect()

    assert resolve_thinking(_Model(thinking=False), False, warn=warn) is None
    assert seen == []


def test_disabling_an_always_reasoning_model_warns():
    warn, seen = _collect()

    assert resolve_thinking(_Model(value="gemini-2.5-pro", optional=False), False, warn=warn) is None
    assert len(seen) == 1
    assert "gemini-2.5-pro" in seen[0]


@pytest.mark.parametrize("value", ["lo", "xhigh", "HIGH", "extreme", "", 3, 1.5])
def test_invalid_values_raise(value):
    warn, _ = _collect()

    with pytest.raises(ValueError):
        resolve_thinking(_Model(levels=True), value, warn=warn)


def test_valid_levels_are_exactly_three():
    assert THINKING_LEVELS == ("low", "medium", "high")


def test_reserved_key_is_underscore_prefixed():
    assert THINKING_KWARG.startswith("_")


def _mixin_host(model):
    """A bare object carrying the mixin, so these tests need no provider client."""
    from aimu.models._internal.chat_state import _ChatStateMixin

    class _Host(_ChatStateMixin):
        def __init__(self, model):
            self.model = model

    return _Host(model)


def test_warn_once_deduplicates(caplog):
    host = _mixin_host(_Model())

    with caplog.at_level("WARNING"):
        host._warn_once("same message")
        host._warn_once("same message")
        host._warn_once("different message")

    assert [r.message for r in caplog.records] == ["same message", "different message"]


def test_apply_thinking_injects_the_reserved_key():
    host = _mixin_host(_Model())

    kwargs = host._apply_thinking({"max_tokens": 10}, False)

    assert kwargs[THINKING_KWARG] == ResolvedThinking(enabled=False, level=None)
    assert kwargs["max_tokens"] == 10


def test_apply_thinking_passes_none_through_untouched():
    host = _mixin_host(_Model())
    original = {"max_tokens": 10}

    assert host._apply_thinking(original, None) is original
    assert host._apply_thinking(None, None) is None


def test_apply_thinking_does_not_mutate_the_callers_dict():
    host = _mixin_host(_Model())
    original = {"max_tokens": 10}

    host._apply_thinking(original, True)

    assert THINKING_KWARG not in original


def test_apply_thinking_rejects_a_caller_supplied_reserved_key():
    """A caller-supplied `_thinking` key must raise here, at the layer where the mistake is
    actionable, rather than reach a provider that does `resolved.enabled` on whatever the
    caller put there and blows up with an opaque AttributeError."""
    host = _mixin_host(_Model())

    with pytest.raises(ValueError, match=THINKING_KWARG):
        host._apply_thinking({"max_tokens": 10, THINKING_KWARG: "anything"}, None)


def _fake_client(model, recorder):
    """A concrete BaseModelClient whose _chat/_generate only record their kwargs."""
    from aimu.models.base import BaseModelClient

    class _Fake(BaseModelClient):
        MODELS = None

        def __init__(self):
            self.model = model
            self.model_kwargs = None
            self._system_message = None
            self.default_generate_kwargs = {}
            self.messages = []
            self.tools = []
            self.last_thinking = ""
            self.last_usage = None
            self.last_output_truncated = False
            self.last_structured = None

        def _resolve_generate_kwargs(self, generate_kwargs=None):
            return dict(generate_kwargs or {})

        def _chat(self, user_message=None, generate_kwargs=None, **kw):
            recorder.append(generate_kwargs)
            return "ok"

        def _generate(self, prompt, generate_kwargs=None, **kw):
            recorder.append(generate_kwargs)
            return "ok"

    return _Fake()


def test_chat_forwards_a_resolved_request():
    seen: list = []
    client = _fake_client(_Model(levels=True), seen)

    client.chat("hi", thinking="low")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="low")


def test_generate_forwards_a_resolved_request():
    seen: list = []
    client = _fake_client(_Model(), seen)

    client.generate("hi", thinking=False)

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=False, level=None)


def test_omitting_thinking_injects_nothing():
    seen: list = []
    client = _fake_client(_Model(levels=True), seen)

    client.chat("hi")
    client.generate("hi")

    assert all(THINKING_KWARG not in (kwargs or {}) for kwargs in seen)


def test_invalid_value_raises_before_any_request():
    seen: list = []
    client = _fake_client(_Model(levels=True), seen)

    with pytest.raises(ValueError, match="Unknown thinking level"):
        client.chat("hi", thinking="xhigh")

    assert seen == []


def _ollama_recorder(monkeypatch, model):
    """Construct an OllamaClient whose chat/generate only record their kwargs."""
    from aimu.models.providers.ollama import OllamaClient

    calls: list[dict] = []

    def record(**kw):
        calls.append(kw)
        # `response["message"]` is accessed with dot notation in `_chat` (it is a pydantic
        # object on the real SDK), so the stand-in needs attribute access too.
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None, thinking=None)
        return {"message": message}

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=record, generate=record),
    )
    monkeypatch.setattr(sync_ollama, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(sync_ollama, "truncated_from_ollama", lambda *a, **k: False)
    return OllamaClient(model), calls


@pytest.mark.parametrize(
    "thinking,expected_think",
    [(None, True), (True, True), (False, False), ("low", "low"), ("high", "high")],
)
def test_ollama_maps_thinking_to_the_think_parameter(monkeypatch, thinking, expected_think):
    client, calls = _ollama_recorder(monkeypatch, OllamaModel.QWEN_3_8_27B)

    client.chat("hi", thinking=thinking)

    assert calls[0]["think"] == expected_think
    assert THINKING_KWARG not in calls[0]["options"]


def test_ollama_selects_the_instruct_profile_when_off(monkeypatch):
    client, calls = _ollama_recorder(monkeypatch, OllamaModel.QWEN_3_8_27B)

    client.chat("hi", thinking=False)

    assert calls[0]["options"]["temperature"] == 0.7
    assert calls[0]["options"]["presence_penalty"] == 1.5


def test_ollama_keeps_the_thinking_profile_when_on(monkeypatch):
    client, calls = _ollama_recorder(monkeypatch, OllamaModel.QWEN_3_8_27B)

    client.chat("hi", thinking=True)

    assert calls[0]["options"]["temperature"] == 1.0
    assert calls[0]["options"]["presence_penalty"] == 0.0


def test_ollama_caller_kwargs_still_win_over_the_selected_profile(monkeypatch):
    client, calls = _ollama_recorder(monkeypatch, OllamaModel.QWEN_3_8_27B)

    client.chat("hi", generate_kwargs={"temperature": 0.15}, thinking=False)

    assert calls[0]["options"]["temperature"] == 0.15
    assert calls[0]["options"]["presence_penalty"] == 1.5


def _openai_compat_client(monkeypatch, cls, model, **ctor):
    """Build an OpenAI-compat client with a recording SDK stub."""
    import openai

    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        message = types.SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        ),
    )
    return cls(model, **ctor), calls


def test_openai_compat_emits_reasoning_effort_with_xhigh_ceiling(monkeypatch):
    from aimu.models.providers.openai_compat import OllamaOpenAIClient, OllamaOpenAIModel

    client, calls = _openai_compat_client(monkeypatch, OllamaOpenAIClient, OllamaOpenAIModel.QWEN_3_8_27B)

    client.chat("hi", thinking="high")

    assert calls[0]["reasoning_effort"] == "xhigh"
    assert THINKING_KWARG not in calls[0]


@pytest.mark.parametrize("level,expected", [("low", "low"), ("medium", "medium"), ("high", "xhigh")])
def test_openai_compat_level_mapping(monkeypatch, level, expected):
    from aimu.models.providers.openai_compat import OllamaOpenAIClient, OllamaOpenAIModel

    client, calls = _openai_compat_client(monkeypatch, OllamaOpenAIClient, OllamaOpenAIModel.QWEN_3_8_27B)

    client.chat("hi", thinking=level)

    assert calls[0]["reasoning_effort"] == expected


def test_openai_compat_disables_via_chat_template_kwargs(monkeypatch):
    from aimu.models.providers.openai_compat import OllamaOpenAIClient, OllamaOpenAIModel

    client, calls = _openai_compat_client(monkeypatch, OllamaOpenAIClient, OllamaOpenAIModel.QWEN_3_8_27B)

    client.chat("hi", thinking=False)

    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_effort" not in calls[0]


def test_openai_compat_merges_a_caller_supplied_extra_body(monkeypatch):
    from aimu.models.providers.openai_compat import OllamaOpenAIClient, OllamaOpenAIModel

    client, calls = _openai_compat_client(monkeypatch, OllamaOpenAIClient, OllamaOpenAIModel.QWEN_3_8_27B)

    client.chat(
        "hi",
        generate_kwargs={"extra_body": {"guided_regex": "[0-9]+"}},
        thinking=False,
    )

    extra = calls[0]["extra_body"]
    assert extra["guided_regex"] == "[0-9]+"  # caller's key survives
    assert extra["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_cloud_never_sends_template_kwargs(monkeypatch):
    from aimu.models.providers.openai.text import OpenAIClient, OpenAIModel

    client, calls = _openai_compat_client(monkeypatch, OpenAIClient, OpenAIModel.GPT_4O)

    client.chat("hi", thinking=False)  # GPT-4o is not a thinking model: warn and ignore

    assert "extra_body" not in calls[0]
    assert "reasoning_effort" not in calls[0]
    assert THINKING_KWARG not in calls[0]


def test_gemini_cloud_gate_blocks_chat_template_kwargs_and_warns(monkeypatch, caplog):
    """GPT_4O above never reaches the gate at all (it isn't a thinking model, so resolution
    short-circuits to None before ``_SUPPORTS_CHAT_TEMPLATE_KWARGS`` is ever consulted). This
    test exercises the gate itself: GEMINI_2_5_FLASH is thinking=True and thinking_optional=True,
    so ``thinking=False`` reaches ``_apply_resolved_thinking`` and must be rejected by the False
    override rather than silently sent as a Qwen/vLLM template kwarg Google's endpoint doesn't
    understand. If ``_SUPPORTS_CHAT_TEMPLATE_KWARGS`` were flipped back to True on GeminiClient,
    this would fail: ``extra_body`` would appear in the request and no warning would be logged.
    """
    from aimu.models.providers.gemini.text import GeminiClient, GeminiModel

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client, calls = _openai_compat_client(monkeypatch, GeminiClient, GeminiModel.GEMINI_2_5_FLASH)

    with caplog.at_level("WARNING"):
        client.chat("hi", thinking=False)

    assert "extra_body" not in calls[0]
    assert "reasoning_effort" not in calls[0]
    assert THINKING_KWARG not in calls[0]
    assert any("has no way to disable reasoning" in r.message for r in caplog.records)


def _anthropic_kwargs(monkeypatch, model, thinking, generate_kwargs=None):
    """Return the generate_kwargs an AnthropicClient would send for a thinking request."""
    import anthropic

    from aimu.models.providers.anthropic import AnthropicClient

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace())
    client = AnthropicClient(model)

    base = {"max_tokens": 1024, **(generate_kwargs or {})}
    kwargs = client._apply_thinking(base, thinking)
    return client._thinking_kwargs(client._resolve_generate_kwargs(kwargs))


def test_anthropic_maps_a_level_to_a_token_budget(monkeypatch):
    from aimu.models.providers.anthropic import AnthropicModel

    kwargs = _anthropic_kwargs(monkeypatch, AnthropicModel.CLAUDE_SONNET_4_6, "low")

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert THINKING_KWARG not in kwargs


def test_anthropic_thinking_off_sends_no_thinking_block(monkeypatch):
    from aimu.models.providers.anthropic import AnthropicModel

    kwargs = _anthropic_kwargs(
        monkeypatch, AnthropicModel.CLAUDE_SONNET_4_6, False, generate_kwargs={"temperature": 0.3}
    )

    assert "thinking" not in kwargs
    # temperature is only stripped/forced to satisfy extended thinking; with thinking off for
    # this call, the caller's own value must survive untouched (not merely be present).
    assert kwargs["temperature"] == 0.3


def test_anthropic_thinking_off_keeps_top_p(monkeypatch):
    """top_p is dropped for the same reason temperature is forced: it conflicts with extended
    thinking. With thinking resolved off for this call, there is no conflict, so the caller's
    own value must survive untouched, mirroring the temperature test above."""
    from aimu.models.providers.anthropic import AnthropicModel

    kwargs = _anthropic_kwargs(monkeypatch, AnthropicModel.CLAUDE_SONNET_4_6, False, generate_kwargs={"top_p": 0.9})

    assert "thinking" not in kwargs
    assert kwargs["top_p"] == 0.9


def test_anthropic_none_still_drops_top_p(monkeypatch):
    """thinking=None leaves the default (thinking-on) behavior untouched, so top_p still
    conflicts with extended thinking and must still be dropped."""
    from aimu.models.providers.anthropic import AnthropicModel

    kwargs = _anthropic_kwargs(monkeypatch, AnthropicModel.CLAUDE_SONNET_4_6, None, generate_kwargs={"top_p": 0.9})

    assert "top_p" not in kwargs


def test_anthropic_adaptive_warns_and_ignores_a_level(monkeypatch, caplog):
    from aimu.models.providers.anthropic import AnthropicModel

    with caplog.at_level("WARNING"):
        kwargs = _anthropic_kwargs(monkeypatch, AnthropicModel.CLAUDE_OPUS_4_7, "low")

    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "budget_tokens" not in kwargs["thinking"]
    assert any("adaptive" in r.message.lower() for r in caplog.records)


def test_anthropic_none_keeps_current_behavior(monkeypatch):
    from aimu.models.providers.anthropic import AnthropicModel

    kwargs = _anthropic_kwargs(monkeypatch, AnthropicModel.CLAUDE_SONNET_4_6, None)

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8000}


def test_anthropic_structured_output_drops_the_reserved_key(monkeypatch, caplog):
    """Regression: chat(schema=..., thinking=...) must not leak `_thinking` into the
    Anthropic request. The structured path forces a tool and cannot combine with extended
    thinking, so `_thinking_kwargs` is never on that path; `_structured_call` must strip the
    reserved key itself."""
    import dataclasses

    import anthropic

    from aimu.models.providers.anthropic import AnthropicClient, AnthropicModel

    @dataclasses.dataclass
    class Person:
        name: str

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace())
    client = AnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="tool_use", input={"name": "Ada"})],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    client._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=fake_create))

    with caplog.at_level("WARNING"):
        result = client.chat("hi", schema=Person, thinking="low")

    assert THINKING_KWARG not in captured
    assert "thinking" not in captured
    assert result == Person(name="Ada")
    assert any("structured output" in r.message.lower() for r in caplog.records)


def test_anthropic_structured_output_thinking_off_does_not_warn(monkeypatch, caplog):
    """chat(schema=..., thinking=False) is entirely consistent: the forced-tool structured
    path already carries no reasoning, so asking for no reasoning got exactly what was asked
    for and must not warn. thinking="low" on the same call asks for something the path
    genuinely cannot honour and must still warn (regression guard for the fix above)."""
    import dataclasses

    import anthropic

    from aimu.models.providers.anthropic import AnthropicClient, AnthropicModel

    @dataclasses.dataclass
    class Person:
        name: str

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace())

    def make_client():
        client = AnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)

        def fake_create(**kwargs):
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="tool_use", input={"name": "Ada"})],
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
            )

        client._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=fake_create))
        return client

    with caplog.at_level("WARNING"):
        result = make_client().chat("hi", schema=Person, thinking=False)

    assert result == Person(name="Ada")
    assert not any("structured output" in r.message.lower() for r in caplog.records)

    caplog.clear()

    with caplog.at_level("WARNING"):
        make_client().chat("hi", schema=Person, thinking="low")

    assert any("structured output" in r.message.lower() for r in caplog.records)


async def test_anthropic_async_structured_output_drops_the_reserved_key(monkeypatch, caplog):
    """Async mirror of the sync structured-path leak regression above."""
    import dataclasses

    import anthropic

    from aimu.aio.providers.anthropic import AsyncAnthropicClient
    from aimu.models.providers.anthropic import AnthropicModel

    @dataclasses.dataclass
    class Person:
        name: str

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace())
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kw: types.SimpleNamespace())
    client = AsyncAnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="tool_use", input={"name": "Ada"})],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    client._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=fake_create))

    with caplog.at_level("WARNING"):
        result = await client.chat("hi", schema=Person, thinking="low")

    assert THINKING_KWARG not in captured
    assert "thinking" not in captured
    assert result == Person(name="Ada")
    assert any("structured output" in r.message.lower() for r in caplog.records)


def _hf_template_recorder(model):
    """A stand-in HuggingFaceClient exposing only what _apply_chat_template touches on the
    processor branch (the one Gemma 4 and Qwen 3.5/3.6 actually take)."""
    from aimu.models.providers.hf.text import HuggingFaceClient

    calls: list[dict] = []

    class _Processor:
        def apply_chat_template(self, messages, **kw):
            calls.append(kw)
            return "rendered"

        def __call__(self, **kw):
            return types.SimpleNamespace(to=lambda device: {})

    client = types.SimpleNamespace(
        model=model,
        _hf_processor=_Processor(),
        _hf_model=types.SimpleNamespace(device="cpu"),
    )
    return HuggingFaceClient, client, calls


def _hf_tokenizer_recorder(model):
    """A stand-in HuggingFaceClient exposing only what _apply_chat_template touches on the
    plain-tokenizer branch. This is the branch QWEN_3_8_27B (the only HF models with
    thinking_levels=True) actually takes in production: its id, "Qwen/Qwen3.8-27B", does not
    match any of the processor-branch prefixes ("Qwen/Qwen3.5", "Qwen/Qwen3.6", "google/gemma-3",
    "google/gemma-4"), so _hf_processor is None for it and _apply_chat_template falls through to
    the tokenizer branches below.
    """
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    calls: list[dict] = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kw):
            calls.append(kw)
            return "rendered"

        def __call__(self, *args, **kw):
            return types.SimpleNamespace(to=lambda device: types.SimpleNamespace(input_ids=[[]]))

    client = types.SimpleNamespace(
        model=model,
        MODELS=HuggingFaceModel,
        _hf_processor=None,
        _hf_tokenizer=_Tokenizer(),
        _hf_model=types.SimpleNamespace(device="cpu"),
    )
    return HuggingFaceClient, client, calls


@pytest.mark.parametrize("enabled", [True, False])
def test_hf_threads_enable_thinking_per_call(enabled):
    from aimu.models._internal.thinking import ResolvedThinking
    from aimu.models.providers.hf.text import HuggingFaceModel

    cls, client, calls = _hf_template_recorder(HuggingFaceModel.QWEN_3_8_27B)

    cls._apply_chat_template(client, [], thinking=ResolvedThinking(enabled=enabled))

    assert calls[0]["enable_thinking"] is enabled


def test_hf_defaults_to_the_capability_flag_when_unset():
    from aimu.models.providers.hf.text import HuggingFaceModel

    cls, client, calls = _hf_template_recorder(HuggingFaceModel.QWEN_3_8_27B)

    cls._apply_chat_template(client, [], thinking=None)

    assert calls[0]["enable_thinking"] is True  # QWEN_3_8_27B supports thinking


def test_hf_selects_the_instruct_profile_when_off():
    from aimu.models._internal.thinking import THINKING_KWARG, ResolvedThinking
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    client = client_stand_in(HuggingFaceClient, HuggingFaceModel.QWEN_3_8_27B)

    merged = client._resolve_generate_kwargs({THINKING_KWARG: ResolvedThinking(enabled=False)})

    assert merged["temperature"] == 0.7
    assert merged["top_p"] == 0.80
    assert THINKING_KWARG in merged  # peeked, not popped: the template path needs it


@pytest.mark.parametrize("level,expected", [("low", "low"), ("medium", "medium"), ("high", "xhigh")])
def test_hf_maps_a_level_to_the_templates_effort_vocabulary_on_the_processor_branch(level, expected):
    """Qwen's own effort vocabulary tops out at "xhigh"; the shared QWEN_REASONING_EFFORT
    table must be applied before the value reaches apply_chat_template. Exercised here on the
    processor branch (mirrors the shape used for the OpenAI-compat wire-field mapping test)."""
    from aimu.models._internal.thinking import ResolvedThinking
    from aimu.models.providers.hf.text import HuggingFaceModel

    cls, client, calls = _hf_template_recorder(HuggingFaceModel.QWEN_3_8_27B)

    cls._apply_chat_template(client, [], thinking=ResolvedThinking(enabled=True, level=level))

    assert calls[0]["reasoning_effort"] == expected


@pytest.mark.parametrize("level,expected", [("low", "low"), ("medium", "medium"), ("high", "xhigh")])
def test_hf_maps_a_level_on_the_tokenizer_branch_the_model_actually_takes(level, expected):
    """QWEN_3_8_27B's id does not match any processor-branch prefix, so in production it
    templates through the plain tokenizer, not the processor. This is the branch that must
    carry reasoning_effort without raising: Qwen 3.8's chat_template.jinja validates the value
    against {"xhigh", "medium", "low"} and calls raise_exception on anything else."""
    from aimu.models._internal.thinking import ResolvedThinking
    from aimu.models.providers.hf.text import HuggingFaceModel

    cls, client, calls = _hf_tokenizer_recorder(HuggingFaceModel.QWEN_3_8_27B)

    cls._apply_chat_template(client, [], thinking=ResolvedThinking(enabled=True, level=level))

    assert calls[0]["enable_thinking"] is True
    assert calls[0]["reasoning_effort"] == expected


def test_hf_sends_no_effort_kwarg_when_the_model_has_no_levels():
    """A model with thinking_levels=False must never receive reasoning_effort: some chat
    templates raise on an unrecognised kwarg, and this one has no vocabulary for it at all."""
    from aimu.models._internal.thinking import ResolvedThinking
    from aimu.models.providers.hf.text import HuggingFaceModel

    assert HuggingFaceModel.QWEN_3_6_27B.thinking_levels is False

    cls, client, calls = _hf_template_recorder(HuggingFaceModel.QWEN_3_6_27B)

    cls._apply_chat_template(client, [], thinking=ResolvedThinking(enabled=True, level="high"))

    assert "reasoning_effort" not in calls[0]


def test_llamacpp_drops_the_reserved_key(monkeypatch):
    from aimu.models._internal.thinking import ResolvedThinking
    from aimu.models.providers.llamacpp import LlamaCppClient, LlamaCppModel

    client = client_stand_in(LlamaCppClient, LlamaCppModel.QWEN_3_8B, {"max_tokens": 128})

    merged = client._resolve_generate_kwargs({THINKING_KWARG: ResolvedThinking(enabled=False)})

    assert THINKING_KWARG not in merged


def test_fallback_forwards_thinking_to_the_first_client():
    from aimu.models.fallback import FallbackClient

    seen: list = []
    primary = _fake_client(_Model(levels=True), seen)

    FallbackClient([primary]).chat("hi", thinking="low")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="low")


def test_agentic_view_accepts_thinking():
    from aimu.agents import Agent

    seen: list = []
    client = _fake_client(_Model(levels=True), seen)

    Agent(client).as_model_client().chat("hi", thinking="low")

    assert any(THINKING_KWARG in (kwargs or {}) for kwargs in seen)


def test_top_level_chat_forwards_thinking(monkeypatch):
    import aimu

    seen: list = []

    def fake_client(model=None, **kw):
        return _fake_client(_Model(levels=True), seen)

    monkeypatch.setattr(aimu, "client", fake_client)

    aimu.chat("hi", model="ollama:qwen3.8:27b", thinking="low")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="low")


def test_pop_thinking_removes_and_returns_the_reserved_key():
    from aimu.models._internal.thinking import pop_thinking

    resolved = ResolvedThinking(enabled=True, level="low")
    kwargs = {"temperature": 0.5, THINKING_KWARG: resolved}

    assert pop_thinking(kwargs) is resolved
    assert THINKING_KWARG not in kwargs
    assert kwargs == {"temperature": 0.5}


def test_pop_thinking_returns_none_when_absent():
    from aimu.models._internal.thinking import pop_thinking

    kwargs = {"temperature": 0.5}

    assert pop_thinking(kwargs) is None
    assert kwargs == {"temperature": 0.5}


def test_every_provider_consumes_the_reserved_key_through_the_helper():
    """The reserved key must be removed via pop_thinking, not a hand-rolled pop.

    One greppable helper is what a fifth provider author can find. Hand-rolled pops were
    reimplemented three different ways across four providers, and on the Ollama path a
    missed one is silently serialized into the request body rather than rejected.
    """
    import pathlib

    provider_files = [
        "aimu/models/providers/ollama.py",
        "aimu/aio/providers/ollama.py",
        "aimu/models/providers/openai_compat.py",
        "aimu/models/providers/anthropic.py",
        "aimu/models/providers/hf/text.py",
        "aimu/models/providers/llamacpp.py",
    ]

    hand_rolled = [path for path in provider_files if "pop(THINKING_KWARG" in pathlib.Path(path).read_text()]

    assert hand_rolled == []


def test_non_thinking_warning_does_not_claim_a_reasoning_model_cannot_reason():
    """o3 reasons; AIMU just cannot expose or steer that reasoning.

    `supports_thinking` means "reasoning is visible through AIMU", so the warning must not
    assert something false about a well-known reasoning family. A user who reads
    "o3 is not a thinking model" has been told a falsehood and will distrust the rest.
    """
    from aimu.models.providers.openai.text import OpenAIModel

    warn, seen = _collect()

    resolve_thinking(_Model(value=OpenAIModel.O3.value, thinking=False), True, warn=warn)

    assert len(seen) == 1
    assert "is not a thinking model" not in seen[0]
    assert OpenAIModel.O3.value in seen[0]


@pytest.mark.parametrize("bad", ["xhigh", 3])
def test_both_invalid_value_paths_name_the_valid_levels_the_same_way(bad):
    warn, _ = _collect()

    with pytest.raises(ValueError) as exc:
        resolve_thinking(_Model(levels=True), bad, warn=warn)

    assert "low, medium, high" in str(exc.value)


@pytest.mark.parametrize("method,stream", [("chat", False), ("chat", True), ("generate", False), ("generate", True)])
def test_ollama_threads_thinking_through_every_entry_point(monkeypatch, method, stream):
    """All four request builders must translate and strip the reserved key, not just chat().

    They share a one-line `_pop_think` call, but nothing stopped a future edit from touching one
    and not the others, and a leaked key on this provider is serialized silently rather than
    rejected (Ollama types `options` as an open mapping).
    """
    from aimu.models.providers.ollama import OllamaClient

    calls: list[dict] = []

    class _Part(dict):
        """Ollama parts are read both by key and by attribute, depending on the path."""

        def __init__(self, **fields):
            super().__init__(**fields)
            self.__dict__.update(fields)

    def record(**kw):
        calls.append(kw)
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None, thinking=None)
        part = _Part(message=message, response="ok", thinking=None)
        return iter([part]) if kw.get("stream") else part

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=record, generate=record),
    )
    monkeypatch.setattr(sync_ollama, "usage_from_ollama", lambda *a, **k: None)
    monkeypatch.setattr(sync_ollama, "truncated_from_ollama", lambda *a, **k: False)
    client = OllamaClient(OllamaModel.QWEN_3_8_27B)

    result = getattr(client, method)("hi", thinking="low", stream=stream)
    if stream:
        list(result)

    assert calls[0]["think"] == "low"
    assert THINKING_KWARG not in calls[0]["options"]


def test_ollama_thinking_none_on_a_non_thinking_model_still_sends_think_false(monkeypatch):
    """The no-op guarantee has two sides; this is the one that was resting on inference."""
    client, calls = _ollama_recorder(monkeypatch, OllamaModel.LLAMA_3_2_3B)

    client.chat("hi", thinking=None)

    assert calls[0]["think"] is False


def test_anthropic_explicit_budget_wins_over_a_level(monkeypatch):
    """`thinking_budget_tokens` is the documented escape hatch, so a level must not override it."""
    from aimu.models.providers.anthropic import AnthropicClient, AnthropicModel

    monkeypatch.setattr(anthropic_sdk, "Anthropic", lambda **kw: types.SimpleNamespace())
    client = AnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)

    kwargs = client._apply_thinking({"max_tokens": 20000, "thinking_budget_tokens": 12000}, "low")
    kwargs = client._thinking_kwargs(client._resolve_generate_kwargs(kwargs))

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 12000}


@pytest.mark.parametrize("method,stream", [("chat", False), ("chat", True), ("generate", False), ("generate", True)])
def test_fallback_forwards_thinking_on_every_path(method, stream):
    from aimu.models.fallback import FallbackClient

    seen: list = []
    primary = _fake_client(_Model(levels=True), seen)

    result = getattr(FallbackClient([primary]), method)("hi", thinking="low", stream=stream)
    if stream:
        list(result)

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="low")


def test_fallback_forwards_thinking_after_a_failover():
    from aimu.models.fallback import FallbackClient

    seen: list = []

    class _Boom:
        model = _Model(levels=True)
        messages: list = []
        tools: list = []
        system_message = None

        def reset(self, system_message="__keep__"):
            pass

        def chat(self, *a, **kw):
            raise RuntimeError("primary down")

    backup = _fake_client(_Model(levels=True), seen)

    FallbackClient([_Boom(), backup]).chat("hi", thinking="high")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="high")


def test_hf_never_passes_the_reserved_key_to_transformers_generate():
    """The pop and the generate() splat are adjacent lines; nothing else pins their order.

    Transformers raises on an unused model_kwarg, so a leak here is loud rather than silent,
    but the other three providers all have a wire-leak test and this path had none.
    """
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    seen: list[dict] = []

    class _Inputs(dict):
        input_ids = [[0, 1, 2]]

        def to(self, device):
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return "rendered"

        def __call__(self, *a, **kw):
            return _Inputs()

        def decode(self, *a, **kw):
            return "answer"

    def generate(**kw):
        seen.append(kw)
        return [[0, 1, 2, 3]]

    client = types.SimpleNamespace(
        model=HuggingFaceModel.QWEN_3_8_27B,
        _hf_processor=None,
        _hf_tokenizer=_Tokenizer(),
        _hf_model=types.SimpleNamespace(device="cpu", generate=generate),
        _uses_processor_parse_response=False,
        last_thinking=None,
        _pending_thinking_tokens=[],
        _parsed_tool_calls=None,
        MODELS=HuggingFaceModel,
    )
    # bind the real template method so the pop-then-splat ordering is the code under test
    client._apply_chat_template = HuggingFaceClient._apply_chat_template.__get__(client, type(client))
    client._record_request = lambda *a, **k: None
    kwargs = {"max_new_tokens": 16, THINKING_KWARG: ResolvedThinking(enabled=True, level="high")}

    HuggingFaceClient._generate_sync(client, [{"role": "user", "content": "hi"}], kwargs, None)

    assert THINKING_KWARG not in seen[0]
    assert seen[0]["max_new_tokens"] == 16


# ---------------------------------------------------------------------------
# Agent-level threading: thinking= reaches every chat() the agent loop makes
# ---------------------------------------------------------------------------


class _RecordingClient(MockModelClient):
    """A thinking-capable mock client that records the generate_kwargs of every turn.

    The agent threads the public ``thinking=`` argument, so each turn's ``_thinking`` entry is
    what the client's own ``chat()`` resolved. Recording the dict per turn is therefore the way
    to prove the request reached *every* round, not just the first.
    """

    def __init__(self, responses: list):
        super().__init__(responses)
        self.model.value = "mock-thinker"
        self.model.supports_thinking = True
        self.model.thinking_levels = True
        self.model.thinking_optional = True
        self.model.supports_structured_output = False
        self.model.supports_vision = False
        self.model.supports_audio = False
        self.seen: list[dict] = []

    def _chat(self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        # Record only the leaf (non-streaming) call: the streamed path re-enters _chat through
        # _chat_streamed, so recording both would double-count every streamed turn.
        if not stream:
            self.seen.append(dict(generate_kwargs or {}))
        return super()._chat(user_message, generate_kwargs, use_tools, stream, images, audio)

    def thinking_per_turn(self) -> list:
        return [kwargs.get(THINKING_KWARG) for kwargs in self.seen]


@tool
def ping() -> str:
    """Answer a ping."""
    return "pong"


@dataclass
class _Verdict:
    passed: bool


def test_agent_run_thinking_reaches_every_turn_of_the_tool_loop():
    client = _RecordingClient(["tool", "final answer"])
    agent = Agent(client, tools=[ping])

    agent.run("q", thinking="high")

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="high")] * 2


def test_agent_run_thinking_reaches_the_forced_wrap_up_turn():
    client = _RecordingClient(["tool", "tool", "wrapped up"])
    agent = Agent(client, tools=[ping], max_iterations=2)

    agent.run("q", thinking="high")

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="high")] * 3


def test_agent_thinking_field_is_the_default_for_every_run():
    client = _RecordingClient(["answer"])
    agent = Agent(client, thinking="low")

    agent.run("q")

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="low")]


def test_agent_run_thinking_overrides_the_field():
    client = _RecordingClient(["answer"])
    agent = Agent(client, thinking="low")

    agent.run("q", thinking="high")

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="high")]


def test_agent_run_thinking_false_overrides_a_field_level():
    """``False`` is a real request, so the override cannot be an ``or``-style truthiness test."""
    client = _RecordingClient(["answer"])
    agent = Agent(client, thinking="high")

    agent.run("q", thinking=False)

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=False, level=None)]


def test_agent_run_without_thinking_leaves_the_request_untouched():
    client = _RecordingClient(["answer"])
    agent = Agent(client)

    agent.run("q")

    assert client.thinking_per_turn() == [None]


def test_agent_run_thinking_reaches_the_structured_turn():
    client = _RecordingClient(['{"passed": true}'])
    agent = Agent(client)

    verdict = agent.run("q", thinking="low", schema=_Verdict)

    assert verdict.passed is True
    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="low")]


def test_agent_run_thinking_reaches_every_turn_of_the_streamed_loop():
    client = _RecordingClient(["tool", "final answer"])
    agent = Agent(client, tools=[ping])

    list(agent.run("q", stream=True, thinking="high"))

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="high")] * 2


def test_agent_run_thinking_reaches_the_streamed_structured_turn():
    client = _RecordingClient(['{"passed": true}'])
    agent = Agent(client)

    list(agent.run("q", stream=True, thinking="medium", schema=_Verdict))

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="medium")]


def test_agentic_view_thinking_loses_to_the_agents_field():
    """Pins the precedence the how-to documents for ``agent.as_model_client()``.

    The view's ``chat()`` resolves its own ``thinking=`` into the request, but the agent then
    applies its field to the inner turns it drives, so the field wins. Not a regression guard on
    a choice, a guard on a documented consequence of the field being applied per turn.
    """
    client = _RecordingClient(["answer"])
    agent = Agent(client, thinking="high")

    agent.as_model_client().chat("q", thinking="low")

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="high")]


def test_agentic_view_thinking_applies_when_the_agent_has_no_field():
    client = _RecordingClient(["answer"])
    agent = Agent(client)

    agent.as_model_client().chat("q", thinking="low")

    assert client.thinking_per_turn() == [ResolvedThinking(enabled=True, level="low")]
