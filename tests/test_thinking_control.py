"""Mock-only tests for the portable `thinking=` control surface.

Covers the capability declaration on ModelSpec, argument resolution, warning behavior, and
the per-provider wire translation. No server and no model weights required.
"""

from __future__ import annotations

import types

import ollama
import pytest

import aimu.models.providers.ollama as sync_ollama
from aimu.models._internal.thinking import (
    THINKING_KWARG,
    THINKING_LEVELS,
    ResolvedThinking,
    resolve_thinking,
    select_profile,
)
from aimu.models.base import AdHocModel, Model, ModelSpec
from aimu.models.providers.ollama import OllamaModel


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


def test_profile_selection_prefers_the_instruct_profile_when_off():
    model = _Model(nonthinking={"temperature": 0.7, "presence_penalty": 1.5})

    off = select_profile(model, ResolvedThinking(enabled=False))
    on = select_profile(model, ResolvedThinking(enabled=True))

    assert off == {"temperature": 0.7, "presence_penalty": 1.5}
    assert on == {"temperature": 1.0, "presence_penalty": 0.0}


def test_profile_selection_falls_back_when_no_instruct_profile_exists():
    model = _Model()

    assert select_profile(model, ResolvedThinking(enabled=False)) == model.generation_kwargs
    assert select_profile(model, None) == model.generation_kwargs


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

        def _update_generate_kwargs(self, generate_kwargs=None):
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
    return client._thinking_kwargs(client._update_generate_kwargs(kwargs))


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

    client = types.SimpleNamespace(model=HuggingFaceModel.QWEN_3_8_27B)

    merged = HuggingFaceClient._update_generate_kwargs(client, {THINKING_KWARG: ResolvedThinking(enabled=False)})

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
