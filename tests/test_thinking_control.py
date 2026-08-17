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
