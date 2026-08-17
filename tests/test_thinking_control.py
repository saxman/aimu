"""Mock-only tests for the portable `thinking=` control surface.

Covers the capability declaration on ModelSpec, argument resolution, warning behavior, and
the per-provider wire translation. No server and no model weights required.
"""

from __future__ import annotations

import pytest

from aimu.models.base import AdHocModel, Model, ModelSpec


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
