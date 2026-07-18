"""Tests for AdHocModel, the Model-like wrapper for ids not in any provider enum."""

from aimu.models._base.text import AdHocModel, ModelSpec


def test_adhoc_model_mirrors_spec():
    m = AdHocModel(ModelSpec(id="my-model.gguf", tools=True, thinking=True))
    assert m.value == "my-model.gguf"
    assert m.name == "my-model.gguf"
    assert m.spec.id == "my-model.gguf"
    assert m.supports_tools is True
    assert m.supports_thinking is True
    assert m.supports_vision is False
    assert m.supports_audio is False
    assert m.supports_structured_output is False
    assert m.generation_kwargs == {}
