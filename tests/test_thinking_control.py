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
