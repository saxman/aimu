"""Naming a deprecated or retired model raises ``ModelDeprecatedError``, on every route.

The routes are a model string, an enum attribute, an enum subscript, a value lookup, a bare
member name, and the client factories built on them. The registry is exercised through a small
fake catalog so the status wording does not depend on today's date, and through the real
Anthropic table so the wiring is proven where it is used.
"""

from __future__ import annotations

from datetime import date

import pytest

import aimu
from aimu.models import HAS_ANTHROPIC, ModelDeprecatedError
from aimu.models._internal.deprecations import DeprecatedModel, deprecation_error, register_deprecated
from aimu.models.base import Model, ModelSpec


class FakeModel(Model):
    LIVE = ModelSpec("live-1")


register_deprecated(
    FakeModel,
    (
        DeprecatedModel("OLD_SOON", ("old-soon",), "LIVE", deprecated_on=date(2026, 1, 1), retires_on=date(2999, 1, 1)),
        DeprecatedModel("OLD_GONE", ("old-gone",), "LIVE", deprecated_on=date(2020, 1, 1), retires_on=date(2020, 6, 1)),
        DeprecatedModel("OLD_TBA", ("old-tba",), "LIVE", deprecated_on=date(2026, 1, 1)),
    ),
)


def test_is_a_value_error():
    """The same calls raised ValueError for an unknown id before, so existing handlers stay correct."""
    assert issubclass(ModelDeprecatedError, ValueError)


@pytest.mark.parametrize(
    "lookup",
    [lambda: FakeModel.OLD_SOON, lambda: FakeModel["OLD_SOON"], lambda: FakeModel("old-soon")],
    ids=["attribute", "subscript", "value"],
)
def test_every_enum_route_raises(lookup):
    with pytest.raises(ModelDeprecatedError) as excinfo:
        lookup()

    assert excinfo.value.replacement == "live-1"
    assert excinfo.value.retires_on == date(2999, 1, 1)
    assert "FakeModel.LIVE" in str(excinfo.value)


def test_live_members_and_unknown_names_are_unaffected():
    assert FakeModel.LIVE.value == "live-1"
    assert FakeModel["LIVE"] is FakeModel.LIVE
    with pytest.raises(AttributeError):
        FakeModel.NEVER_EXISTED
    with pytest.raises(KeyError):
        FakeModel["NEVER_EXISTED"]
    with pytest.raises(ValueError) as excinfo:
        FakeModel("never-existed")
    assert not isinstance(excinfo.value, ModelDeprecatedError)


@pytest.mark.parametrize(
    "name, wording",
    [
        ("OLD_SOON", "retires on 2999-01-01"),
        ("OLD_GONE", "was retired on 2020-06-01"),
        ("OLD_TBA", "retirement date not yet announced"),
    ],
)
def test_message_states_the_lifecycle(name, wording):
    with pytest.raises(ModelDeprecatedError, match=wording):
        FakeModel[name]


def test_registration_rejects_a_replacement_outside_the_catalog():
    class Other(Model):
        LIVE = ModelSpec("other-live")

    with pytest.raises(ValueError, match="not a member"):
        register_deprecated(Other, (DeprecatedModel("OLD", ("old",), "MISSPELLED", deprecated_on=date(2026, 1, 1)),))


def test_registration_rejects_an_entry_that_is_still_live():
    class Other(Model):
        LIVE = ModelSpec("other-live")
        STILL_HERE = ModelSpec("still-here")

    with pytest.raises(ValueError, match="still a live catalog member"):
        register_deprecated(
            Other, (DeprecatedModel("STILL_HERE", ("still-here",), "LIVE", deprecated_on=date(2026, 1, 1)),)
        )


def test_error_reports_the_primary_id():
    entry = DeprecatedModel("X", ("alias", "alias-20250101"), "LIVE", deprecated_on=date(2026, 1, 1))

    assert deprecation_error(FakeModel, entry, "alias-20250101").model == "alias"


# --------------------------------------------------------------------------------------------
# The Anthropic table
# --------------------------------------------------------------------------------------------

needs_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")


def _anthropic_attribute(name):
    from aimu.models import AnthropicModel

    return getattr(AnthropicModel, name)


@needs_anthropic
@pytest.mark.parametrize(
    "resolve",
    [
        lambda: aimu.resolve_model_string("anthropic:claude-sonnet-4-5"),
        lambda: aimu.resolve_model_string("anthropic:claude-sonnet-4-5-20250929"),
        lambda: aimu.resolve_model_enum("anthropic:claude-sonnet-4-5"),
        lambda: aimu.resolve_model_enum("CLAUDE_SONNET_4_5"),
        lambda: _anthropic_attribute("CLAUDE_SONNET_4_5"),
        lambda: aimu.client("anthropic:claude-sonnet-4-5"),
    ],
    ids=["string", "dated-string", "enum-from-string", "bare-name", "attribute", "client"],
)
def test_sonnet_4_5_raises_on_every_route(resolve):
    with pytest.raises(ModelDeprecatedError) as excinfo:
        resolve()

    assert excinfo.value.replacement == "claude-sonnet-5-5"
    assert excinfo.value.retires_on == date(2026, 11, 30)


@needs_anthropic
def test_async_client_raises_too():
    from aimu import aio

    with pytest.raises(ModelDeprecatedError):
        aio.client("anthropic:claude-opus-4-1")


@needs_anthropic
def test_a_retired_model_names_anthropics_replacement():
    with pytest.raises(ModelDeprecatedError, match="CLAUDE_OPUS_4_8"):
        aimu.resolve_model_string("anthropic:claude-3-opus-20240229")
