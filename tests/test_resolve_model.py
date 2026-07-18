"""Tests for resolve_model: enum-match vs ad-hoc, base_url rules, flag/error handling."""

from __future__ import annotations

import pytest

from aimu.models import HAS_OPENAI_COMPAT
from aimu.models._base.text import AdHocModel
from aimu.models.model_client import resolve_model

pytestmark = pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")


def test_known_id_no_url():
    r = resolve_model("llamaserver:qwen3-8b.gguf")
    assert not isinstance(r.model, AdHocModel)
    assert r.model.value == "qwen3-8b.gguf"
    assert r.base_url is None
    assert r.provider == "llamaserver"


def test_known_id_with_url_override():
    r = resolve_model("llamaserver:qwen3-8b.gguf@http://gpu-box:8080/v1")
    assert not isinstance(r.model, AdHocModel)
    assert r.base_url == "http://gpu-box:8080/v1"


def test_adhoc_id_under_known_provider():
    r = resolve_model("llamaserver:my-finetune.gguf@http://gpu-box:8080/v1;tools,thinking")
    assert isinstance(r.model, AdHocModel)
    assert r.model.value == "my-finetune.gguf"
    assert r.model.supports_tools is True
    assert r.model.supports_thinking is True
    assert r.base_url == "http://gpu-box:8080/v1"


def test_adhoc_id_defaults_capabilities_false():
    r = resolve_model("llamaserver:custom.gguf")
    assert isinstance(r.model, AdHocModel)
    assert r.model.supports_tools is False


def test_generic_prefix_requires_url():
    with pytest.raises(ValueError, match="requires an endpoint"):
        resolve_model("openai-compat:anything")


def test_generic_prefix_adhoc():
    r = resolve_model("openai-compat:anything@http://gpu-box:9000/v1;tools")
    assert isinstance(r.model, AdHocModel)
    assert r.provider == "openai-compat"
    assert r.base_url == "http://gpu-box:9000/v1"
    assert r.model.supports_tools is True


def test_flags_with_known_id_error():
    with pytest.raises(ValueError, match="not allowed with the known model id"):
        resolve_model("llamaserver:qwen3-8b.gguf@http://h:8080/v1;tools")


def test_unknown_flag_error():
    with pytest.raises(ValueError, match="Unknown capability flag"):
        resolve_model("openai-compat:m@http://h/v1;bogus")


def test_base_url_on_non_compat_provider_error():
    with pytest.raises(ValueError, match="does not accept an @base_url"):
        resolve_model("anthropic:claude-sonnet-4-6@http://proxy/v1")


def test_unknown_id_on_non_compat_provider_error():
    with pytest.raises(ValueError, match="has no model id"):
        resolve_model("anthropic:not-a-real-model")
