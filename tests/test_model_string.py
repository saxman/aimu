"""Tests for the extended model-string parser (provider:model_id[@base_url][;flags])."""

from __future__ import annotations

import pytest

from aimu.models._internal.model_string import parse_model_string


def test_plain_provider_id():
    p = parse_model_string("anthropic:claude-sonnet-4-6")
    assert (p.provider, p.model_id, p.base_url, p.flags) == ("anthropic", "claude-sonnet-4-6", None, ())


def test_colon_in_model_id_preserved():
    p = parse_model_string("ollama:qwen3.5:9b")
    assert p.provider == "ollama"
    assert p.model_id == "qwen3.5:9b"
    assert p.base_url is None


def test_base_url_with_port_and_path():
    p = parse_model_string("llamaserver:qwen3-8b.gguf@http://gpu-box:8080/v1")
    assert p.model_id == "qwen3-8b.gguf"
    assert p.base_url == "http://gpu-box:8080/v1"
    assert p.flags == ()


def test_base_url_with_userinfo_at_sign():
    p = parse_model_string("openai-compat:m@http://user:pass@host:9000/v1")
    assert p.model_id == "m"
    assert p.base_url == "http://user:pass@host:9000/v1"


def test_flags_parsed_and_stripped():
    p = parse_model_string("llamaserver:m@http://h:8080/v1;tools, thinking")
    assert p.base_url == "http://h:8080/v1"
    assert p.flags == ("tools", "thinking")


def test_flags_without_base_url():
    p = parse_model_string("llamaserver:m;tools")
    assert p.base_url is None
    assert p.flags == ("tools",)


def test_empty_flag_segments_dropped():
    p = parse_model_string("openai-compat:m@http://h/v1;tools,,")
    assert p.flags == ("tools",)


def test_missing_colon_raises():
    with pytest.raises(ValueError, match="provider:model_id"):
        parse_model_string("just-an-id")
