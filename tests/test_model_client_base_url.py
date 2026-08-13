"""ModelClient builds clients from extended model strings without network I/O."""

from __future__ import annotations

import pytest

from aimu.models import HAS_OPENAI_COMPAT, ModelClient

pytestmark = pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")


def _openai_base_url(client: ModelClient) -> str:
    # ModelClient -> concrete OpenAICompatClient (._client) -> openai.OpenAI (._client)
    return str(client._client._client.base_url)


def test_known_id_base_url_override():
    c = ModelClient("llamaserver:qwen3-8b.gguf@http://gpu-box:8080/v1")
    assert c.model.value == "qwen3-8b.gguf"
    assert "gpu-box:8080" in _openai_base_url(c)


def test_adhoc_known_provider_capabilities_and_url():
    c = ModelClient("llamaserver:my-finetune.gguf@http://gpu-box:8080/v1;tools,thinking")
    assert c.model.value == "my-finetune.gguf"
    assert c.model.supports_tools is True
    assert c.model.supports_thinking is True
    assert "gpu-box:8080" in _openai_base_url(c)


def test_generic_prefix_builds_compat_client():
    c = ModelClient("openai-compat:whatever@http://gpu-box:9000/v1;tools")
    assert c.model.value == "whatever"
    assert c.model.supports_tools is True
    assert "gpu-box:9000" in _openai_base_url(c)


def test_default_localhost_when_no_url():
    c = ModelClient("llamaserver:custom.gguf;tools")
    assert "localhost:8080" in _openai_base_url(c)


# oMLX (MLX inference on Apple Silicon). These three double as the sync wiring canary for the
# provider: together they prove the guarded import, the _provider_registry entry, the isinstance
# dispatch branch, and _BASE_URL_PROVIDERS membership all landed.


def test_omlx_known_id_default_localhost():
    c = ModelClient("omlx:Qwen3.6-35B-A3B-4bit")
    assert c.model.value == "Qwen3.6-35B-A3B-4bit"
    assert c.model.supports_vision is True
    assert "localhost:8000" in _openai_base_url(c)


def test_omlx_base_url_override():
    # The canonical oMLX deployment is a headless Mac on the LAN driven from a laptop.
    c = ModelClient("omlx:Qwen3.6-35B-A3B-4bit@http://mac-studio:8000/v1")
    assert "mac-studio:8000" in _openai_base_url(c)


def test_omlx_adhoc_directory_id():
    # oMLX ids are user-chosen --model-dir subdirectory names, so the ad-hoc form is a primary
    # way in rather than an escape hatch. It only works because "omlx" is in _BASE_URL_PROVIDERS.
    c = ModelClient("omlx:my-own-conversion-4bit;tools,thinking,vision")
    assert c.model.value == "my-own-conversion-4bit"
    assert c.model.supports_tools is True
    assert c.model.supports_vision is True
    assert "localhost:8000" in _openai_base_url(c)
