"""AsyncModelClient builds clients from extended model strings without network I/O."""

from __future__ import annotations

import pytest

from aimu.aio import AsyncModelClient
from aimu.models import HAS_OPENAI_COMPAT

pytestmark = pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")


def _openai_base_url(client: AsyncModelClient) -> str:
    return str(client._client._client.base_url)


def test_known_id_base_url_override():
    c = AsyncModelClient("llamaserver:qwen3-8b.gguf@http://gpu-box:8080/v1")
    assert c.model.value == "qwen3-8b.gguf"
    assert "gpu-box:8080" in _openai_base_url(c)


def test_adhoc_known_provider():
    c = AsyncModelClient("llamaserver:my-finetune.gguf@http://gpu-box:8080/v1;tools")
    assert c.model.value == "my-finetune.gguf"
    assert c.model.supports_tools is True
    assert "gpu-box:8080" in _openai_base_url(c)


def test_generic_prefix():
    c = AsyncModelClient("openai-compat:whatever@http://gpu-box:9000/v1;tools,thinking")
    assert c.model.supports_thinking is True
    assert "gpu-box:9000" in _openai_base_url(c)


# oMLX. It is a *network* server, so unlike HuggingFace/LlamaCpp it constructs directly from an
# enum or string rather than wrapping a sync client.


def test_omlx_known_id_default_localhost():
    c = AsyncModelClient("omlx:Qwen3.6-35B-A3B-4bit")
    assert c.model.value == "Qwen3.6-35B-A3B-4bit"
    assert c.model.supports_vision is True
    assert "localhost:8000" in _openai_base_url(c)


def test_omlx_adhoc_directory_id():
    # The async ad-hoc path routes by provider prefix through the hand-maintained
    # _ASYNC_COMPAT_CLIENTS dict (the sync side reads _provider_registry() and cannot have this
    # gap). A missing "omlx" entry there raises KeyError, so this is that entry's only guard.
    c = AsyncModelClient("omlx:my-own-conversion-4bit;tools,thinking,vision")
    assert c.model.value == "my-own-conversion-4bit"
    assert c.model.supports_vision is True
    assert "localhost:8000" in _openai_base_url(c)


def test_omlx_base_url_override():
    c = AsyncModelClient("omlx:Qwen3.6-35B-A3B-4bit@http://mac-studio:8000/v1")
    assert "mac-studio:8000" in _openai_base_url(c)
