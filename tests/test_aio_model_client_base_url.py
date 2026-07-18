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
