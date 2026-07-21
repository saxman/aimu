"""A server-unreachable failure surfaces as ModelConnectionError, with the root cause preserved.

No network I/O: the SDK's ``chat.completions.create`` is monkeypatched to raise the same
``openai.APIConnectionError`` the SDK raises when the endpoint is down.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from aimu.aio import AsyncModelClient
from aimu.models import HAS_OPENAI_COMPAT, ModelClient, ModelConnectionError

pytestmark = pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")

_URL = "http://gpu-box:8080/v1"
_ROOT_CAUSE = "[Errno 61] Connection refused"


def _raise_connection_error(*args, **kwargs):
    request = httpx.Request("POST", _URL + "/chat/completions")
    raise openai.APIConnectionError(request=request) from httpx.ConnectError(_ROOT_CAUSE)


async def _araise_connection_error(*args, **kwargs):
    _raise_connection_error()


def _client_with_dead_endpoint() -> ModelClient:
    c = ModelClient(f"llamaserver:custom.gguf@{_URL};tools")
    c._client._client.chat.completions.create = _raise_connection_error
    return c


def _async_client_with_dead_endpoint() -> AsyncModelClient:
    c = AsyncModelClient(f"llamaserver:custom.gguf@{_URL};tools")
    c._client._client.chat.completions.create = _araise_connection_error
    return c


def _assert_wrapped(exc: ModelConnectionError) -> None:
    assert isinstance(exc.__cause__, openai.APIConnectionError)
    # The specific OS reason lives one link deeper, on the SDK error's own cause.
    assert _ROOT_CAUSE in str(exc.__cause__.__cause__)


def test_generate_wraps_connection_error():
    c = _client_with_dead_endpoint()
    with pytest.raises(ModelConnectionError) as info:
        c._client._generate("hi")
    _assert_wrapped(info.value)


def test_chat_wraps_connection_error():
    c = _client_with_dead_endpoint()
    with pytest.raises(ModelConnectionError) as info:
        c._client._chat("hi")
    _assert_wrapped(info.value)


def test_streamed_generate_wraps_connection_error():
    c = _client_with_dead_endpoint()
    with pytest.raises(ModelConnectionError) as info:
        list(c._client._generate("hi", stream=True))
    _assert_wrapped(info.value)


async def test_async_generate_wraps_connection_error():
    c = _async_client_with_dead_endpoint()
    with pytest.raises(ModelConnectionError) as info:
        await c._client._generate("hi")
    _assert_wrapped(info.value)


async def test_async_streamed_generate_wraps_connection_error():
    c = _async_client_with_dead_endpoint()
    with pytest.raises(ModelConnectionError) as info:
        stream = await c._client._generate("hi", stream=True)
        async for _ in stream:
            pass
    _assert_wrapped(info.value)
