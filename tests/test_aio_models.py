"""Live-backend async model tests. Mirrors ``tests/test_models.py`` for ``aimu.aio``.

Usage:
- All async-supported models:    pytest tests/test_aio_models.py
- Just Ollama:                   pytest tests/test_aio_models.py --client=ollama
- A specific model:              pytest tests/test_aio_models.py --client=ollama --model=QWEN_3_8B
"""

from __future__ import annotations

from typing import AsyncIterable

import pytest
from fastmcp import FastMCP

from aimu.aio import MCPClient
from aimu.aio._base import AsyncBaseModelClient
from aimu.models import StreamChunk, StreamingContentType
from helpers_aio import create_real_async_model_client, resolve_async_model_params

# A multi-step reasoning prompt so adaptive-thinking models (e.g. Anthropic Opus 4.7+,
# Fable 5) actually engage thinking — they may emit none on trivial prompts. Mirrors
# ``_THINKING_PROMPT`` in tests/test_models.py.
_THINKING_PROMPT = (
    "Compute the sum of all integers from 1 to 200 that are divisible by 7. "
    "Show your work, then state the final total."
)


def pytest_generate_tests(metafunc):
    if "async_model_client" not in metafunc.fixturenames:
        return
    params = resolve_async_model_params(metafunc.config, default_params=None)
    metafunc.parametrize("async_model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def async_model_client(request):
    yield create_real_async_model_client(request)


# ---------------------------------------------------------------------------
# Class wiring
# ---------------------------------------------------------------------------


async def test_class_properties(async_model_client):
    assert isinstance(async_model_client, AsyncBaseModelClient)
    assert async_model_client.default_generate_kwargs is not None


# ---------------------------------------------------------------------------
# generate / generate(stream=True)
# ---------------------------------------------------------------------------


async def test_generate(async_model_client):
    response = await async_model_client.generate("What is the capital of France?")
    assert isinstance(response, str)
    assert "Paris" in response


async def test_generate_streamed(async_model_client):
    stream = await async_model_client.generate("What is the capital of France?", stream=True)
    assert isinstance(stream, AsyncIterable)

    content = ""
    async for chunk in stream:
        assert isinstance(chunk, StreamChunk)
        if chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content
    assert "Paris" in content


async def test_generate_with_parameters(async_model_client):
    response = await async_model_client.generate("What is the capital of France?", generate_kwargs={"max_tokens": 1000})
    assert isinstance(response, str)
    assert "Paris" in response


# ---------------------------------------------------------------------------
# chat / chat(stream=True), single + multi-turn
# ---------------------------------------------------------------------------


async def test_chat(async_model_client):
    async_model_client.messages = []
    response = await async_model_client.chat("What is the capital of France?")
    assert "Paris" in response
    assert len(async_model_client.messages) == 3  # system, user, assistant


async def test_chat_multiple_turns(async_model_client):
    async_model_client.messages = []

    response1 = await async_model_client.chat("What is the capital of France?")
    assert "Paris" in response1
    assert len(async_model_client.messages) == 3

    response2 = (await async_model_client.chat("What is the population of that city?")).lower()
    assert "population" in response2 or "inhabitants" in response2
    assert len(async_model_client.messages) == 5

    response3 = (await async_model_client.chat("What is the climate like there?")).lower()
    assert "climate" in response3 or "temperature" in response3
    assert len(async_model_client.messages) == 7


async def test_chat_streamed(async_model_client):
    async_model_client.messages = []
    stream = await async_model_client.chat("What is the capital of France?", stream=True)
    assert isinstance(stream, AsyncIterable)

    content = ""
    async for chunk in stream:
        if chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert "Paris" in content
    assert len(async_model_client.messages) == 3


async def test_chat_streamed_multiple_turns(async_model_client):
    async_model_client.messages = []

    content1 = ""
    async for chunk in await async_model_client.chat("What is the capital of France?", stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            content1 += chunk.content
    assert "Paris" in content1
    assert len(async_model_client.messages) == 3

    content2 = ""
    async for chunk in await async_model_client.chat("What is the population of that city?", stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            content2 += chunk.content
    assert "population" in content2.lower() or "inhabitants" in content2.lower()
    assert len(async_model_client.messages) == 5

    content3 = ""
    async for chunk in await async_model_client.chat("What is the climate like there?", stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            content3 += chunk.content
    assert "climate" in content3.lower() or "temperature" in content3.lower()
    assert len(async_model_client.messages) == 7


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------


async def test_generate_streamed_thinking(async_model_client):
    """Thinking models populate last_thinking and yield THINKING chunks."""
    if not async_model_client.model.supports_thinking:
        pytest.skip("Model does not support thinking")
    if async_model_client.model.name.startswith("GEMMA"):
        pytest.skip("Gemma models do not consistently include thinking in responses")

    content = ""
    thinking = ""
    async for chunk in await async_model_client.generate(_THINKING_PROMPT, stream=True):
        if chunk.phase == StreamingContentType.THINKING:
            thinking += chunk.content
        elif chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert async_model_client.last_thinking, "last_thinking should be populated for thinking models"
    assert thinking, "thinking chunks should be yielded for thinking models"
    assert content, "generation should produce non-empty output"


async def test_generate_streamed_exclude_thinking(async_model_client):
    """include= without THINKING should drop THINKING chunks while keeping last_thinking."""
    if not async_model_client.model.supports_thinking:
        pytest.skip("Model does not support thinking")
    if async_model_client.model.name.startswith("GEMMA"):
        pytest.skip("Gemma models do not consistently include thinking in responses")

    content = ""
    thinking = ""
    stream = await async_model_client.generate(
        _THINKING_PROMPT,
        stream=True,
        include=["generating", "tool_calling", "done"],
    )
    async for chunk in stream:
        if chunk.phase == StreamingContentType.THINKING:
            thinking += chunk.content
        elif chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert not thinking, "thinking chunks should not be yielded when THINKING excluded"
    assert async_model_client.last_thinking, "last_thinking should still be populated even when THINKING filtered"
    assert content, "generation should produce non-empty output"


# ---------------------------------------------------------------------------
# Tool calling — via aio.MCPClient
# ---------------------------------------------------------------------------


async def test_chat_with_tools(async_model_client):
    # reset() clears messages and sets the system_message in one call.
    async_model_client.reset(
        system_message="You are a helpful assistant that uses tools to answer questions from the user."
    )

    mcp_server = FastMCP("Test Tools")

    @mcp_server.tool()
    def temperature(location: str) -> float:
        """Return the current temperature in Celsius for a given location."""
        return 27.0

    mcp = await MCPClient.connect(server=mcp_server)
    try:
        async_model_client.tools = await mcp.as_tools()

        response = await async_model_client.chat("What is the temperature in Paris?")

        if not async_model_client.model.supports_tools:
            assert len(async_model_client.messages) == 3
            return

        assert len(async_model_client.messages) == 5  # system, user, tool_call, tool, assistant
        assert async_model_client.messages[-2]["role"] == "tool"
        assert "27" in response
    finally:
        await mcp.aclose()


async def test_chat_streamed_with_tools(async_model_client):
    async_model_client.reset(
        system_message="You are a helpful assistant that uses tools to answer questions from the user."
    )

    mcp_server = FastMCP("Test Tools")

    @mcp_server.tool()
    def temperature(location: str) -> float:
        """Return the current temperature in Celsius for a given location."""
        return 27.0

    mcp = await MCPClient.connect(server=mcp_server)
    try:
        async_model_client.tools = await mcp.as_tools()

        content = ""
        async for chunk in await async_model_client.chat("What is the temperature in Paris?", stream=True):
            if chunk.phase == StreamingContentType.GENERATING:
                content += chunk.content

        if not async_model_client.model.supports_tools:
            assert len(async_model_client.messages) == 3
            return

        assert len(async_model_client.messages) == 5
        assert async_model_client.messages[-2]["role"] == "tool"
        assert "27" in content
    finally:
        await mcp.aclose()
