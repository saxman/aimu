"""
Model client tests that can run with different client types.

Usage:
- Run all tests with all models: pytest tests/test_models.py
- Run tests with only Ollama models: pytest tests/test_models.py --client=ollama
- Run tests with only HuggingFace models: pytest tests/test_models.py --client=hf
- Run tests with only OpenAI-compatible models: pytest tests/test_models.py --client=openai_compat
- Run tests with only LM Studio models: pytest tests/test_models.py --client=lmstudio_openai
- Run tests with only Ollama OpenAI models: pytest tests/test_models.py --client=ollama_openai
- Run tests with only HuggingFace Transformers Serve models: pytest tests/test_models.py --client=hf_openai
"""

import pytest
from typing import Iterable
from fastmcp import FastMCP

from helpers import create_real_model_client, resolve_model_params
from aimu.models import BaseModelClient, StreamingContentType, StreamChunk
from aimu.tools.client import MCPClient


# A multi-step reasoning prompt used by the thinking tests. It reliably elicits thinking
# from both forced-thinking models (budget_tokens always thinks) and adaptive models
# (Anthropic Opus 4.7+/Fable 5), which only think when they judge a task non-trivial and so
# skip thinking on a factual prompt like "capital of France". The thinking tests assert
# thinking emission, not answer correctness (that is covered by test_generate/test_chat),
# so the computed result is intentionally not checked -- keeping the tests robust across
# smaller local thinking models that may not compute it exactly.
_THINKING_PROMPT = (
    "Compute the sum of all integers from 1 to 200 that are divisible by 7. "
    "Show your work, then state the final total."
)


def pytest_generate_tests(metafunc):
    if "model_client" not in metafunc.fixturenames:
        return
    # default_params=None → full cross-client list when --client=all
    params = resolve_model_params(metafunc.config, default_params=None)
    metafunc.parametrize("model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[BaseModelClient]:
    yield from create_real_model_client(request)


def test_class_properties(model_client):
    """Test that the model client is a subclass of BaseModelClient."""

    assert issubclass(model_client.__class__, BaseModelClient)
    assert isinstance(model_client, BaseModelClient)

    assert model_client.default_generate_kwargs is not None


def test_generate(model_client):
    """Test that the model can generate a response to a prompt."""

    prompt = "What is the capital of France?"
    response = model_client.generate(prompt)

    assert isinstance(response, str)
    assert "Paris" in response


def test_generate_streamed(model_client):
    """Test that the model can generate a response to a prompt in a streamed manner."""

    prompt = "What is the capital of France?"
    response = model_client.generate(prompt, stream=True)

    assert isinstance(response, Iterable)

    content = ""
    for chunk in response:
        assert isinstance(chunk, StreamChunk)
        if chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert "Paris" in content


def test_generate_with_parameters(model_client):
    """Test that the model can generate a response to a prompt with specific generation parameters."""

    prompt = "What is the capital of France?"
    response = model_client.generate(prompt, generate_kwargs={"max_tokens": 1000})
    # TODO validate the response length in tokens

    assert isinstance(response, str)
    assert "Paris" in response


def test_chat(model_client):
    """Test that the model can chat with a user, maintaining a conversation history."""

    model_client.messages = []

    response = model_client.chat("What is the capital of France?")

    assert "Paris" in response
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant


def test_chat_multiple_turns(model_client):
    """Test that the model can chat with a user over multiple turns, maintaining a conversation history."""

    model_client.messages = []

    response1 = model_client.chat("What is the capital of France?")
    assert "Paris" in response1
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant

    response2 = model_client.chat("What is the population of that city?").lower()
    assert "population" in response2 or "inhabitants" in response2
    assert len(model_client.messages) == 5  # system (auto-added), user, assistant, user, assistant

    response3 = model_client.chat("What is the climate like there?").lower()
    assert "climate" in response3 or "temperature" in response3
    assert len(model_client.messages) == 7  # system (auto-added), user, assistant, user, assistant, user, assistant


def test_chat_streamed(model_client):
    """Test that the model can chat with a user in a streamed manner, maintaining a conversation history."""

    model_client.messages = []

    response = model_client.chat("What is the capital of France?", stream=True)

    assert isinstance(response, Iterable)

    content = ""
    for chunk in response:
        if chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert "Paris" in content
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant


def test_chat_streamed_multiple_turns(model_client):
    """Test that the model can chat with a user over multiple turns in a streamed manner, maintaining a conversation history."""

    model_client.messages = []

    # Generous budget: thinking models (e.g. Qwen3.5) reason before answering, and
    # HF's default max_new_tokens=1024 can truncate a follow-up turn mid-thinking
    # (before </think>), leaving an empty answer. Matches the budget used by the
    # dedicated thinking-streaming tests.
    kwargs = {"max_tokens": 4096}

    content1 = ""
    for chunk in model_client.chat("What is the capital of France?", generate_kwargs=kwargs, stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            content1 += chunk.content
    assert "Paris" in content1
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant

    content2 = ""
    for chunk in model_client.chat("What is the population of that city?", generate_kwargs=kwargs, stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            content2 += chunk.content
    assert "population" in content2.lower() or "inhabitants" in content2.lower()
    assert len(model_client.messages) == 5  # system (auto-added), user, assistant, user, assistant

    content3 = ""
    for chunk in model_client.chat("What is the climate like there?", generate_kwargs=kwargs, stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            content3 += chunk.content
    assert "climate" in content3.lower() or "temperature" in content3.lower()
    assert len(model_client.messages) == 7  # system (auto-added), user, assistant, user, assistant, user, assistant


def test_chat_with_tools(model_client):
    """Test that the model can use tools to answer questions from the user."""

    # ensure the chat history is reset and set system message for tools
    model_client.messages = []
    model_client.system_message = "You are a helpful assistant that uses tools to answer questions from the user."

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        """Return the current temperature in Celsius for a given location."""
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.tools = mcp_client.as_tools()

    response = model_client.chat("What is the temperature in Paris?")

    # If the model does not support tools, we shouldn't see tools being used
    if not model_client.model.supports_tools:
        assert len(model_client.messages) == 3  # system (auto-added), user, assistant
        return

    assert len(model_client.messages) == 5  # system (auto-added), user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in response

    # If the model supports thinking, we should have a thinking messages in the last message and in the tool call
    if model_client.model.supports_thinking:
        is_gemma = model_client.model.name.startswith("GEMMA")
        is_gpt_oss = model_client.model.name.startswith("GPT_OSS")
        is_magistral = model_client.model.name.startswith("MAGISTRAL")
        is_smollm = model_client.model.name.startswith("SMOLLM")
        # Adaptive-thinking models (Anthropic Opus 4.7+/Fable 5) decide per request whether
        # to think and skip it on a simple tool prompt like this, so thinking isn't guaranteed.
        style = getattr(model_client.model, "thinking_style", None)
        is_adaptive = style is not None and style.name == "ADAPTIVE"

        # these models do not consistently include thinking in the final response
        if not (is_gemma or is_gpt_oss or is_magistral or is_smollm or is_adaptive):
            assert "thinking" in model_client.messages[-1]

        # these models do not consistently include thinking in the tool call response
        if not (is_gemma or is_magistral or is_smollm or is_adaptive):
            assert "thinking" in model_client.messages[-3]


def test_generate_streamed_thinking(model_client):
    """Test that thinking models populate last_thinking after streamed generation."""

    if not model_client.model.supports_thinking:
        pytest.skip("Model does not support thinking")
    if model_client.model.name.startswith("GEMMA"):
        pytest.skip("Gemma models do not consistently include thinking in responses")

    content = ""
    thinking = ""
    # Generous budget: this prompt induces long reasoning, and HF's default
    # max_new_tokens=1024 truncates an 8B model mid-thinking (before </think>),
    # which leaves last_thinking unset and leaks the reasoning into GENERATING.
    for chunk in model_client.generate(_THINKING_PROMPT, stream=True, generate_kwargs={"max_tokens": 4096}):
        if chunk.phase == StreamingContentType.THINKING:
            thinking += chunk.content
        elif chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert model_client.last_thinking, "last_thinking should be populated for thinking models"
    assert thinking, "thinking chunks should be yielded for thinking models"
    assert content, "generation should produce non-empty output"


def test_generate_streamed_exclude_thinking(model_client):
    """Test that thinking chunks are excluded when include= omits THINKING."""

    if not model_client.model.supports_thinking:
        pytest.skip("Model does not support thinking")
    if model_client.model.name.startswith("GEMMA"):
        pytest.skip("Gemma models do not consistently include thinking in responses")

    content = ""
    thinking = ""
    stream = model_client.generate(
        _THINKING_PROMPT,
        stream=True,
        include=["generating", "tool_calling", "done"],
        generate_kwargs={"max_tokens": 4096},
    )
    for chunk in stream:
        if chunk.phase == StreamingContentType.THINKING:
            thinking += chunk.content
        elif chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    assert not thinking, "thinking chunks should not be yielded when THINKING excluded"
    assert model_client.last_thinking, "last_thinking should still be populated even when THINKING filtered"
    assert content, "generation should produce non-empty output"


def test_chat_streamed_with_tools(model_client):
    """Test that the model can use tools to answer questions from the user in a streamed manner."""

    # ensure the chat history is reset and set system message for tools
    model_client.messages = []
    model_client.system_message = "You are a helpful assistant that uses tools to answer questions from the user."

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        """Return the current temperature in Celsius for a given location."""
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.tools = mcp_client.as_tools()

    response = model_client.chat("What is the temperature in Paris?", stream=True)

    content = ""
    for chunk in response:
        if chunk.phase == StreamingContentType.GENERATING:
            content += chunk.content

    # If the model does not support tools, we shouldn't see tools being used
    if not model_client.model.supports_tools:
        assert len(model_client.messages) == 3  # system (auto-added), user, assistant
        return

    assert len(model_client.messages) == 5  # system (auto-added), user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in content

    # If the model supports thinking, we should have a thinking messages in the last message and in the tool call
    if model_client.model.supports_thinking:
        is_gemma = model_client.model.name.startswith("GEMMA")
        is_gpt_oss = model_client.model.name.startswith("GPT_OSS")
        is_magistral = model_client.model.name.startswith("MAGISTRAL")
        is_smollm = model_client.model.name.startswith("SMOLLM")
        # Adaptive-thinking models (Anthropic Opus 4.7+/Fable 5) decide per request whether
        # to think and skip it on a simple tool prompt like this, so thinking isn't guaranteed.
        style = getattr(model_client.model, "thinking_style", None)
        is_adaptive = style is not None and style.name == "ADAPTIVE"

        # these models do not consistently include thinking in the final response
        if not (is_gemma or is_gpt_oss or is_magistral or is_smollm or is_adaptive):
            assert "thinking" in model_client.messages[-1]

        # these models do not consistently include thinking in the tool call response
        if not (is_gemma or is_magistral or is_smollm or is_adaptive):
            assert "thinking" in model_client.messages[-3]
