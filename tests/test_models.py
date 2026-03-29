"""
Model client tests that can run with different client types.

Usage:
- Run all tests with all models: pytest tests/test_models.py
- Run tests with only Ollama models: pytest tests/test_models.py --client=ollama
- Run tests with only HuggingFace models: pytest tests/test_models.py --client=hf
- Run tests with only Aisuite models: pytest tests/test_models.py --client=aisuite
"""

import pytest
from typing import Iterable
from fastmcp import FastMCP
import time

from aimu.models import ModelClient, HuggingFaceClient, OllamaClient, AisuiteClient, StreamingContentType
from aimu.tools.client import MCPClient


def pytest_generate_tests(metafunc):
    """Generate tests based on command line options."""

    if "model_client" in metafunc.fixturenames:
        model = metafunc.config.getoption("--model", "all")
        client_type = metafunc.config.getoption("--client", "all").lower()

        if client_type in ["hf", "huggingface"]:
            client = HuggingFaceClient
        elif client_type == "aisuite":
            client = AisuiteClient
        else:
            client = OllamaClient  # default

        if model == "all":
            if client_type == "all":
                test_models = list(OllamaClient.MODELS) + list(HuggingFaceClient.MODELS) + list(AisuiteClient.MODELS)
            else:
                test_models = client.MODELS
        else:
            if client_type == "all":
                test_models = []
                if model in OllamaClient.MODELS:
                    test_models.append(OllamaClient.MODELS[model])
                if model in HuggingFaceClient.MODELS:
                    test_models.append(HuggingFaceClient.MODELS[model])
                if model in AisuiteClient.MODELS:
                    test_models.append(AisuiteClient.MODELS[model])
            else:
                test_models = [client.MODELS[model]]

        metafunc.parametrize("model_client", test_models, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[ModelClient]:
    """Create model client based on the model parameter."""

    model = request.param

    if model in OllamaClient.MODELS:
        time.sleep(2)  # give Ollama some time to free memory from the prior model
        client = OllamaClient(model, system_message="You are a helpful assistant.", model_keep_alive_seconds=2)
    elif model in HuggingFaceClient.MODELS:
        client = HuggingFaceClient(model, system_message="You are a helpful assistant.")
    elif model in AisuiteClient.MODELS:
        client = AisuiteClient(model, system_message="You are a helpful assistant.")
    else:
        raise ValueError(f"Unknown model: {model}")

    yield client

    del client


def test_class_properties(model_client):
    """Test that the model client is a subclass of ModelClient."""

    assert issubclass(model_client.__class__, ModelClient)
    assert isinstance(model_client, ModelClient)

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
    response = model_client.generate_streamed(prompt)

    assert isinstance(response, Iterable)

    content = ""
    for chunk in response:
        assert isinstance(chunk, str)
        if model_client.streaming_content_type == StreamingContentType.GENERATING:
            content += chunk

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

    response = model_client.chat_streamed("What is the capital of France?")

    assert isinstance(response, Iterable)

    content = ""
    for chunk in response:
        if model_client.streaming_content_type == StreamingContentType.GENERATING:
            content += chunk

    assert "Paris" in content
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant


def test_chat_streamed_multiple_turns(model_client):
    """Test that the model can chat with a user over multiple turns in a streamed manner, maintaining a conversation history."""

    model_client.messages = []

    content1 = ""
    for chunk in model_client.chat_streamed("What is the capital of France?"):
        if model_client.streaming_content_type == StreamingContentType.GENERATING:
            content1 += chunk
    assert "Paris" in content1
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant

    content2 = ""
    for chunk in model_client.chat_streamed("What is the population of that city?"):
        if model_client.streaming_content_type == StreamingContentType.GENERATING:
            content2 += chunk
    assert "population" in content2.lower() or "inhabitants" in content2.lower()
    assert len(model_client.messages) == 5  # system (auto-added), user, assistant, user, assistant

    content3 = ""
    for chunk in model_client.chat_streamed("What is the climate like there?"):
        if model_client.streaming_content_type == StreamingContentType.GENERATING:
            content3 += chunk
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
    model_client.mcp_client = mcp_client

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
        if (
            model_client.model != OllamaClient.MODELS.GPT_OSS_20B
            and model_client.model != OllamaClient.MODELS.MAGISTRAL_SMALL_24B
        ):
            assert "thinking" in model_client.messages[-1]

        if model_client.model != OllamaClient.MODELS.MAGISTRAL_SMALL_24B:
            assert "thinking" in model_client.messages[-3]


def test_generate_streamed_thinking(model_client):
    """Test that thinking models populate last_thinking after streamed generation."""

    if not model_client.model.supports_thinking:
        pytest.skip("Model does not support thinking")

    content = ""
    thinking = ""
    for chunk in model_client.generate_streamed("What is the capital of France?"):
        if model_client.streaming_content_type == StreamingContentType.THINKING:
            thinking += chunk
        elif model_client.streaming_content_type == StreamingContentType.GENERATING:
            content += chunk

    assert model_client.last_thinking, "last_thinking should be populated for thinking models"
    assert thinking, "thinking chunks should be yielded for thinking models"
    assert "Paris" in content


def test_generate_streamed_include_thinking_false(model_client):
    """Test that thinking chunks are excluded when include_thinking=False."""

    if not model_client.model.supports_thinking:
        pytest.skip("Model does not support thinking")

    content = ""
    thinking = ""
    for chunk in model_client.generate_streamed("What is the capital of France?", include_thinking=False):
        if model_client.streaming_content_type == StreamingContentType.THINKING:
            thinking += chunk
        elif model_client.streaming_content_type == StreamingContentType.GENERATING:
            content += chunk

    assert not thinking, "thinking chunks should not be yielded when include_thinking=False"
    assert model_client.last_thinking, "last_thinking should still be populated even when include_thinking=False"
    assert "Paris" in content


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
    model_client.mcp_client = mcp_client

    response = model_client.chat_streamed("What is the temperature in Paris?")

    content = ""
    for chunk in response:
        if model_client.streaming_content_type == StreamingContentType.GENERATING:
            content += chunk

    # If the model does not support tools, we shouldn't see tools being used
    if not model_client.model.supports_tools:
        assert len(model_client.messages) == 3  # system (auto-added), user, assistant
        return

    assert len(model_client.messages) == 5  # system (auto-added), user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in content

    if model_client.model.supports_thinking:
        if (
            model_client.model != OllamaClient.MODELS.GPT_OSS_20B
            and model_client.model != OllamaClient.MODELS.MAGISTRAL_SMALL_24B
        ):
            assert "thinking" in model_client.messages[-1]

        if model_client.model != OllamaClient.MODELS.MAGISTRAL_SMALL_24B:
            assert "thinking" in model_client.messages[-3]
