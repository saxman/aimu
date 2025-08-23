"""
Model client tests that can run with different client types.

Usage:
- Run all tests with all models: pytest tests/test_models.py
- Run tests with only Ollama models: pytest tests/test_models.py --client=ollama
- Run tests with only HuggingFace models: pytest tests/test_models.py --client=hf
"""

import pytest
from typing import Iterable
from fastmcp import FastMCP

from aimu.models import ModelClient, HuggingFaceClient, OllamaClient
from aimu.tools.client import MCPClient


def pytest_generate_tests(metafunc):
    """Generate tests based on command line options."""

    if "model_client" in metafunc.fixturenames:
        client_type = metafunc.config.getoption("--client", "all").lower()

        if client_type == "ollama":
            test_models = OllamaClient.MODELS
        elif client_type in ["hf", "huggingface"]:
            test_models = HuggingFaceClient.MODELS
        else:
            test_models = list(OllamaClient.MODELS) + list(HuggingFaceClient.MODELS)

        metafunc.parametrize("model_client", test_models, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[ModelClient]:
    """Create model client based on the model parameter."""

    model = request.param

    if model in OllamaClient.MODELS:
        client = OllamaClient(model, system_message="You are a helpful assistant.", model_keep_alive_seconds=2)
    elif model in HuggingFaceClient.MODELS:
        client = HuggingFaceClient(model, system_message="You are a helpful assistant.")
    else:
        raise ValueError(f"Unknown model: {model}")

    yield client

    del client


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
    content = next(response)  # type: ignore
    assert isinstance(content, str)

    for response_part in response:
        content += response_part

    assert "Paris" in content
    assert isinstance(content, str)


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


def test_chat_streamed(model_client):
    """Test that the model can chat with a user in a streamed manner, maintaining a conversation history."""

    model_client.messages = []

    response = model_client.chat_streamed("What is the capital of France?")

    assert isinstance(response, Iterable)
    content = next(response)  # type: ignore
    assert isinstance(content, str)

    for response_part in response:
        content += response_part

    assert "Paris" in content
    assert len(model_client.messages) == 3  # system (auto-added), user, assistant


def test_chat_with_tools(model_client):
    """Test that the model can use tools to answer questions from the user."""

    # ensure the chat history is reset and set system message for tools
    model_client.messages = []
    model_client.system_message = "You are a helpful assistant that uses tools to answer questions from the user."

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.mcp_client = mcp_client

    response = model_client.chat("What is the temperature in Paris?")

    # If the model does not support tools, we shouldn't see tools being used
    if model_client.model not in model_client.TOOL_MODELS:
        assert len(model_client.messages) == 3  # system (auto-added), user, assistant
        return

    assert len(model_client.messages) == 5  # system (auto-added), user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in response

    if model_client.model in model_client.THINKING_MODELS:
        # If the model supports thinking, we should have a thinking messages in the last message and in the tool call
        # assert "thinking" in model_client.messages[-1] ## GPT OSS does not seem to add thinking to the last message
        assert "thinking" in model_client.messages[-3]


def test_chat_streamed_with_tools(model_client):
    """Test that the model can use tools to answer questions from the user in a streamed manner."""

    # ensure the chat history is reset and set system message for tools
    model_client.messages = []
    model_client.system_message = "You are a helpful assistant that uses tools to answer questions from the user."

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.mcp_client = mcp_client

    response = model_client.chat_streamed("What is the temperature in Paris?")

    content = ""
    for response_part in response:
        content += response_part

    # If the model does not support tools, we shouldn't see tools being used
    if model_client.model not in model_client.TOOL_MODELS:
        assert len(model_client.messages) == 3  # system (auto-added), user, assistant
        return

    assert len(model_client.messages) == 5  # system (auto-added), user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in content

    if model_client.model in model_client.THINKING_MODELS:
        # If the model supports thinking, we should have a thinking messages in the last message and in the tool call
        # assert "thinking" in model_client.messages[-1] ## GPT OSS does not seem to add thinking to the last message
        assert "thinking" in model_client.messages[-3]
