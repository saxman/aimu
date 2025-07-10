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
from aimu.tools import MCPClient

OLLAMA_TEST_MODELS = [
    OllamaClient.MODEL_LLAMA_3_2_3B,
    OllamaClient.MODEL_MISTRAL_SMALL_3_2_24B,
    OllamaClient.MODEL_QWEN_3_8B,
    OllamaClient.MODEL_PHI_4_14B,
    OllamaClient.MODEL_DEEPSEEK_R1_8B,
]

HUGGINGFACE_TEST_MODELS = [
    HuggingFaceClient.MODEL_LLAMA_3_1_8B,
    HuggingFaceClient.MODEL_MISTRAL_7B,
    HuggingFaceClient.MODEL_QWEN_3_8B,
    HuggingFaceClient.MODEL_PHI_4_MINI_3_8B,
]


def pytest_generate_tests(metafunc):
    """Generate tests based on command line options."""
    if "model_client" in metafunc.fixturenames:
        client_type = metafunc.config.getoption("--client", "all").lower()

        if client_type == "ollama":
            test_models = OLLAMA_TEST_MODELS
        elif client_type in ["hf", "huggingface"]:
            test_models = HUGGINGFACE_TEST_MODELS
        else:
            test_models = OLLAMA_TEST_MODELS + HUGGINGFACE_TEST_MODELS

        metafunc.parametrize("model_client", test_models, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[ModelClient]:
    """Create model client based on the model parameter."""
    model_id = request.param

    if ":" in model_id:
        client = OllamaClient(model_id)
    else:
        client = HuggingFaceClient(model_id)

    yield client

    del client


def test_generate(model_client):
    prompt = "What is the capital of France?"
    response = model_client.generate(prompt)

    assert isinstance(response, str)
    assert "Paris" in response


def test_generate_streamed(model_client):
    prompt = "What is the capital of France?"
    response = model_client.generate_streamed(prompt)

    assert isinstance(response, Iterable)
    content = next(response)
    assert isinstance(content, str)

    for response_part in response:
        content += response_part

    assert "Paris" in content
    assert isinstance(content, str)


def test_generate_with_parameters(model_client):
    prompt = "What is the capital of France?"
    response = model_client.generate(prompt, generate_kwargs={"max_tokens": 1000})
    # TODO validate the response length in tokens

    assert isinstance(response, str)
    assert "Paris" in response


def test_chat(model_client):
    # ensure the chat history is reset
    model_client.messages = []

    message = {"role": model_client.system_role, "content": "You are a helpful assistant."}
    response = model_client.chat(message)

    assert len(model_client.messages) == 2

    message = {"role": "user", "content": "What is the capital of France?"}
    response = model_client.chat(message)

    assert "Paris" in response
    assert len(model_client.messages) == 4


def test_chat_streamed(model_client):
    # ensure the chat history is reset
    model_client.messages = []

    message = {"role": model_client.system_role, "content": "You are a helpful assistant."}
    response = model_client.chat(message)

    assert len(model_client.messages) == 2

    message = {"role": "user", "content": "What is the capital of France?"}
    response = model_client.chat_streamed(message)

    assert isinstance(response, Iterable)
    content = next(response)
    assert isinstance(content, str)

    for response_part in response:
        content += response_part

    assert "Paris" in content
    assert len(model_client.messages) == 4


def test_chat_with_tools(model_client):
    # ensure the chat history is reset
    model_client.messages = []

    message = {
        "role": model_client.system_role,
        "content": "You are a helpful assistant that uses tools to answer questions from the user.",
    }
    response = model_client.chat(message)

    assert len(model_client.messages) == 2

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.mcp_client = mcp_client

    message = {"role": "user", "content": "What is the temperature in Paris?"}

    # If the model does not support tools, we should see an error
    if model_client.model_id not in model_client.TOOL_MODELS:
        with pytest.raises(ValueError):
            model_client.chat(message, tools=mcp_client.get_tools())
        return

    response = model_client.chat(message, tools=mcp_client.get_tools())

    assert len(model_client.messages) == 6  # system, assistant, user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in response


def test_chat_streamed_with_tools(model_client):
    # ensure the chat history is reset
    model_client.messages = []

    message = {
        "role": model_client.system_role,
        "content": "You are a helpful assistant that uses tools to answer questions from the user.",
    }
    response = model_client.chat_streamed(message)

    content = ""
    for response_part in response:
        content += response_part

    assert len(model_client.messages) == 2

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.mcp_client = mcp_client

    message = {"role": "user", "content": "What is the temperature in Paris?"}

    # If the model does not support tools, we should see an error
    if model_client.model_id not in model_client.TOOL_MODELS:
        with pytest.raises(ValueError):
            model_client.chat(message, tools=mcp_client.get_tools())
        return

    response = model_client.chat(message, tools=mcp_client.get_tools())

    for response_part in response:
        content += response_part

    assert len(model_client.messages) == 6  # system, assistant, user, tool call, tool response, assistant
    assert model_client.messages[-2]["role"] == "tool"  # second to last message should be tool response
    assert "27" in content


def test_chat_with_tools_from_system_message(model_client):
    """
    Test that tools can be used when they are referenced in the system message.
    """

    if "llama" in model_client.model_id or "mistral" in model_client.model_id:
        # Llama and Mistral models require a system message to be sent before using tools
        pytest.skip("Llama and Mistral models require a system message to be sent before using tools.")

    # ensure the chat history is reset
    model_client.messages = []

    message = {
        "role": model_client.system_role,
        "content": "You are a helpful assistant that uses tools to answer questions from the user. Tell the user what the temperature is in Paris.",
    }

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.mcp_client = mcp_client

    # If the model does not support tools, we should see an error
    if model_client.model_id not in model_client.TOOL_MODELS:
        with pytest.raises(ValueError):
            model_client.chat(message, tools=mcp_client.get_tools())
        return

    response = model_client.chat(message, tools=mcp_client.get_tools())

    assert "27" in response


def test_thiking(model_client):
    """
    Test that the model can think before responding.
    """

    if model_client.model_id not in model_client.THINKING_MODELS:
        return

    # ensure the chat history is reset
    model_client.messages = []

    message = {"role": model_client.system_role, "content": "You are a helpful assistant."}
    response = model_client.chat(message)

    assert len(model_client.messages) == 2

    message = {"role": "user", "content": "What is the capital of France?"}
    response = model_client.chat(message)

    assert "Paris" in response
    assert len(model_client.messages) == 4
