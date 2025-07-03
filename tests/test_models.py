import pytest
from typing import Iterable
from fastmcp import FastMCP

from aimu.models import ModelClient, HuggingFaceClient, OllamaClient
from aimu.tools import MCPClient

TEST_MODELS = [
    HuggingFaceClient.MODEL_LLAMA_3_1_8B,
    HuggingFaceClient.MODEL_MISTRAL_7B,
    OllamaClient.MODEL_LLAMA_3_2_3B,
    OllamaClient.MODEL_MISTRAL_7B,
]


@pytest.fixture(params=TEST_MODELS)
def model_client(request) -> Iterable[ModelClient]:
    if ":" in request.param:
        client = OllamaClient(request.param)
    else:
        client = HuggingFaceClient(request.param)

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
    response = model_client.generate(prompt, generate_kwargs={"max_tokens": 10})

    assert isinstance(response, str)
    assert "Paris" in response


def test_chat(model_client):
    message = {"role": model_client.system_role, "content": "You are a helpful assistant."}
    response = model_client.chat(message)

    assert len(model_client.messages) == 2

    message = {"role": "user", "content": "What is the capital of France?"}
    response = model_client.chat(message)

    assert "Paris" in response
    assert len(model_client.messages) == 4


def test_chat_streamed(model_client):
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
    message = {"role": model_client.system_role, "content": "You are a helpful assistant."}
    response = model_client.chat(message)

    assert len(model_client.messages) == 2

    mcp = FastMCP("Test Tools")

    @mcp.tool()
    def temperature(location: str) -> float:
        return 27.0

    mcp_client = MCPClient(server=mcp)
    model_client.mcp_client = mcp_client

    message = {
        "role": "user",
        "content": "What is the temperature in Paris?"
    }
    response = model_client.chat(message, tools=mcp_client.get_tools())

    assert "27" in response
    assert len(model_client.messages) == 6 # 1 system, 1 user, 2 assistant, 2 tool messages
    assert model_client.messages[-2]["role"] == "tool" # second to last message should be tool response
