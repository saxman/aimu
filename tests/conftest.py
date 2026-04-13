"""
Pytest configuration for model tests.
"""

import time
from typing import Iterator

import pytest

from aimu.models import (
    HAS_ANTHROPIC,
    HAS_LLAMACPP,
    HAS_OPENAI_COMPAT,
    HFOpenAIClient,
    HuggingFaceClient,
    LMStudioOpenAIClient,
    LlamaCppClient,
    ModelClient,
    OllamaClient,
    OllamaOpenAIClient,
    VLLMOpenAIClient,
)

if HAS_LLAMACPP:
    from aimu.models.llamacpp import LlamaCppModel

if HAS_ANTHROPIC:
    from aimu.models import AnthropicClient

if HAS_OPENAI_COMPAT:
    from aimu.models import OpenAIClient, GeminiClient


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--client",
        action="store",
        default="all",
        help=(
            "Client type to test: 'ollama', 'hf'/'huggingface', "
            "'openai', 'anthropic', 'gemini', or 'all' (default: all)"
        ),
    )

    parser.addoption(
        "--model",
        action="store",
        default="all",
        help="Model name to test (default: all)",
    )

    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to a local GGUF model file (required for --client=llamacpp)",
    )


# ---------------------------------------------------------------------------
# Shared parametrization helpers
# ---------------------------------------------------------------------------


def _resolve_client(client_type: str, config):
    """Return the client class for a given --client value (must not be 'all')."""
    if client_type in ["hf", "huggingface"]:
        return HuggingFaceClient
    if client_type == "openai":
        if not HAS_OPENAI_COMPAT:
            pytest.skip("openai package not installed")
        return OpenAIClient
    if client_type == "anthropic":
        if not HAS_ANTHROPIC:
            pytest.skip("anthropic package not installed")
        return AnthropicClient
    if client_type == "gemini":
        if not HAS_OPENAI_COMPAT:
            pytest.skip("openai package not installed")
        return GeminiClient
    if client_type == "openai_compat":
        from aimu.models import OpenAICompatClient

        return OpenAICompatClient
    if client_type == "lmstudio_openai":
        return LMStudioOpenAIClient
    if client_type == "ollama_openai":
        return OllamaOpenAIClient
    if client_type == "hf_openai":
        return HFOpenAIClient
    if client_type == "vllm_openai":
        return VLLMOpenAIClient
    if client_type == "llamacpp":
        if not HAS_LLAMACPP:
            pytest.skip("llama-cpp-python not installed")
        return LlamaCppClient
    return OllamaClient  # default


def resolve_model_params(config, default_params=None) -> list:
    """
    Build the list of params for parametrizing the model_client fixture.

    Args:
        config:         pytest config object (from metafunc.config).
        default_params: Params to use when --client=all. Pass ``["mock"]`` for
                        files that want a mock by default; pass ``None`` to get
                        the full cross-client model list (test_models.py).

    Returns:
        List of model enum values or sentinel strings.
    """
    client_type = config.getoption("--client", "all").lower()
    model = config.getoption("--model", "all")

    if client_type == "all":
        if default_params is not None:
            return default_params
        # Full cross-client list (test_models.py behaviour)
        if model == "all":
            params = (
                list(OllamaClient.MODELS)
                + list(HuggingFaceClient.MODELS)
                + (list(AnthropicClient.MODELS) if HAS_ANTHROPIC else [])
                + list(LMStudioOpenAIClient.MODELS)
                + list(OllamaOpenAIClient.MODELS)
                + list(HFOpenAIClient.MODELS)
                + list(VLLMOpenAIClient.MODELS)
            )
            if HAS_OPENAI_COMPAT:
                params += list(OpenAIClient.MODELS) + list(GeminiClient.MODELS)
            if HAS_LLAMACPP:
                params += list(LlamaCppModel)
            return params
        # Search for named model across all clients
        extra_clients = []
        if HAS_ANTHROPIC:
            extra_clients.append(AnthropicClient)
        if HAS_OPENAI_COMPAT:
            extra_clients.extend([OpenAIClient, GeminiClient])

        params = []
        for client in (
            OllamaClient,
            HuggingFaceClient,
            *extra_clients,
            LMStudioOpenAIClient,
            OllamaOpenAIClient,
            HFOpenAIClient,
            VLLMOpenAIClient,
        ):
            if model in client.MODELS:
                params.append(client.MODELS[model])
        return params

    client = _resolve_client(client_type, config)
    if model == "all":
        return list(client.MODELS)
    return [client.MODELS[model]]


def create_real_model_client(request) -> Iterator[ModelClient]:
    """
    Dispatch request.param (a model enum value) to the correct client class,
    yield the constructed client, then delete it.

    Intended for use inside a ``model_client`` fixture after the mock branch
    has been handled.
    """
    model = request.param

    if model in OllamaClient.MODELS:
        time.sleep(2)  # give Ollama time to free memory from the prior model
        client = OllamaClient(model, system_message="You are a helpful assistant.", model_keep_alive_seconds=2)
    elif model in HuggingFaceClient.MODELS:
        client = HuggingFaceClient(model, system_message="You are a helpful assistant.")
    elif HAS_ANTHROPIC and model in AnthropicClient.MODELS:
        client = AnthropicClient(model, system_message="You are a helpful assistant.")
    elif HAS_OPENAI_COMPAT and model in OpenAIClient.MODELS:
        client = OpenAIClient(model, system_message="You are a helpful assistant.")
    elif HAS_OPENAI_COMPAT and model in GeminiClient.MODELS:
        client = GeminiClient(model, system_message="You are a helpful assistant.")
    elif model in LMStudioOpenAIClient.MODELS:
        client = LMStudioOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in OllamaOpenAIClient.MODELS:
        client = OllamaOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in HFOpenAIClient.MODELS:
        client = HFOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in VLLMOpenAIClient.MODELS:
        client = VLLMOpenAIClient(model, system_message="You are a helpful assistant.")
    elif HAS_LLAMACPP and model in LlamaCppModel:
        model_path = request.config.getoption("--model-path")
        if not model_path:
            pytest.skip("--model-path is required for llamacpp client tests")
        client = LlamaCppClient(model, model_path=model_path, system_message="You are a helpful assistant.")
    else:
        raise ValueError(f"Unknown model: {model}")

    yield client
    del client
