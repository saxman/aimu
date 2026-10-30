"""Which models the chat apps offer, and which one they pick on their own.

The apps' automatic default must never land on a client whose constructor downloads weights: with
Ollama installed but empty, falling through to the next installed client picked HuggingFaceClient,
whose constructor started a multi-gigabyte download at app start (the failure issue #9 removed from
Ollama itself).
"""

from __future__ import annotations

import enum

from model_choice import default_client_and_model, model_key, model_options


class _OllamaModel(enum.Enum):
    QWEN_3_5_9B = "qwen3.5:9b"
    QWEN_3_8B = "qwen3:8b"


class _HFModel(enum.Enum):
    QWEN_3_8_27B = "Qwen/Qwen3.8-27B"


class _CloudModel(enum.Enum):
    CLAUDE = "claude-sonnet-5"


def _client(name, models):
    return type(name, (), {"TOOL_MODELS": list(models)})


OllamaClient = _client("OllamaClient", _OllamaModel)
HuggingFaceClient = _client("HuggingFaceClient", _HFModel)
LlamaCppClient = _client("LlamaCppClient", _HFModel)
AnthropicClient = _client("AnthropicClient", _CloudModel)


def test_ollama_offers_only_installed_models():
    installed = {model_key(_OllamaModel.QWEN_3_8B)}
    assert model_options(OllamaClient, installed, ollama_client=OllamaClient) == [_OllamaModel.QWEN_3_8B]


def test_other_clients_offer_their_catalog():
    assert model_options(AnthropicClient, set(), ollama_client=OllamaClient) == [_CloudModel.CLAUDE]


def test_default_prefers_the_preferred_installed_ollama_model():
    installed = {model_key(m) for m in _OllamaModel}
    choice = default_client_and_model(
        [OllamaClient, HuggingFaceClient], installed, ollama_client=OllamaClient, preferred=_OllamaModel.QWEN_3_5_9B
    )
    assert choice == (OllamaClient, _OllamaModel.QWEN_3_5_9B)


def test_empty_ollama_never_falls_back_to_a_downloading_client():
    choice = default_client_and_model(
        [OllamaClient, HuggingFaceClient, LlamaCppClient], set(), ollama_client=OllamaClient
    )
    assert choice == (None, None)


def test_empty_ollama_falls_back_to_a_client_that_downloads_nothing():
    choice = default_client_and_model(
        [OllamaClient, HuggingFaceClient, AnthropicClient], set(), ollama_client=OllamaClient
    )
    assert choice == (AnthropicClient, _CloudModel.CLAUDE)
