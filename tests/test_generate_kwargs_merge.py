"""Mock-only regression tests: a caller's partial generate_kwargs must not discard the
model's sampling profile.

Before this fix, `_update_generate_kwargs` returned the caller's dict verbatim whenever it
was non-empty, so passing one key silently dropped temperature/top_p/top_k/min_p.
"""

from __future__ import annotations

import types

import ollama
import pytest

from aimu.models.providers.ollama import OllamaClient, OllamaModel


@pytest.fixture
def ollama_client(monkeypatch):
    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    return OllamaClient(OllamaModel.QWEN_3_5_9B)


def test_partial_kwargs_keep_the_model_profile(ollama_client):
    merged = ollama_client._update_generate_kwargs({"max_tokens": 2000})

    # the caller's key wins, translated to Ollama's name
    assert merged["num_predict"] == 2000
    # and the profile survives
    assert merged["temperature"] == 1.0
    assert merged["top_p"] == 0.95
    assert merged["top_k"] == 20


def test_caller_overrides_a_profile_key(ollama_client):
    merged = ollama_client._update_generate_kwargs({"temperature": 0.2})

    assert merged["temperature"] == 0.2
    assert merged["top_p"] == 0.95


def test_no_kwargs_yields_the_profile_unchanged(ollama_client):
    merged = ollama_client._update_generate_kwargs()

    assert merged["temperature"] == 1.0
    assert merged["top_p"] == 0.95


def test_merge_does_not_mutate_the_client_profile(ollama_client):
    ollama_client._update_generate_kwargs({"temperature": 0.2})

    assert ollama_client.default_generate_kwargs["temperature"] == 1.0


async def test_aio_ollama_merges(monkeypatch):
    from aimu import aio

    monkeypatch.setattr(ollama, "AsyncClient", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    client = aio.client("ollama:qwen3.5:9b")

    merged = client._client._update_generate_kwargs({"max_tokens": 2000})

    assert merged["num_predict"] == 2000
    assert merged["top_p"] == 0.95


def test_hf_merges_without_loading_weights(monkeypatch):
    """Exercise the merge directly on the unbound method, so no weights load."""
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    client = types.SimpleNamespace(model=HuggingFaceModel.QWEN_3_5_9B)

    merged = HuggingFaceClient._update_generate_kwargs(client, {"temperature": 0.2})

    assert merged["temperature"] == 0.2
    assert merged["top_k"] == 20  # from the Qwen profile
    assert merged["max_new_tokens"] == 4096  # from HF DEFAULT_GENERATE_KWARGS
