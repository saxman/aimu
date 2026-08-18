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


def test_merge_does_not_mutate_the_shared_catalog_profile(ollama_client):
    """The profile lives on an ENUM MEMBER, so corrupting it would leak across clients.

    `model.generation_kwargs` is process-global: every client of that model reads the same dict.
    Ollama's merge reads it through `select_profile`, which copies, and the old code returned
    `self.default_generate_kwargs` itself and then `pop`ped `max_tokens` out of it, corrupting
    the client's profile after one call. Asserting on `default_generate_kwargs` no longer
    guards anything, since this path stopped reading it once per-mode profile selection landed.

    This pins the property, not one line: it holds while EITHER defense stands (select_profile
    copying, or the merge building a fresh dict), and fails when both are removed.
    """
    profile = OllamaModel.QWEN_3_5_9B.generation_kwargs

    ollama_client._update_generate_kwargs({"temperature": 0.2, "max_tokens": 2000})
    ollama_client._update_generate_kwargs({"temperature": 0.3})

    assert profile["temperature"] == 1.0
    assert "max_tokens" not in profile
    assert "num_predict" not in profile


def test_merge_does_not_mutate_the_shared_profile_on_the_async_client(monkeypatch):
    from aimu import aio

    monkeypatch.setattr(ollama, "AsyncClient", lambda **kw: types.SimpleNamespace())
    client = aio.client("ollama:qwen3.5:9b")
    profile = OllamaModel.QWEN_3_5_9B.generation_kwargs

    client._client._update_generate_kwargs({"max_tokens": 2000})

    assert profile["temperature"] == 1.0
    assert "num_predict" not in profile


def test_merge_does_not_mutate_the_shared_profile_on_the_hf_client():
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    client = types.SimpleNamespace(model=HuggingFaceModel.QWEN_3_5_9B)
    profile = HuggingFaceModel.QWEN_3_5_9B.generation_kwargs

    HuggingFaceClient._update_generate_kwargs(client, {"max_tokens": 2000})

    assert profile["temperature"] == 1.0
    assert "max_new_tokens" not in profile


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


def test_ollama_reports_the_model_profile_as_its_defaults(ollama_client):
    """`default_generate_kwargs` is mirrored onto ModelClient, _AgenticView and the in-process
    wrappers, so it is a real read surface. For Ollama the defaults come from the model rather
    than a class constant, so reporting an empty dict here would claim there are none."""
    assert ollama_client.default_generate_kwargs["temperature"] == 1.0
    assert ollama_client.default_generate_kwargs["top_p"] == 0.95


def test_ollama_defaults_are_a_copy_callers_cannot_corrupt(ollama_client):
    ollama_client.default_generate_kwargs["temperature"] = 99

    assert OllamaModel.QWEN_3_5_9B.generation_kwargs["temperature"] == 1.0
