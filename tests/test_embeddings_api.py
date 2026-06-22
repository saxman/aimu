"""Mock-only unit tests for the embedding-client surface.

Covers spec equality, the factory + string resolver, OpenAI / Ollama provider
``_embed`` with stubbed SDKs (no network), the ``embed()`` single-vs-list
normalization, ``SemanticMemoryStore(embedding_client=...)`` wiring, and the
top-level ``aimu.embedding_client()`` / ``aimu.embed()`` dispatch.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from aimu.models import (
    EmbeddingSpec,
    OpenAIEmbeddingSpec,
    resolve_embedding_model_string,
)


# ---------------------------------------------------------------------------
# Spec + resolver
# ---------------------------------------------------------------------------


def test_embedding_spec_equality_and_hash_by_id():
    a = EmbeddingSpec("m", dimensions=10)
    b = EmbeddingSpec("m", dimensions=999)
    assert a == b
    assert hash(a) == hash(b)
    assert a != EmbeddingSpec("other")


def test_resolve_embedding_model_string_known():
    from aimu.models import HAS_OPENAI_EMBEDDING

    if not HAS_OPENAI_EMBEDDING:
        pytest.skip("openai not installed")
    member = resolve_embedding_model_string("openai:text-embedding-3-small")
    assert member.value == "text-embedding-3-small"
    assert member.spec.dimensions == 1536


def test_resolve_embedding_model_string_requires_colon():
    with pytest.raises(ValueError, match="provider:model_id"):
        resolve_embedding_model_string("text-embedding-3-small")


def test_resolve_embedding_model_string_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        resolve_embedding_model_string("nope:foo")


# ---------------------------------------------------------------------------
# Factory construction
# ---------------------------------------------------------------------------


def _require_openai():
    from aimu.models import HAS_OPENAI_EMBEDDING

    if not HAS_OPENAI_EMBEDDING:
        pytest.skip("openai not installed")


def test_factory_rejects_bare_spec():
    _require_openai()
    from aimu.models import EmbeddingClient

    with pytest.raises(TypeError, match="EmbeddingModel enum member"):
        EmbeddingClient(OpenAIEmbeddingSpec("text-embedding-3-small"))


def test_factory_unknown_provider_string():
    from aimu.models import EmbeddingClient

    with pytest.raises((ValueError, ImportError)):
        EmbeddingClient("madeup:model")


def test_factory_builds_openai_from_enum_and_string():
    _require_openai()
    from aimu.models import EmbeddingClient, OpenAIEmbeddingModel

    c1 = EmbeddingClient(OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL)
    c2 = EmbeddingClient("openai:text-embedding-3-small")
    assert c1.spec.id == c2.spec.id == "text-embedding-3-small"
    assert c1.dimensions == 1536


# ---------------------------------------------------------------------------
# OpenAI provider _embed + embed() normalization (stubbed SDK)
# ---------------------------------------------------------------------------


def _openai_client():
    _require_openai()
    from aimu.models import OpenAIEmbeddingModel
    from aimu.models.providers.openai.embedding import OpenAIEmbeddingClient

    client = OpenAIEmbeddingClient(OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL)
    # Replace the real SDK object with a fake namespace; touching the real client's lazy
    # `.embeddings` attr imports `openai.resources.*`, which sibling mock tests stub.
    client._client = SimpleNamespace(embeddings=SimpleNamespace(create=None))
    return client


def test_openai_embed_single_returns_one_vector(monkeypatch):
    client = _openai_client()
    monkeypatch.setattr(
        client._client.embeddings,
        "create",
        lambda **_: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]),
    )
    out = client.embed("hello")
    assert out == [0.1, 0.2, 0.3]


def test_openai_embed_list_returns_list_of_vectors(monkeypatch):
    client = _openai_client()
    monkeypatch.setattr(
        client._client.embeddings,
        "create",
        lambda **_: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0]), SimpleNamespace(embedding=[2.0])]),
    )
    out = client.embed(["a", "b"])
    assert out == [[1.0], [2.0]]


def test_embed_empty_list_returns_empty():
    client = _openai_client()
    assert client.embed([]) == []


def test_embed_rejects_non_string_items():
    client = _openai_client()
    with pytest.raises(ValueError, match="string or a list of strings"):
        client.embed([1, 2, 3])


# ---------------------------------------------------------------------------
# Ollama provider (stubbed ollama module)
# ---------------------------------------------------------------------------


def test_ollama_embed(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.models.providers import ollama as ollama_module
    from aimu.models.providers.ollama import OllamaEmbeddingClient, OllamaEmbeddingModel

    monkeypatch.setattr(ollama_module.ollama, "pull", lambda *_a, **_k: None)
    monkeypatch.setattr(ollama_module.ollama, "embed", lambda **_: {"embeddings": [[0.5, 0.6], [0.7, 0.8]]})
    client = OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT)
    assert client.embed(["x", "y"]) == [[0.5, 0.6], [0.7, 0.8]]
    assert client.embed("x") == [0.5, 0.6]


# ---------------------------------------------------------------------------
# SemanticMemoryStore wiring
# ---------------------------------------------------------------------------


def test_semantic_store_uses_provided_embedding_client():
    from aimu.memory.semantic_store import SemanticMemoryStore

    calls = []

    class FakeEmbeddingClient:
        def embed(self, texts):
            calls.append(list(texts))
            return [[float(len(t)), 1.0, 0.0] for t in texts]

    store = SemanticMemoryStore(collection_name="api_test_custom", embedding_client=FakeEmbeddingClient())
    store.store("Paul works at Google")
    assert store.search("work", n_results=1)  # non-empty
    assert calls  # the custom client was invoked for embedding


def test_semantic_store_default_embedding_unchanged():
    from aimu.memory.semantic_store import SemanticMemoryStore

    store = SemanticMemoryStore(collection_name="api_test_default")
    store.store("hello world")
    assert store.search("hello", n_results=1) == ["hello world"]


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def test_top_level_embed_dispatch(monkeypatch):
    _require_openai()
    import aimu

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def fake_create(**_):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[9.0])])

    # Replace the real SDK client with a fake namespace so we never touch openai's lazy
    # `.embeddings` import (stubbed by sibling tests in a full-suite run).
    from aimu.models.providers.openai import embedding as emb_mod

    real_init = emb_mod.OpenAIEmbeddingClient.__init__

    def patched_init(self, model, model_kwargs=None):
        real_init(self, model, model_kwargs)
        self._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

    monkeypatch.setattr(emb_mod.OpenAIEmbeddingClient, "__init__", patched_init)

    out = aimu.embed("hi", model="openai:text-embedding-3-small")
    assert out == [9.0]


# ---------------------------------------------------------------------------
# HuggingFace provider (stubbed sentence-transformers, works whether or not it's installed)
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_embedding_module(monkeypatch):
    """Stub ``sentence_transformers`` and load the HF embedding provider module fresh.

    Lets the provider's logic be tested without the real (large) dependency installed.
    """
    import numpy as np

    class _FakeSentenceTransformer:
        def __init__(self, model_id, device=None, **kwargs):
            self.model_id = model_id
            self.device = device
            self.kwargs = kwargs

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
            _FakeSentenceTransformer.last_normalize = normalize_embeddings
            return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype="float32")

    st_stub = ModuleType("sentence_transformers")
    st_stub.SentenceTransformer = _FakeSentenceTransformer
    st_stub._aimu_stub = True
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)
    monkeypatch.delitem(sys.modules, "aimu.models.providers.hf.embedding", raising=False)

    module = importlib.import_module("aimu.models.providers.hf.embedding")
    module._model_registry.clear()
    return module, _FakeSentenceTransformer


def test_hf_construction_from_enum_string_spec(hf_embedding_module):
    module, _ = hf_embedding_module
    from aimu.models import HuggingFaceEmbeddingSpec

    c_enum = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.BGE_SMALL_EN_V1_5)
    c_str = module.HuggingFaceEmbeddingClient("hf:BAAI/bge-small-en-v1.5")
    c_spec = module.HuggingFaceEmbeddingClient(HuggingFaceEmbeddingSpec("BAAI/bge-small-en-v1.5", dimensions=384))
    assert c_enum.spec.id == c_str.spec.id == c_spec.spec.id == "BAAI/bge-small-en-v1.5"
    assert c_enum.dimensions == 384


def test_hf_unknown_string_raises(hf_embedding_module):
    module, _ = hf_embedding_module
    with pytest.raises(ValueError, match="Unknown HuggingFace embedding model id"):
        module.HuggingFaceEmbeddingClient("hf:some/unknown-repo")


def test_hf_embed_single_and_list(hf_embedding_module):
    module, _ = hf_embedding_module
    client = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.ALL_MINILM_L6_V2)
    assert client.embed("hello") == [5.0, 1.0, 0.0]
    out = client.embed(["a", "bb"])
    assert out == [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]


def test_hf_embed_normalizes_by_default(hf_embedding_module):
    module, fake = hf_embedding_module
    client = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.BGE_BASE_EN_V1_5)
    client.embed(["x"])
    assert fake.last_normalize is True
    client.embed(["x"], normalize_embeddings=False)
    assert fake.last_normalize is False


def test_hf_lazy_load_and_weight_cache(hf_embedding_module):
    module, _ = hf_embedding_module
    c1 = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.ALL_MINILM_L6_V2)
    assert c1._model is None  # not loaded until first embed
    c1.embed("trigger load")
    assert c1._model is not None
    c2 = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.ALL_MINILM_L6_V2)
    c2.embed("again")
    assert c2._model is c1._model  # shared via module-level registry
