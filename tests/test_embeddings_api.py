"""Mock-only unit tests for the embedding-client surface.

Covers spec equality, the factory + string resolver, OpenAI / Ollama provider
``_embed`` with stubbed SDKs (no network), the ``embed()`` single-vs-list
normalization, ``SemanticMemoryStore(embedding_client=...)`` wiring, and the
top-level ``aimu.embedding_client()`` / ``aimu.embed()`` dispatch.
"""

from __future__ import annotations

import importlib
import importlib.machinery
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


def test_resolve_embedding_model_string_uncatalogued_id_points_at_the_client():
    """The pointer is emitted by the shared modality resolver, so every modality gets it."""
    with pytest.raises(ValueError, match="EmbeddingClient"):
        resolve_embedding_model_string("ollama:not-in-the-catalog")


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


class _RecordingEmbeddingClient:
    """A minimal concrete client recording exactly the texts its provider call receives."""

    @staticmethod
    def build(spec):
        from aimu.models import BaseEmbeddingClient

        class Recording(BaseEmbeddingClient):
            def __init__(self):
                super().__init__(model=spec)
                self.spec = spec
                self.sent = []

            def _embed(self, texts, **kwargs):
                self.sent.append(list(texts))
                return [[0.0] for _ in texts]

        return Recording()


def test_input_type_prepends_the_spec_prompts():
    client = _RecordingEmbeddingClient.build(EmbeddingSpec("m", query_prompt="q: ", document_prompt="d: "))
    client.embed("hi", input_type="query")
    client.embed(["a", "b"], input_type="document")
    client.embed("raw")
    assert client.sent == [["q: hi"], ["d: a", "d: b"], ["raw"]]


def test_input_type_is_a_no_op_for_a_side_the_model_has_no_prompt_for():
    client = _RecordingEmbeddingClient.build(EmbeddingSpec("m", query_prompt="q: "))
    client.embed("doc", input_type="document")
    assert client.sent == [["doc"]]


def test_input_type_rejects_an_unknown_value():
    client = _RecordingEmbeddingClient.build(EmbeddingSpec("m", query_prompt="q: "))
    with pytest.raises(ValueError, match="input_type"):
        client.embed("x", input_type="passage")


@pytest.mark.parametrize(
    "member, query_prompt, document_prompt",
    [
        ("E5_LARGE_V2", "query: ", "passage: "),
        ("BGE_SMALL_EN_V1_5", "Represent this sentence for searching relevant passages: ", None),
        ("MXBAI_EMBED_LARGE_V1", "Represent this sentence for searching relevant passages: ", None),
        ("EMBEDDING_GEMMA_2", "task: search result | query: ", "title: none | text: "),
        ("ALL_MINILM_L6_V2", None, None),
    ],
)
def test_hf_catalog_declares_each_model_card_prompt(hf_embedding_module, member, query_prompt, document_prompt):
    module, _ = hf_embedding_module
    spec = module.HuggingFaceEmbeddingModel[member].spec
    assert (spec.query_prompt, spec.document_prompt) == (query_prompt, document_prompt)


def test_hf_input_type_reaches_encode_prefixed(hf_embedding_module):
    module, _ = hf_embedding_module
    client = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.E5_LARGE_V2)
    # The stub encodes each text as [len(text), 1, 0], so the prefix shows up in the length.
    assert client.embed("abc", input_type="query")[0] == len("query: abc")


def test_ollama_nomic_declares_its_search_prefixes():
    from aimu.models.providers.ollama import OllamaEmbeddingModel

    spec = OllamaEmbeddingModel.NOMIC_EMBED_TEXT.spec
    assert (spec.query_prompt, spec.document_prompt) == ("search_query: ", "search_document: ")


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

    class FakeClient:
        def pull(self, *_a, **_k):
            return None

        def embed(self, **_):
            return {"embeddings": [[0.5, 0.6], [0.7, 0.8]]}

    # The embedding client is host-bound (see tests/test_ollama_host.py), so pull and embed go
    # through an `ollama.Client` instance rather than the module-level functions.
    monkeypatch.setattr(ollama_module.ollama, "Client", lambda **_k: FakeClient())
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
        def embed(self, texts, input_type=None):
            calls.append((input_type, list(texts)))
            return [[float(len(t)), 1.0, 0.0] for t in texts]

    store = SemanticMemoryStore(collection_name="api_test_custom", embedding_client=FakeEmbeddingClient())
    store.store("Paul works at Google")
    assert store.search("work", n_results=1)  # non-empty
    assert ("document", ["Paul works at Google"]) in calls
    assert ("query", ["work"]) in calls


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
    st_stub.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
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


def test_hf_embeddinggemma_2_loads_the_text_encoder_only(hf_embedding_module):
    module, _ = hf_embedding_module
    client = module.HuggingFaceEmbeddingClient(module.HuggingFaceEmbeddingModel.EMBEDDING_GEMMA_2)
    assert client.dimensions == 768
    client.embed("trigger load")
    assert client.sentence_transformer.kwargs["config_kwargs"] == {"vision_config": None, "audio_config": None}


def test_hf_caller_model_kwargs_win_over_spec_load_kwargs(hf_embedding_module):
    module, _ = hf_embedding_module
    client = module.HuggingFaceEmbeddingClient(
        module.HuggingFaceEmbeddingModel.EMBEDDING_GEMMA_2, model_kwargs={"config_kwargs": {}}
    )
    client.embed("trigger load")
    assert client.sentence_transformer.kwargs["config_kwargs"] == {}


@pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
@pytest.mark.parametrize("dtype", ["float16", "fp16", "half", "torch.float16"])
def test_hf_rejected_dtype_raises_at_construction(hf_embedding_module, key, dtype):
    module, _ = hf_embedding_module
    with pytest.raises(ValueError, match="float16"):
        module.HuggingFaceEmbeddingClient(
            module.HuggingFaceEmbeddingModel.EMBEDDING_GEMMA_2, model_kwargs={"model_kwargs": {key: dtype}}
        )


def test_hf_rejected_dtype_accepts_a_torch_dtype_object(hf_embedding_module):
    torch = pytest.importorskip("torch")
    module, _ = hf_embedding_module
    with pytest.raises(ValueError, match="float16"):
        module.HuggingFaceEmbeddingClient(
            module.HuggingFaceEmbeddingModel.EMBEDDING_GEMMA_2, model_kwargs={"model_kwargs": {"dtype": torch.float16}}
        )
    module.HuggingFaceEmbeddingClient(
        module.HuggingFaceEmbeddingModel.EMBEDDING_GEMMA_2, model_kwargs={"model_kwargs": {"dtype": torch.bfloat16}}
    )


def test_hf_dtype_guard_only_applies_to_models_that_declare_it(hf_embedding_module):
    module, _ = hf_embedding_module
    module.HuggingFaceEmbeddingClient(
        module.HuggingFaceEmbeddingModel.BGE_SMALL_EN_V1_5, model_kwargs={"model_kwargs": {"dtype": "float16"}}
    )
