"""Live, opt-in embedding tests, parametrized via --embedding-client / --embedding-model.

Skipped entirely unless --embedding-client is passed. Examples::

    pytest tests/test_embeddings.py --embedding-client=openai
    pytest tests/test_embeddings.py --embedding-client=ollama --embedding-model=NOMIC_EMBED_TEXT
    pytest tests/test_embeddings.py --embedding-client=all
"""

from __future__ import annotations

import pytest

import aimu


def _params(config):
    selected = config.getoption("--embedding-client")
    if not selected:
        return []
    model_name = config.getoption("--embedding-model")
    from aimu.models import HAS_HF_EMBEDDING, HAS_OLLAMA_EMBEDDING, HAS_OPENAI_EMBEDDING

    providers: list[tuple] = []
    if selected in ("openai", "all") and HAS_OPENAI_EMBEDDING:
        from aimu.models import OpenAIEmbeddingModel

        providers.append(("openai", OpenAIEmbeddingModel))
    if selected in ("ollama", "all") and HAS_OLLAMA_EMBEDDING:
        from aimu.models import OllamaEmbeddingModel

        providers.append(("ollama", OllamaEmbeddingModel))
    if selected in ("hf", "all") and HAS_HF_EMBEDDING:
        from aimu.models import HuggingFaceEmbeddingModel

        providers.append(("hf", HuggingFaceEmbeddingModel))

    params = []
    for provider, enum in providers:
        members = list(enum) if model_name == "all" else [enum[model_name]]
        for member in members:
            params.append(pytest.param(member, id=f"{provider}:{member.name}"))
    return params


def pytest_generate_tests(metafunc):
    if "embedding_model" in metafunc.fixturenames:
        params = _params(metafunc.config)
        if not params:
            pytest.skip("pass --embedding-client to run live embedding tests")
        metafunc.parametrize("embedding_model", params)


@pytest.fixture
def client(embedding_model):
    return aimu.embedding_client(embedding_model)


def test_embed_single_returns_vector(client):
    vec = client.embed("the quick brown fox")
    assert isinstance(vec, list)
    assert vec and all(isinstance(x, float) for x in vec)


def test_embed_list_preserves_order_and_count(client):
    vecs = client.embed(["alpha", "beta", "gamma"])
    assert len(vecs) == 3
    assert all(isinstance(v, list) and v for v in vecs)
    # All vectors share one width.
    assert len({len(v) for v in vecs}) == 1


def test_embed_similar_texts_closer_than_dissimilar(client):
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    cat, kitten, finance = client.embed(["a small cat", "a tiny kitten", "quarterly tax filing"])
    assert cosine(cat, kitten) > cosine(cat, finance)
