"""`host=` on the native Ollama clients, and the `ollama:<model_id>@<host>` string form.

The `ollama` SDK constructors are monkeypatched throughout, so these need no server. They
assert what AIMU forwards, that an unset host is omitted (leaving `OLLAMA_HOST` and the
SDK's 127.0.0.1 default in charge), and that the `/v1` suffix habit from the OpenAI-compat
providers is refused rather than 404ing at request time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ollama")

from aimu.models import ModelClient, OllamaModel  # noqa: E402
from aimu.models.providers.ollama import OllamaEmbeddingClient, OllamaEmbeddingModel  # noqa: E402

REMOTE = "http://gpu-box:11434"


class FakeClient:
    """Stands in for `ollama.Client` / `ollama.AsyncClient`, recording construction kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pulled: list[str] = []

    def pull(self, model, *args, **kwargs):
        self.pulled.append(model)

    def embed(self, model=None, input=None, **kwargs):
        self.embedded = (model, input)
        return {"embeddings": [[0.1, 0.2] for _ in (input or [])]}


@pytest.fixture
def sync_clients(monkeypatch):
    """Every `ollama.Client` the sync provider module builds, in construction order."""
    from aimu.models.providers import ollama as ollama_mod

    built: list[FakeClient] = []

    def factory(**kwargs):
        built.append(FakeClient(**kwargs))
        return built[-1]

    monkeypatch.setattr(ollama_mod.ollama, "Client", factory)
    return built


@pytest.fixture
def async_clients(monkeypatch):
    """Every `ollama.AsyncClient` the async provider module builds, in construction order."""
    from aimu.aio.providers import ollama as ollama_mod

    built: list[FakeClient] = []

    def factory(**kwargs):
        built.append(FakeClient(**kwargs))
        return built[-1]

    monkeypatch.setattr(ollama_mod.ollama, "AsyncClient", factory)
    return built


# ---------------------------------------------------------------------------
# Text clients
# ---------------------------------------------------------------------------


def test_sync_client_forwards_host(sync_clients):
    from aimu.models.providers.ollama import OllamaClient

    OllamaClient(OllamaModel.LLAMA_3_2_3B, host=REMOTE)
    assert sync_clients[0].kwargs["host"] == REMOTE


def test_sync_client_omits_host_when_unset(sync_clients):
    # An absent host must not become `host=None`: the SDK's own OLLAMA_HOST / 127.0.0.1
    # resolution has to stay in charge.
    from aimu.models.providers.ollama import OllamaClient

    OllamaClient(OllamaModel.LLAMA_3_2_3B)
    assert "host" not in sync_clients[0].kwargs


def test_sync_client_forwards_host_with_timeout(sync_clients):
    from aimu.models.providers.ollama import OllamaClient

    OllamaClient(OllamaModel.LLAMA_3_2_3B, host=REMOTE, timeout=15)
    assert sync_clients[0].kwargs == {"host": REMOTE, "timeout": 15}


def test_sync_client_pulls_on_the_remote_host(sync_clients):
    # The eager pull runs through the host-bound client, so it lands on the remote server.
    from aimu.models.providers.ollama import OllamaClient

    OllamaClient(OllamaModel.LLAMA_3_2_3B, host=REMOTE)
    assert sync_clients[0].pulled == [OllamaModel.LLAMA_3_2_3B.value]


def test_async_client_forwards_host(async_clients):
    from aimu.aio.providers.ollama import AsyncOllamaClient

    AsyncOllamaClient(OllamaModel.LLAMA_3_2_3B, host=REMOTE)
    assert async_clients[0].kwargs["host"] == REMOTE


def test_async_client_omits_host_when_unset(async_clients):
    from aimu.aio.providers.ollama import AsyncOllamaClient

    AsyncOllamaClient(OllamaModel.LLAMA_3_2_3B)
    assert "host" not in async_clients[0].kwargs


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------


def test_embedding_client_forwards_host_and_timeout(sync_clients):
    OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT, host=REMOTE, timeout=15)
    assert sync_clients[0].kwargs == {"host": REMOTE, "timeout": 15}


def test_embedding_client_omits_host_when_unset(sync_clients):
    OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT)
    assert "host" not in sync_clients[0].kwargs


def test_embedding_client_pulls_and_embeds_through_the_host_bound_client(sync_clients):
    # The regression this closes: module-level `ollama.pull` / `ollama.embed` ignored `host`,
    # so a remote text client silently paired with a localhost embedder.
    client = OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT, host=REMOTE)
    vectors = client.embed(["one", "two"])

    fake = sync_clients[0]
    assert fake.pulled == [OllamaEmbeddingModel.NOMIC_EMBED_TEXT.value]
    assert fake.embedded == (OllamaEmbeddingModel.NOMIC_EMBED_TEXT.value, ["one", "two"])
    assert len(vectors) == 2


# ---------------------------------------------------------------------------
# The /v1 foot-gun: the native API is not the OpenAI-compat endpoint
# ---------------------------------------------------------------------------


def test_sync_client_rejects_v1_suffix(sync_clients):
    from aimu.models.providers.ollama import OllamaClient

    with pytest.raises(ValueError, match="ollama-openai"):
        OllamaClient(OllamaModel.LLAMA_3_2_3B, host=f"{REMOTE}/v1")


def test_async_client_rejects_v1_suffix(async_clients):
    from aimu.aio.providers.ollama import AsyncOllamaClient

    with pytest.raises(ValueError, match="ollama-openai"):
        AsyncOllamaClient(OllamaModel.LLAMA_3_2_3B, host=f"{REMOTE}/v1")


def test_embedding_client_rejects_v1_suffix(sync_clients):
    with pytest.raises(ValueError, match="ollama-openai"):
        OllamaEmbeddingClient(OllamaEmbeddingModel.NOMIC_EMBED_TEXT, host=f"{REMOTE}/v1")


# ---------------------------------------------------------------------------
# Model-string form: ollama:<model_id>@<host>
# ---------------------------------------------------------------------------


def test_model_string_endpoint_becomes_host(sync_clients):
    ModelClient(f"ollama:{OllamaModel.LLAMA_3_2_3B.value}@{REMOTE}")
    assert sync_clients[0].kwargs["host"] == REMOTE


def test_model_string_accepts_bare_host_and_port(sync_clients):
    # The SDK's own host parsing accepts these forms; AIMU forwards them verbatim.
    ModelClient(f"ollama:{OllamaModel.LLAMA_3_2_3B.value}@gpu-box:11434")
    assert sync_clients[0].kwargs["host"] == "gpu-box:11434"


def test_model_string_without_endpoint_omits_host(sync_clients):
    ModelClient(f"ollama:{OllamaModel.LLAMA_3_2_3B.value}")
    assert "host" not in sync_clients[0].kwargs


def test_model_string_endpoint_does_not_open_the_adhoc_form(sync_clients):
    # `ollama` accepts an endpoint but stays curated-catalog: an unknown tag still raises,
    # with or without capability flags.
    with pytest.raises(ValueError, match="no model id"):
        ModelClient(f"ollama:not-a-real-tag:latest@{REMOTE}")
    with pytest.raises(ValueError, match="no model id"):
        ModelClient(f"ollama:not-a-real-tag:latest@{REMOTE};tools,thinking")


def test_async_model_string_endpoint_becomes_host(async_clients):
    from aimu.aio import AsyncModelClient

    AsyncModelClient(f"ollama:{OllamaModel.LLAMA_3_2_3B.value}@{REMOTE}")
    assert async_clients[0].kwargs["host"] == REMOTE


# ---------------------------------------------------------------------------
# Factory path: aimu.embedding_client() must reach the real constructor params
# ---------------------------------------------------------------------------


def test_embedding_factory_forwards_host_from_string(sync_clients):
    import aimu

    aimu.embedding_client("ollama:nomic-embed-text", host=REMOTE)
    assert sync_clients[0].kwargs["host"] == REMOTE


def test_embedding_factory_forwards_host_from_enum(sync_clients):
    import aimu

    aimu.embedding_client(OllamaEmbeddingModel.NOMIC_EMBED_TEXT, host=REMOTE, timeout=9)
    assert sync_clients[0].kwargs == {"host": REMOTE, "timeout": 9}


def test_provider_entry_splits_declared_kwargs_from_model_kwargs():
    # The modality factories bundle unrecognized kwargs into `model_kwargs` (where a
    # weight-loading client wants `device=`), and forward only declared params directly.
    from aimu.models.embedding_client import _entries

    ollama_entry = next(e for e in _entries() if e.prefix == "ollama")
    assert ollama_entry.split_kwargs({"host": REMOTE, "device": "cpu"}) == ({"host": REMOTE}, {"device": "cpu"})
    assert ollama_entry.split_kwargs(None) == ({}, None)

    hf_entry = next(e for e in _entries() if e.prefix == "hf")
    assert hf_entry.split_kwargs({"device": "cpu"}) == ({}, {"device": "cpu"})
