"""Embedding-client factory paralleling :mod:`aimu.models.transcription_client`.

Exposes:

- :func:`resolve_embedding_model_string`: parse ``"provider:model_id"`` for embedding
  providers.
- :class:`EmbeddingClient`: factory :class:`BaseEmbeddingClient` that dispatches to the
  right concrete client based on the model enum / spec / string passed in.

Mirrors the transcription-side dispatch. The shared dispatch logic lives in
:mod:`aimu.models._internal.factory`.
"""

from __future__ import annotations

from typing import Any

from ._internal.factory import FactoryDelegate, ProviderEntry, available_registry, build_client, resolve_model_string
from .base import BaseEmbeddingClient, EmbeddingModel, EmbeddingSpec

# --- Optional provider imports ---

try:
    from .providers.openai.embedding import OpenAIEmbeddingClient, OpenAIEmbeddingModel

    _HAS_OPENAI_EMBEDDING = True
except ImportError:
    _HAS_OPENAI_EMBEDDING = False
    OpenAIEmbeddingClient = None  # type: ignore[assignment,misc]
    OpenAIEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from .providers.ollama import OllamaEmbeddingClient, OllamaEmbeddingModel

    _HAS_OLLAMA_EMBEDDING = True
except ImportError:
    _HAS_OLLAMA_EMBEDDING = False
    OllamaEmbeddingClient = None  # type: ignore[assignment,misc]
    OllamaEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from .providers.hf.embedding import HuggingFaceEmbeddingClient, HuggingFaceEmbeddingModel

    _HAS_HF_EMBEDDING = True
except ImportError:
    _HAS_HF_EMBEDDING = False
    HuggingFaceEmbeddingClient = None  # type: ignore[assignment,misc]
    HuggingFaceEmbeddingModel = None  # type: ignore[assignment,misc]

_OPENAI_HINT = "OpenAI embedding support requires the [openai_compat] extra (openai): pip install -e '.[openai_compat]'"
_OLLAMA_HINT = "Ollama embedding support requires the [ollama] extra (ollama): pip install -e '.[ollama]'"
_HF_HINT = "HuggingFace embedding support requires the [hf] extra (sentence-transformers): pip install -e '.[hf]'"


def _entries() -> list[ProviderEntry]:
    return [
        ProviderEntry("openai", _HAS_OPENAI_EMBEDDING, OpenAIEmbeddingModel, OpenAIEmbeddingClient, _OPENAI_HINT),
        ProviderEntry("ollama", _HAS_OLLAMA_EMBEDDING, OllamaEmbeddingModel, OllamaEmbeddingClient, _OLLAMA_HINT),
        ProviderEntry("hf", _HAS_HF_EMBEDDING, HuggingFaceEmbeddingModel, HuggingFaceEmbeddingClient, _HF_HINT),
    ]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(EmbeddingModel subclass, client subclass)`` (installed only)."""
    return available_registry(_entries())


def resolve_embedding_model_string(model_str: str) -> EmbeddingModel:
    """Look up an embedding-provider model enum from a ``"provider:model_id"`` string.

    Only matches *exact* enum-member values; for ad-hoc model ids pass the
    ``"provider:..."`` string directly to :class:`EmbeddingClient`.
    """
    return resolve_model_string(model_str, _entries(), modality="embedding")


class EmbeddingClient(FactoryDelegate):
    """Public factory for text-embedding provider clients.

    Parallel to :class:`aimu.models.TranscriptionClient`. Accepts a provider's
    :class:`EmbeddingModel` enum member, an :class:`EmbeddingSpec`, or a
    ``"provider:model_id"`` string (``"openai:..."``, ``"ollama:..."`` or ``"hf:..."``).
    Provider-specific construction kwargs are forwarded as ``model_kwargs``.

    Examples::

        from aimu.models import EmbeddingClient, OpenAIEmbeddingModel

        client = EmbeddingClient(OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL)
        client = EmbeddingClient("openai:text-embedding-3-small")
        client = EmbeddingClient("ollama:nomic-embed-text")
    """

    def __init__(self, model: EmbeddingModel | EmbeddingSpec | str, model_kwargs: dict | None = None) -> None:
        self._client: BaseEmbeddingClient = build_client(
            model, model_kwargs, _entries(), modality="embedding", model_base=EmbeddingModel, spec_base=EmbeddingSpec
        )

    @property
    def dimensions(self) -> int | None:
        return self._client.dimensions

    def embed(self, texts: str | list[str], **kwargs: Any) -> Any:
        """Embed text. Forwarded to the inner client's :meth:`BaseEmbeddingClient.embed`."""
        return self._client.embed(texts, **kwargs)

    def __repr__(self) -> str:
        return f"EmbeddingClient({self._client!r})"
