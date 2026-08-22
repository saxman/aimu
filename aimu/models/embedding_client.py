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

from ._internal.factory import (
    FactoryDelegate,
    ProviderEntry,
    available_registry,
    build_client,
    resolve_model_string,
)
from .base import BaseEmbeddingClient, EmbeddingModel, EmbeddingSpec

_OPENAI_HINT = "OpenAI embedding support requires the [openai_compat] extra (openai): pip install -e '.[openai_compat]'"
_OLLAMA_HINT = "Ollama embedding support requires the [ollama] extra (ollama): pip install -e '.[ollama]'"
_HF_HINT = "HuggingFace embedding support requires the [hf] extra (sentence-transformers): pip install -e '.[hf]'"


def _entries() -> list[ProviderEntry]:
    return [
        ProviderEntry(
            prefix="openai",
            module="aimu.models.providers.openai.embedding",
            enum_name="OpenAIEmbeddingModel",
            client_name="OpenAIEmbeddingClient",
            requires="openai",
            install_hint=_OPENAI_HINT,
        ),
        ProviderEntry(
            prefix="ollama",
            module="aimu.models.providers.ollama",
            enum_name="OllamaEmbeddingModel",
            client_name="OllamaEmbeddingClient",
            requires="ollama",
            install_hint=_OLLAMA_HINT,
            # Ollama has no weight loader, so these are real constructor params rather than
            # loader kwargs; without this the factory would bundle them into an ignored
            # model_kwargs and a remote host would silently fall back to localhost.
            direct_kwargs=frozenset({"host", "timeout"}),
        ),
        ProviderEntry(
            prefix="hf",
            module="aimu.models.providers.hf.embedding",
            enum_name="HuggingFaceEmbeddingModel",
            client_name="HuggingFaceEmbeddingClient",
            requires="sentence_transformers",
            install_hint=_HF_HINT,
        ),
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

    Examples::

        from aimu.models import EmbeddingClient, OpenAIEmbeddingModel

        client = EmbeddingClient(OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL)
        client = EmbeddingClient("openai:text-embedding-3-small")
        client = EmbeddingClient("ollama:nomic-embed-text")

    Provider-specific construction kwargs are forwarded. A param the concrete client declares
    itself reaches it directly (``ProviderEntry.direct_kwargs``); anything else is bundled into
    ``model_kwargs``, which is where a weight-loading client wants it::

        EmbeddingClient("ollama:nomic-embed-text", host="gpu-box")                  # -> host=
        EmbeddingClient(HuggingFaceEmbeddingModel.BGE_SMALL_EN_V1_5, device="cpu")  # -> model_kwargs=
    """

    def __init__(self, model: EmbeddingModel | EmbeddingSpec | str, **kwargs: Any) -> None:
        self._client: BaseEmbeddingClient = build_client(
            model,
            kwargs or None,
            _entries(),
            modality="embedding",
            model_base=EmbeddingModel,
            spec_base=EmbeddingSpec,
        )

    @property
    def dimensions(self) -> int | None:
        return self._client.dimensions

    def embed(self, texts: str | list[str], **kwargs: Any) -> Any:
        """Embed text. Forwarded to the inner client's :meth:`BaseEmbeddingClient.embed`."""
        return self._client.embed(texts, **kwargs)

    def __repr__(self) -> str:
        return f"EmbeddingClient({self._client!r})"
