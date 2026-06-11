"""Embedding-client factory paralleling :mod:`aimu.models.transcription_client`.

Exposes:

- :func:`resolve_embedding_model_string` — parse ``"provider:model_id"`` for embedding
  providers.
- :class:`EmbeddingClient` — factory :class:`BaseEmbeddingClient` that dispatches to the
  right concrete client based on the model enum / spec / string passed in.

Mirrors the transcription-side dispatch (:class:`aimu.models.TranscriptionClient`).
"""

from __future__ import annotations

from typing import Any

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


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(EmbeddingModel subclass, EmbeddingClient subclass)``."""
    registry: dict[str, tuple] = {}
    if _HAS_OPENAI_EMBEDDING:
        registry["openai"] = (OpenAIEmbeddingModel, OpenAIEmbeddingClient)
    if _HAS_OLLAMA_EMBEDDING:
        registry["ollama"] = (OllamaEmbeddingModel, OllamaEmbeddingClient)
    if _HAS_HF_EMBEDDING:
        registry["hf"] = (HuggingFaceEmbeddingModel, HuggingFaceEmbeddingClient)
    return registry


def resolve_embedding_model_string(model_str: str) -> EmbeddingModel:
    """Look up an embedding-provider model enum from a ``"provider:model_id"`` string.

    Only matches *exact* enum-member values; for ad-hoc model ids pass the
    ``"provider:..."`` string directly to :class:`EmbeddingClient`.
    """
    if ":" not in model_str:
        raise ValueError(
            f"Embedding model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(_provider_registry())}"
        )
    provider, _, model_id = model_str.partition(":")
    registry = _provider_registry()
    if provider not in registry:
        raise ValueError(
            f"Unknown embedding provider {provider!r}. Available providers (with installed deps): {sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(f"Provider {provider!r} has no embedding model id {model_id!r}. Available: {available}")


class EmbeddingClient:
    """Public factory for text-embedding provider clients.

    Parallel to :class:`aimu.models.TranscriptionClient`. Accepts a provider's
    :class:`EmbeddingModel` enum member, an :class:`EmbeddingSpec`, or a
    ``"provider:model_id"`` string (``"openai:..."`` or ``"ollama:..."``).
    Provider-specific construction kwargs are forwarded as ``model_kwargs``.

    Examples::

        from aimu.models import EmbeddingClient, OpenAIEmbeddingModel

        client = EmbeddingClient(OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL)
        client = EmbeddingClient("openai:text-embedding-3-small")
        client = EmbeddingClient("ollama:nomic-embed-text")
    """

    def __init__(self, model: EmbeddingModel | EmbeddingSpec | str, model_kwargs: dict | None = None) -> None:
        if isinstance(model, str):
            if ":" not in model:
                raise ValueError(f"Embedding model string must be in 'provider:model_id' form, got: {model!r}")
            provider, _, _model_id = model.partition(":")
            if provider == "openai":
                if not _HAS_OPENAI_EMBEDDING:
                    raise ImportError(
                        "OpenAI embedding support requires the [openai_compat] extra (openai): "
                        "pip install -e '.[openai_compat]'"
                    )
                self._client: BaseEmbeddingClient = OpenAIEmbeddingClient(model, model_kwargs=model_kwargs)
                return
            if provider == "ollama":
                if not _HAS_OLLAMA_EMBEDDING:
                    raise ImportError(
                        "Ollama embedding support requires the [ollama] extra (ollama): pip install -e '.[ollama]'"
                    )
                self._client = OllamaEmbeddingClient(model, model_kwargs=model_kwargs)
                return
            if provider == "hf":
                if not _HAS_HF_EMBEDDING:
                    raise ImportError(
                        "HuggingFace embedding support requires the [hf] extra (sentence-transformers): "
                        "pip install -e '.[hf]'"
                    )
                self._client = HuggingFaceEmbeddingClient(model, model_kwargs=model_kwargs)
                return
            raise ValueError(f"Unknown embedding provider {provider!r}. Available: {sorted(_provider_registry())}")

        if isinstance(model, EmbeddingSpec) and not isinstance(model, EmbeddingModel):
            raise TypeError(
                "Pass an EmbeddingModel enum member (e.g. OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL) or a "
                "'provider:model_id' string. EmbeddingSpec is the value type held by enum members."
            )

        if _HAS_OPENAI_EMBEDDING and OpenAIEmbeddingModel is not None and isinstance(model, OpenAIEmbeddingModel):
            self._client = OpenAIEmbeddingClient(model, model_kwargs=model_kwargs)
        elif _HAS_OLLAMA_EMBEDDING and OllamaEmbeddingModel is not None and isinstance(model, OllamaEmbeddingModel):
            self._client = OllamaEmbeddingClient(model, model_kwargs=model_kwargs)
        elif _HAS_HF_EMBEDDING and HuggingFaceEmbeddingModel is not None and isinstance(model, HuggingFaceEmbeddingModel):
            self._client = HuggingFaceEmbeddingClient(model, model_kwargs=model_kwargs)
        else:
            raise ValueError(
                f"No available client for embedding-model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed."
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> EmbeddingSpec:
        return self._client.spec

    @property
    def dimensions(self) -> int | None:
        return self._client.dimensions

    @property
    def model_kwargs(self) -> dict | None:
        return self._client.model_kwargs

    def embed(self, texts: str | list[str], **kwargs: Any) -> Any:
        """Embed text. Forwarded to the inner client's :meth:`BaseEmbeddingClient.embed`."""
        return self._client.embed(texts, **kwargs)

    def __repr__(self) -> str:
        return f"EmbeddingClient({self._client!r})"
