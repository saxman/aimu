"""Embedding-modality base types: the embedding specs, the ``EmbeddingModel`` enum,
and ``BaseEmbeddingClient``.

Text-embedding generation as a parallel surface to text/image/audio/speech/transcription.
Disjoint from the chat surface (no message history, no streaming) -- an embedding client
maps text to fixed-length vectors. ``BaseEmbeddingClient`` is its own ABC for that reason.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union


@dataclass
class EmbeddingSpec:
    """Descriptor for a single text-embedding model.

    Sibling to :class:`ModelSpec` / :class:`AudioSpec`. ``dimensions`` and
    ``max_input_tokens`` are informational (vector width and the per-input token
    budget); both may be ``None`` when not pinned. Provider-specific subclasses add
    their own fields.

    Equality and hash are by ``id`` only so the spec can be used directly as an enum
    value.
    """

    id: str
    dimensions: Optional[int] = None
    max_input_tokens: Optional[int] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EmbeddingSpec):
            return self.id == other.id
        return NotImplemented


@dataclass(eq=False)
class OpenAIEmbeddingSpec(EmbeddingSpec):
    """Descriptor for an OpenAI embedding model. ``eq=False`` keeps id-only equality."""


@dataclass(eq=False)
class OllamaEmbeddingSpec(EmbeddingSpec):
    """Descriptor for an Ollama embedding model. ``eq=False`` keeps id-only equality."""


@dataclass(eq=False)
class HuggingFaceEmbeddingSpec(EmbeddingSpec):
    """Descriptor for a HuggingFace (sentence-transformers) embedding model.

    ``normalize`` is the default L2-normalization applied by the client (overridable per
    call via ``embed(normalize_embeddings=...)``); cosine-similarity retrieval wants
    normalized vectors. Pooling is read from the model's own config by sentence-transformers,
    so it is not pinned here. ``eq=False`` keeps id-only equality.
    """

    normalize: bool = True


class EmbeddingModel(Enum):
    """Base enum for embedding-provider model catalogs.

    Parallel to :class:`AudioModel`. Each member's value is an :class:`EmbeddingSpec`
    (or subclass); the constructor sets ``_value_`` from ``spec.id`` and stores the spec.
    """

    def __init__(self, spec: EmbeddingSpec):
        self._value_ = spec.id
        self.spec = spec


class BaseEmbeddingClient(ABC):
    """Abstract base for text-embedding provider clients.

    Subclasses implement :meth:`_embed`, which takes a non-empty list of strings and
    returns one vector (``list[float]``) per input. The public :meth:`embed` normalizes
    a single-string call to a single vector and a list call to a list of vectors, so
    every provider offers the same ergonomic surface.
    """

    model: Any
    spec: EmbeddingSpec

    @abstractmethod
    def __init__(self, model: Any, model_kwargs: Optional[dict] = None):
        self.model = model
        self.model_kwargs = model_kwargs

    @property
    def dimensions(self) -> Optional[int]:
        """The embedding vector width declared by the spec, or ``None`` if unspecified."""
        return self.spec.dimensions

    @abstractmethod
    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Provider-specific embedding. ``texts`` is always a non-empty list; returns one
        vector per input in the same order."""

    def embed(self, texts: Union[str, list[str]], **kwargs: Any) -> Union[list[float], list[list[float]]]:
        """Embed one string or a list of strings.

        A single ``str`` returns one vector (``list[float]``); a list returns a list of
        vectors (``list[list[float]]``), preserving order. An empty list returns ``[]``.
        Extra ``**kwargs`` are forwarded to the provider call.
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return []
        if any(not isinstance(t, str) for t in items):
            raise ValueError("embed() expects a string or a list of strings.")
        vectors = self._embed(items, **kwargs)
        return vectors[0] if single else vectors

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.spec.id!r})"
