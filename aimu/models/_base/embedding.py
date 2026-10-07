"""Embedding-modality base types: the embedding specs, the ``EmbeddingModel`` enum,
and ``BaseEmbeddingClient``.

Text-embedding generation as a parallel surface to text/image/audio/speech/transcription.
Disjoint from the chat surface (no message history, no streaming) -- an embedding client
maps text to fixed-length vectors. ``BaseEmbeddingClient`` is its own ABC for that reason.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, get_args

InputType = Literal["query", "document"]
INPUT_TYPES: tuple[str, ...] = get_args(InputType)


@dataclass
class EmbeddingSpec:
    """Descriptor for a single text-embedding model.

    Sibling to :class:`ModelSpec` / :class:`AudioSpec`. ``dimensions`` and
    ``max_input_tokens`` are informational (vector width and the per-input token
    budget); both may be ``None`` when not pinned. Provider-specific subclasses add
    their own fields.

    ``query_prompt`` / ``document_prompt`` are the prefixes the model's card says to prepend
    for asymmetric retrieval, applied by ``embed(input_type="query" | "document")``. They are
    declared here, per model, rather than read from a provider's config, so the exact text
    sent is visible in the catalog and is the same whichever provider serves the model.

    ``matryoshka_dimensions`` lists the output widths the model was trained to be truncated to
    (Matryoshka Representation Learning), the only widths a client's ``dimensions=`` accepts.
    ``None`` means the model is not truncatable. A ``range`` expresses "any width up to native"
    for an API that truncates server-side at any size (OpenAI's text-embedding-3).

    Equality and hash are by ``id`` only so the spec can be used directly as an enum
    value.
    """

    id: str
    dimensions: Optional[int] = None
    max_input_tokens: Optional[int] = None
    query_prompt: Optional[str] = None
    document_prompt: Optional[str] = None
    matryoshka_dimensions: Optional[Sequence[int]] = None

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
    so it is not pinned here.

    ``load_kwargs`` are default ``SentenceTransformer`` constructor kwargs for this model, merged
    *under* a caller's ``model_kwargs`` (a caller's key replaces the spec's whole, so passing
    ``config_kwargs={}`` restores everything a spec's ``config_kwargs`` switched off).

    ``rejected_dtypes`` names weight dtypes the model cannot run in. Requesting one raises at
    construction, because the failure it prevents is silent: EmbeddingGemma 2 in float16 returns
    NaN or degraded vectors rather than an error. ``eq=False`` keeps id-only equality.
    """

    normalize: bool = True
    load_kwargs: Optional[dict] = field(default=None)
    rejected_dtypes: tuple[str, ...] = ()


class EmbeddingModel(Enum):
    """Base enum for embedding-provider model catalogs.

    Parallel to :class:`AudioModel`. Each member's value is an :class:`EmbeddingSpec`
    (or subclass); the constructor sets ``_value_`` from ``spec.id`` and stores the spec.
    """

    def __init__(self, spec: EmbeddingSpec):
        self._value_ = spec.id
        self.spec = spec


def _describe_widths(widths: Sequence[int]) -> str:
    if isinstance(widths, range) and widths.step == 1:
        return f"any width from {widths.start} to {widths.stop - 1}"
    return ", ".join(str(width) for width in widths)


def _check_dimensions(spec: EmbeddingSpec, dimensions: Any) -> None:
    """Raise unless ``dimensions`` is a width ``spec`` was trained to be truncated to.

    Raising rather than warning is deliberate: a vector of the wrong width, or one truncated
    where the model was not trained for it, corrupts a persisted store without an error.
    """
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError(f"dimensions must be a positive int. Got: {dimensions!r}")
    if not spec.matryoshka_dimensions:
        raise ValueError(
            f"{spec.id} declares no widths it can be truncated to (its spec's matryoshka_dimensions "
            f"is empty); omit dimensions= to get its native {spec.dimensions}-wide vectors."
        )
    if dimensions not in spec.matryoshka_dimensions:
        raise ValueError(
            f"{spec.id} accepts dimensions= of {_describe_widths(spec.matryoshka_dimensions)}, "
            f"the widths it was trained to be truncated to. Got: {dimensions}"
        )


class BaseEmbeddingClient(ABC):
    """Abstract base for text-embedding provider clients.

    Subclasses implement :meth:`_embed`, which takes a non-empty list of strings and
    returns one vector (``list[float]``) per input. The public :meth:`embed` normalizes
    a single-string call to a single vector and a list call to a list of vectors, so
    every provider offers the same ergonomic surface.
    """

    model: Any
    spec: EmbeddingSpec
    # The width requested at construction, which providers send to their own truncation
    # parameter. A class attribute so a subclass that skips super().__init__ still reads None.
    _output_dimensions: Optional[int] = None

    @abstractmethod
    def __init__(
        self,
        model: Any,
        model_kwargs: Optional[dict] = None,
        *,
        spec: Optional[EmbeddingSpec] = None,
        dimensions: Optional[int] = None,
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        if spec is not None:
            self.spec = spec
        if dimensions is not None:
            if spec is None:
                raise TypeError("BaseEmbeddingClient needs spec= to validate dimensions=.")
            _check_dimensions(spec, dimensions)
            self._output_dimensions = dimensions

    @property
    def dimensions(self) -> Optional[int]:
        """The width of the vectors this client returns: the ``dimensions=`` it was built with,
        else the spec's native width, or ``None`` if neither is pinned."""
        return self._output_dimensions or self.spec.dimensions

    @abstractmethod
    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Provider-specific embedding. ``texts`` is always a non-empty list; returns one
        vector per input in the same order."""

    def embed(
        self,
        texts: Union[str, list[str]],
        *,
        input_type: Optional[InputType] = None,
        **kwargs: Any,
    ) -> Union[list[float], list[list[float]]]:
        """Embed one string or a list of strings.

        A single ``str`` returns one vector (``list[float]``); a list returns a list of
        vectors (``list[list[float]]``), preserving order. An empty list returns ``[]``.

        ``input_type`` says which side of an asymmetric retrieval the texts are: ``"query"``
        or ``"document"`` prepends the spec's ``query_prompt`` / ``document_prompt``, and is a
        no-op when the model declares none for that side (a symmetric model needs none).
        ``None`` sends the texts as given. Embed a corpus and its queries with the same
        convention: vectors from prompted and unprompted text are not comparable.

        Extra ``**kwargs`` are forwarded to the provider call.
        """
        if "dimensions" in kwargs:
            raise ValueError(
                "dimensions= is set once per client, so every vector it returns has the same width: "
                "aimu.embedding_client(model, dimensions=N)."
            )
        if input_type not in (None, *INPUT_TYPES):
            raise ValueError(f"input_type must be one of {INPUT_TYPES} or None. Got: {input_type!r}")
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return []
        if any(not isinstance(t, str) for t in items):
            raise ValueError("embed() expects a string or a list of strings.")
        prompt = getattr(self.spec, f"{input_type}_prompt") if input_type else None
        if prompt:
            items = [prompt + text for text in items]
        vectors = self._embed(items, **kwargs)
        return vectors[0] if single else vectors

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.spec.id!r})"
