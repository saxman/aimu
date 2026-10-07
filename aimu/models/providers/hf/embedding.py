"""HuggingFace text-embedding client backed by sentence-transformers.

Uses ``sentence_transformers.SentenceTransformer`` so each model's own pooling and
normalization config is honoured (BGE uses CLS pooling, E5 / GTE / MiniLM use mean
pooling) instead of a hand-rolled strategy that would be silently wrong per model.

Lazy model load on first ``embed()`` call; weights are cached in a module-level registry
so a second client for the same model reuses already-loaded weights (same pattern as the
other in-process HuggingFace clients). Free them with ``aimu.clear_hf_cache()``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

# Hard import so HAS_HF_EMBEDDING in aimu/models/__init__.py accurately reflects whether
# the required dep is installed.
import sentence_transformers  # noqa: F401

from ._device import pop_device_hint
from ...base import BaseEmbeddingClient, EmbeddingModel, HuggingFaceEmbeddingSpec

logger = logging.getLogger(__name__)

_model_registry: dict[tuple, Any] = {}
_registry_lock = threading.Lock()


def _make_cache_key(spec_id: str, model_kwargs: dict | None) -> tuple:
    return (spec_id, *sorted((k, str(v)) for k, v in (model_kwargs or {}).items()))


_BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Catalog of curated HuggingFace (sentence-transformers) embedding models.

    Each member's value is a :class:`HuggingFaceEmbeddingSpec`. ``.value`` returns the
    HuggingFace repo id; ``.spec`` returns the full spec. Power users can bypass the enum
    by passing a ``"hf:<repo_id>"`` string or a hand-built :class:`HuggingFaceEmbeddingSpec`.
    """

    ALL_MINILM_L6_V2 = HuggingFaceEmbeddingSpec(
        "sentence-transformers/all-MiniLM-L6-v2", dimensions=384, max_input_tokens=256
    )
    BGE_SMALL_EN_V1_5 = HuggingFaceEmbeddingSpec(
        "BAAI/bge-small-en-v1.5", dimensions=384, max_input_tokens=512, query_prompt=_BGE_QUERY_PROMPT
    )
    BGE_BASE_EN_V1_5 = HuggingFaceEmbeddingSpec(
        "BAAI/bge-base-en-v1.5", dimensions=768, max_input_tokens=512, query_prompt=_BGE_QUERY_PROMPT
    )
    BGE_LARGE_EN_V1_5 = HuggingFaceEmbeddingSpec(
        "BAAI/bge-large-en-v1.5", dimensions=1024, max_input_tokens=512, query_prompt=_BGE_QUERY_PROMPT
    )
    GTE_LARGE = HuggingFaceEmbeddingSpec("thenlper/gte-large", dimensions=1024, max_input_tokens=512)
    E5_LARGE_V2 = HuggingFaceEmbeddingSpec(
        "intfloat/e5-large-v2",
        dimensions=1024,
        max_input_tokens=512,
        query_prompt="query: ",
        document_prompt="passage: ",
    )
    MXBAI_EMBED_LARGE_V1 = HuggingFaceEmbeddingSpec(
        "mixedbread-ai/mxbai-embed-large-v1", dimensions=1024, max_input_tokens=512, query_prompt=_BGE_QUERY_PROMPT
    )
    # Ships vision and audio encoders too (740M parameters in all). embed() takes text only, so
    # the spec loads the 270M text backbone; pass model_kwargs={"config_kwargs": {}} for the rest.
    EMBEDDING_GEMMA_2 = HuggingFaceEmbeddingSpec(
        "google/embeddinggemma-2",
        dimensions=768,
        max_input_tokens=8192,
        query_prompt="task: search result | query: ",
        document_prompt="title: none | text: ",
        load_kwargs={"config_kwargs": {"vision_config": None, "audio_config": None}},
        rejected_dtypes=("float16",),
    )


_DTYPE_ALIASES = {"half": "float16", "fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "float": "float32"}


def _dtype_name(dtype: Any) -> str:
    """Canonical name for a dtype given as a ``torch.dtype`` or any of its usual string spellings."""
    name = str(dtype).removeprefix("torch.")
    return _DTYPE_ALIASES.get(name, name)


def _check_dtype(spec: HuggingFaceEmbeddingSpec, model_kwargs: dict | None) -> None:
    """Raise if ``model_kwargs`` requests a weight dtype the spec declares unusable.

    sentence-transformers takes the dtype nested under its own ``model_kwargs``, spelled either
    ``dtype`` or the older ``torch_dtype``.
    """
    if not spec.rejected_dtypes:
        return
    transformer_kwargs = (model_kwargs or {}).get("model_kwargs") or {}
    for key in ("dtype", "torch_dtype"):
        if key in transformer_kwargs and _dtype_name(transformer_kwargs[key]) in spec.rejected_dtypes:
            raise ValueError(
                f"{spec.id} cannot run in {_dtype_name(transformer_kwargs[key])}: it returns NaN or degraded "
                f"vectors rather than an error. Pass a dtype other than {', '.join(spec.rejected_dtypes)}, "
                f"or omit it."
            )


def _parse_model_string(s: str) -> HuggingFaceEmbeddingSpec:
    """Resolve a ``"hf:<repo_id>"`` string to a known :class:`HuggingFaceEmbeddingModel` spec.

    AIMU ships curated specs only; an arbitrary repo id is not supported via string (the
    declared ``dimensions`` can't be inferred reliably). For a custom model hand-build a
    :class:`HuggingFaceEmbeddingSpec` and pass that object.
    """
    if ":" not in s:
        raise ValueError(
            f"HuggingFace embedding model string must be in 'provider:repo_id' form "
            f"(e.g. 'hf:BAAI/bge-small-en-v1.5'). Got: {s!r}"
        )
    provider, repo_id = s.split(":", 1)
    if provider != "hf":
        raise ValueError(f"Only 'hf:' provider is supported for HuggingFaceEmbeddingClient. Got provider: {provider!r}")
    if not repo_id:
        raise ValueError(f"Empty model id after 'hf:': {s!r}")
    for member in HuggingFaceEmbeddingModel:
        if member.value == repo_id:
            return member.spec
    available = sorted(m.value for m in HuggingFaceEmbeddingModel)
    raise ValueError(
        f"Unknown HuggingFace embedding model id {repo_id!r}. AIMU supports curated models only; "
        f"pass a known id, a HuggingFaceEmbeddingModel member, or a hand-built "
        f"HuggingFaceEmbeddingSpec for a custom model. Available ids: {available}"
    )


class HuggingFaceEmbeddingClient(BaseEmbeddingClient):
    """Local text-embedding client backed by sentence-transformers.

    Loads model weights lazily on the first :meth:`embed` call. Pass a
    :class:`HuggingFaceEmbeddingModel` member, a :class:`HuggingFaceEmbeddingSpec`, or a
    ``"hf:<repo_id>"`` string. Target a device with ``model_kwargs={"device": "cuda:1"}``;
    other ``model_kwargs`` are forwarded to ``SentenceTransformer``.

    Retrieval-tuned models (E5, BGE, mxbai, EmbeddingGemma 2) want a query or document prefix
    for asymmetric retrieval; ``embed(texts, input_type="query" | "document")`` prepends the
    one the spec declares. The model's own sentence-transformers ``prompts`` config is not
    consulted, so the spec is the one place that text comes from.
    """

    MODELS = HuggingFaceEmbeddingModel

    def __init__(
        self,
        model: "HuggingFaceEmbeddingModel | HuggingFaceEmbeddingSpec | str",
        model_kwargs: dict | None = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, HuggingFaceEmbeddingModel):
            spec = model.spec
        elif isinstance(model, HuggingFaceEmbeddingSpec):
            spec = model
        else:
            raise TypeError(
                f"HuggingFaceEmbeddingClient expects a HuggingFaceEmbeddingModel member, "
                f"HuggingFaceEmbeddingSpec, or 'hf:<repo_id>' string. Got: {type(model).__name__}"
            )
        _check_dtype(spec, model_kwargs)
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec
        self._model: Any = None  # lazy
        self._cache_key = _make_cache_key(spec.id, model_kwargs)

    def _load_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {**(self.spec.load_kwargs or {}), **(self.model_kwargs or {})}
        device = pop_device_hint(kwargs)

        logger.info("Loading sentence-transformers model %s", self.spec.id)
        return SentenceTransformer(self.spec.id, device=device, **kwargs)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with _registry_lock:
            if self._cache_key in _model_registry:
                self._model = _model_registry[self._cache_key]
                return
        self._model = self._load_model()
        with _registry_lock:
            _model_registry[self._cache_key] = self._model

    @property
    def sentence_transformer(self) -> Any:
        """The loaded SentenceTransformer (triggers weight load on first access)."""
        self._ensure_loaded()
        return self._model

    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self._ensure_loaded()
        kwargs.setdefault("normalize_embeddings", self.spec.normalize)
        vectors = self._model.encode(texts, convert_to_numpy=True, **kwargs)
        return [vector.tolist() for vector in vectors]

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "unloaded"
        return f"HuggingFaceEmbeddingClient(model={self.spec.id!r}, {loaded})"
