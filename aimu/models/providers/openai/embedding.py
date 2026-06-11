"""OpenAI text-embedding client.

Uses the same ``openai`` SDK as :class:`OpenAICompatClient`, calling
``client.embeddings.create()``. Reads ``OPENAI_API_KEY`` from the environment
(or a ``.env`` file). Mirrors the structure of the OpenAI transcription client.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv

from ...base import BaseEmbeddingClient, EmbeddingModel, OpenAIEmbeddingSpec

OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbeddingModel(EmbeddingModel):
    """Catalog of OpenAI embedding models.

    Each member's value is an :class:`OpenAIEmbeddingSpec`. ``.value`` returns the
    OpenAI model id; ``.spec`` returns the full spec. ``text-embedding-3-*`` support a
    ``dimensions=`` override at call time (passed through ``embed()``).
    """

    TEXT_EMBEDDING_3_SMALL = OpenAIEmbeddingSpec("text-embedding-3-small", dimensions=1536, max_input_tokens=8191)
    TEXT_EMBEDDING_3_LARGE = OpenAIEmbeddingSpec("text-embedding-3-large", dimensions=3072, max_input_tokens=8191)
    TEXT_EMBEDDING_ADA_002 = OpenAIEmbeddingSpec("text-embedding-ada-002", dimensions=1536, max_input_tokens=8191)


def _parse_model_string(s: str) -> OpenAIEmbeddingSpec:
    """Resolve an ``"openai:<model_id>"`` string to a known :class:`OpenAIEmbeddingModel` spec."""
    if ":" not in s:
        raise ValueError(
            f"OpenAI embedding model string must be in 'provider:model_id' form "
            f"(e.g. 'openai:text-embedding-3-small'). Got: {s!r}"
        )
    provider, model_id = s.split(":", 1)
    if provider != "openai":
        raise ValueError(f"Only 'openai:' provider is supported for OpenAIEmbeddingClient. Got provider: {provider!r}")
    for member in OpenAIEmbeddingModel:
        if member.value == model_id:
            return member.spec
    available = sorted(m.value for m in OpenAIEmbeddingModel)
    raise ValueError(
        f"Unknown OpenAI embedding model id {model_id!r}. AIMU supports curated models only; "
        f"pass a known id, an OpenAIEmbeddingModel member, or a hand-built OpenAIEmbeddingSpec. "
        f"Available ids: {available}"
    )


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """Text-embedding client for the OpenAI API.

    Pass an :class:`OpenAIEmbeddingModel` member, an :class:`OpenAIEmbeddingSpec`, or an
    ``"openai:<model_id>"`` string. ``model_kwargs`` may carry ``api_key=`` / ``base_url=``
    overrides; otherwise ``OPENAI_API_KEY`` is read from the environment.
    """

    MODELS = OpenAIEmbeddingModel

    def __init__(
        self,
        model: "OpenAIEmbeddingModel | OpenAIEmbeddingSpec | str",
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, OpenAIEmbeddingModel):
            spec = model.spec
        elif isinstance(model, OpenAIEmbeddingSpec):
            spec = model
        else:
            raise TypeError(
                f"OpenAIEmbeddingClient expects an OpenAIEmbeddingModel member, OpenAIEmbeddingSpec, "
                f"or 'openai:<model_id>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec

        import openai

        kwargs = dict(model_kwargs or {})
        load_dotenv()
        api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY", "not-set")
        base_url = kwargs.pop("base_url", OPENAI_BASE_URL)
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.spec.id, input=texts, **kwargs)
        return [item.embedding for item in response.data]

    def __repr__(self) -> str:
        return f"OpenAIEmbeddingClient(model={self.spec.id!r})"
