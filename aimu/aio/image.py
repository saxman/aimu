"""Async image-generation surface mirroring :mod:`aimu.models.image_client`.

Exposes:

- :class:`AsyncImageClient` — factory paralleling the sync :class:`ImageClient`,
  wrapping an existing sync :class:`BaseImageClient` (any provider).
- :func:`image_client` / :func:`generate_image` — convenience functions matching
  the shape of :func:`aimu.image_client` / :func:`aimu.generate_image`.

Because image providers either load weights in-process (HuggingFace) or hold a
cloud-API client (Gemini), the factory follows the established wrap pattern: pass
an existing sync client. Direct enum / string construction is refused with a
helpful error pointing at the wrap pattern.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from aimu.models.base import HuggingFaceImageSpec, ImageModel, ImageSpec
    from aimu.models.providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    from .providers.hf.image import AsyncHuggingFaceImageClient

    _HAS_HF_IMAGE = True
except ImportError:
    _HAS_HF_IMAGE = False
    HuggingFaceImageClient = None  # type: ignore[assignment,misc]
    HuggingFaceImageModel = None  # type: ignore[assignment,misc]
    HuggingFaceImageSpec = None  # type: ignore[assignment,misc]
    ImageModel = None  # type: ignore[assignment,misc]
    ImageSpec = None  # type: ignore[assignment,misc]
    AsyncHuggingFaceImageClient = None  # type: ignore[assignment,misc]

try:
    from aimu.models.base import GeminiImageSpec
    from aimu.models.providers.gemini.image import GeminiImageClient, GeminiImageModel

    from .providers.gemini.image import AsyncGeminiImageClient

    _HAS_GEMINI_IMAGE = True
except ImportError:
    _HAS_GEMINI_IMAGE = False
    GeminiImageClient = None  # type: ignore[assignment,misc]
    GeminiImageModel = None  # type: ignore[assignment,misc]
    GeminiImageSpec = None  # type: ignore[assignment,misc]
    AsyncGeminiImageClient = None  # type: ignore[assignment,misc]


_WRAP_GUIDANCE = (
    "Build a sync image client first and pass it to aio.image_client():\n"
    "    sync_client = aimu.image_client({model})\n"
    "    async_client = aio.image_client(sync_client)\n"
    "(For in-process providers like HF diffusers this also avoids loading weights twice.)"
)


def _refuse(model: Any) -> None:
    """Raise the wrap-pattern guidance error for non-client inputs."""
    if _HAS_HF_IMAGE and HuggingFaceImageModel is not None and isinstance(model, HuggingFaceImageModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceImageModel.{model.name}"))
    if _HAS_HF_IMAGE and HuggingFaceImageSpec is not None and isinstance(model, HuggingFaceImageSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceImageSpec({model.id!r})"))
    if _HAS_GEMINI_IMAGE and GeminiImageModel is not None and isinstance(model, GeminiImageModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"GeminiImageModel.{model.name}"))
    if _HAS_GEMINI_IMAGE and GeminiImageSpec is not None and isinstance(model, GeminiImageSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"GeminiImageSpec({model.id!r})"))
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))


class AsyncImageClient:
    """Public async factory for image-generation provider clients.

    Parallel to :class:`aimu.models.ImageClient` for the async surface. Wraps an
    existing sync :class:`BaseImageClient` so weights / API clients are shared.

    Passing an :class:`ImageModel` enum / :class:`ImageSpec` / string raises
    ``ValueError`` pointing at the sync-then-wrap pattern (same convention as
    HuggingFace / LlamaCpp text clients).
    """

    def __init__(self, sync_client: Any):
        if _HAS_HF_IMAGE and isinstance(sync_client, HuggingFaceImageClient):
            self._client: Any = AsyncHuggingFaceImageClient(sync_client)
        elif _HAS_GEMINI_IMAGE and isinstance(sync_client, GeminiImageClient):
            self._client = AsyncGeminiImageClient(sync_client)
        else:
            _refuse(sync_client)
            raise TypeError(
                f"AsyncImageClient expects a sync HuggingFaceImageClient or GeminiImageClient. "
                f"Got: {type(sync_client).__name__}"
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> Any:
        return self._client.spec

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate one or more images. Forwarded to the inner async provider client.

        When ``stream=True`` is in ``**kwargs``, the inner client returns an
        ``AsyncIterator[StreamChunk]`` synchronously (after ``await`` only because
        it had to set up the to-thread pump). Callers should then ``async for``
        the result to consume progress chunks.
        """
        return await self._client.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncImageClient({self._client!r})"


def image_client(model: Any) -> AsyncImageClient:
    """Construct an :class:`AsyncImageClient` by wrapping an existing sync client.

    Accepts a sync :class:`HuggingFaceImageClient` or :class:`GeminiImageClient`.
    Passing an enum / spec / string raises ``ValueError`` pointing at the wrap pattern.
    """
    if not _HAS_HF_IMAGE and not _HAS_GEMINI_IMAGE:
        raise ImportError(
            "Image support requires the [hf] or [google] extra: pip install -e '.[hf]' or pip install -e '.[google]'"
        )
    return AsyncImageClient(model)


async def generate_image(
    prompt: str,
    *,
    model: Any = None,
    format: str = "pil",
    **kwargs: Any,
) -> Any:
    """One-shot async image generation across HF diffusers or Google Gemini.

    Accepts either an existing sync image client (preferred — weights / API client
    reused across calls) or a model spec/string (constructs a fresh sync client
    inside, which loads weights each call for diffusers).

    When ``model`` is omitted, the ``AIMU_IMAGE_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).
    """
    if not _HAS_HF_IMAGE and not _HAS_GEMINI_IMAGE:
        raise ImportError(
            "Image support requires the [hf] or [google] extra: pip install -e '.[hf]' or pip install -e '.[google]'"
        )

    if model is None:
        from aimu.models._internal.model_defaults import IMAGE_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(IMAGE_MODEL_ENV)

    sync_client: Optional[Any] = None
    if _HAS_HF_IMAGE and isinstance(model, HuggingFaceImageClient):
        sync_client = model
    elif _HAS_GEMINI_IMAGE and isinstance(model, GeminiImageClient):
        sync_client = model
    elif isinstance(model, str):
        if model.startswith("gemini:"):
            if not _HAS_GEMINI_IMAGE:
                raise ImportError("Gemini image support requires the [google] extra: pip install -e '.[google]'")
            sync_client = GeminiImageClient(model)
        elif model.startswith("hf:"):
            if not _HAS_HF_IMAGE:
                raise ImportError("HuggingFace image support requires the [hf] extra: pip install -e '.[hf]'")
            sync_client = HuggingFaceImageClient(model)
        else:
            raise ValueError(f"Unrecognised image model string: {model!r}")
    elif _HAS_GEMINI_IMAGE and GeminiImageModel is not None and isinstance(model, GeminiImageModel):
        sync_client = GeminiImageClient(model)
    elif _HAS_HF_IMAGE and HuggingFaceImageModel is not None and isinstance(model, HuggingFaceImageModel):
        sync_client = HuggingFaceImageClient(model)
    else:
        raise TypeError(f"Unrecognised image model: {type(model).__name__}")

    async_client = image_client(sync_client)
    return await async_client.generate(prompt, format=format, **kwargs)
