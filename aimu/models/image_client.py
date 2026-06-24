"""Image-client factory paralleling :mod:`aimu.models.model_client`.

Exposes:

- :func:`resolve_image_model_string`: parse ``"provider:model_id"`` for image providers.
- :func:`resolve_image_model_enum`: enum / string / bare-name resolution.
- :class:`ImageClient`: factory ``BaseImageClient`` that dispatches to the right
  concrete client based on the model enum / spec / string passed in.

Mirrors the text-side dispatch (:class:`aimu.models.ModelClient`,
:func:`aimu.models.resolve_model_string`). The shared dispatch logic lives in
:mod:`aimu.models._internal.factory`.
"""

from __future__ import annotations

from typing import Any, Optional

from ._internal.factory import (
    FactoryDelegate,
    ProviderEntry,
    available_registry,
    build_client,
    factory_model_kwargs,
    resolve_model_string,
)
from .base import BaseImageClient, ImageModel, ImageSpec

# --- Optional provider imports ---

try:
    from .providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    _HAS_HF_IMAGE = True
except ImportError:
    _HAS_HF_IMAGE = False
    HuggingFaceImageClient = None  # type: ignore[assignment,misc]
    HuggingFaceImageModel = None  # type: ignore[assignment,misc]

try:
    from .providers.gemini.image import GeminiImageClient, GeminiImageModel

    _HAS_GEMINI_IMAGE = True
except ImportError:
    _HAS_GEMINI_IMAGE = False
    GeminiImageClient = None  # type: ignore[assignment,misc]
    GeminiImageModel = None  # type: ignore[assignment,misc]

_HF_HINT = (
    "HuggingFace image support requires the [hf] extra (diffusers, torch, transformers, Pillow): pip install -e '.[hf]'"
)
_GEMINI_HINT = "Gemini image support requires the [google] extra (google-genai, Pillow): pip install -e '.[google]'"


def _entries() -> list[ProviderEntry]:
    return [
        ProviderEntry("hf", _HAS_HF_IMAGE, HuggingFaceImageModel, HuggingFaceImageClient, _HF_HINT),
        ProviderEntry("gemini", _HAS_GEMINI_IMAGE, GeminiImageModel, GeminiImageClient, _GEMINI_HINT),
    ]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(ImageModel subclass, ImageClient subclass)`` (installed only)."""
    return available_registry(_entries())


def resolve_image_model_string(model_str: str) -> ImageModel:
    """Look up an image-provider model enum from a ``"provider:model_id"`` string.

    Mirrors :func:`aimu.models.resolve_model_string` for image providers. Only matches
    *exact* enum-member values; for ad-hoc HuggingFace repos or Gemini aliases, pass the
    ``"provider:..."`` string directly to :class:`ImageClient`.
    """
    return resolve_model_string(model_str, _entries(), modality="image")


def resolve_image_model_enum(model: ImageModel | str) -> ImageModel:
    """Resolve an image model to an ``ImageModel`` enum member.

    Accepts, in order:

    - an ``ImageModel`` enum member (returned unchanged);
    - a ``"provider:model_id"`` string (delegates to :func:`resolve_image_model_string`);
    - a bare enum-member *name* (e.g. ``"FLUX_2_KLEIN_4B"`` / ``"NANO_BANANA"``), looked
      up across every installed image-provider enum.

    A bare name present in more than one provider's enum is **ambiguous** and raises
    ``ValueError``: pass an explicit ``"provider:model_id"`` string to disambiguate.
    Parallel to :func:`aimu.resolve_model_enum` for the text modality.
    """
    if isinstance(model, ImageModel):
        return model
    if not isinstance(model, str):
        raise TypeError(f"Expected an ImageModel enum member or a string, got {type(model).__name__}.")
    if ":" in model:
        return resolve_image_model_string(model)

    registry = _provider_registry()
    matches = [
        (prefix, enum_cls[model]) for prefix, (enum_cls, _client) in registry.items() if model in enum_cls.__members__
    ]
    if len(matches) == 1:
        return matches[0][1]
    if not matches:
        available = sorted({name for _, (enum_cls, _client) in registry.items() for name in enum_cls.__members__})
        raise ValueError(
            f"Unknown image model name {model!r}. Pass a 'provider:model_id' string or one of: {available}"
        )
    providers = ", ".join(prefix for prefix, _ in matches)
    options = ", ".join(f"{prefix}:{member.value}" for prefix, member in matches)
    raise ValueError(
        f"Image model name {model!r} is ambiguous across providers ({providers}). "
        f"Disambiguate with a 'provider:model_id' string, e.g. one of: {options}"
    )


class ImageClient(FactoryDelegate):
    """Public factory for image-generation provider clients.

    Parallel to :class:`aimu.models.ModelClient` for the image modality. Accepts
    a provider's :class:`ImageModel` enum member, an :class:`ImageSpec`, or a
    ``"provider:model_id"`` string. Provider-specific construction kwargs are
    passed directly (the legacy ``model_kwargs={...}`` form is deprecated).

    Examples::

        from aimu.models import ImageClient, HuggingFaceImageModel, GeminiImageModel

        # Enum form
        client = ImageClient(HuggingFaceImageModel.SD_1_5)
        client = ImageClient(GeminiImageModel.NANO_BANANA)

        # String form (no enum import needed)
        client = ImageClient("hf:runwayml/stable-diffusion-v1-5")
        client = ImageClient("gemini:nano-banana")

    Provider-specific construction kwargs are passed directly::

        ImageClient(HuggingFaceImageModel.SDXL_BASE, variant="fp16")
        ImageClient(GeminiImageModel.NANO_BANANA, api_key="...")
    """

    def __init__(self, model: ImageModel | ImageSpec | str, **kwargs: Any) -> None:
        self._client: BaseImageClient = build_client(
            model,
            factory_model_kwargs(kwargs),
            _entries(),
            modality="image",
            model_base=ImageModel,
            spec_base=ImageSpec,
        )

    @property
    def max_prompt_tokens(self) -> Optional[int]:
        return self._client.max_prompt_tokens

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate one or more images. Forwarded to the inner client's
        :meth:`BaseImageClient.generate`."""
        return self._client.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"ImageClient({self._client!r})"
