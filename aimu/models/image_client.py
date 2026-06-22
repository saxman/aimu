"""Image-client factory paralleling :mod:`aimu.models.model_client`.

Exposes:

- :func:`resolve_image_model_string`: parse ``"provider:model_id"`` for image providers.
- :class:`ImageClient`: factory ``BaseImageClient`` that dispatches to the right
  concrete client based on the model enum / spec / string passed in.

Mirrors the text-side dispatch (:class:`aimu.models.ModelClient`,
:func:`aimu.models.resolve_model_string`).
"""

from __future__ import annotations

from typing import Any, Optional, Union

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


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(ImageModel subclass, ImageClient subclass)``."""
    registry: dict[str, tuple] = {}
    if _HAS_HF_IMAGE:
        registry["hf"] = (HuggingFaceImageModel, HuggingFaceImageClient)
    if _HAS_GEMINI_IMAGE:
        registry["gemini"] = (GeminiImageModel, GeminiImageClient)
    return registry


def resolve_image_model_string(model_str: str) -> ImageModel:
    """Look up an image-provider model enum from a ``"provider:model_id"`` string.

    Mirrors :func:`aimu.models.resolve_model_string` for image providers.

    Examples::

        resolve_image_model_string("hf:runwayml/stable-diffusion-v1-5")
        resolve_image_model_string("gemini:nano-banana")
        resolve_image_model_string("gemini:gemini-2.5-flash-image")

    Note that string-form construction with arbitrary repo ids / Gemini aliases
    is handled inside each concrete client's ``__init__``; this function only
    matches *exact* enum-member values. For ad-hoc HuggingFace repos or Gemini
    aliases, pass the ``"provider:..."`` string directly to :class:`ImageClient`
    instead of calling this helper.
    """
    if ":" not in model_str:
        raise ValueError(
            f"Image model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(_provider_registry())}"
        )
    provider, _, model_id = model_str.partition(":")
    registry = _provider_registry()
    if provider not in registry:
        raise ValueError(
            f"Unknown image provider {provider!r}. Available providers (with installed deps): {sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(f"Provider {provider!r} has no image model id {model_id!r}. Available: {available}")


def resolve_image_model_enum(model: Union[ImageModel, str]) -> ImageModel:
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


class ImageClient:
    """Public factory for image-generation provider clients.

    Parallel to :class:`aimu.models.ModelClient` for the image modality. Accepts
    a provider's :class:`ImageModel` enum member, an :class:`ImageSpec`, or a
    ``"provider:model_id"`` string. Provider-specific construction kwargs are
    forwarded as ``model_kwargs`` to the concrete client.

    Examples::

        from aimu.models import ImageClient, HuggingFaceImageModel, GeminiImageModel

        # Enum form
        client = ImageClient(HuggingFaceImageModel.SD_1_5)
        client = ImageClient(GeminiImageModel.NANO_BANANA)

        # String form (no enum import needed)
        client = ImageClient("hf:runwayml/stable-diffusion-v1-5")
        client = ImageClient("gemini:nano-banana")

    Provider-specific kwargs are forwarded::

        ImageClient(HuggingFaceImageModel.SDXL_BASE, model_kwargs={"variant": "fp16"})
        ImageClient(GeminiImageModel.NANO_BANANA, model_kwargs={"api_key": "..."})
    """

    def __init__(self, model: Union[ImageModel, ImageSpec, str], model_kwargs: Optional[dict] = None) -> None:
        # String-form: route by prefix so ad-hoc ids (HuggingFace repo, Gemini alias)
        # are accepted without requiring the model to be in our enum catalogs.
        if isinstance(model, str):
            if ":" not in model:
                raise ValueError(f"Image model string must be in 'provider:model_id' form, got: {model!r}")
            provider, _, _model_id = model.partition(":")
            if provider == "hf":
                if not _HAS_HF_IMAGE:
                    raise ImportError(
                        "HuggingFace image support requires the [hf] extra (diffusers, torch, transformers, Pillow): "
                        "pip install -e '.[hf]'"
                    )
                self._client: BaseImageClient = HuggingFaceImageClient(model, model_kwargs=model_kwargs)
                return
            if provider == "gemini":
                if not _HAS_GEMINI_IMAGE:
                    raise ImportError(
                        "Gemini image support requires the [google] extra (google-genai, Pillow): "
                        "pip install -e '.[google]'"
                    )
                self._client = GeminiImageClient(model, model_kwargs=model_kwargs)
                return
            raise ValueError(f"Unknown image provider {provider!r}. Available: {sorted(_provider_registry())}")

        if isinstance(model, ImageSpec) and not isinstance(model, ImageModel):
            raise TypeError(
                "Pass an ImageModel enum member (e.g. HuggingFaceImageModel.SD_1_5) or a "
                "'provider:model_id' string. ImageSpec is the value type held by enum members."
            )

        # Enum form: isinstance against the concrete enum classes (guarded by _HAS_* flags).
        if _HAS_HF_IMAGE and isinstance(model, HuggingFaceImageModel):
            self._client = HuggingFaceImageClient(model, model_kwargs=model_kwargs)
        elif _HAS_GEMINI_IMAGE and isinstance(model, GeminiImageModel):
            self._client = GeminiImageClient(model, model_kwargs=model_kwargs)
        else:
            raise ValueError(
                f"No available client for image-model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed (pip install 'aimu[hf]' or 'aimu[google]')."
            )

    # --- Delegate the common interface to the inner concrete client ---

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> ImageSpec:
        return self._client.spec

    @property
    def model_kwargs(self) -> Optional[dict]:
        return self._client.model_kwargs

    @property
    def max_prompt_tokens(self) -> Optional[int]:
        return self._client.max_prompt_tokens

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate one or more images. Forwarded to the inner client's
        :meth:`BaseImageClient.generate`."""
        return self._client.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"ImageClient({self._client!r})"
