"""Google Gemini image-generation client (Nano Banana) via the ``google-genai`` SDK.

Inherits :class:`BaseImageClient` — public :meth:`generate` (format conversion,
num_images plumbing) lives on the base; this subclass implements
:meth:`_generate` to call ``genai.Client().models.generate_content`` with
``response_modalities=[Modality.IMAGE]`` and decode the returned inline_data.

Authentication: reads ``GOOGLE_API_KEY`` from the environment (matches the
existing text :class:`GeminiClient`), or accepts ``api_key=`` in the
``model_kwargs`` dict.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any, Optional, Union

from dotenv import load_dotenv
from PIL import Image

# Hard imports so ``HAS_GEMINI_IMAGE`` accurately reflects "the dep is installed"
# (matches the convention used by every other provider in this package).
from google import genai
from google.genai import types as genai_types

from ..base import BaseImageClient, GeminiImageSpec, ImageModel

logger = logging.getLogger(__name__)
load_dotenv()


class GeminiImageModel(ImageModel):
    """Catalog of supported Google Gemini image-generation models.

    Each member's value is a :class:`GeminiImageSpec`. ``.value`` returns the
    Gemini API model id string.
    """

    # "Nano Banana" — Gemini 2.5 Flash Image (GA).
    NANO_BANANA = GeminiImageSpec("gemini-2.5-flash-image")
    # The preview channel, retained for users who pinned it earlier.
    NANO_BANANA_PREVIEW = GeminiImageSpec("gemini-2.5-flash-image-preview")


# Short-name aliases users can pass via "gemini:<alias>".
_ALIASES = {
    "nano-banana": "gemini-2.5-flash-image",
    "nano-banana-preview": "gemini-2.5-flash-image-preview",
}


def _parse_model_string(s: str) -> GeminiImageSpec:
    """Parse a ``"gemini:<model_id>"`` string into a :class:`GeminiImageSpec`."""
    if ":" not in s:
        raise ValueError(
            f"Gemini image model string must be in 'provider:model_id' form (e.g. "
            f"'gemini:nano-banana' or 'gemini:gemini-2.5-flash-image'). Got: {s!r}"
        )
    provider, model_id = s.split(":", 1)
    if provider != "gemini":
        raise ValueError(f"Expected 'gemini:' provider for Gemini image. Got: {provider!r}")
    if not model_id:
        raise ValueError(f"Empty model id after 'gemini:': {s!r}")
    resolved = _ALIASES.get(model_id, model_id)
    return GeminiImageSpec(resolved)


class GeminiImageClient(BaseImageClient):
    """Text-to-image client backed by Google's Gemini API (e.g. Nano Banana).

    Accepts a :class:`GeminiImageModel` enum member, a :class:`GeminiImageSpec`,
    or a ``"gemini:<id_or_alias>"`` string. ``model_kwargs`` may contain
    ``api_key=`` to override the ``GOOGLE_API_KEY`` env var.
    """

    MODELS = GeminiImageModel

    def __init__(
        self,
        model: Union[GeminiImageModel, GeminiImageSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, GeminiImageModel):
            spec = model.spec
        elif isinstance(model, GeminiImageSpec):
            spec = model
        else:
            raise TypeError(
                f"GeminiImageClient expects a GeminiImageModel member, GeminiImageSpec, or "
                f"'gemini:<id_or_alias>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=dict(model_kwargs or {}))
        self.spec = spec
        api_key = self.model_kwargs.pop("api_key", None) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Export the env var, put it in a .env file, "
                "or pass api_key= via model_kwargs to GeminiImageClient."
            )
        self._api_key = api_key
        self._client: Any = None  # lazy

    @property
    def client(self) -> Any:
        """The underlying ``google.genai.Client``. Constructed lazily on first access."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_image_config(self, aspect_ratio: Optional[str], image_size: Optional[str]) -> Optional[Any]:
        kwargs = dict(self.spec.image_config_kwargs or {})
        if aspect_ratio is None:
            aspect_ratio = self.spec.default_aspect_ratio
        if image_size is None:
            image_size = self.spec.default_image_size
        if aspect_ratio is not None:
            kwargs["aspect_ratio"] = aspect_ratio
        if image_size is not None:
            kwargs["image_size"] = image_size
        return genai_types.ImageConfig(**kwargs) if kwargs else None

    def _generate_one(self, prompt: str, image_config: Any) -> Image.Image:
        """Issue one generate_content call and extract the first image part."""
        config = genai_types.GenerateContentConfig(
            response_modalities=[genai_types.Modality.IMAGE],
            image_config=image_config,
        )
        response = self.client.models.generate_content(
            model=self.spec.id,
            contents=[prompt],
            config=config,
        )

        # Walk parts looking for inline_data with image mime.
        for cand in response.candidates or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    return Image.open(BytesIO(inline.data))

        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"Gemini API returned no image data — the model may have refused the prompt. prompt_feedback={feedback!r}"
        )

    def _generate(
        self,
        prompt: str,
        *,
        num_images: int = 1,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        **_ignored: Any,
    ) -> list:
        """Provider-specific generation. Returns a list of PIL Images.

        Nano Banana's ``generate_content`` returns one image per call, so
        ``num_images > 1`` issues N calls.
        """
        image_config = self._build_image_config(aspect_ratio, image_size)
        return [self._generate_one(prompt, image_config) for _ in range(num_images)]

    def _generate_streamed(
        self,
        prompt: str,
        *,
        num_images: int = 1,
        format: str = "pil",
        output_dir: Optional[Any] = None,
        preview_every: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        **_ignored: Any,
    ):
        """Coarse streaming: one start chunk + one final chunk per image.

        Gemini Nano Banana doesn't expose per-step progress via its API; this
        emits a "started" :attr:`StreamingContentType.IMAGE_GENERATING` chunk
        before each API call and a ``final=True`` chunk after, carrying the
        encoded result. ``preview_every`` is accepted but has no effect (no
        intermediate latents exist on the cloud side).
        """
        del preview_every  # not applicable for cloud-API providers

        # Local imports keep base.py light and avoid circular import paths.
        from .._image_output import encode_image
        from ..base import StreamChunk, StreamingContentType

        image_config = self._build_image_config(aspect_ratio, image_size)

        for i in range(num_images):
            yield StreamChunk(
                StreamingContentType.IMAGE_GENERATING,
                {
                    "step": 0,
                    "total_steps": 1,
                    "image": None,
                    "final": False,
                    "result": None,
                    "image_index": i,
                    "num_images": num_images,
                },
            )
            img = self._generate_one(prompt, image_config)
            encoded = encode_image(img, format=format, prompt=prompt, output_dir=output_dir)
            yield StreamChunk(
                StreamingContentType.IMAGE_GENERATING,
                {
                    "step": 1,
                    "total_steps": 1,
                    "image": img,
                    "final": True,
                    "result": encoded,
                    "image_index": i,
                    "num_images": num_images,
                },
            )

    def __repr__(self) -> str:
        return f"GeminiImageClient(model={self.spec.id!r})"
