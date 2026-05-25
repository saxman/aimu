"""HuggingFace ``diffusers``-backed text-to-image client.

``DiffusionClient`` deliberately does *not* inherit ``BaseModelClient``: text and
image are different modalities with disjoint interfaces. ``chat()`` / ``messages``
/ ``StreamChunk`` make no sense for stateless text-to-image generation, and the
"plain data" principle is better served by a focused class than by a forced fit.

Usage::

    from aimu.models.diffusion import DiffusionClient, DiffusionModel

    client = DiffusionClient(DiffusionModel.SDXL_BASE)
    image = client.generate("a watercolor of a fox in a snowy forest")
    image.save("fox.png")

    # Or save directly:
    path = client.generate("a watercolor of a fox", format="path")
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

# Hard import so `HAS_DIFFUSION` in aimu/models/__init__.py accurately reflects
# whether the user can actually generate images. Heavy: pulls torch. Matches the
# convention used by HuggingFaceClient (which imports torch + transformers at
# module load). Pipeline weights are still loaded lazily inside ``_load_pipeline``.
import diffusers  # noqa: F401  # used dynamically via getattr in _load_pipeline

from .._image_output import encode_image
from ..base import DiffusionSpec

logger = logging.getLogger(__name__)


DEFAULT_MODEL_KWARGS: dict[str, Any] = {
    "torch_dtype": "auto",
}


class DiffusionModel(Enum):
    """Catalog of supported text-to-image diffusion models.

    Each member's value is a :class:`DiffusionSpec`. ``.value`` returns the
    HuggingFace repo id; ``.spec`` returns the full spec for inspection.
    Power users can bypass the enum by passing ``"hf:<repo_id>"`` directly to
    :class:`DiffusionClient` for ad-hoc models.
    """

    def __init__(self, spec: DiffusionSpec):
        self._value_ = spec.id
        self.spec = spec

    SD_1_5 = DiffusionSpec(
        "runwayml/stable-diffusion-v1-5",
        pipeline_class="StableDiffusionPipeline",
        default_steps=25,
        default_guidance=7.5,
        default_width=512,
        default_height=512,
    )
    SDXL_BASE = DiffusionSpec(
        "stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=30,
        default_guidance=5.0,
        default_width=1024,
        default_height=1024,
        pipeline_kwargs={"variant": "fp16", "use_safetensors": True},
    )
    SD_3_5_MEDIUM = DiffusionSpec(
        "stabilityai/stable-diffusion-3.5-medium",
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance=3.5,
        default_width=1024,
        default_height=1024,
    )
    FLUX_DEV = DiffusionSpec(
        "black-forest-labs/FLUX.1-dev",
        pipeline_class="FluxPipeline",
        default_steps=28,
        default_guidance=3.5,
        default_width=1024,
        default_height=1024,
    )
    FLUX_SCHNELL = DiffusionSpec(
        "black-forest-labs/FLUX.1-schnell",
        pipeline_class="FluxPipeline",
        default_steps=4,
        default_guidance=0.0,
        default_width=1024,
        default_height=1024,
    )


def _parse_model_string(s: str) -> DiffusionSpec:
    """Parse a ``"hf:<repo_id>"`` string into an ad-hoc :class:`DiffusionSpec`.

    Used to support arbitrary HuggingFace diffusion models not in the
    :class:`DiffusionModel` enum. The auto-detect ``DiffusionPipeline`` loader
    is used; pass an explicit ``DiffusionSpec`` (or a `DiffusionModel` enum
    member) for non-standard pipelines.
    """
    if ":" not in s:
        raise ValueError(
            f"Diffusion model string must be in 'provider:repo_id' form (e.g. "
            f"'hf:runwayml/stable-diffusion-v1-5'). Got: {s!r}"
        )
    provider, repo_id = s.split(":", 1)
    if provider != "hf":
        raise ValueError(
            f"Only 'hf:' provider is supported for diffusion in v1. Got provider: {provider!r}"
        )
    if not repo_id:
        raise ValueError(f"Empty model id after 'hf:': {s!r}")
    return DiffusionSpec(repo_id)


class DiffusionClient:
    """Text-to-image client backed by HuggingFace ``diffusers``.

    Loads pipeline weights lazily on the first :meth:`generate` call. Pass
    a :class:`DiffusionModel` enum member, a :class:`DiffusionSpec`, or a
    ``"hf:<repo_id>"`` string. ``model_kwargs`` is forwarded to
    ``pipeline_cls.from_pretrained()`` (overrides on top of
    :data:`DEFAULT_MODEL_KWARGS` and ``spec.pipeline_kwargs``).
    """

    def __init__(
        self,
        model: Union[DiffusionModel, DiffusionSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, DiffusionModel):
            spec = model.spec
        elif isinstance(model, DiffusionSpec):
            spec = model
        else:
            raise TypeError(
                f"DiffusionClient expects a DiffusionModel member, DiffusionSpec, or "
                f"'hf:<repo_id>' string. Got: {type(model).__name__}"
            )
        self.model = model
        self.spec = spec
        self.model_kwargs = model_kwargs
        self._pipe: Any = None  # lazy

    def _load_pipeline(self) -> Any:
        """Resolve the pipeline class from the diffusers namespace and load weights."""
        try:
            pipeline_cls = getattr(diffusers, self.spec.pipeline_class)
        except AttributeError as exc:
            raise ValueError(
                f"diffusers has no pipeline class {self.spec.pipeline_class!r}. "
                f"Check spec.pipeline_class for {self.spec.id!r}."
            ) from exc

        kwargs = dict(DEFAULT_MODEL_KWARGS)
        if self.spec.pipeline_kwargs:
            kwargs.update(self.spec.pipeline_kwargs)
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)

        logger.info("Loading diffusion pipeline %s (%s)", self.spec.id, self.spec.pipeline_class)
        pipe = pipeline_cls.from_pretrained(self.spec.id, **kwargs)

        try:
            import torch

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                pipe = pipe.to("mps")
        except ImportError:
            pass

        return pipe

    @property
    def pipeline(self) -> Any:
        """The loaded ``diffusers`` pipeline. Triggers weight load on first access."""
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        format: str = "pil",
        output_dir: Optional[Path] = None,
    ) -> Union[Any, list[Any], str, list[str], bytes, list[bytes]]:
        """Generate one or more images from a text prompt.

        Args:
            prompt: Text describing the desired image.
            negative_prompt: Things to avoid (provider-dependent; ignored by FLUX).
                Defaults to the spec's ``default_negative_prompt``.
            width, height: Image dimensions. Default to the spec's defaults.
            num_inference_steps: Sampler step count; spec default if omitted.
            guidance_scale: Classifier-free guidance scale; spec default if omitted.
            seed: Optional RNG seed for reproducibility.
            num_images: How many images to generate in one call.
            format: One of ``"pil"`` / ``"path"`` / ``"bytes"`` / ``"data_url"``.
                See :func:`aimu.models._image_output.encode_image`.
            output_dir: Override the save directory when ``format="path"``.

        Returns:
            A single image (``num_images=1``) or a list of images (``num_images>1``),
            in the encoding selected by ``format``.
        """
        spec = self.spec
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps if num_inference_steps is not None else spec.default_steps,
            "guidance_scale": guidance_scale if guidance_scale is not None else spec.default_guidance,
            "width": width if width is not None else spec.default_width,
            "height": height if height is not None else spec.default_height,
            "num_images_per_prompt": num_images,
        }

        effective_negative = negative_prompt if negative_prompt is not None else spec.default_negative_prompt
        if effective_negative is not None:
            call_kwargs["negative_prompt"] = effective_negative

        if seed is not None:
            import torch

            device = getattr(self.pipeline, "device", None)
            generator = torch.Generator(device=device.type if device is not None else "cpu").manual_seed(seed)
            call_kwargs["generator"] = generator

        result = self.pipeline(**call_kwargs)
        images = result.images if hasattr(result, "images") else result

        encoded = [encode_image(img, format=format, prompt=prompt, output_dir=output_dir) for img in images]
        return encoded[0] if num_images == 1 else encoded

    def __repr__(self) -> str:
        loaded = "loaded" if self._pipe is not None else "unloaded"
        return f"DiffusionClient(model={self.spec.id!r}, {loaded})"
