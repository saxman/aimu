"""Image-modality base types: the image specs, the ``ImageModel`` enum, and ``BaseImageClient``.

Disjoint from the text stack — image models have no concept of tools / thinking /
vision-input — so ``BaseImageClient`` is its own ABC rather than a ``BaseModelClient``
subclass. Only ``StreamChunk`` / ``StreamingContentType`` are shared (via ``.shared``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

from .shared import StreamChunk, StreamingContentType


@dataclass
class ImageSpec:
    """Base descriptor for a single image-generation model.

    Sibling to :class:`ModelSpec`; deliberately disjoint because image models have
    no concept of tools / thinking / vision-input. Provider-specific subclasses
    (:class:`HuggingFaceImageSpec`, :class:`GeminiImageSpec`) add their own defaults.

    Equality and hash are by ``id`` only — matching :class:`ModelSpec` — so the spec
    can be used directly as an enum value even when carrying dict fields.

    ``max_prompt_tokens`` is the model's text-encoder prompt budget — the image-side
    analog of :class:`ModelSpec`'s capability flags. ``77`` is the CLIP default (SD,
    SDXL); T5-based models carry more (SD3 ≈ 256, FLUX-dev ≈ 512, PixArt-Σ ≈ 300);
    ``None`` means no practical cap (cloud providers that take natural-language prompts).
    """

    id: str
    max_prompt_tokens: Optional[int] = 77
    # Whether the model's generation API accepts a separate negative prompt. False for
    # guidance-distilled / conversational models (FLUX.2 Klein, Gemini Nano Banana) that
    # take a single prose prompt; for those, BaseImageClient folds any caller-supplied
    # negative prompt into the main prompt instead of passing it as a kwarg.
    supports_negative_prompt: bool = True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageSpec):
            return self.id == other.id
        return NotImplemented


@dataclass(eq=False)
class HuggingFaceImageSpec(ImageSpec):
    """Descriptor for a single HuggingFace ``diffusers``-backed image model.

    Carries image-generation defaults (steps, guidance, dimensions) plus the
    loader's pipeline class name. ``pipeline_class`` names a class in the
    ``diffusers`` namespace (e.g. ``"StableDiffusionXLPipeline"``, ``"FluxPipeline"``);
    the client resolves it lazily via ``getattr(diffusers, pipeline_class)``.

    ``eq=False`` inherits :class:`ImageSpec`'s id-only equality + hash; the
    dataclass-default per-field ``__eq__`` would otherwise shadow it.
    """

    pipeline_class: str = "DiffusionPipeline"
    img2img_pipeline_class: Optional[str] = None
    img2img_uses_strength: bool = True
    default_steps: int = 30
    default_guidance: float = 7.5
    default_width: int = 1024
    default_height: int = 1024
    default_negative_prompt: Optional[str] = None
    pipeline_kwargs: Optional[dict] = field(default=None)


@dataclass(eq=False)
class GeminiImageSpec(ImageSpec):
    """Descriptor for a single Google Gemini image-generation model (e.g. Nano Banana).

    Cloud-API models don't have a pipeline class or denoising steps. Carries the
    API-side ``id`` (e.g. ``"gemini-2.5-flash-image"``) plus optional defaults the
    underlying SDK uses to build :class:`google.genai.types.ImageConfig`.

    ``eq=False`` inherits :class:`ImageSpec`'s id-only equality + hash; the
    dataclass-default per-field ``__eq__`` would otherwise shadow it.
    """

    # Cloud models take natural-language prompts with no CLIP-style token cap.
    max_prompt_tokens: Optional[int] = None
    # Nano Banana (and the Gemini image family generally) take a single prose prompt;
    # there is no negative-prompt parameter, so a caller-supplied one is folded into the
    # prompt by BaseImageClient. Applies to enum members and ad-hoc "gemini:..." strings.
    supports_negative_prompt: bool = False
    default_aspect_ratio: Optional[str] = None  # e.g. "1:1", "16:9"
    default_image_size: Optional[str] = None  # e.g. "1024x1024" (SDK-dependent)
    image_config_kwargs: Optional[dict] = field(default=None)


class ImageModel(Enum):
    """Base enum for image-generation provider model catalogs.

    Parallel to :class:`Model`. Each member's value is an :class:`ImageSpec`
    (or subclass); the constructor sets ``_value_`` from ``spec.id`` and stores
    the spec on the member.
    """

    def __init__(self, spec: ImageSpec):
        self._value_ = spec.id
        self.spec = spec


class BaseImageClient(ABC):
    """Abstract base for image-generation provider clients.

    Parallel to :class:`BaseModelClient` for image modality. Subclasses implement
    :meth:`_generate` returning a list of PIL Images; the public :meth:`generate`
    wraps that with the shared format-conversion helper
    (:func:`aimu.models._image_output.encode_image`) so every image client offers
    the same ``format="pil"|"path"|"bytes"|"data_url"`` surface.
    """

    MODELS = ImageModel

    model: Any
    spec: ImageSpec
    model_kwargs: Optional[dict]

    @abstractmethod
    def __init__(self, model: Any, model_kwargs: Optional[dict] = None):
        self.model = model
        self.model_kwargs = model_kwargs

    @property
    def max_prompt_tokens(self) -> Optional[int]:
        """The model's text-encoder prompt budget in tokens; ``None`` means no practical cap."""
        return self.spec.max_prompt_tokens

    @abstractmethod
    def _generate(self, prompt: str, *, num_images: int = 1, **kwargs: Any) -> list:
        """Provider-specific generation. Returns a list of PIL ``Image.Image`` objects.

        Provider-specific keyword args (e.g. ``aspect_ratio`` for Gemini,
        ``num_inference_steps`` for diffusers) are accepted via ``**kwargs`` and
        validated/applied by each subclass.
        """

    def generate(
        self,
        prompt: str,
        *,
        num_images: int = 1,
        format: str = "pil",
        output_dir: Optional[Any] = None,
        stream: bool = False,
        preview_every: Optional[int] = None,
        reference_image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate one or more images from a text prompt.

        Subclasses define the provider-specific ``**kwargs``. The base handles
        ``num_images`` validation, format conversion via
        :func:`aimu.models._image_output.encode_image`, and the single-image-vs-list
        return convention.

        When ``stream=True``, returns an iterator of :class:`StreamChunk` objects with
        phase :attr:`StreamingContentType.IMAGE_GENERATING`. Providers that expose
        step-level callbacks (e.g. HuggingFace diffusers) emit one chunk per step;
        providers without (Gemini Nano Banana) emit coarse start/done chunks per image.
        ``preview_every=N`` opts in to per-step latent-decoded previews (carried in
        ``chunk.content["image"]``); default ``None`` keeps step chunks lightweight.

        ``reference_image`` enables img2img generation: the output is guided by the
        reference alongside the prompt. Accepts a PIL Image, file path string,
        :class:`pathlib.Path`, raw bytes, ``data:image/...`` URL, or ``http(s)://`` URL.
        For HuggingFace, the model spec must have ``img2img_pipeline_class`` set and
        the ``strength`` kwarg controls deviation from the reference (default ``0.75``).
        For Gemini, the reference is included as inline image data in the request.
        Output size for HuggingFace img2img is determined by the reference image;
        ``width`` and ``height`` kwargs are ignored.
        """
        if num_images < 1:
            raise ValueError(f"num_images must be >= 1, got {num_images}")

        if kwargs.get("negative_prompt") and not self.spec.supports_negative_prompt:
            raise ValueError(
                f"Image model {self.spec.id!r} has no negative-prompt support "
                f"(spec.supports_negative_prompt is False). Omit negative_prompt, or fold the "
                f"avoidance into the prompt as prose. Check spec.supports_negative_prompt to "
                f"decide which to use."
            )

        if stream:
            return self._generate_streamed(
                prompt,
                num_images=num_images,
                format=format,
                output_dir=output_dir,
                preview_every=preview_every,
                reference_image=reference_image,
                **kwargs,
            )

        from .._image_output import encode_image  # local import keeps base.py light

        images = self._generate(prompt, num_images=num_images, reference_image=reference_image, **kwargs)
        encoded = [encode_image(img, format=format, prompt=prompt, output_dir=output_dir) for img in images]
        return encoded[0] if num_images == 1 else encoded

    def _generate_streamed(
        self,
        prompt: str,
        *,
        num_images: int = 1,
        format: str = "pil",
        output_dir: Optional[Any] = None,
        preview_every: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Default streaming implementation: no per-step progress.

        Calls the blocking :meth:`_generate` and emits one
        :attr:`StreamingContentType.IMAGE_GENERATING` chunk per completed image, each
        marked ``final=True`` with the encoded ``result``. Providers that expose
        step-level callbacks override this to emit progress mid-generation.

        ``preview_every`` is accepted but ignored at this default level — the base
        has no notion of intermediate steps. Subclasses honour it.
        """
        from .._image_output import encode_image  # local import keeps base.py light

        del preview_every  # unused at the default level; provider overrides honour it

        images = self._generate(prompt, num_images=num_images, **kwargs)
        for i, img in enumerate(images):
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
        return f"{type(self).__name__}(model={self.spec.id!r})"
