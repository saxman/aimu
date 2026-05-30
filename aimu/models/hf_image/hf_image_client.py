"""HuggingFace ``diffusers``-backed text-to-image client.

``HuggingFaceImageClient`` inherits :class:`BaseImageClient` — the public
:meth:`generate` (format conversion, num_images plumbing) lives on the base;
this subclass just implements :meth:`_generate` to call the loaded diffusers
pipeline and return PIL Images.

Usage::

    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    client = HuggingFaceImageClient(HuggingFaceImageModel.SDXL_BASE)
    image = client.generate("a watercolor of a fox in a snowy forest")
    image.save("fox.png")

    # Or save directly via the base's format selector:
    path = client.generate("a watercolor of a fox", format="path")
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

# Hard import so ``HAS_HF_IMAGE`` in aimu/models/__init__.py accurately reflects
# whether the optional dep is installed. Heavy: pulls torch via diffusers' lazy
# loader. Pipeline weights are still loaded lazily inside ``_load_pipeline``.
import diffusers  # noqa: F401  # used dynamically via getattr in _load_pipeline

from .._hf_device import auto_place_pipeline, move_to_device, pop_device_hint
from ..base import BaseImageClient, HuggingFaceImageSpec, ImageModel

logger = logging.getLogger(__name__)


DEFAULT_MODEL_KWARGS: dict[str, Any] = {
    "torch_dtype": "auto",
}


class HuggingFaceImageModel(ImageModel):
    """Catalog of supported HuggingFace ``diffusers`` text-to-image models.

    Each member's value is a :class:`HuggingFaceImageSpec`. ``.value`` returns
    the HuggingFace repo id; ``.spec`` returns the full spec. Power users can
    bypass the enum by passing a ``"hf:<repo_id>"`` string directly to
    :class:`HuggingFaceImageClient`.
    """

    SD_1_5 = HuggingFaceImageSpec(
        "runwayml/stable-diffusion-v1-5",
        pipeline_class="StableDiffusionPipeline",
        default_steps=25,
        default_guidance=7.5,
        default_width=512,
        default_height=512,
    )
    SDXL_BASE = HuggingFaceImageSpec(
        "stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=30,
        default_guidance=5.0,
        default_width=1024,
        default_height=1024,
        pipeline_kwargs={"variant": "fp16", "use_safetensors": True},
    )
    SD_3_5_MEDIUM = HuggingFaceImageSpec(
        "stabilityai/stable-diffusion-3.5-medium",
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance=3.5,
        default_width=1024,
        default_height=1024,
        max_prompt_tokens=256,  # T5-XXL encoder
    )
    FLUX_DEV = HuggingFaceImageSpec(
        "black-forest-labs/FLUX.1-dev",
        pipeline_class="FluxPipeline",
        default_steps=28,
        default_guidance=3.5,
        default_width=1024,
        default_height=1024,
        max_prompt_tokens=512,  # T5-XXL encoder
    )
    FLUX_SCHNELL = HuggingFaceImageSpec(
        "black-forest-labs/FLUX.1-schnell",
        pipeline_class="FluxPipeline",
        default_steps=4,
        default_guidance=0.0,
        default_width=1024,
        default_height=1024,
        max_prompt_tokens=256,  # T5-XXL encoder
    )


def _parse_model_string(s: str) -> HuggingFaceImageSpec:
    """Parse a ``"hf:<repo_id>"`` string into an ad-hoc :class:`HuggingFaceImageSpec`."""
    if ":" not in s:
        raise ValueError(
            f"HuggingFace image model string must be in 'provider:repo_id' form (e.g. "
            f"'hf:runwayml/stable-diffusion-v1-5'). Got: {s!r}"
        )
    provider, repo_id = s.split(":", 1)
    if provider != "hf":
        raise ValueError(f"Only 'hf:' provider is supported for HuggingFaceImageClient. Got provider: {provider!r}")
    if not repo_id:
        raise ValueError(f"Empty model id after 'hf:': {s!r}")
    return HuggingFaceImageSpec(repo_id)


class HuggingFaceImageClient(BaseImageClient):
    """Text-to-image client backed by HuggingFace ``diffusers``.

    Loads pipeline weights lazily on the first :meth:`_generate` call. Pass a
    :class:`HuggingFaceImageModel` enum member, a :class:`HuggingFaceImageSpec`,
    or a ``"hf:<repo_id>"`` string. ``model_kwargs`` is forwarded to
    ``pipeline_cls.from_pretrained()`` (overrides on top of
    :data:`DEFAULT_MODEL_KWARGS` and ``spec.pipeline_kwargs``).

    Placement is automatic by default: on first generation the client measures the
    loaded pipeline's size and the *free* memory on each visible GPU (so it accounts
    for other processes, e.g. a local LLM server), then picks the cheapest strategy
    that fits — pin to the freest GPU, else model CPU offload, else sequential CPU
    offload (see :func:`aimu.models._hf_device.auto_place_pipeline`). Override it:

    - ``model_kwargs={"device": "cuda:1"}`` — pin the whole pipeline to one GPU.
    - ``model_kwargs={"device_map": "balanced"}`` — hand placement to diffusers/
      accelerate (or any other diffusers-supported ``device_map`` value).

    CPU offload requires ``accelerate``; without it, an oversized model is pinned to
    the freest GPU with a warning.
    """

    MODELS = HuggingFaceImageModel

    def __init__(
        self,
        model: Union[HuggingFaceImageModel, HuggingFaceImageSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, HuggingFaceImageModel):
            spec = model.spec
        elif isinstance(model, HuggingFaceImageSpec):
            spec = model
        else:
            raise TypeError(
                f"HuggingFaceImageClient expects a HuggingFaceImageModel member, "
                f"HuggingFaceImageSpec, or 'hf:<repo_id>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec
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

        device = pop_device_hint(kwargs)

        logger.info("Loading diffusion pipeline %s (%s)", self.spec.id, self.spec.pipeline_class)
        pipe = pipeline_cls.from_pretrained(self.spec.id, **kwargs)

        # Placement precedence (most → least explicit):
        #   1. ``device_map`` in model_kwargs — caller shards manually; diffusers owns
        #      placement, so ``.to()`` must not be called afterward.
        #   2. ``device`` hint — pin the whole pipeline to one GPU.
        #   3. neither — auto-plan from the live free VRAM across all visible GPUs.
        if "device_map" in kwargs:
            if device is not None:
                logger.warning("Ignoring device=%r because device_map=%r was set.", device, kwargs["device_map"])
            return pipe
        if device is not None:
            return move_to_device(pipe, device)
        return auto_place_pipeline(pipe)

    @property
    def pipeline(self) -> Any:
        """The loaded ``diffusers`` pipeline. Triggers weight load on first access."""
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    def _build_call_kwargs(
        self,
        prompt: str,
        *,
        num_images: int,
        negative_prompt: Optional[str],
        width: Optional[int],
        height: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        seed: Optional[int],
    ) -> dict[str, Any]:
        """Compose the kwargs dict passed to the diffusers pipeline.

        Shared by :meth:`_generate` (blocking) and :meth:`_generate_streamed`
        (callback-driven). Defaults fall back to ``self.spec``.
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

        return call_kwargs

    def _generate(
        self,
        prompt: str,
        *,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **_ignored: Any,
    ) -> list:
        """Provider-specific generation. Returns a list of PIL Images.

        Public callers should use :meth:`generate` (inherited from
        :class:`BaseImageClient`), which adds format conversion via
        :func:`encode_image`.
        """
        call_kwargs = self._build_call_kwargs(
            prompt,
            num_images=num_images,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        result = self.pipeline(**call_kwargs)
        images = result.images if hasattr(result, "images") else result
        return list(images)

    def _decode_latents_to_pil(self, latents: Any) -> list:
        """Decode an intermediate latents tensor to PIL Images via the pipeline's VAE.

        Expensive (~50–200 ms per call on GPU). Called only when ``preview_every``
        opts in for a given step.
        """
        import torch

        pipe = self.pipeline
        if not hasattr(pipe, "vae") or not hasattr(pipe, "image_processor"):
            return []  # not all pipelines expose these; preview is best-effort
        with torch.no_grad():
            scaled = latents / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        images = pipe.image_processor.postprocess(decoded, output_type="pil")
        return list(images) if isinstance(images, (list, tuple)) else [images]

    def _generate_streamed(
        self,
        prompt: str,
        *,
        num_images: int = 1,
        format: str = "pil",
        output_dir: Optional[Any] = None,
        preview_every: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **_ignored: Any,
    ):
        """Streaming generation: yields one IMAGE_GENERATING chunk per denoising step.

        Implementation runs the diffusers pipeline in a background thread; a
        ``callback_on_step_end`` closure pushes per-step ``StreamChunk``s onto a
        :class:`queue.Queue`. The main thread (this generator) drains the queue,
        forwarding chunks to the caller. When ``preview_every=N`` is set, latents
        at matching steps are decoded via ``pipe.vae`` and included in the chunk.

        Final ``final=True`` chunks (one per image when ``num_images > 1``) carry
        the encoded ``result`` per the requested ``format=``.
        """
        import queue
        import threading

        # Local imports keep the module light.
        from .._image_output import encode_image
        from ..base import StreamChunk, StreamingContentType

        call_kwargs = self._build_call_kwargs(
            prompt,
            num_images=num_images,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        total_steps = call_kwargs["num_inference_steps"]

        q: "queue.Queue[Any]" = queue.Queue()
        _SENTINEL = object()
        _EXC = object()

        wants_preview = preview_every is not None and preview_every > 0
        request_latents = wants_preview  # tell diffusers to expose latents

        def _callback(pipe_, step_index, timestep, callback_kwargs):  # noqa: ARG001
            # diffusers passes step_index as 0-based; promote to 1-based for display.
            step = step_index + 1
            include_preview = wants_preview and (step % preview_every == 0 or step == total_steps)
            image_preview = None
            if include_preview and "latents" in callback_kwargs:
                try:
                    decoded = self._decode_latents_to_pil(callback_kwargs["latents"])
                    image_preview = decoded[0] if decoded else None
                except Exception:  # noqa: BLE001
                    image_preview = None  # never break generation on a preview failure
            q.put(
                StreamChunk(
                    StreamingContentType.IMAGE_GENERATING,
                    {
                        "step": step,
                        "total_steps": total_steps,
                        "image": image_preview,
                        "final": False,
                        "result": None,
                    },
                )
            )
            return callback_kwargs

        call_kwargs["callback_on_step_end"] = _callback
        if request_latents:
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        def _run():
            try:
                result = self.pipeline(**call_kwargs)
                images = result.images if hasattr(result, "images") else result
                q.put(("_done", list(images)))
            except Exception as exc:  # noqa: BLE001
                q.put((_EXC, exc))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Drain step chunks until the run thread reports done (or fails).
        final_images: list = []
        while True:
            item = q.get()
            if isinstance(item, tuple) and item and item[0] == "_done":
                final_images = item[1]
                break
            if isinstance(item, tuple) and item and item[0] is _EXC:
                raise item[1]
            yield item  # IMAGE_GENERATING progress chunk

        thread.join()

        # Emit one final chunk per produced image, carrying the encoded result.
        for i, img in enumerate(final_images):
            encoded = encode_image(img, format=format, prompt=prompt, output_dir=output_dir)
            yield StreamChunk(
                StreamingContentType.IMAGE_GENERATING,
                {
                    "step": total_steps,
                    "total_steps": total_steps,
                    "image": img,
                    "final": True,
                    "result": encoded,
                    "image_index": i,
                    "num_images": len(final_images),
                },
            )

    def __repr__(self) -> str:
        loaded = "loaded" if self._pipe is not None else "unloaded"
        return f"HuggingFaceImageClient(model={self.spec.id!r}, {loaded})"
