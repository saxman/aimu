"""Async wrapper around a sync :class:`HuggingFaceImageClient`.

In-process providers (HF transformers, llama-cpp, diffusers) load model weights
into memory; constructing two clients for the same model would double-load. The
established pattern (see :mod:`aimu.aio.providers.hf`) is to construct a sync
client and pass it to the async wrapper, which delegates work via
``asyncio.to_thread`` and shares all state.

Usage::

    sync_client = aimu.image_client(HuggingFaceImageModel.SDXL_BASE)
    async_client = aimu.aio.image_client(sync_client)
    image = await async_client.generate("a fox in a snowy forest")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional, Union

from aimu.models.hf_image import HuggingFaceImageClient


class AsyncHuggingFaceImageClient:
    """Async facade over a sync :class:`HuggingFaceImageClient`.

    Does *not* inherit ``AsyncBaseModelClient`` — image generation has no message
    history or tool-calling lifecycle, so the text-shaped base class doesn't apply.
    """

    def __init__(self, sync_client: HuggingFaceImageClient):
        if not isinstance(sync_client, HuggingFaceImageClient):
            raise TypeError(
                f"AsyncHuggingFaceImageClient requires an existing sync HuggingFaceImageClient. "
                f"Got {type(sync_client).__name__}."
            )
        self._sync = sync_client

    @property
    def model(self) -> Any:
        return self._sync.model

    @property
    def spec(self) -> Any:
        return self._sync.spec

    @property
    def model_kwargs(self) -> Optional[dict]:
        return self._sync.model_kwargs

    @property
    def pipeline(self) -> Any:
        """The loaded pipeline (triggers sync load on first access)."""
        return self._sync.pipeline

    async def generate(
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
        """Async equivalent of :meth:`HuggingFaceImageClient.generate`.

        Routes the heavy pipeline call through :func:`asyncio.to_thread` so the
        event loop stays free during the (GIL/CUDA-serialised) inference.
        """
        return await asyncio.to_thread(
            self._sync.generate,
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            format=format,
            output_dir=output_dir,
        )

    def __repr__(self) -> str:
        return f"AsyncHuggingFaceImageClient({self._sync!r})"
