"""Async wrapper around a sync :class:`GeminiImageClient`.

Nano Banana is a cloud API, so technically a native ``async`` client (via
``google.genai.aio``) would beat ``asyncio.to_thread``. We use the wrap pattern
here for shape consistency with :class:`AsyncHuggingFaceImageClient` and to
share the sync client's API-key resolution. Switching to native async is a
follow-up.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional, Union

from aimu.models.providers.gemini.image import GeminiImageClient


class AsyncGeminiImageClient:
    """Async facade over a sync :class:`GeminiImageClient`."""

    def __init__(self, sync_client: GeminiImageClient):
        if not isinstance(sync_client, GeminiImageClient):
            raise TypeError(
                f"AsyncGeminiImageClient requires an existing sync GeminiImageClient. Got {type(sync_client).__name__}."
            )
        self._sync = sync_client

    @property
    def model(self) -> Any:
        return self._sync.model

    @property
    def spec(self) -> Any:
        return self._sync.spec

    @property
    def client(self) -> Any:
        return self._sync.client

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        num_images: int = 1,
        format: str = "pil",
        output_dir: Optional[Path] = None,
        stream: bool = False,
        preview_every: Optional[int] = None,
    ) -> Union[Any, list[Any], str, list[str], bytes, list[bytes]]:
        """Async equivalent of :meth:`GeminiImageClient.generate`.

        When ``stream=True``, returns an async iterator that yields the coarse
        start/done ``IMAGE_GENERATING`` chunks emitted by the sync client (one
        pair per image). Each ``next()`` is routed through ``asyncio.to_thread``
        so blocking API calls don't stall the event loop.
        """
        if stream:
            sync_iter = self._sync.generate(
                prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                num_images=num_images,
                format=format,
                output_dir=output_dir,
                stream=True,
                preview_every=preview_every,
            )
            return _stream_via_thread(sync_iter)
        return await asyncio.to_thread(
            self._sync.generate,
            prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            num_images=num_images,
            format=format,
            output_dir=output_dir,
        )

    def __repr__(self) -> str:
        return f"AsyncGeminiImageClient({self._sync!r})"


async def _stream_via_thread(sync_iter):
    """Pull from a sync iterator via ``asyncio.to_thread`` between yields."""
    sentinel = object()
    loop_iter = iter(sync_iter)
    while True:
        chunk = await asyncio.to_thread(next, loop_iter, sentinel)
        if chunk is sentinel:
            break
        yield chunk
