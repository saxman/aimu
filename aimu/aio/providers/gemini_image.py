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

from aimu.models.gemini_image import GeminiImageClient


class AsyncGeminiImageClient:
    """Async facade over a sync :class:`GeminiImageClient`."""

    def __init__(self, sync_client: GeminiImageClient):
        if not isinstance(sync_client, GeminiImageClient):
            raise TypeError(
                f"AsyncGeminiImageClient requires an existing sync GeminiImageClient. "
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
    ) -> Union[Any, list[Any], str, list[str], bytes, list[bytes]]:
        """Async equivalent of :meth:`GeminiImageClient.generate`."""
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
