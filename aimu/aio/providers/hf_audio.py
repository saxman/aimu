"""Async wrapper around a sync :class:`HuggingFaceAudioClient`.

In-process providers (HF transformers, diffusers) load model weights into memory;
constructing two clients for the same model would double-load. The established
pattern (see :mod:`aimu.aio.providers.hf_image`) is to construct a sync client
and pass it to the async wrapper, which delegates work via ``asyncio.to_thread``
and shares all state.

Usage::

    sync_client = aimu.audio_client(HuggingFaceAudioModel.MUSICGEN_SMALL)
    async_client = aimu.aio.audio_client(sync_client)
    path = await async_client.generate("upbeat lo-fi jazz")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from aimu.models.hf_audio import HuggingFaceAudioClient


class AsyncHuggingFaceAudioClient:
    """Async facade over a sync :class:`HuggingFaceAudioClient`.

    Does *not* inherit ``AsyncBaseModelClient`` — audio generation has no message
    history or tool-calling lifecycle, so the text-shaped base class doesn't apply.
    """

    def __init__(self, sync_client: HuggingFaceAudioClient):
        if not isinstance(sync_client, HuggingFaceAudioClient):
            raise TypeError(
                f"AsyncHuggingFaceAudioClient requires an existing sync HuggingFaceAudioClient. "
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
        duration_s: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Any:
        """Async equivalent of :meth:`HuggingFaceAudioClient.generate`.

        Routes the heavy pipeline call through :func:`asyncio.to_thread` so the
        event loop stays free during inference. When ``stream=True``, returns an
        async iterator that pulls each ``AUDIO_GENERATING`` chunk from the
        underlying sync generator via ``to_thread`` between yields.
        """
        if stream:
            sync_iter = self._sync.generate(
                prompt,
                duration_s=duration_s,
                num_inference_steps=num_inference_steps,
                num_audio=num_audio,
                format=format,
                output_dir=output_dir,
                seed=seed,
                stream=True,
            )
            return _stream_via_thread(sync_iter)
        return await asyncio.to_thread(
            self._sync.generate,
            prompt,
            duration_s=duration_s,
            num_inference_steps=num_inference_steps,
            num_audio=num_audio,
            format=format,
            output_dir=output_dir,
            seed=seed,
        )

    def __repr__(self) -> str:
        return f"AsyncHuggingFaceAudioClient({self._sync!r})"


async def _stream_via_thread(sync_iter):
    """Yield from a sync iterator by hopping each ``next()`` to a worker thread.

    The pipeline callback fires on the run thread; the public sync generator drains
    a ``queue.Queue``. Each ``next()`` here is a brief wait on that queue, offloaded
    to ``asyncio.to_thread`` so the event loop stays free between generation steps.
    """
    sentinel = object()
    loop_iter = iter(sync_iter)
    while True:
        chunk = await asyncio.to_thread(next, loop_iter, sentinel)
        if chunk is sentinel:
            break
        yield chunk
