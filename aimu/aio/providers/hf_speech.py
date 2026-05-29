"""Async wrapper around a sync :class:`HuggingFaceSpeechClient`.

In-process providers (HF transformers) load model weights into memory;
constructing two clients for the same model would double-load. The established
pattern (see :mod:`aimu.aio.providers.hf_audio`) is to construct a sync client
and pass it to the async wrapper, which delegates work via ``asyncio.to_thread``
and shares all state.

Usage::

    sync_client = aimu.speech_client(HuggingFaceSpeechModel.MMS_TTS_ENG)
    async_client = aimu.aio.speech_client(sync_client)
    path = await async_client.generate("Hello, world!")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from aimu.models.hf_speech import HuggingFaceSpeechClient


class AsyncHuggingFaceSpeechClient:
    """Async facade over a sync :class:`HuggingFaceSpeechClient`.

    Does *not* inherit ``AsyncBaseModelClient`` — speech generation has no message
    history or tool-calling lifecycle, so the text-shaped base class doesn't apply.
    """

    def __init__(self, sync_client: HuggingFaceSpeechClient):
        if not isinstance(sync_client, HuggingFaceSpeechClient):
            raise TypeError(
                f"AsyncHuggingFaceSpeechClient requires an existing sync HuggingFaceSpeechClient. "
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
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Path] = None,
        stream: bool = False,
    ) -> Any:
        """Async equivalent of :meth:`HuggingFaceSpeechClient.generate`.

        Routes the heavy pipeline call through :func:`asyncio.to_thread` so the
        event loop stays free during inference. When ``stream=True``, returns an
        async iterator that pulls each ``SPEECH_GENERATING`` chunk from the
        underlying sync generator via ``to_thread`` between yields.
        """
        if stream:
            sync_iter = self._sync.generate(
                text,
                voice=voice,
                speed=speed,
                num_audio=num_audio,
                format=format,
                output_dir=output_dir,
                stream=True,
            )
            return _stream_via_thread(sync_iter)
        return await asyncio.to_thread(
            self._sync.generate,
            text,
            voice=voice,
            speed=speed,
            num_audio=num_audio,
            format=format,
            output_dir=output_dir,
        )

    def __repr__(self) -> str:
        return f"AsyncHuggingFaceSpeechClient({self._sync!r})"


async def _stream_via_thread(sync_iter):
    """Yield from a sync iterator by hopping each ``next()`` to a worker thread."""
    sentinel = object()
    loop_iter = iter(sync_iter)
    while True:
        chunk = await asyncio.to_thread(next, loop_iter, sentinel)
        if chunk is sentinel:
            break
        yield chunk
