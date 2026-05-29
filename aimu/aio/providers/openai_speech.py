"""Async wrapper around a sync :class:`OpenAISpeechClient`.

OpenAI TTS is a cloud API, so technically a native async client would beat
``asyncio.to_thread``. We use the wrap pattern here for shape consistency with
other async provider wrappers. Switching to native async is a follow-up.

Usage::

    sync_client = aimu.speech_client(OpenAISpeechModel.TTS_1)
    async_client = aimu.aio.speech_client(sync_client)
    path = await async_client.generate("Hello, world!")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from aimu.models.openai_speech import OpenAISpeechClient


class AsyncOpenAISpeechClient:
    """Async facade over a sync :class:`OpenAISpeechClient`."""

    def __init__(self, sync_client: OpenAISpeechClient):
        if not isinstance(sync_client, OpenAISpeechClient):
            raise TypeError(
                f"AsyncOpenAISpeechClient requires an existing sync OpenAISpeechClient. "
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
        """Async equivalent of :meth:`OpenAISpeechClient.generate`.

        Routes the API call through :func:`asyncio.to_thread` so the event loop
        stays free during the HTTP request. When ``stream=True``, returns an
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
        return f"AsyncOpenAISpeechClient({self._sync!r})"


async def _stream_via_thread(sync_iter):
    """Yield from a sync iterator by hopping each ``next()`` to a worker thread."""
    sentinel = object()
    loop_iter = iter(sync_iter)
    while True:
        chunk = await asyncio.to_thread(next, loop_iter, sentinel)
        if chunk is sentinel:
            break
        yield chunk
