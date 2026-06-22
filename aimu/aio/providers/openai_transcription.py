"""Async OpenAI transcription client: wraps sync via asyncio.to_thread (Decision 7)."""

from __future__ import annotations

import asyncio
from typing import Any

from aimu.models.providers.openai.transcription import OpenAITranscriptionClient


class AsyncOpenAITranscriptionClient:
    """Async wrapper around :class:`OpenAITranscriptionClient`.

    Routes :meth:`transcribe` through ``asyncio.to_thread`` so the event loop
    stays free during HTTP I/O. Properties delegate to the wrapped sync client.
    """

    def __init__(self, sync_client: OpenAITranscriptionClient):
        # Use class-name check so this survives monkeypatched module reloads in tests.
        if type(sync_client).__name__ != "OpenAITranscriptionClient":
            raise TypeError(
                f"AsyncOpenAITranscriptionClient requires an existing sync OpenAITranscriptionClient. "
                f"Got {type(sync_client).__name__}."
            )
        self._sync = sync_client

    @property
    def model(self) -> Any:
        return self._sync.model

    @property
    def spec(self) -> Any:
        return self._sync.spec

    async def transcribe(self, audio: Any, **kwargs: Any) -> str | dict:
        return await asyncio.to_thread(self._sync.transcribe, audio, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncOpenAITranscriptionClient(model={self.spec.id!r})"
