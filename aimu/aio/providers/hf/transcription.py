"""Async HuggingFace transcription client: wraps sync via asyncio.to_thread (Decision 7)."""

from __future__ import annotations

import asyncio
from typing import Any

from aimu.models.providers.hf.transcription import HuggingFaceTranscriptionClient


class AsyncHuggingFaceTranscriptionClient:
    """Async wrapper around :class:`HuggingFaceTranscriptionClient`.

    Routes :meth:`transcribe` through ``asyncio.to_thread``. Shares pipeline weights
    with the wrapped sync client -- no second weight load.
    """

    def __init__(self, sync_client: HuggingFaceTranscriptionClient):
        # Use class-name check so this survives monkeypatched module reloads in tests.
        if type(sync_client).__name__ != "HuggingFaceTranscriptionClient":
            raise TypeError(
                f"AsyncHuggingFaceTranscriptionClient requires an existing sync HuggingFaceTranscriptionClient. "
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
        return f"AsyncHuggingFaceTranscriptionClient(model={self.spec.id!r})"
