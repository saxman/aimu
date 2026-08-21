"""Async transcription surface mirroring :mod:`aimu.models.transcription_client`.

Both providers' symbols are resolved lazily -- on first call, not on ``import
aimu.aio`` -- via :func:`_hf` / :func:`_openai`, the same ``installed()`` +
import-on-demand shape used throughout ``aimu.models`` and the rest of ``aimu.aio``.
An absent dependency yields ``None``; an installed-but-broken one raises with the
original cause chained.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from aimu.models._internal.factory import installed

_HF_MODULE = "aimu.models.providers.hf.transcription"
_HF_REQUIRES = "transformers"
_OPENAI_MODULE = "aimu.models.providers.openai.transcription"
_OPENAI_REQUIRES = "openai"


def _hf() -> Optional[SimpleNamespace]:
    """Lazily resolve the HuggingFace transcription symbols, or ``None`` if unavailable."""
    if not installed(_HF_REQUIRES):
        return None
    try:
        from aimu.models.providers.hf.transcription import HuggingFaceTranscriptionClient

        from .providers.hf.transcription import AsyncHuggingFaceTranscriptionClient
    except ImportError as exc:
        raise ImportError(
            f"Transcription support could not be loaded from {_HF_MODULE!r} ({_HF_REQUIRES!r} is installed): {exc}"
        ) from exc
    return SimpleNamespace(
        HuggingFaceTranscriptionClient=HuggingFaceTranscriptionClient,
        AsyncHuggingFaceTranscriptionClient=AsyncHuggingFaceTranscriptionClient,
    )


def _openai() -> Optional[SimpleNamespace]:
    """Lazily resolve the OpenAI transcription symbols, or ``None`` if unavailable."""
    if not installed(_OPENAI_REQUIRES):
        return None
    try:
        from aimu.models.providers.openai.transcription import OpenAITranscriptionClient

        from .providers.openai_transcription import AsyncOpenAITranscriptionClient
    except ImportError as exc:
        raise ImportError(
            f"Transcription support could not be loaded from {_OPENAI_MODULE!r} "
            f"({_OPENAI_REQUIRES!r} is installed): {exc}"
        ) from exc
    return SimpleNamespace(
        OpenAITranscriptionClient=OpenAITranscriptionClient,
        AsyncOpenAITranscriptionClient=AsyncOpenAITranscriptionClient,
    )


_WRAP_GUIDANCE = (
    "Build a sync transcription client first and pass it to aio.transcription_client():\n"
    "    sync_client = aimu.transcription_client({model})\n"
    "    async_client = aio.transcription_client(sync_client)\n"
    "(This also avoids loading weights twice for in-process providers.)"
)


def _refuse(model: Any) -> None:
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))
    raise TypeError(
        f"AsyncTranscriptionClient expects a sync HuggingFaceTranscriptionClient or "
        f"OpenAITranscriptionClient. Got: {type(model).__name__}. " + _WRAP_GUIDANCE.format(model=repr(model))
    )


def _is_hf_transcription_client(obj: Any) -> bool:
    hf = _hf()
    return hf is not None and isinstance(obj, hf.HuggingFaceTranscriptionClient)


def _is_openai_transcription_client(obj: Any) -> bool:
    openai = _openai()
    return openai is not None and isinstance(obj, openai.OpenAITranscriptionClient)


class AsyncTranscriptionClient:
    """Async transcription client. Wraps an existing sync client.

    Passing a spec, enum member, or string raises ``ValueError`` pointing at the
    sync-then-wrap pattern.
    """

    def __init__(self, sync_client: Any):
        if _is_hf_transcription_client(sync_client):
            self._client: Any = _hf().AsyncHuggingFaceTranscriptionClient(sync_client)
        elif _is_openai_transcription_client(sync_client):
            self._client = _openai().AsyncOpenAITranscriptionClient(sync_client)
        else:
            _refuse(sync_client)

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> Any:
        return self._client.spec

    async def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        return await self._client.transcribe(audio, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncTranscriptionClient({self._client!r})"


def transcription_client(sync_client: Any) -> AsyncTranscriptionClient:
    """Wrap an existing sync transcription client for async use."""
    if _hf() is None and _openai() is None:
        raise ImportError(
            "Transcription support requires the [hf] or [openai_compat] extra: "
            "pip install -e '.[hf]' or pip install -e '.[openai_compat]'"
        )
    return AsyncTranscriptionClient(sync_client)


async def transcribe(audio: Any, *, model: Any = None, **kwargs: Any) -> str | dict:
    """One-shot async transcription.

    ``model`` may be an existing sync :class:`HuggingFaceTranscriptionClient` or
    :class:`OpenAITranscriptionClient` (preferred), a model enum member, or a
    ``"provider:model_id"`` string. When ``model`` is omitted, the
    ``AIMU_TRANSCRIPTION_MODEL`` env var is used; if unset a ``ValueError`` is raised.
    """
    if _hf() is None and _openai() is None:
        raise ImportError(
            "Transcription support requires the [hf] or [openai_compat] extra: "
            "pip install -e '.[hf]' or pip install -e '.[openai_compat]'"
        )

    if model is None:
        from aimu.models._internal.model_defaults import TRANSCRIPTION_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(TRANSCRIPTION_MODEL_ENV)

    if _is_hf_transcription_client(model) or _is_openai_transcription_client(model):
        sync_client: Any = model
    elif isinstance(model, str) or hasattr(model, "spec"):
        import aimu

        sync_client = aimu.transcription_client(model)
    else:
        raise TypeError(f"Unrecognised transcription model: {type(model).__name__}")

    return await transcription_client(sync_client).transcribe(audio, **kwargs)
