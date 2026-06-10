"""Async transcription surface mirroring :mod:`aimu.models.transcription_client`."""

from __future__ import annotations

from typing import Any

try:
    from aimu.models.providers.hf.transcription import HuggingFaceTranscriptionClient

    from .providers.hf.transcription import AsyncHuggingFaceTranscriptionClient

    _HAS_HF_TRANSCRIPTION = True
except ImportError:
    _HAS_HF_TRANSCRIPTION = False
    HuggingFaceTranscriptionClient = None  # type: ignore[assignment,misc]
    AsyncHuggingFaceTranscriptionClient = None  # type: ignore[assignment,misc]

try:
    from aimu.models.providers.openai.transcription import OpenAITranscriptionClient

    from .providers.openai_transcription import AsyncOpenAITranscriptionClient

    _HAS_OPENAI_TRANSCRIPTION = True
except ImportError:
    _HAS_OPENAI_TRANSCRIPTION = False
    OpenAITranscriptionClient = None  # type: ignore[assignment,misc]
    AsyncOpenAITranscriptionClient = None  # type: ignore[assignment,misc]


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
        f"OpenAITranscriptionClient. Got: {type(model).__name__}. "
        + _WRAP_GUIDANCE.format(model=repr(model))
    )


def _is_hf_transcription_client(obj: Any) -> bool:
    try:
        from aimu.models.providers.hf.transcription import HuggingFaceTranscriptionClient as _Cls

        return isinstance(obj, _Cls)
    except ImportError:
        return False


def _is_openai_transcription_client(obj: Any) -> bool:
    try:
        from aimu.models.providers.openai.transcription import OpenAITranscriptionClient as _Cls

        return isinstance(obj, _Cls)
    except ImportError:
        return False


class AsyncTranscriptionClient:
    """Async transcription client. Wraps an existing sync client.

    Passing a spec, enum member, or string raises ``ValueError`` pointing at the
    sync-then-wrap pattern.
    """

    def __init__(self, sync_client: Any):
        if _HAS_HF_TRANSCRIPTION and _is_hf_transcription_client(sync_client):
            self._client: Any = AsyncHuggingFaceTranscriptionClient(sync_client)
        elif _HAS_OPENAI_TRANSCRIPTION and _is_openai_transcription_client(sync_client):
            from .providers.openai_transcription import AsyncOpenAITranscriptionClient as _Cls

            self._client = _Cls(sync_client)
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
    if not _HAS_HF_TRANSCRIPTION and not _HAS_OPENAI_TRANSCRIPTION:
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
    if not _HAS_HF_TRANSCRIPTION and not _HAS_OPENAI_TRANSCRIPTION:
        raise ImportError(
            "Transcription support requires the [hf] or [openai_compat] extra: "
            "pip install -e '.[hf]' or pip install -e '.[openai_compat]'"
        )

    if model is None:
        from aimu.models._internal.model_defaults import TRANSCRIPTION_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(TRANSCRIPTION_MODEL_ENV)

    if (
        (_HAS_HF_TRANSCRIPTION and _is_hf_transcription_client(model))
        or (_HAS_OPENAI_TRANSCRIPTION and _is_openai_transcription_client(model))
    ):
        sync_client: Any = model
    elif isinstance(model, str) or hasattr(model, "spec"):
        import aimu

        sync_client = aimu.transcription_client(model)
    else:
        raise TypeError(f"Unrecognised transcription model: {type(model).__name__}")

    return await transcription_client(sync_client).transcribe(audio, **kwargs)
