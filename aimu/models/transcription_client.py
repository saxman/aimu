"""Transcription-client factory paralleling :mod:`aimu.models.speech_client`.

Exposes:

- :func:`resolve_transcription_model_string`: parse ``"provider:model_id"`` for
  transcription providers.
- :class:`TranscriptionClient`: factory :class:`BaseTranscriptionClient` that
  dispatches to the right concrete client based on the model enum / spec / string
  passed in.

Mirrors the speech-side dispatch. The shared dispatch logic lives in
:mod:`aimu.models._internal.factory`.
"""

from __future__ import annotations

from typing import Any

from ._internal.factory import (
    FactoryDelegate,
    ProviderEntry,
    available_registry,
    build_client,
    factory_model_kwargs,
    resolve_model_string,
)
from .base import BaseTranscriptionClient, TranscriptionModel, TranscriptionSpec

# --- Optional provider imports ---

try:
    from .providers.hf.transcription import HuggingFaceTranscriptionClient, HuggingFaceTranscriptionModel

    _HAS_HF_TRANSCRIPTION = True
except ImportError:
    _HAS_HF_TRANSCRIPTION = False
    HuggingFaceTranscriptionClient = None  # type: ignore[assignment,misc]
    HuggingFaceTranscriptionModel = None  # type: ignore[assignment,misc]

try:
    from .providers.openai.transcription import OpenAITranscriptionClient, OpenAITranscriptionModel

    _HAS_OPENAI_TRANSCRIPTION = True
except ImportError:
    _HAS_OPENAI_TRANSCRIPTION = False
    OpenAITranscriptionClient = None  # type: ignore[assignment,misc]
    OpenAITranscriptionModel = None  # type: ignore[assignment,misc]

_HF_HINT = (
    "HuggingFace transcription support requires the [hf] extra (soundfile, torch, transformers): pip install -e '.[hf]'"
)
_OPENAI_HINT = (
    "OpenAI transcription support requires the [openai_compat] extra (openai): pip install -e '.[openai_compat]'"
)


def _entries() -> list[ProviderEntry]:
    return [
        ProviderEntry(
            "hf", _HAS_HF_TRANSCRIPTION, HuggingFaceTranscriptionModel, HuggingFaceTranscriptionClient, _HF_HINT
        ),
        ProviderEntry(
            "openai", _HAS_OPENAI_TRANSCRIPTION, OpenAITranscriptionModel, OpenAITranscriptionClient, _OPENAI_HINT
        ),
    ]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(TranscriptionModel subclass, client subclass)`` (installed only)."""
    return available_registry(_entries())


def resolve_transcription_model_string(model_str: str) -> TranscriptionModel:
    """Look up a transcription-provider model enum from a ``"provider:model_id"`` string.

    Only matches *exact* enum-member values; for ad-hoc model ids pass the
    ``"provider:..."`` string directly to :class:`TranscriptionClient`.
    """
    return resolve_model_string(model_str, _entries(), modality="transcription")


class TranscriptionClient(FactoryDelegate):
    """Public factory for ASR/STT provider clients.

    Parallel to :class:`aimu.models.SpeechClient` for the transcription modality.
    Accepts a provider's :class:`TranscriptionModel` enum member, a
    :class:`TranscriptionSpec`, or a ``"provider:model_id"`` string
    (``"hf:..."`` or ``"openai:..."``). Provider-specific construction kwargs are
    passed directly (the legacy ``model_kwargs={...}`` form is deprecated).

    Examples::

        from aimu.models import TranscriptionClient, OpenAITranscriptionModel

        client = TranscriptionClient(OpenAITranscriptionModel.WHISPER_1)
        client = TranscriptionClient("openai:whisper-1")
        client = TranscriptionClient("hf:openai/whisper-tiny")

    Provider-specific construction kwargs are passed directly::

        TranscriptionClient(HuggingFaceTranscriptionModel.WHISPER_TINY, device="cpu")
    """

    def __init__(self, model: TranscriptionModel | TranscriptionSpec | str, **kwargs: Any) -> None:
        self._client: BaseTranscriptionClient = build_client(
            model,
            factory_model_kwargs(kwargs),
            _entries(),
            modality="transcription",
            model_base=TranscriptionModel,
            spec_base=TranscriptionSpec,
        )

    def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        """Transcribe audio to text. Forwarded to the inner client's
        :meth:`BaseTranscriptionClient.transcribe`."""
        return self._client.transcribe(audio, **kwargs)

    def __repr__(self) -> str:
        return f"TranscriptionClient({self._client!r})"
