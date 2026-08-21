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
    resolve_model_string,
)
from .base import BaseTranscriptionClient, TranscriptionModel, TranscriptionSpec

_HF_HINT = (
    "HuggingFace transcription support requires the [hf] extra (soundfile, torch, transformers): pip install -e '.[hf]'"
)
_OPENAI_HINT = (
    "OpenAI transcription support requires the [openai_compat] extra (openai): pip install -e '.[openai_compat]'"
)


def _entries() -> list[ProviderEntry]:
    return [
        ProviderEntry(
            prefix="hf",
            module="aimu.models.providers.hf.transcription",
            enum_name="HuggingFaceTranscriptionModel",
            client_name="HuggingFaceTranscriptionClient",
            requires="transformers",
            install_hint=_HF_HINT,
        ),
        ProviderEntry(
            prefix="openai",
            module="aimu.models.providers.openai.transcription",
            enum_name="OpenAITranscriptionModel",
            client_name="OpenAITranscriptionClient",
            requires="openai",
            install_hint=_OPENAI_HINT,
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
    passed directly, e.g. ``TranscriptionClient(model, device="cpu")``.

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
            kwargs or None,
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
