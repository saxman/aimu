"""Transcription-client factory paralleling :mod:`aimu.models.speech_client`.

Exposes:

- :func:`resolve_transcription_model_string` — parse ``"provider:model_id"`` for
  transcription providers.
- :class:`TranscriptionClient` — factory :class:`BaseTranscriptionClient` that
  dispatches to the right concrete client based on the model enum / spec / string
  passed in.

Mirrors the speech-side dispatch (:class:`aimu.models.SpeechClient`,
:func:`aimu.models.resolve_speech_model_string`).
"""

from __future__ import annotations

from typing import Any

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


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(TranscriptionModel subclass, TranscriptionClient subclass)``."""
    registry: dict[str, tuple] = {}
    if _HAS_HF_TRANSCRIPTION:
        registry["hf"] = (HuggingFaceTranscriptionModel, HuggingFaceTranscriptionClient)
    if _HAS_OPENAI_TRANSCRIPTION:
        registry["openai"] = (OpenAITranscriptionModel, OpenAITranscriptionClient)
    return registry


def resolve_transcription_model_string(model_str: str) -> TranscriptionModel:
    """Look up a transcription-provider model enum from a ``"provider:model_id"`` string.

    Mirrors :func:`aimu.models.resolve_speech_model_string` for transcription providers.
    Only matches *exact* enum-member values; for ad-hoc model ids pass the
    ``"provider:..."`` string directly to :class:`TranscriptionClient`.
    """
    if ":" not in model_str:
        raise ValueError(
            f"Transcription model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(_provider_registry())}"
        )
    provider, _, model_id = model_str.partition(":")
    registry = _provider_registry()
    if provider not in registry:
        raise ValueError(
            f"Unknown transcription provider {provider!r}. "
            f"Available providers (with installed deps): {sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(f"Provider {provider!r} has no transcription model id {model_id!r}. Available: {available}")


class TranscriptionClient:
    """Public factory for ASR/STT provider clients.

    Parallel to :class:`aimu.models.SpeechClient` for the transcription modality.
    Accepts a provider's :class:`TranscriptionModel` enum member, a
    :class:`TranscriptionSpec`, or a ``"provider:model_id"`` string
    (``"hf:..."`` or ``"openai:..."``). Provider-specific construction kwargs are
    forwarded as ``model_kwargs``.

    Examples::

        from aimu.models import TranscriptionClient, OpenAITranscriptionModel

        client = TranscriptionClient(OpenAITranscriptionModel.WHISPER_1)
        client = TranscriptionClient("openai:whisper-1")
        client = TranscriptionClient("hf:openai/whisper-tiny")
    """

    def __init__(self, model: TranscriptionModel | TranscriptionSpec | str, model_kwargs: dict | None = None) -> None:
        if isinstance(model, str):
            if ":" not in model:
                raise ValueError(f"Transcription model string must be in 'provider:model_id' form, got: {model!r}")
            provider, _, _model_id = model.partition(":")
            if provider == "hf":
                if not _HAS_HF_TRANSCRIPTION:
                    raise ImportError(
                        "HuggingFace transcription support requires the [hf] extra "
                        "(soundfile, torch, transformers): pip install -e '.[hf]'"
                    )
                self._client: BaseTranscriptionClient = HuggingFaceTranscriptionClient(model, model_kwargs=model_kwargs)
                return
            if provider == "openai":
                if not _HAS_OPENAI_TRANSCRIPTION:
                    raise ImportError(
                        "OpenAI transcription support requires the [openai_compat] extra "
                        "(openai): pip install -e '.[openai_compat]'"
                    )
                self._client = OpenAITranscriptionClient(model, model_kwargs=model_kwargs)
                return
            raise ValueError(f"Unknown transcription provider {provider!r}. Available: {sorted(_provider_registry())}")

        if isinstance(model, TranscriptionSpec) and not isinstance(model, TranscriptionModel):
            raise TypeError(
                "Pass a TranscriptionModel enum member (e.g. OpenAITranscriptionModel.WHISPER_1) or a "
                "'provider:model_id' string. TranscriptionSpec is the value type held by enum members."
            )

        if (
            _HAS_HF_TRANSCRIPTION
            and HuggingFaceTranscriptionModel is not None
            and isinstance(model, HuggingFaceTranscriptionModel)
        ):
            self._client = HuggingFaceTranscriptionClient(model, model_kwargs=model_kwargs)
        elif (
            _HAS_OPENAI_TRANSCRIPTION
            and OpenAITranscriptionModel is not None
            and isinstance(model, OpenAITranscriptionModel)
        ):
            self._client = OpenAITranscriptionClient(model, model_kwargs=model_kwargs)
        else:
            raise ValueError(
                f"No available client for transcription-model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed."
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> TranscriptionSpec:
        return self._client.spec

    @property
    def model_kwargs(self) -> dict | None:
        return self._client.model_kwargs

    def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        """Transcribe audio to text. Forwarded to the inner client's
        :meth:`BaseTranscriptionClient.transcribe`."""
        return self._client.transcribe(audio, **kwargs)

    def __repr__(self) -> str:
        return f"TranscriptionClient({self._client!r})"
