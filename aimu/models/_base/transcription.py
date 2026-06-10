"""Transcription-modality base types.

Speech-to-text (ASR/STT) only. Disjoint from audio generation (BaseAudioClient) and
text-to-speech (BaseSpeechClient). Input normalization reuses _normalize_audio() from
models/_internal/audio_input.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


_VALID_RESPONSE_FORMATS = ("text", "json", "verbose_json")


@dataclass
class TranscriptionSpec:
    """Base descriptor for a single speech-to-text (ASR/STT) model.

    Sibling to :class:`SpeechSpec`; disjoint because transcription converts audio
    to text rather than text to audio. Language defaults, timestamp support, and
    translation support live here so :class:`BaseTranscriptionClient` can validate
    requests without knowing the concrete subclass.

    Equality and hash are by ``id`` only so the spec can be used directly
    as an enum value.
    """

    id: str
    default_language: Optional[str] = None
    supports_timestamps: bool = False
    supports_translation: bool = False

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TranscriptionSpec):
            return self.id == other.id
        return NotImplemented


@dataclass(eq=False)
class HuggingFaceTranscriptionSpec(TranscriptionSpec):
    """Descriptor for a HuggingFace ASR model (e.g. Whisper via transformers).

    ``eq=False`` inherits :class:`TranscriptionSpec`'s id-only equality + hash.
    """


@dataclass(eq=False)
class OpenAITranscriptionSpec(TranscriptionSpec):
    """Descriptor for an OpenAI transcription model (e.g. whisper-1).

    ``eq=False`` inherits :class:`TranscriptionSpec`'s id-only equality + hash.
    """


class TranscriptionModel(Enum):
    """Base enum for ASR provider model catalogs.

    Parallel to :class:`SpeechModel`. Each member's value is a
    :class:`TranscriptionSpec` (or subclass); the constructor sets ``_value_``
    from ``spec.id`` and stores the spec on the member.
    """

    def __init__(self, spec: TranscriptionSpec):
        self._value_ = spec.id
        self.spec = spec


class BaseTranscriptionClient(ABC):
    """Abstract base for speech-to-text (ASR/STT) provider clients.

    Parallel to :class:`BaseSpeechClient` for the transcription modality.
    Subclasses implement ``_transcribe()`` returning ``str`` or ``dict``.
    The public ``transcribe()`` validates arguments and delegates.

    ``BaseTranscriptionClient`` is the STT base only. Text-to-speech uses
    the separate :class:`BaseSpeechClient`.
    """

    model: Any
    spec: "TranscriptionSpec"

    @abstractmethod
    def __init__(self, model: Any, model_kwargs: Optional[dict] = None):
        self.model = model
        self.model_kwargs = model_kwargs

    @abstractmethod
    def _transcribe(
        self,
        audio: Any,
        *,
        language: Optional[str] = None,
        response_format: str = "text",
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str | dict:
        """Provider-specific transcription."""

    def transcribe(
        self,
        audio: Any,
        *,
        language: Optional[str] = None,
        response_format: str = "text",
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str | dict:
        """Transcribe an audio clip to text.

        ``audio`` accepts: file path (str or pathlib.Path), raw bytes (WAV assumed),
        an ``https://`` URL (fetched eagerly), or a ``data:audio/...;base64,...`` URL.

        ``response_format``:
          ``"text"`` -> plain string
          ``"json"`` -> ``{"text": "..."}``
          ``"verbose_json"`` -> ``{"text": "...", "segments": [...], "language": "...", "duration": float}``
        """
        if response_format not in _VALID_RESPONSE_FORMATS:
            raise ValueError(f"response_format must be one of {list(_VALID_RESPONSE_FORMATS)}, got {response_format!r}")
        if response_format == "verbose_json" and not self.spec.supports_timestamps:
            raise ValueError(
                f"Model {self.spec.id!r} does not support timestamps "
                f"(supports_timestamps=False). Use response_format='text' or 'json'."
            )
        return self._transcribe(
            audio,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.spec.id!r})"
