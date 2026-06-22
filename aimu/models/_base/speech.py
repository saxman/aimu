"""Speech-modality base types: the speech specs, the ``SpeechModel`` enum, and ``BaseSpeechClient``.

Text-to-speech (TTS) only. Disjoint from audio generation (which is duration-based);
TTS converts literal text to spoken audio. Only ``StreamChunk`` / ``StreamingContentType``
are shared. Output encoding reuses :func:`aimu.models._internal.audio_output.encode_audio`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional

from .shared import StreamChunk, StreamingContentType


@dataclass
class SpeechSpec:
    """Base descriptor for a single text-to-speech (TTS) model.

    Sibling to :class:`AudioSpec`; disjoint because TTS has no concept of
    duration; it converts text to speech directly. Voice and speed defaults
    live here so :class:`BaseSpeechClient` can resolve them without knowing
    the concrete subclass.

    Equality and hash are by ``id`` only so the spec can be used directly
    as an enum value.
    """

    id: str
    default_voice: Optional[str] = None
    default_speed: float = 1.0

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SpeechSpec):
            return self.id == other.id
        return NotImplemented


@dataclass(eq=False)
class HuggingFaceSpeechSpec(SpeechSpec):
    """Descriptor for a HuggingFace-backed TTS model.

    ``pipeline_type`` selects the loader/runner strategy:
    - ``"tts_pipeline"``: ``transformers.pipeline("text-to-speech")``.
    - ``"bark"``: ``BarkModel`` + ``AutoProcessor`` (zero-shot voice codes).

    ``eq=False`` inherits :class:`SpeechSpec`'s id-only equality + hash.
    """

    pipeline_type: str = "tts_pipeline"


@dataclass(eq=False)
class OpenAISpeechSpec(SpeechSpec):
    """Descriptor for an OpenAI TTS model (tts-1, tts-1-hd).

    ``eq=False`` inherits :class:`SpeechSpec`'s id-only equality + hash.
    Default voice and speed are carried on :class:`SpeechSpec`; this subclass
    exists for type-dispatch in :class:`SpeechClient`.
    """


class SpeechModel(Enum):
    """Base enum for TTS provider model catalogs.

    Parallel to :class:`AudioModel`. Each member's value is a :class:`SpeechSpec`
    (or subclass); the constructor sets ``_value_`` from ``spec.id`` and stores
    the spec on the member.
    """

    def __init__(self, spec: SpeechSpec):
        self._value_ = spec.id
        self.spec = spec


class BaseSpeechClient(ABC):
    """Abstract base for text-to-speech (TTS) provider clients.

    Parallel to :class:`BaseAudioClient` for the speech modality. Subclasses
    implement :meth:`_generate` returning a list of ``(sample_rate, np.ndarray)``
    tuples; the public :meth:`generate` wraps that with format conversion via
    :func:`aimu.models._internal.audio_output.encode_audio` so every speech client offers
    the same ``format="numpy"|"path"|"bytes"|"data_url"`` surface.

    Voice and speed are speech-specific parameters absent from audio generation.
    ``BaseSpeechClient`` resolves them from ``self.spec`` defaults when the caller
    passes ``None``.

    .. note::
        ``BaseSpeechClient`` is the TTS base only. Future STT (speech-to-text)
        will use a separate ``BaseTranscriptionClient``.
    """

    model: Any
    spec: "SpeechSpec"

    @abstractmethod
    def __init__(self, model: Any, model_kwargs: Optional[dict] = None):
        self.model = model
        self.model_kwargs = model_kwargs

    @abstractmethod
    def _generate(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        **kwargs: Any,
    ) -> list:
        """Provider-specific synthesis. Returns list of ``(sample_rate, np.ndarray)``."""

    def generate(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Any] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Synthesise speech from text.

        ``format`` controls the return type: ``"numpy"`` returns a
        ``(sample_rate, np.ndarray)`` tuple; ``"path"`` saves a WAV file and
        returns its path; ``"bytes"`` and ``"data_url"`` return WAV-encoded
        bytes / base64 data URL respectively.

        When ``stream=True``, returns an iterator of :class:`StreamChunk` with
        phase :attr:`StreamingContentType.SPEECH_GENERATING`. Chunk content:
        ``{"chunk_index": int, "total_chunks": int | None, "final": bool, "result": Any}``.
        ``total_chunks`` is ``None`` for providers that stream HTTP bytes (OpenAI).
        """
        if num_audio < 1:
            raise ValueError(f"num_audio must be >= 1, got {num_audio}")

        resolved_voice = voice if voice is not None else self.spec.default_voice
        resolved_speed = speed if speed is not None else self.spec.default_speed

        if stream:
            return self._generate_streamed(
                text,
                voice=resolved_voice,
                speed=resolved_speed,
                num_audio=num_audio,
                format=format,
                output_dir=output_dir,
                **kwargs,
            )

        from .._internal.audio_output import encode_audio

        results = self._generate(text, voice=resolved_voice, speed=resolved_speed, num_audio=num_audio, **kwargs)
        encoded = [encode_audio(audio, sr, format=format, prompt=text, output_dir=output_dir) for sr, audio in results]
        return encoded[0] if num_audio == 1 else encoded

    def _generate_streamed(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Default streaming: calls blocking :meth:`_generate`, emits one final chunk per clip.

        Providers that expose streaming (e.g. OpenAI HTTP byte streaming) override
        this to emit progress mid-generation. Intermediate chunks carry
        ``final=False`` and ``result=None``; the terminal chunk carries ``final=True``
        and the encoded output.
        """
        from .._internal.audio_output import encode_audio

        results = self._generate(text, voice=voice, speed=speed, num_audio=num_audio, **kwargs)
        for i, (sr, audio) in enumerate(results, start=1):
            encoded = encode_audio(audio, sr, format=format, prompt=text, output_dir=output_dir)
            yield StreamChunk(
                StreamingContentType.SPEECH_GENERATING,
                {"chunk_index": i, "total_chunks": len(results), "final": True, "result": encoded},
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.spec.id!r})"
