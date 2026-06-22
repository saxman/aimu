"""Audio-modality base types: the audio specs, the ``AudioModel`` enum, and ``BaseAudioClient``.

Music / sound generation (not TTS, which lives in :mod:`aimu.models._base.speech`). Disjoint
from text and image; only ``StreamChunk`` / ``StreamingContentType`` are shared.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional

from .shared import StreamChunk, StreamingContentType


@dataclass
class AudioSpec:
    """Base descriptor for a single audio-generation model.

    Sibling to :class:`ImageSpec`; deliberately disjoint because audio models have
    a distinct generation interface (duration-based rather than dimension-based).
    Provider-specific subclasses (:class:`HuggingFaceAudioSpec`) add their own defaults.

    Equality and hash are by ``id`` only so the spec can be used directly as an
    enum value.
    """

    id: str

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AudioSpec):
            return self.id == other.id
        return NotImplemented


@dataclass(eq=False)
class HuggingFaceAudioSpec(AudioSpec):
    """Descriptor for a single HuggingFace-backed audio-generation model.

    ``pipeline_type`` selects the loader/runner strategy:
    - ``"musicgen"``: transformers ``MusicgenForConditionalGeneration``
    - ``"audioldm2"``: diffusers ``AudioLDM2Pipeline``
    - ``"stable_audio"``: diffusers ``StableAudioPipeline``

    ``default_steps`` is ignored for ``"musicgen"`` (token-autoregressive, no denoising
    steps). ``eq=False`` inherits :class:`AudioSpec`'s id-only equality + hash.
    """

    pipeline_type: str = "musicgen"
    default_duration_s: float = 10.0
    default_steps: int = 200


class AudioModel(Enum):
    """Base enum for audio-generation provider model catalogs.

    Parallel to :class:`ImageModel`. Each member's value is an :class:`AudioSpec`
    (or subclass); the constructor sets ``_value_`` from ``spec.id`` and stores
    the spec on the member.
    """

    def __init__(self, spec: AudioSpec):
        self._value_ = spec.id
        self.spec = spec


class BaseAudioClient(ABC):
    """Abstract base for audio-generation provider clients.

    Parallel to :class:`BaseImageClient` for audio modality. Subclasses implement
    :meth:`_generate` returning a list of ``(sample_rate, np.ndarray)`` tuples; the
    public :meth:`generate` wraps that with format conversion via
    :func:`aimu.models._internal.audio_output.encode_audio` so every audio client offers the
    same ``format="numpy"|"path"|"bytes"|"data_url"`` surface.

    ``"numpy"`` is the native format (parallel to ``"pil"`` for images); the others
    produce WAV-encoded output.
    """

    model: Any
    spec: AudioSpec

    @abstractmethod
    def __init__(self, model: Any, model_kwargs: Optional[dict] = None):
        self.model = model
        self.model_kwargs = model_kwargs

    @abstractmethod
    def _generate(self, prompt: str, *, duration_s: float, num_audio: int = 1, **kwargs: Any) -> list:
        """Provider-specific generation. Returns list of ``(sample_rate, np.ndarray)``."""

    def generate(
        self,
        prompt: str,
        *,
        duration_s: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Any] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generate one or more audio clips from a text prompt.

        ``format`` controls the return type: ``"numpy"`` returns a
        ``(sample_rate, np.ndarray)`` tuple (or list of tuples for ``num_audio > 1``);
        ``"path"`` saves a WAV file and returns its path; ``"bytes"`` and
        ``"data_url"`` return WAV-encoded bytes / base64 data URL respectively.

        When ``stream=True``, returns an iterator of :class:`StreamChunk` with phase
        :attr:`StreamingContentType.AUDIO_GENERATING`. Diffusers-backed providers emit
        one chunk per denoising step (no intermediate audio decode); other providers
        emit a single final chunk.
        """
        if num_audio < 1:
            raise ValueError(f"num_audio must be >= 1, got {num_audio}")

        resolved_duration = duration_s if duration_s is not None else self.spec.default_duration_s

        if stream:
            return self._generate_streamed(
                prompt,
                duration_s=resolved_duration,
                num_inference_steps=num_inference_steps,
                num_audio=num_audio,
                format=format,
                output_dir=output_dir,
                seed=seed,
                **kwargs,
            )

        from .._internal.audio_output import encode_audio

        results = self._generate(
            prompt,
            duration_s=resolved_duration,
            num_inference_steps=num_inference_steps,
            num_audio=num_audio,
            seed=seed,
            **kwargs,
        )
        encoded = [
            encode_audio(audio, sr, format=format, prompt=prompt, output_dir=output_dir) for sr, audio in results
        ]
        return encoded[0] if num_audio == 1 else encoded

    def _generate_streamed(
        self,
        prompt: str,
        *,
        duration_s: float,
        num_inference_steps: Optional[int] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Any] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Default streaming: calls blocking :meth:`_generate`, emits one final chunk per clip.

        Providers that expose step-level callbacks (diffusers models) override this to
        emit progress mid-generation. Intermediate chunks carry ``final=False`` and
        ``result=None``; the terminal chunk per clip carries ``final=True`` and the
        encoded output.
        """
        from .._internal.audio_output import encode_audio

        results = self._generate(
            prompt,
            duration_s=duration_s,
            num_inference_steps=num_inference_steps,
            num_audio=num_audio,
            seed=seed,
            **kwargs,
        )
        for sr, audio in results:
            encoded = encode_audio(audio, sr, format=format, prompt=prompt, output_dir=output_dir)
            yield StreamChunk(
                StreamingContentType.AUDIO_GENERATING,
                {
                    "step": 1,
                    "total_steps": 1,
                    "final": True,
                    "result": encoded,
                    "duration_s": duration_s,
                },
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.spec.id!r})"
