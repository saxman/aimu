"""Abstract base types for every model client, re-exported from one place.

The definitions live in the per-modality :mod:`aimu.models._base` package (split out
so each modality stack reads top-to-bottom); this module re-exports them so the public
import location stays ``aimu.models.base`` for the ~50 call sites that depend on it.

- text:          ``ModelSpec``, ``Model``, ``BaseModelClient``
- image:         ``ImageSpec``, ``HuggingFaceImageSpec``, ``GeminiImageSpec``, ``ImageModel``, ``BaseImageClient``
- audio:         ``AudioSpec``, ``HuggingFaceAudioSpec``, ``AudioModel``, ``BaseAudioClient``
- speech:        ``SpeechSpec``, ``HuggingFaceSpeechSpec``, ``OpenAISpeechSpec``, ``SpeechModel``, ``BaseSpeechClient``
- transcription: ``TranscriptionSpec``, ``HuggingFaceTranscriptionSpec``, ``OpenAITranscriptionSpec``, ``TranscriptionModel``, ``BaseTranscriptionClient``
- shared:        ``StreamChunk``, ``StreamingContentType``, ``classproperty``
"""

from ._base.audio import AudioModel, AudioSpec, BaseAudioClient, HuggingFaceAudioSpec
from ._base.image import (
    BaseImageClient,
    GeminiImageSpec,
    HuggingFaceImageSpec,
    ImageModel,
    ImageSpec,
)
from ._base.shared import StreamChunk, StreamingContentType, classproperty
from ._base.speech import (
    BaseSpeechClient,
    HuggingFaceSpeechSpec,
    OpenAISpeechSpec,
    SpeechModel,
    SpeechSpec,
)
from ._base.transcription import (
    BaseTranscriptionClient,
    HuggingFaceTranscriptionSpec,
    OpenAITranscriptionSpec,
    TranscriptionModel,
    TranscriptionSpec,
)
from ._base.text import BaseModelClient, Model, ModelSpec

__all__ = [
    # shared / cross-modality
    "StreamChunk",
    "StreamingContentType",
    "classproperty",
    # text
    "ModelSpec",
    "Model",
    "BaseModelClient",
    # image
    "ImageSpec",
    "HuggingFaceImageSpec",
    "GeminiImageSpec",
    "ImageModel",
    "BaseImageClient",
    # audio
    "AudioSpec",
    "HuggingFaceAudioSpec",
    "AudioModel",
    "BaseAudioClient",
    # speech
    "SpeechSpec",
    "HuggingFaceSpeechSpec",
    "OpenAISpeechSpec",
    "SpeechModel",
    "BaseSpeechClient",
    # transcription
    "TranscriptionSpec",
    "HuggingFaceTranscriptionSpec",
    "OpenAITranscriptionSpec",
    "TranscriptionModel",
    "BaseTranscriptionClient",
]
