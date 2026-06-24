"""Speech-client factory paralleling :mod:`aimu.models.audio_client`.

Exposes:

- :func:`resolve_speech_model_string`: parse ``"provider:model_id"`` for speech providers.
- :class:`SpeechClient`: factory :class:`BaseSpeechClient` that dispatches to the right
  concrete client based on the model enum / spec / string passed in.

Mirrors the audio-side dispatch. The shared dispatch logic lives in
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
from .base import BaseSpeechClient, SpeechModel, SpeechSpec

# --- Optional provider imports ---

try:
    from .providers.hf.speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    _HAS_HF_SPEECH = True
except ImportError:
    _HAS_HF_SPEECH = False
    HuggingFaceSpeechClient = None  # type: ignore[assignment,misc]
    HuggingFaceSpeechModel = None  # type: ignore[assignment,misc]

try:
    from .providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    _HAS_OPENAI_SPEECH = True
except ImportError:
    _HAS_OPENAI_SPEECH = False
    OpenAISpeechClient = None  # type: ignore[assignment,misc]
    OpenAISpeechModel = None  # type: ignore[assignment,misc]

_HF_HINT = "HuggingFace speech support requires the [hf] extra (soundfile, torch, transformers): pip install -e '.[hf]'"
_OPENAI_HINT = "OpenAI speech support requires the [openai_compat] extra (openai): pip install -e '.[openai_compat]'"


def _entries() -> list[ProviderEntry]:
    return [
        ProviderEntry("hf", _HAS_HF_SPEECH, HuggingFaceSpeechModel, HuggingFaceSpeechClient, _HF_HINT),
        ProviderEntry("openai", _HAS_OPENAI_SPEECH, OpenAISpeechModel, OpenAISpeechClient, _OPENAI_HINT),
    ]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(SpeechModel subclass, SpeechClient subclass)`` (installed only)."""
    return available_registry(_entries())


def resolve_speech_model_string(model_str: str) -> SpeechModel:
    """Look up a speech-provider model enum from a ``"provider:model_id"`` string.

    Only matches *exact* enum-member values; for ad-hoc model ids pass the
    ``"provider:..."`` string directly to :class:`SpeechClient`.
    """
    return resolve_model_string(model_str, _entries(), modality="speech")


class SpeechClient(FactoryDelegate):
    """Public factory for TTS provider clients.

    Parallel to :class:`aimu.models.AudioClient` for the speech modality. Accepts
    a provider's :class:`SpeechModel` enum member, a :class:`SpeechSpec`, or a
    ``"provider:model_id"`` string (``"hf:..."`` or ``"openai:..."``).
    Provider-specific construction kwargs are passed directly (the legacy
    ``model_kwargs={...}`` form is deprecated).

    Examples::

        from aimu.models import SpeechClient, OpenAISpeechModel

        client = SpeechClient(OpenAISpeechModel.TTS_1)
        client = SpeechClient("openai:tts-1")
        client = SpeechClient("hf:facebook/mms-tts-eng")

    Provider-specific construction kwargs are passed directly::

        SpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG, device="cpu")
    """

    def __init__(self, model: SpeechModel | SpeechSpec | str, **kwargs: Any) -> None:
        self._client: BaseSpeechClient = build_client(
            model,
            factory_model_kwargs(kwargs),
            _entries(),
            modality="speech",
            model_base=SpeechModel,
            spec_base=SpeechSpec,
        )

    def generate(self, text: str, **kwargs: Any) -> Any:
        """Synthesise speech from text. Forwarded to the inner client's
        :meth:`BaseSpeechClient.generate`."""
        return self._client.generate(text, **kwargs)

    def __repr__(self) -> str:
        return f"SpeechClient({self._client!r})"
