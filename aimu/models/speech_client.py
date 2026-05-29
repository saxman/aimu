"""Speech-client factory paralleling :mod:`aimu.models.audio_client`.

Exposes:

- :func:`resolve_speech_model_string` — parse ``"provider:model_id"`` for speech providers.
- :class:`SpeechClient` — factory :class:`BaseSpeechClient` that dispatches to the right
  concrete client based on the model enum / spec / string passed in.

Mirrors the audio-side dispatch (:class:`aimu.models.AudioClient`,
:func:`aimu.models.resolve_audio_model_string`).
"""

from __future__ import annotations

from typing import Any, Optional, Union

from .base import BaseSpeechClient, SpeechModel, SpeechSpec

# --- Optional provider imports ---

try:
    from .hf_speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    _HAS_HF_SPEECH = True
except ImportError:
    _HAS_HF_SPEECH = False
    HuggingFaceSpeechClient = None  # type: ignore[assignment,misc]
    HuggingFaceSpeechModel = None  # type: ignore[assignment,misc]

try:
    from .openai_speech import OpenAISpeechClient, OpenAISpeechModel

    _HAS_OPENAI_SPEECH = True
except ImportError:
    _HAS_OPENAI_SPEECH = False
    OpenAISpeechClient = None  # type: ignore[assignment,misc]
    OpenAISpeechModel = None  # type: ignore[assignment,misc]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(SpeechModel subclass, SpeechClient subclass)``."""
    registry: dict[str, tuple] = {}
    if _HAS_HF_SPEECH:
        registry["hf"] = (HuggingFaceSpeechModel, HuggingFaceSpeechClient)
    if _HAS_OPENAI_SPEECH:
        registry["openai"] = (OpenAISpeechModel, OpenAISpeechClient)
    return registry


def resolve_speech_model_string(model_str: str) -> SpeechModel:
    """Look up a speech-provider model enum from a ``"provider:model_id"`` string.

    Mirrors :func:`aimu.models.resolve_audio_model_string` for speech providers.
    Only matches *exact* enum-member values; for ad-hoc model ids pass the
    ``"provider:..."`` string directly to :class:`SpeechClient`.
    """
    if ":" not in model_str:
        raise ValueError(
            f"Speech model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(_provider_registry())}"
        )
    provider, _, model_id = model_str.partition(":")
    registry = _provider_registry()
    if provider not in registry:
        raise ValueError(
            f"Unknown speech provider {provider!r}. Available providers (with installed deps): "
            f"{sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(
        f"Provider {provider!r} has no speech model id {model_id!r}. Available: {available}"
    )


class SpeechClient:
    """Public factory for TTS provider clients.

    Parallel to :class:`aimu.models.AudioClient` for the speech modality. Accepts
    a provider's :class:`SpeechModel` enum member, a :class:`SpeechSpec`, or a
    ``"provider:model_id"`` string (``"hf:..."`` or ``"openai:..."``).
    Provider-specific construction kwargs are forwarded as ``model_kwargs``.

    Examples::

        from aimu.models import SpeechClient, OpenAISpeechModel

        client = SpeechClient(OpenAISpeechModel.TTS_1)
        client = SpeechClient("openai:tts-1")
        client = SpeechClient("hf:facebook/mms-tts-eng")
    """

    def __init__(self, model: Union[SpeechModel, SpeechSpec, str], model_kwargs: Optional[dict] = None) -> None:
        if isinstance(model, str):
            if ":" not in model:
                raise ValueError(f"Speech model string must be in 'provider:model_id' form, got: {model!r}")
            provider, _, _model_id = model.partition(":")
            if provider == "hf":
                if not _HAS_HF_SPEECH:
                    raise ImportError(
                        "HuggingFace speech support requires the [hf] extra "
                        "(soundfile, torch, transformers): pip install -e '.[hf]'"
                    )
                self._client: BaseSpeechClient = HuggingFaceSpeechClient(model, model_kwargs=model_kwargs)
                return
            if provider == "openai":
                if not _HAS_OPENAI_SPEECH:
                    raise ImportError(
                        "OpenAI speech support requires the [openai_compat] extra "
                        "(openai): pip install -e '.[openai_compat]'"
                    )
                self._client = OpenAISpeechClient(model, model_kwargs=model_kwargs)
                return
            raise ValueError(
                f"Unknown speech provider {provider!r}. Available: {sorted(_provider_registry())}"
            )

        if isinstance(model, SpeechSpec) and not isinstance(model, SpeechModel):
            raise TypeError(
                "Pass a SpeechModel enum member (e.g. OpenAISpeechModel.TTS_1) or a "
                "'provider:model_id' string. SpeechSpec is the value type held by enum members."
            )

        if _HAS_HF_SPEECH and HuggingFaceSpeechModel is not None and isinstance(model, HuggingFaceSpeechModel):
            self._client = HuggingFaceSpeechClient(model, model_kwargs=model_kwargs)
        elif _HAS_OPENAI_SPEECH and OpenAISpeechModel is not None and isinstance(model, OpenAISpeechModel):
            self._client = OpenAISpeechClient(model, model_kwargs=model_kwargs)
        else:
            raise ValueError(
                f"No available client for speech-model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed."
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> SpeechSpec:
        return self._client.spec

    @property
    def model_kwargs(self) -> Optional[dict]:
        return self._client.model_kwargs

    def generate(self, text: str, **kwargs: Any) -> Any:
        """Synthesise speech from text. Forwarded to the inner client's
        :meth:`BaseSpeechClient.generate`."""
        return self._client.generate(text, **kwargs)

    def __repr__(self) -> str:
        return f"SpeechClient({self._client!r})"
