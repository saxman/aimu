"""Audio-client factory paralleling :mod:`aimu.models.image_client`.

Exposes:

- :func:`resolve_audio_model_string`: parse ``"provider:model_id"`` for audio providers.
- :class:`AudioClient`: factory :class:`BaseAudioClient` that dispatches to the right
  concrete client based on the model enum / spec / string passed in.

Mirrors the image-side dispatch (:class:`aimu.models.ImageClient`,
:func:`aimu.models.resolve_image_model_string`).
"""

from __future__ import annotations

from typing import Any, Optional, Union

from .base import AudioModel, AudioSpec, BaseAudioClient

# --- Optional provider imports ---

try:
    from .providers.hf.audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    _HAS_HF_AUDIO = True
except ImportError:
    _HAS_HF_AUDIO = False
    HuggingFaceAudioClient = None  # type: ignore[assignment,misc]
    HuggingFaceAudioModel = None  # type: ignore[assignment,misc]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(AudioModel subclass, AudioClient subclass)``."""
    registry: dict[str, tuple] = {}
    if _HAS_HF_AUDIO:
        registry["hf"] = (HuggingFaceAudioModel, HuggingFaceAudioClient)
    return registry


def resolve_audio_model_string(model_str: str) -> AudioModel:
    """Look up an audio-provider model enum from a ``"provider:model_id"`` string.

    Mirrors :func:`aimu.models.resolve_image_model_string` for audio providers.

    Examples::

        resolve_audio_model_string("hf:facebook/musicgen-small")
        resolve_audio_model_string("hf:cvssp/audioldm2")

    Note that string-form construction with arbitrary HuggingFace repo ids is
    handled inside the concrete client's ``__init__``; this function only matches
    *exact* enum-member values. For ad-hoc repos pass the ``"hf:..."`` string
    directly to :class:`AudioClient` instead of calling this helper.
    """
    if ":" not in model_str:
        raise ValueError(
            f"Audio model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(_provider_registry())}"
        )
    provider, _, model_id = model_str.partition(":")
    registry = _provider_registry()
    if provider not in registry:
        raise ValueError(
            f"Unknown audio provider {provider!r}. Available providers (with installed deps): {sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(f"Provider {provider!r} has no audio model id {model_id!r}. Available: {available}")


class AudioClient:
    """Public factory for audio-generation provider clients.

    Parallel to :class:`aimu.models.ImageClient` for the audio modality. Accepts
    a provider's :class:`AudioModel` enum member, an :class:`AudioSpec`, or a
    ``"provider:model_id"`` string. Provider-specific construction kwargs are
    forwarded as ``model_kwargs`` to the concrete client.

    Examples::

        from aimu.models import AudioClient, HuggingFaceAudioModel

        # Enum form
        client = AudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)

        # String form (supports ad-hoc HuggingFace repo ids)
        client = AudioClient("hf:facebook/musicgen-small")
        client = AudioClient("hf:myorg/custom-audioldm2")
    """

    def __init__(self, model: Union[AudioModel, AudioSpec, str], model_kwargs: Optional[dict] = None) -> None:
        if isinstance(model, str):
            if ":" not in model:
                raise ValueError(f"Audio model string must be in 'provider:model_id' form, got: {model!r}")
            provider, _, _model_id = model.partition(":")
            if provider == "hf":
                if not _HAS_HF_AUDIO:
                    raise ImportError(
                        "HuggingFace audio support requires the [hf] extra (soundfile, torch, transformers, diffusers): "
                        "pip install -e '.[hf]'"
                    )
                self._client: BaseAudioClient = HuggingFaceAudioClient(model, model_kwargs=model_kwargs)
                return
            raise ValueError(f"Unknown audio provider {provider!r}. Available: {sorted(_provider_registry())}")

        if isinstance(model, AudioSpec) and not isinstance(model, AudioModel):
            raise TypeError(
                "Pass an AudioModel enum member (e.g. HuggingFaceAudioModel.MUSICGEN_SMALL) or a "
                "'provider:model_id' string. AudioSpec is the value type held by enum members."
            )

        if _HAS_HF_AUDIO and isinstance(model, HuggingFaceAudioModel):
            self._client = HuggingFaceAudioClient(model, model_kwargs=model_kwargs)
        else:
            raise ValueError(
                f"No available client for audio-model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed (pip install 'aimu[hf]')."
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> AudioSpec:
        return self._client.spec

    @property
    def model_kwargs(self) -> Optional[dict]:
        return self._client.model_kwargs

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate one or more audio clips. Forwarded to the inner client's
        :meth:`BaseAudioClient.generate`."""
        return self._client.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"AudioClient({self._client!r})"
