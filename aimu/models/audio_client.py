"""Audio-client factory paralleling :mod:`aimu.models.image_client`.

Exposes:

- :func:`resolve_audio_model_string`: parse ``"provider:model_id"`` for audio providers.
- :class:`AudioClient`: factory :class:`BaseAudioClient` that dispatches to the right
  concrete client based on the model enum / spec / string passed in.

Mirrors the image-side dispatch (:class:`aimu.models.ImageClient`,
:func:`aimu.models.resolve_image_model_string`). The shared dispatch logic lives in
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
from .base import AudioModel, AudioSpec, BaseAudioClient

# --- Optional provider imports ---

try:
    from .providers.hf.audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    _HAS_HF_AUDIO = True
except ImportError:
    _HAS_HF_AUDIO = False
    HuggingFaceAudioClient = None  # type: ignore[assignment,misc]
    HuggingFaceAudioModel = None  # type: ignore[assignment,misc]

_HF_HINT = (
    "HuggingFace audio support requires the [hf] extra (soundfile, torch, transformers, diffusers): "
    "pip install -e '.[hf]'"
)


def _entries() -> list[ProviderEntry]:
    return [ProviderEntry("hf", _HAS_HF_AUDIO, HuggingFaceAudioModel, HuggingFaceAudioClient, _HF_HINT)]


def _provider_registry() -> dict[str, tuple]:
    """Map ``provider`` string → ``(AudioModel subclass, AudioClient subclass)`` (installed only)."""
    return available_registry(_entries())


def resolve_audio_model_string(model_str: str) -> AudioModel:
    """Look up an audio-provider model enum from a ``"provider:model_id"`` string.

    Mirrors :func:`aimu.models.resolve_image_model_string` for audio providers. Only
    matches *exact* enum-member values; for ad-hoc HuggingFace repos pass the
    ``"hf:..."`` string directly to :class:`AudioClient`.
    """
    return resolve_model_string(model_str, _entries(), modality="audio")


class AudioClient(FactoryDelegate):
    """Public factory for audio-generation provider clients.

    Parallel to :class:`aimu.models.ImageClient` for the audio modality. Accepts
    a provider's :class:`AudioModel` enum member, an :class:`AudioSpec`, or a
    ``"provider:model_id"`` string. Provider-specific construction kwargs are
    passed directly (the legacy ``model_kwargs={...}`` form is deprecated).

    Examples::

        from aimu.models import AudioClient, HuggingFaceAudioModel

        # Enum form
        client = AudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)

        # String form (supports ad-hoc HuggingFace repo ids)
        client = AudioClient("hf:facebook/musicgen-small")
        client = AudioClient("hf:myorg/custom-audioldm2")

    Provider-specific construction kwargs are passed directly::

        AudioClient(HuggingFaceAudioModel.STABLE_AUDIO_OPEN, torch_dtype="auto")
    """

    def __init__(self, model: AudioModel | AudioSpec | str, **kwargs: Any) -> None:
        self._client: BaseAudioClient = build_client(
            model,
            factory_model_kwargs(kwargs),
            _entries(),
            modality="audio",
            model_base=AudioModel,
            spec_base=AudioSpec,
        )

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate one or more audio clips. Forwarded to the inner client's
        :meth:`BaseAudioClient.generate`."""
        return self._client.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"AudioClient({self._client!r})"
