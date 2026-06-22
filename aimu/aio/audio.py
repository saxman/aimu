"""Async audio-generation surface mirroring :mod:`aimu.models.audio_client`.

Exposes:

- :class:`AsyncAudioClient`: factory paralleling the sync :class:`AudioClient`,
  wrapping an existing sync :class:`BaseAudioClient` (any provider).
- :func:`audio_client` / :func:`generate_audio`: convenience functions matching
  the shape of :func:`aimu.audio_client` / :func:`aimu.generate_audio`.

Because audio providers load weights in-process (HuggingFace transformers/diffusers),
the factory follows the established wrap pattern: pass an existing sync client. Direct
enum / string construction is refused with a helpful error pointing at the wrap pattern.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from aimu.models.base import AudioModel, AudioSpec, HuggingFaceAudioSpec
    from aimu.models.providers.hf.audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    from .providers.hf.audio import AsyncHuggingFaceAudioClient

    _HAS_HF_AUDIO = True
except ImportError:
    _HAS_HF_AUDIO = False
    HuggingFaceAudioClient = None  # type: ignore[assignment,misc]
    HuggingFaceAudioModel = None  # type: ignore[assignment,misc]
    HuggingFaceAudioSpec = None  # type: ignore[assignment,misc]
    AudioModel = None  # type: ignore[assignment,misc]
    AudioSpec = None  # type: ignore[assignment,misc]
    AsyncHuggingFaceAudioClient = None  # type: ignore[assignment,misc]


_WRAP_GUIDANCE = (
    "Build a sync audio client first and pass it to aio.audio_client():\n"
    "    sync_client = aimu.audio_client({model})\n"
    "    async_client = aio.audio_client(sync_client)\n"
    "(This also avoids loading weights twice for in-process providers.)"
)


def _refuse(model: Any) -> None:
    """Raise the wrap-pattern guidance error for non-client inputs."""
    if _HAS_HF_AUDIO and HuggingFaceAudioModel is not None and isinstance(model, HuggingFaceAudioModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceAudioModel.{model.name}"))
    if _HAS_HF_AUDIO and HuggingFaceAudioSpec is not None and isinstance(model, HuggingFaceAudioSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceAudioSpec({model.id!r})"))
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))


class AsyncAudioClient:
    """Public async factory for audio-generation provider clients.

    Parallel to :class:`aimu.models.AudioClient` for the async surface. Wraps an
    existing sync :class:`BaseAudioClient` so weights are shared.

    Passing an :class:`AudioModel` enum / :class:`AudioSpec` / string raises
    ``ValueError`` pointing at the sync-then-wrap pattern (same convention as
    HuggingFace text/image clients).
    """

    def __init__(self, sync_client: Any):
        if _HAS_HF_AUDIO and isinstance(sync_client, HuggingFaceAudioClient):
            self._client: Any = AsyncHuggingFaceAudioClient(sync_client)
        else:
            _refuse(sync_client)
            raise TypeError(
                f"AsyncAudioClient expects a sync HuggingFaceAudioClient. Got: {type(sync_client).__name__}"
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> Any:
        return self._client.spec

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate one or more audio clips. Forwarded to the inner async provider client.

        When ``stream=True`` is in ``**kwargs``, the inner client returns an
        ``AsyncIterator[StreamChunk]``, consumed with ``async for``.
        """
        return await self._client.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncAudioClient({self._client!r})"


def audio_client(model: Any) -> AsyncAudioClient:
    """Construct an :class:`AsyncAudioClient` by wrapping an existing sync client.

    Accepts a sync :class:`HuggingFaceAudioClient`. Passing an enum / spec / string
    raises ``ValueError`` pointing at the wrap pattern.
    """
    if not _HAS_HF_AUDIO:
        raise ImportError("Audio support requires the [hf] extra: pip install -e '.[hf]'")
    return AsyncAudioClient(model)


async def generate_audio(
    prompt: str,
    *,
    model: Any = None,
    format: str = "path",
    **kwargs: Any,
) -> Any:
    """One-shot async audio generation.

    Accepts either an existing sync audio client (preferred, weights reused across
    calls) or a model enum/spec/string (constructs a fresh sync client inside, which
    loads weights each call).

    When ``model`` is omitted, the ``AIMU_AUDIO_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).
    """
    if not _HAS_HF_AUDIO:
        raise ImportError("Audio support requires the [hf] extra: pip install -e '.[hf]'")

    if model is None:
        from aimu.models._internal.model_defaults import AUDIO_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(AUDIO_MODEL_ENV)

    sync_client: Optional[Any] = None
    if _HAS_HF_AUDIO and isinstance(model, HuggingFaceAudioClient):
        sync_client = model
    elif isinstance(model, str):
        if not model.startswith("hf:"):
            raise ValueError(f"Unrecognised audio model string: {model!r}")
        if not _HAS_HF_AUDIO:
            raise ImportError("HuggingFace audio support requires the [hf] extra: pip install -e '.[hf]'")
        sync_client = HuggingFaceAudioClient(model)
    elif _HAS_HF_AUDIO and HuggingFaceAudioModel is not None and isinstance(model, HuggingFaceAudioModel):
        sync_client = HuggingFaceAudioClient(model)
    else:
        raise TypeError(f"Unrecognised audio model: {type(model).__name__}")

    async_client = audio_client(sync_client)
    return await async_client.generate(prompt, format=format, **kwargs)
