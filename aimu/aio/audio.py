"""Async audio-generation surface mirroring :mod:`aimu.models.audio_client`.

Exposes:

- :class:`AsyncAudioClient`: factory paralleling the sync :class:`AudioClient`,
  wrapping an existing sync :class:`BaseAudioClient` (any provider).
- :func:`audio_client` / :func:`generate_audio`: convenience functions matching
  the shape of :func:`aimu.audio_client` / :func:`aimu.generate_audio`.

Because audio providers load weights in-process (HuggingFace transformers/diffusers),
the factory follows the established wrap pattern: pass an existing sync client. Direct
enum / string construction is refused with a helpful error pointing at the wrap pattern.

HuggingFace symbols are resolved lazily -- on first call, not on ``import aimu.aio`` --
via :func:`_hf`, the same ``installed()`` + import-on-demand shape used throughout
``aimu.models`` and the rest of ``aimu.aio``. An absent ``soundfile`` yields ``None``;
an installed-but-broken one raises with the original cause chained.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from aimu.models._internal.factory import installed

_HF_MODULE = "aimu.models.providers.hf.audio"
_HF_REQUIRES = "soundfile"


def _hf() -> Optional[SimpleNamespace]:
    """Lazily resolve the HuggingFace audio symbols, or ``None`` if unavailable."""
    if not installed(_HF_REQUIRES):
        return None
    try:
        from aimu.models.base import HuggingFaceAudioSpec
        from aimu.models.providers.hf.audio import HuggingFaceAudioClient, HuggingFaceAudioModel

        from .providers.hf.audio import AsyncHuggingFaceAudioClient
    except ImportError as exc:
        raise ImportError(
            f"Audio support could not be loaded from {_HF_MODULE!r} ({_HF_REQUIRES!r} is installed): {exc}"
        ) from exc
    return SimpleNamespace(
        HuggingFaceAudioClient=HuggingFaceAudioClient,
        HuggingFaceAudioModel=HuggingFaceAudioModel,
        HuggingFaceAudioSpec=HuggingFaceAudioSpec,
        AsyncHuggingFaceAudioClient=AsyncHuggingFaceAudioClient,
    )


_WRAP_GUIDANCE = (
    "Build a sync audio client first and pass it to aio.audio_client():\n"
    "    sync_client = aimu.audio_client({model})\n"
    "    async_client = aio.audio_client(sync_client)\n"
    "(This also avoids loading weights twice for in-process providers.)"
)


def _refuse(model: Any) -> None:
    """Raise the wrap-pattern guidance error for non-client inputs."""
    hf = _hf()
    if hf is not None and isinstance(model, hf.HuggingFaceAudioModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceAudioModel.{model.name}"))
    if hf is not None and isinstance(model, hf.HuggingFaceAudioSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceAudioSpec({model.id!r})"))
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))


class AsyncAudioClient:
    """Public async factory for audio-generation provider clients.

    Parallel to :class:`aimu.models.AudioClient` for the async surface. Wraps an
    existing sync :class:`BaseAudioClient` so weights are shared.

    Passing an audio model enum / spec / string raises ``ValueError`` pointing at the
    sync-then-wrap pattern (same convention as HuggingFace text/image clients).
    """

    def __init__(self, sync_client: Any):
        hf = _hf()
        if hf is not None and isinstance(sync_client, hf.HuggingFaceAudioClient):
            self._client: Any = hf.AsyncHuggingFaceAudioClient(sync_client)
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
    if _hf() is None:
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
    hf = _hf()
    if hf is None:
        raise ImportError("Audio support requires the [hf] extra: pip install -e '.[hf]'")

    if model is None:
        from aimu.models._internal.model_defaults import AUDIO_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(AUDIO_MODEL_ENV)

    sync_client: Optional[Any] = None
    if isinstance(model, hf.HuggingFaceAudioClient):
        sync_client = model
    elif isinstance(model, str):
        if not model.startswith("hf:"):
            raise ValueError(f"Unrecognised audio model string: {model!r}")
        sync_client = hf.HuggingFaceAudioClient(model)
    elif isinstance(model, hf.HuggingFaceAudioModel):
        sync_client = hf.HuggingFaceAudioClient(model)
    else:
        raise TypeError(f"Unrecognised audio model: {type(model).__name__}")

    async_client = audio_client(sync_client)
    return await async_client.generate(prompt, format=format, **kwargs)
