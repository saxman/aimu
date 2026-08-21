"""Async speech-generation surface mirroring :mod:`aimu.models.speech_client`.

Exposes:

- :class:`AsyncSpeechClient`: factory paralleling the sync :class:`SpeechClient`,
  wrapping an existing sync :class:`BaseSpeechClient` (any provider).
- :func:`speech_client` / :func:`generate_speech`: convenience functions matching
  the shape of :func:`aimu.speech_client` / :func:`aimu.generate_speech`.

Direct enum / string construction is refused with a helpful error pointing at the
wrap pattern (same convention as HuggingFace text/image/audio clients).

Both providers' symbols are resolved lazily -- on first call, not on ``import
aimu.aio`` -- via :func:`_hf` / :func:`_openai`, the same ``installed()`` +
import-on-demand shape used throughout ``aimu.models`` and the rest of ``aimu.aio``.
An absent dependency yields ``None``; an installed-but-broken one raises with the
original cause chained.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from aimu.models._internal.factory import installed

_HF_MODULE = "aimu.models.providers.hf.speech"
_HF_REQUIRES = "soundfile"
_OPENAI_MODULE = "aimu.models.providers.openai.speech"
_OPENAI_REQUIRES = "openai"


def _hf() -> Optional[SimpleNamespace]:
    """Lazily resolve the HuggingFace speech symbols, or ``None`` if unavailable."""
    if not installed(_HF_REQUIRES):
        return None
    try:
        from aimu.models.base import HuggingFaceSpeechSpec
        from aimu.models.providers.hf.speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

        from .providers.hf.speech import AsyncHuggingFaceSpeechClient
    except ImportError as exc:
        raise ImportError(
            f"Speech support could not be loaded from {_HF_MODULE!r} ({_HF_REQUIRES!r} is installed): {exc}"
        ) from exc
    return SimpleNamespace(
        HuggingFaceSpeechClient=HuggingFaceSpeechClient,
        HuggingFaceSpeechModel=HuggingFaceSpeechModel,
        HuggingFaceSpeechSpec=HuggingFaceSpeechSpec,
        AsyncHuggingFaceSpeechClient=AsyncHuggingFaceSpeechClient,
    )


def _openai() -> Optional[SimpleNamespace]:
    """Lazily resolve the OpenAI speech symbols, or ``None`` if unavailable."""
    if not installed(_OPENAI_REQUIRES):
        return None
    try:
        from aimu.models.base import OpenAISpeechSpec
        from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

        from .providers.openai.speech import AsyncOpenAISpeechClient
    except ImportError as exc:
        raise ImportError(
            f"Speech support could not be loaded from {_OPENAI_MODULE!r} ({_OPENAI_REQUIRES!r} is installed): {exc}"
        ) from exc
    return SimpleNamespace(
        OpenAISpeechClient=OpenAISpeechClient,
        OpenAISpeechModel=OpenAISpeechModel,
        OpenAISpeechSpec=OpenAISpeechSpec,
        AsyncOpenAISpeechClient=AsyncOpenAISpeechClient,
    )


_WRAP_GUIDANCE = (
    "Build a sync speech client first and pass it to aio.speech_client():\n"
    "    sync_client = aimu.speech_client({model})\n"
    "    async_client = aio.speech_client(sync_client)\n"
    "(This also avoids loading weights twice for in-process providers.)"
)


def _refuse(model: Any) -> None:
    """Raise the wrap-pattern guidance error for non-client inputs."""
    hf = _hf()
    if hf is not None and isinstance(model, hf.HuggingFaceSpeechModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceSpeechModel.{model.name}"))
    openai = _openai()
    if openai is not None and isinstance(model, openai.OpenAISpeechModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"OpenAISpeechModel.{model.name}"))
    if hf is not None and isinstance(model, hf.HuggingFaceSpeechSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceSpeechSpec({model.id!r})"))
    if openai is not None and isinstance(model, openai.OpenAISpeechSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"OpenAISpeechSpec({model.id!r})"))
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))


class AsyncSpeechClient:
    """Public async factory for speech-generation provider clients.

    Parallel to :class:`aimu.models.SpeechClient` for the async surface. Wraps an
    existing sync :class:`BaseSpeechClient` so weights are shared.

    Passing a speech model enum / spec / string raises ``ValueError`` pointing at the
    sync-then-wrap pattern.
    """

    def __init__(self, sync_client: Any):
        hf = _hf()
        openai = _openai()
        if hf is not None and isinstance(sync_client, hf.HuggingFaceSpeechClient):
            self._client: Any = hf.AsyncHuggingFaceSpeechClient(sync_client)
        elif openai is not None and isinstance(sync_client, openai.OpenAISpeechClient):
            self._client = openai.AsyncOpenAISpeechClient(sync_client)
        else:
            _refuse(sync_client)
            raise TypeError(
                f"AsyncSpeechClient expects a sync HuggingFaceSpeechClient or OpenAISpeechClient. "
                f"Got: {type(sync_client).__name__}"
            )

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> Any:
        return self._client.spec

    async def generate(self, text: str, **kwargs: Any) -> Any:
        """Generate speech. Forwarded to the inner async provider client.

        When ``stream=True`` is in ``**kwargs``, the inner client returns an
        ``AsyncIterator[StreamChunk]``, consumed with ``async for``.
        """
        return await self._client.generate(text, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncSpeechClient({self._client!r})"


def speech_client(model: Any) -> AsyncSpeechClient:
    """Construct an :class:`AsyncSpeechClient` by wrapping an existing sync client.

    Accepts a sync :class:`HuggingFaceSpeechClient` or :class:`OpenAISpeechClient`.
    Passing an enum / spec / string raises ``ValueError`` pointing at the wrap pattern.
    """
    if _hf() is None and _openai() is None:
        raise ImportError(
            "Speech support requires the [hf] or [openai_compat] extra: "
            "pip install -e '.[hf]' or pip install -e '.[openai_compat]'"
        )
    return AsyncSpeechClient(model)


async def generate_speech(
    text: str,
    *,
    model: Any = None,
    format: str = "path",
    **kwargs: Any,
) -> Any:
    """One-shot async speech generation.

    ``model`` may be an existing sync :class:`HuggingFaceSpeechClient` or
    :class:`OpenAISpeechClient` (preferred, since state is reused across calls), a
    :class:`HuggingFaceSpeechModel` / :class:`OpenAISpeechModel` enum member, or
    a ``"provider:model_id"`` string (same dispatch as :func:`aimu.generate_speech`).

    When ``model`` is omitted, the ``AIMU_SPEECH_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).
    """
    hf = _hf()
    openai = _openai()
    if hf is None and openai is None:
        raise ImportError(
            "Speech support requires the [hf] or [openai_compat] extra: "
            "pip install -e '.[hf]' or pip install -e '.[openai_compat]'"
        )

    if model is None:
        from aimu.models._internal.model_defaults import SPEECH_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(SPEECH_MODEL_ENV)

    if hf is not None and isinstance(model, hf.HuggingFaceSpeechClient):
        sync_client: Any = model
    elif openai is not None and isinstance(model, openai.OpenAISpeechClient):
        sync_client = model
    elif isinstance(model, str) or (
        (hf is not None and isinstance(model, (hf.HuggingFaceSpeechModel, hf.HuggingFaceSpeechSpec)))
        or (openai is not None and isinstance(model, (openai.OpenAISpeechModel, openai.OpenAISpeechSpec)))
    ):
        import aimu

        sync_client = aimu.speech_client(model)
    else:
        raise TypeError(f"Unrecognised speech model: {type(model).__name__}")

    async_client = speech_client(sync_client)
    return await async_client.generate(text, format=format, **kwargs)
