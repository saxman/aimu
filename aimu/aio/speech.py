"""Async speech-generation surface mirroring :mod:`aimu.models.speech_client`.

Exposes:

- :class:`AsyncSpeechClient` — factory paralleling the sync :class:`SpeechClient`,
  wrapping an existing sync :class:`BaseSpeechClient` (any provider).
- :func:`speech_client` / :func:`generate_speech` — convenience functions matching
  the shape of :func:`aimu.speech_client` / :func:`aimu.generate_speech`.

Direct enum / string construction is refused with a helpful error pointing at the
wrap pattern (same convention as HuggingFace text/image/audio clients).
"""

from __future__ import annotations

from typing import Any

try:
    from aimu.models.base import HuggingFaceSpeechSpec, SpeechModel, SpeechSpec
    from aimu.models.providers.hf.speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    from .providers.hf.speech import AsyncHuggingFaceSpeechClient

    _HAS_HF_SPEECH = True
except ImportError:
    _HAS_HF_SPEECH = False
    HuggingFaceSpeechClient = None  # type: ignore[assignment,misc]
    HuggingFaceSpeechModel = None  # type: ignore[assignment,misc]
    HuggingFaceSpeechSpec = None  # type: ignore[assignment,misc]
    SpeechModel = None  # type: ignore[assignment,misc]
    SpeechSpec = None  # type: ignore[assignment,misc]
    AsyncHuggingFaceSpeechClient = None  # type: ignore[assignment,misc]

try:
    from aimu.models.base import OpenAISpeechSpec
    from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    from .providers.openai.speech import AsyncOpenAISpeechClient

    _HAS_OPENAI_SPEECH = True
except ImportError:
    _HAS_OPENAI_SPEECH = False
    OpenAISpeechClient = None  # type: ignore[assignment,misc]
    OpenAISpeechModel = None  # type: ignore[assignment,misc]
    OpenAISpeechSpec = None  # type: ignore[assignment,misc]
    AsyncOpenAISpeechClient = None  # type: ignore[assignment,misc]


_WRAP_GUIDANCE = (
    "Build a sync speech client first and pass it to aio.speech_client():\n"
    "    sync_client = aimu.speech_client({model})\n"
    "    async_client = aio.speech_client(sync_client)\n"
    "(This also avoids loading weights twice for in-process providers.)"
)


def _refuse(model: Any) -> None:
    """Raise the wrap-pattern guidance error for non-client inputs."""
    if _HAS_HF_SPEECH and HuggingFaceSpeechModel is not None and isinstance(model, HuggingFaceSpeechModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceSpeechModel.{model.name}"))
    if _HAS_OPENAI_SPEECH and OpenAISpeechModel is not None and isinstance(model, OpenAISpeechModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"OpenAISpeechModel.{model.name}"))
    if _HAS_HF_SPEECH and HuggingFaceSpeechSpec is not None and isinstance(model, HuggingFaceSpeechSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"HuggingFaceSpeechSpec({model.id!r})"))
    if _HAS_OPENAI_SPEECH and OpenAISpeechSpec is not None and isinstance(model, OpenAISpeechSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"OpenAISpeechSpec({model.id!r})"))
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))


class AsyncSpeechClient:
    """Public async factory for speech-generation provider clients.

    Parallel to :class:`aimu.models.SpeechClient` for the async surface. Wraps an
    existing sync :class:`BaseSpeechClient` so weights are shared.

    Passing a :class:`SpeechModel` enum / :class:`SpeechSpec` / string raises
    ``ValueError`` pointing at the sync-then-wrap pattern.
    """

    def __init__(self, sync_client: Any):
        if _HAS_HF_SPEECH and HuggingFaceSpeechClient is not None and isinstance(sync_client, HuggingFaceSpeechClient):
            self._client: Any = AsyncHuggingFaceSpeechClient(sync_client)
        elif _HAS_OPENAI_SPEECH and OpenAISpeechClient is not None and isinstance(sync_client, OpenAISpeechClient):
            self._client = AsyncOpenAISpeechClient(sync_client)
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
        ``AsyncIterator[StreamChunk]`` — consume with ``async for``.
        """
        return await self._client.generate(text, **kwargs)

    def __repr__(self) -> str:
        return f"AsyncSpeechClient({self._client!r})"


def speech_client(model: Any) -> AsyncSpeechClient:
    """Construct an :class:`AsyncSpeechClient` by wrapping an existing sync client.

    Accepts a sync :class:`HuggingFaceSpeechClient` or :class:`OpenAISpeechClient`.
    Passing an enum / spec / string raises ``ValueError`` pointing at the wrap pattern.
    """
    if not _HAS_HF_SPEECH and not _HAS_OPENAI_SPEECH:
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
    :class:`OpenAISpeechClient` (preferred — state reused across calls), a
    :class:`HuggingFaceSpeechModel` / :class:`OpenAISpeechModel` enum member, or
    a ``"provider:model_id"`` string — same dispatch as :func:`aimu.generate_speech`.

    When ``model`` is omitted, the ``AIMU_SPEECH_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).
    """
    if not _HAS_HF_SPEECH and not _HAS_OPENAI_SPEECH:
        raise ImportError(
            "Speech support requires the [hf] or [openai_compat] extra: "
            "pip install -e '.[hf]' or pip install -e '.[openai_compat]'"
        )

    if model is None:
        from aimu.models._internal.model_defaults import SPEECH_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(SPEECH_MODEL_ENV)

    if _HAS_HF_SPEECH and HuggingFaceSpeechClient is not None and isinstance(model, HuggingFaceSpeechClient):
        sync_client: Any = model
    elif _HAS_OPENAI_SPEECH and OpenAISpeechClient is not None and isinstance(model, OpenAISpeechClient):
        sync_client = model
    elif isinstance(model, str) or (
        (_HAS_HF_SPEECH and HuggingFaceSpeechModel is not None and isinstance(model, HuggingFaceSpeechModel))
        or (_HAS_OPENAI_SPEECH and OpenAISpeechModel is not None and isinstance(model, OpenAISpeechModel))
        or (_HAS_HF_SPEECH and HuggingFaceSpeechSpec is not None and isinstance(model, HuggingFaceSpeechSpec))
        or (_HAS_OPENAI_SPEECH and OpenAISpeechSpec is not None and isinstance(model, OpenAISpeechSpec))
    ):
        import aimu

        sync_client = aimu.speech_client(model)
    else:
        raise TypeError(f"Unrecognised speech model: {type(model).__name__}")

    async_client = speech_client(sync_client)
    return await async_client.generate(text, format=format, **kwargs)
