"""HuggingFace-backed speech-to-text (ASR) client.

Uses ``transformers.pipeline("automatic-speech-recognition")`` with the Whisper
model family. Audio is normalised via ``_normalize_audio()``, decoded to float32
via soundfile, and passed directly to the pipeline.

Lazy pipeline load on first ``transcribe()`` call; weights are cached in a module-level
registry so a second client instance with the same model reuses already-loaded weights.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from typing import Any

# Hard import so HAS_HF_TRANSCRIPTION in aimu/models/__init__.py accurately reflects
# whether the required deps are installed.
import soundfile  # noqa: F401

from ._device import pop_device_hint
from ...base import BaseTranscriptionClient, HuggingFaceTranscriptionSpec, TranscriptionModel

logger = logging.getLogger(__name__)

_model_registry: dict[tuple, Any] = {}
_registry_lock = threading.Lock()


def _make_cache_key(spec_id: str, model_kwargs: dict | None) -> tuple:
    return (spec_id, *sorted((k, str(v)) for k, v in (model_kwargs or {}).items()))


class HuggingFaceTranscriptionModel(TranscriptionModel):
    """Catalog of supported HuggingFace ASR models.

    Each member's value is a :class:`HuggingFaceTranscriptionSpec`. ``.value`` returns
    the HuggingFace repo id; ``.spec`` returns the full spec. Power users can bypass
    the enum by passing a ``"hf:<repo_id>"`` string directly to
    :class:`HuggingFaceTranscriptionClient`.
    """

    WHISPER_TINY = HuggingFaceTranscriptionSpec(
        "openai/whisper-tiny", supports_timestamps=True, supports_translation=True
    )
    WHISPER_BASE = HuggingFaceTranscriptionSpec(
        "openai/whisper-base", supports_timestamps=True, supports_translation=True
    )
    WHISPER_SMALL = HuggingFaceTranscriptionSpec(
        "openai/whisper-small", supports_timestamps=True, supports_translation=True
    )
    WHISPER_MEDIUM = HuggingFaceTranscriptionSpec(
        "openai/whisper-medium", supports_timestamps=True, supports_translation=True
    )
    WHISPER_LARGE_V3 = HuggingFaceTranscriptionSpec(
        "openai/whisper-large-v3", supports_timestamps=True, supports_translation=True
    )
    DISTIL_WHISPER_LARGE_V3 = HuggingFaceTranscriptionSpec(
        "distil-whisper/distil-large-v3", supports_timestamps=True, supports_translation=True
    )


def _parse_model_string(s: str) -> HuggingFaceTranscriptionSpec:
    """Resolve a ``"hf:<repo_id>"`` string to a known :class:`HuggingFaceTranscriptionModel` spec.

    AIMU ships curated specs for best-of-class models only; an arbitrary repo id is **not**
    supported via string (capability flags can't be inferred reliably). For a custom model,
    hand-build a :class:`HuggingFaceTranscriptionSpec` and pass that object.
    """
    if ":" not in s:
        raise ValueError(
            f"HuggingFace transcription model string must be in 'provider:repo_id' form "
            f"(e.g. 'hf:openai/whisper-tiny'). Got: {s!r}"
        )
    provider, repo_id = s.split(":", 1)
    if provider != "hf":
        raise ValueError(
            f"Only 'hf:' provider is supported for HuggingFaceTranscriptionClient. Got provider: {provider!r}"
        )
    if not repo_id:
        raise ValueError(f"Empty model id after 'hf:': {s!r}")

    for member in HuggingFaceTranscriptionModel:
        if member.value == repo_id:
            return member.spec
    available = sorted(m.value for m in HuggingFaceTranscriptionModel)
    raise ValueError(
        f"Unknown HuggingFace transcription model id {repo_id!r}. AIMU supports curated models only; "
        f"pass a known id, a HuggingFaceTranscriptionModel member, or a hand-built "
        f"HuggingFaceTranscriptionSpec for a custom model. Available ids: {available}"
    )


class HuggingFaceTranscriptionClient(BaseTranscriptionClient):
    """Speech-to-text client backed by HuggingFace transformers.

    Loads pipeline weights lazily on the first :meth:`_transcribe` call. Pass a
    :class:`HuggingFaceTranscriptionModel` enum member, a :class:`HuggingFaceTranscriptionSpec`,
    or a ``"hf:<repo_id>"`` string. ``model_kwargs`` is forwarded to the loader.

    Target a specific device with ``model_kwargs={"device": "cuda:1"}``.
    """

    MODELS = HuggingFaceTranscriptionModel

    def __init__(
        self,
        model: "HuggingFaceTranscriptionModel | HuggingFaceTranscriptionSpec | str",
        model_kwargs: dict | None = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, HuggingFaceTranscriptionModel):
            spec = model.spec
        elif isinstance(model, HuggingFaceTranscriptionSpec):
            spec = model
        else:
            raise TypeError(
                f"HuggingFaceTranscriptionClient expects a HuggingFaceTranscriptionModel member, "
                f"HuggingFaceTranscriptionSpec, or 'hf:<repo_id>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec
        self._pipe: Any = None  # lazy
        self._cache_key = _make_cache_key(spec.id, model_kwargs)

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> Any:
        from transformers import pipeline

        kwargs: dict[str, Any] = {}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)
        device = pop_device_hint(kwargs)

        logger.info("Loading ASR pipeline %s", self.spec.id)
        return pipeline("automatic-speech-recognition", model=self.spec.id, device=device, **kwargs)

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        with _registry_lock:
            if self._cache_key in _model_registry:
                self._pipe = _model_registry[self._cache_key]
                return
        self._pipe = self._load_pipeline()
        with _registry_lock:
            _model_registry[self._cache_key] = self._pipe

    @property
    def pipeline(self) -> Any:
        """The loaded pipeline (triggers weight load on first access)."""
        self._ensure_loaded()
        return self._pipe

    # ------------------------------------------------------------------
    # BaseTranscriptionClient interface
    # ------------------------------------------------------------------

    def _transcribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        response_format: str = "text",
        prompt: str | None = None,
        temperature: float | None = None,
    ) -> str | dict:
        import soundfile as sf

        from aimu.models._internal.audio_input import _normalize_audio

        self._ensure_loaded()

        b64, fmt = _normalize_audio(audio)
        raw_bytes = base64.b64decode(b64)
        audio_array, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")

        pipe_kwargs: dict[str, Any] = {}
        if response_format == "verbose_json":
            pipe_kwargs["return_timestamps"] = True
        if language is not None:
            pipe_kwargs["generate_kwargs"] = {"language": language}

        result = self._pipe({"raw": audio_array, "sampling_rate": sample_rate}, **pipe_kwargs)

        if response_format == "text":
            return result["text"]

        if response_format == "json":
            return {"text": result["text"]}

        # verbose_json: normalize HF chunks to the shared segment schema
        chunks = result.get("chunks", [])
        segments = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]} for c in chunks]
        duration = segments[-1]["end"] if segments else 0.0
        return {
            "text": result["text"],
            "language": language or "auto",
            "duration": duration,
            "segments": segments,
        }

    def __repr__(self) -> str:
        loaded = "loaded" if self._pipe is not None else "unloaded"
        return f"HuggingFaceTranscriptionClient(model={self.spec.id!r}, {loaded})"
