"""HuggingFace-backed text-to-speech (TTS) client.

``HuggingFaceSpeechClient`` inherits :class:`BaseSpeechClient` — the public
:meth:`generate` (format conversion, voice/speed plumbing) lives on the base;
this subclass implements :meth:`_generate` to call the loaded pipeline and return
``(sample_rate, np.ndarray)`` pairs.

Three pipeline backends are supported, selected by ``spec.pipeline_type``:

- ``"tts_pipeline"`` — ``transformers.pipeline("text-to-speech")``.
  Works with any HF model in the TTS pipeline format (e.g. MMS-TTS).
  Returns ``{"audio": ndarray, "sampling_rate": int}``.
- ``"speecht5"`` — ``SpeechT5ForTextToSpeech`` + ``SpeechT5HifiGan`` vocoder.
  Fast, good English quality. Uses speaker embeddings (xvectors) for voice
  selection. Default embedding is index 7306 from the CMU Arctic xvectors
  dataset (loaded from ``datasets`` on first generate call). Pass ``voice="N"``
  to use a different index.
- ``"bark"`` — ``BarkModel`` + ``AutoProcessor``.
  Zero-shot voice cloning via voice codes (``"v2/en_speaker_6"``, etc.).

Usage::

    from aimu.models import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    path = client.generate("Hello, world!")

    # Keep as numpy:
    sr, audio = client.generate("Hello", format="numpy")
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Union

# Hard import so HAS_HF_SPEECH in aimu/models/__init__.py accurately reflects
# whether the optional deps are installed. soundfile is the meaningful sentinel
# since torch/transformers are pulled via the [hf] extra.
import soundfile  # noqa: F401

from ..._hf_device import move_to_device, pop_device_hint, resolve_device
from ...base import BaseSpeechClient, HuggingFaceSpeechSpec, SpeechModel

logger = logging.getLogger(__name__)

# Cache tuple stores (pipe, processor, vocoder, default_speaker_embedding).
# vocoder and default_speaker_embedding are None for non-speecht5 pipeline types.
_model_registry: dict[tuple, tuple] = {}
_registry_lock = threading.Lock()


def _make_cache_key(spec_id: str, pipeline_type: str, model_kwargs: dict | None) -> tuple:
    return (spec_id, pipeline_type, *sorted((k, str(v)) for k, v in (model_kwargs or {}).items()))


_BARK_SAMPLE_RATE = 24_000
_SPEECHT5_SAMPLE_RATE = 16_000
# Index into the CMU Arctic xvectors dataset used as the default speaker embedding.
_SPEECHT5_DEFAULT_SPEAKER_IDX = 7306


class HuggingFaceSpeechModel(SpeechModel):
    """Catalog of supported HuggingFace TTS models.

    Each member's value is a :class:`HuggingFaceSpeechSpec`. ``.value`` returns
    the HuggingFace repo id; ``.spec`` returns the full spec. Power users can
    bypass the enum by passing a ``"hf:<repo_id>"`` string directly to
    :class:`HuggingFaceSpeechClient`.
    """

    MMS_TTS_ENG = HuggingFaceSpeechSpec(
        "facebook/mms-tts-eng",
        pipeline_type="tts_pipeline",
        default_voice=None,
        default_speed=1.0,
    )
    SPEECHT5 = HuggingFaceSpeechSpec(
        "microsoft/speecht5_tts",
        pipeline_type="speecht5",
        default_voice=None,  # uses CMU Arctic xvectors index 7306
        default_speed=1.0,
    )
    BARK = HuggingFaceSpeechSpec(
        "suno/bark",
        pipeline_type="bark",
        default_voice="v2/en_speaker_6",
        default_speed=1.0,
    )


def _parse_model_string(s: str) -> HuggingFaceSpeechSpec:
    """Resolve a ``"hf:<repo_id>"`` string to a known :class:`HuggingFaceSpeechModel` spec.

    AIMU ships curated specs for best-of-class models only; an arbitrary repo id is **not**
    supported via string (``pipeline_type`` / ``default_voice`` can't be inferred reliably).
    For a custom model, hand-build a :class:`HuggingFaceSpeechSpec` and pass that object.
    """
    if ":" not in s:
        raise ValueError(
            f"HuggingFace speech model string must be in 'provider:repo_id' form "
            f"(e.g. 'hf:facebook/mms-tts-eng'). Got: {s!r}"
        )
    provider, repo_id = s.split(":", 1)
    if provider != "hf":
        raise ValueError(f"Only 'hf:' provider is supported for HuggingFaceSpeechClient. Got provider: {provider!r}")
    if not repo_id:
        raise ValueError(f"Empty model id after 'hf:': {s!r}")

    for member in HuggingFaceSpeechModel:
        if member.value == repo_id:
            return member.spec
    available = sorted(m.value for m in HuggingFaceSpeechModel)
    raise ValueError(
        f"Unknown HuggingFace speech model id {repo_id!r}. AIMU supports curated models only; "
        f"pass a known id, a HuggingFaceSpeechModel member, or a hand-built HuggingFaceSpeechSpec "
        f"for a custom model. Available ids: {available}"
    )


class HuggingFaceSpeechClient(BaseSpeechClient):
    """Text-to-speech client backed by HuggingFace transformers.

    Loads pipeline weights lazily on the first :meth:`_generate` call. Pass a
    :class:`HuggingFaceSpeechModel` enum member, a :class:`HuggingFaceSpeechSpec`,
    or a ``"hf:<repo_id>"`` string. ``model_kwargs`` is forwarded to the loader.

    Target a specific GPU with ``model_kwargs={"device": "cuda:1"}`` (TTS models are
    small, so there is no sharding default). Note the SpeechT5 and BARK loaders move
    to the autodetected accelerator by default; the MMS-TTS ``pipeline`` stays on CPU
    unless a ``device`` hint is given.
    """

    MODELS = HuggingFaceSpeechModel

    def __init__(
        self,
        model: Union[HuggingFaceSpeechModel, HuggingFaceSpeechSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, HuggingFaceSpeechModel):
            spec = model.spec
        elif isinstance(model, HuggingFaceSpeechSpec):
            spec = model
        else:
            raise TypeError(
                f"HuggingFaceSpeechClient expects a HuggingFaceSpeechModel member, "
                f"HuggingFaceSpeechSpec, or 'hf:<repo_id>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec
        self._pipe: Any = None  # lazy
        self._processor: Any = None  # lazy; used by speecht5 and bark
        self._vocoder: Any = None  # lazy; used by speecht5
        self._default_speaker_embedding: Any = None  # lazy; used by speecht5
        self._xvectors_cache: Any = None  # lazy; used by speecht5 for non-default voices
        self._cache_key = _make_cache_key(spec.id, spec.pipeline_type, model_kwargs)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_tts_pipeline(self) -> Any:
        from transformers import pipeline

        kwargs: dict[str, Any] = {}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)
        # transformers' pipeline() takes placement via its own ``device=`` arg; None
        # preserves the default (CPU). A ``device`` hint (e.g. "cuda:1") targets one GPU.
        device = pop_device_hint(kwargs)

        logger.info("Loading TTS pipeline %s", self.spec.id)
        return pipeline("text-to-speech", model=self.spec.id, device=device, **kwargs)

    def _load_speecht5(self) -> tuple[Any, Any, Any, Any]:
        """Load SpeechT5 TTS model, HiFi-GAN vocoder, and default speaker embedding."""
        import torch
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        kwargs: dict[str, Any] = {}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)
        device = resolve_device(pop_device_hint(kwargs))

        logger.info("Loading SpeechT5 %s", self.spec.id)
        processor = SpeechT5Processor.from_pretrained(self.spec.id, **kwargs)
        model = SpeechT5ForTextToSpeech.from_pretrained(self.spec.id, **kwargs)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5-hifigan", **kwargs)

        model = model.to(device)
        vocoder = vocoder.to(device)

        # Load default speaker embedding from CMU Arctic xvectors dataset.
        from datasets import load_dataset

        xvectors = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        default_embedding = torch.tensor(xvectors[_SPEECHT5_DEFAULT_SPEAKER_IDX]["xvector"]).unsqueeze(0).to(device)

        return processor, model, vocoder, default_embedding

    def _load_bark(self) -> tuple[Any, Any]:
        from transformers import AutoProcessor, BarkModel

        kwargs: dict[str, Any] = {}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)
        device = pop_device_hint(kwargs)

        logger.info("Loading Bark model %s", self.spec.id)
        processor = AutoProcessor.from_pretrained(self.spec.id, **kwargs)
        model = BarkModel.from_pretrained(self.spec.id, **kwargs)
        return processor, move_to_device(model, device)

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        with _registry_lock:
            if self._cache_key in _model_registry:
                self._pipe, self._processor, self._vocoder, self._default_speaker_embedding = _model_registry[
                    self._cache_key
                ]
                return
        ptype = self.spec.pipeline_type
        if ptype == "tts_pipeline":
            self._pipe = self._load_tts_pipeline()
        elif ptype == "speecht5":
            self._processor, self._pipe, self._vocoder, self._default_speaker_embedding = self._load_speecht5()
        elif ptype == "bark":
            self._processor, self._pipe = self._load_bark()
        else:
            raise ValueError(f"Unknown pipeline_type: {ptype!r}")
        with _registry_lock:
            _model_registry[self._cache_key] = (
                self._pipe,
                self._processor,
                self._vocoder,
                self._default_speaker_embedding,
            )

    @property
    def pipeline(self) -> Any:
        """The loaded pipeline (triggers weight load on first access)."""
        self._ensure_loaded()
        return self._pipe

    # ------------------------------------------------------------------
    # Runner methods
    # ------------------------------------------------------------------

    def _run_tts_pipeline(
        self, text: str, *, voice: Optional[str], speed: Optional[float], num_audio: int
    ) -> list[tuple[int, Any]]:
        import numpy as np

        self._ensure_loaded()
        results = []
        for _ in range(num_audio):
            out = self._pipe(text)
            audio = np.array(out["audio"])
            if audio.ndim > 1:
                audio = audio.squeeze()
            sr: int = out["sampling_rate"]
            results.append((sr, audio))
        return results

    def _run_speecht5(
        self, text: str, *, voice: Optional[str], speed: Optional[float], num_audio: int
    ) -> list[tuple[int, Any]]:
        import torch

        self._ensure_loaded()

        # Resolve speaker embedding: None → pre-loaded default; numeric string → xvectors index.
        if voice is not None:
            try:
                idx = int(voice)
                if self._xvectors_cache is None:
                    from datasets import load_dataset

                    self._xvectors_cache = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                device = next(self._pipe.parameters()).device
                speaker_embedding = torch.tensor(self._xvectors_cache[idx]["xvector"]).unsqueeze(0).to(device)
            except (ValueError, IndexError):
                logger.warning("Invalid SpeechT5 voice index %r — using default embedding.", voice)
                speaker_embedding = self._default_speaker_embedding
        else:
            speaker_embedding = self._default_speaker_embedding

        results = []
        device = next(self._pipe.parameters()).device
        for _ in range(num_audio):
            inputs = self._processor(text=text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            speech = self._pipe.generate_speech(input_ids, speaker_embedding, vocoder=self._vocoder)
            audio = speech.cpu().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()
            results.append((_SPEECHT5_SAMPLE_RATE, audio))
        return results

    def _run_bark(
        self, text: str, *, voice: Optional[str], speed: Optional[float], num_audio: int
    ) -> list[tuple[int, Any]]:
        import numpy as np

        self._ensure_loaded()
        voice_preset = voice or self.spec.default_voice
        results = []
        for _ in range(num_audio):
            inputs = self._processor(text, voice_preset=voice_preset, return_tensors="pt")
            try:
                import torch

                device = next(self._pipe.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            except (ImportError, StopIteration):
                pass
            speech_values = self._pipe.generate(**inputs, do_sample=True)
            sr = self._pipe.generation_config.sample_rate
            audio = speech_values.cpu().numpy().squeeze()
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            results.append((int(sr), audio))
        return results

    # ------------------------------------------------------------------
    # BaseSpeechClient interface
    # ------------------------------------------------------------------

    def _generate(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        **_ignored: Any,
    ) -> list:
        ptype = self.spec.pipeline_type
        if ptype == "tts_pipeline":
            return self._run_tts_pipeline(text, voice=voice, speed=speed, num_audio=num_audio)
        elif ptype == "speecht5":
            return self._run_speecht5(text, voice=voice, speed=speed, num_audio=num_audio)
        elif ptype == "bark":
            return self._run_bark(text, voice=voice, speed=speed, num_audio=num_audio)
        else:
            raise ValueError(f"Unknown pipeline_type: {ptype!r}")

    def __repr__(self) -> str:
        loaded = "loaded" if self._pipe is not None else "unloaded"
        return f"HuggingFaceSpeechClient(model={self.spec.id!r}, {loaded})"
