"""HuggingFace-backed text-to-audio (music/sound) client.

``HuggingFaceAudioClient`` inherits :class:`BaseAudioClient` — the public
:meth:`generate` (format conversion, num_audio plumbing) lives on the base;
this subclass implements :meth:`_generate` to call the loaded pipeline and return
``(sample_rate, np.ndarray)`` pairs.

Three pipeline backends are supported, selected by ``spec.pipeline_type``:

- ``"musicgen"`` — ``transformers`` ``MusicgenForConditionalGeneration``.
  Token-autoregressive; duration is controlled via ``max_new_tokens``.
  No denoising steps; streaming yields a single final chunk.
- ``"audioldm2"`` — ``diffusers`` ``AudioLDM2Pipeline``.
  Latent diffusion; ``audio_length_in_s`` controls duration.
  Streaming yields one chunk per denoising step.
- ``"stable_audio"`` — ``diffusers`` ``StableAudioPipeline``.
  Latent diffusion; ``audio_end_in_s`` controls duration. Produces stereo output.
  Streaming yields one chunk per denoising step.

Usage::

    from aimu.models import HuggingFaceAudioClient, HuggingFaceAudioModel

    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    path = client.generate("upbeat lo-fi jazz", duration_s=10)

    # Or keep as numpy for downstream processing:
    sr, audio = client.generate("upbeat lo-fi jazz", format="numpy")
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Optional, Union

# Hard import so HAS_HF_AUDIO in aimu/models/__init__.py accurately reflects whether
# the optional deps are installed. soundfile is the meaningful sentinel here since
# torch/transformers/diffusers are pulled via the [hf] extra.
import soundfile  # noqa: F401

from ..._hf_device import move_to_device, pop_device_hint
from ...base import AudioModel, BaseAudioClient, HuggingFaceAudioSpec, StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)

_model_registry: dict[tuple, tuple] = {}  # cache_key → (pipe, processor)
_registry_lock = threading.Lock()


def _make_cache_key(spec_id: str, pipeline_type: str, model_kwargs: dict | None) -> tuple:
    return (spec_id, pipeline_type, *sorted((k, str(v)) for k, v in (model_kwargs or {}).items()))


# MusicGen generates at 50 tokens/second (all model sizes, 32 kHz output).
_MUSICGEN_TOKENS_PER_SECOND = 50
_MUSICGEN_SAMPLE_RATE = 32_000


class HuggingFaceAudioModel(AudioModel):
    """Catalog of supported HuggingFace audio-generation models.

    Each member's value is a :class:`HuggingFaceAudioSpec`. ``.value`` returns
    the HuggingFace repo id; ``.spec`` returns the full spec. Power users can
    bypass the enum by passing a ``"hf:<repo_id>"`` string directly to
    :class:`HuggingFaceAudioClient`.
    """

    MUSICGEN_SMALL = HuggingFaceAudioSpec(
        "facebook/musicgen-small",
        pipeline_type="musicgen",
        default_duration_s=10.0,
    )
    MUSICGEN_MEDIUM = HuggingFaceAudioSpec(
        "facebook/musicgen-medium",
        pipeline_type="musicgen",
        default_duration_s=10.0,
    )
    MUSICGEN_LARGE = HuggingFaceAudioSpec(
        "facebook/musicgen-large",
        pipeline_type="musicgen",
        default_duration_s=10.0,
    )
    AUDIOLDM2 = HuggingFaceAudioSpec(
        "cvssp/audioldm2",
        pipeline_type="audioldm2",
        default_duration_s=10.0,
        default_steps=200,
    )
    STABLE_AUDIO_OPEN = HuggingFaceAudioSpec(
        "stabilityai/stable-audio-open-1.0",
        pipeline_type="stable_audio",
        default_duration_s=10.0,
        default_steps=200,
    )


def _parse_model_string(s: str) -> HuggingFaceAudioSpec:
    """Resolve a ``"hf:<repo_id>"`` string to a known :class:`HuggingFaceAudioModel` spec.

    AIMU ships curated specs for best-of-class models only; an arbitrary repo id is **not**
    supported via string (``pipeline_type`` and generation defaults can't be inferred reliably).
    For a custom model, hand-build a :class:`HuggingFaceAudioSpec` and pass that object.
    """
    if ":" not in s:
        raise ValueError(
            f"HuggingFace audio model string must be in 'provider:repo_id' form "
            f"(e.g. 'hf:facebook/musicgen-small'). Got: {s!r}"
        )
    provider, repo_id = s.split(":", 1)
    if provider != "hf":
        raise ValueError(f"Only 'hf:' provider is supported for HuggingFaceAudioClient. Got provider: {provider!r}")
    if not repo_id:
        raise ValueError(f"Empty model id after 'hf:': {s!r}")

    for member in HuggingFaceAudioModel:
        if member.value == repo_id:
            return member.spec
    available = sorted(m.value for m in HuggingFaceAudioModel)
    raise ValueError(
        f"Unknown HuggingFace audio model id {repo_id!r}. AIMU supports curated models only; "
        f"pass a known id, a HuggingFaceAudioModel member, or a hand-built HuggingFaceAudioSpec "
        f"for a custom model. Available ids: {available}"
    )


class HuggingFaceAudioClient(BaseAudioClient):
    """Text-to-audio (music/sound) client backed by HuggingFace transformers or diffusers.

    Loads pipeline weights lazily on the first :meth:`_generate` call. Pass a
    :class:`HuggingFaceAudioModel` enum member, a :class:`HuggingFaceAudioSpec`,
    or a ``"hf:<repo_id>"`` string. ``model_kwargs`` is forwarded to
    the loader (``from_pretrained`` for all pipeline types).

    By default the model moves to the autodetected accelerator (``cuda``/``mps``,
    i.e. ``cuda:0``). On a busy multi-GPU box, target a specific GPU with
    ``model_kwargs={"device": "cuda:1"}``. These models are small enough to fit one
    card, so there is no sharding default (unlike the image client).
    """

    MODELS = HuggingFaceAudioModel

    def __init__(
        self,
        model: Union[HuggingFaceAudioModel, HuggingFaceAudioSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_model_string(model)
        elif isinstance(model, HuggingFaceAudioModel):
            spec = model.spec
        elif isinstance(model, HuggingFaceAudioSpec):
            spec = model
        else:
            raise TypeError(
                f"HuggingFaceAudioClient expects a HuggingFaceAudioModel member, "
                f"HuggingFaceAudioSpec, or 'hf:<repo_id>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec
        self._pipe: Any = None  # lazy
        self._processor: Any = None  # lazy; used only by musicgen
        self._cache_key = _make_cache_key(spec.id, spec.pipeline_type, model_kwargs)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_musicgen(self) -> tuple[Any, Any]:
        """Load MusicGen processor + model from transformers."""
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        model_id = self.spec.id
        kwargs: dict[str, Any] = {}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)
        device = pop_device_hint(kwargs)

        logger.info("Loading MusicGen model %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = MusicgenForConditionalGeneration.from_pretrained(model_id, **kwargs)
        return processor, move_to_device(model, device)

    def _load_diffusers_pipeline(self, pipeline_class_name: str) -> Any:
        """Load a diffusers audio pipeline by class name."""
        import diffusers

        try:
            pipeline_cls = getattr(diffusers, pipeline_class_name)
        except AttributeError as exc:
            raise ValueError(
                f"diffusers has no pipeline class {pipeline_class_name!r}. "
                f"Check spec.pipeline_type for {self.spec.id!r}."
            ) from exc

        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)
        device = pop_device_hint(kwargs)

        logger.info("Loading %s pipeline %s", pipeline_class_name, self.spec.id)
        pipe = pipeline_cls.from_pretrained(self.spec.id, **kwargs)
        return move_to_device(pipe, device)

    def _ensure_loaded(self) -> None:
        """Trigger lazy weight loading if not already done."""
        if self._pipe is not None:
            return
        with _registry_lock:
            if self._cache_key in _model_registry:
                self._pipe, self._processor = _model_registry[self._cache_key]
                return
        ptype = self.spec.pipeline_type
        if ptype == "musicgen":
            self._processor, self._pipe = self._load_musicgen()
        elif ptype == "audioldm2":
            self._pipe = self._load_diffusers_pipeline("AudioLDM2Pipeline")
        elif ptype == "stable_audio":
            self._pipe = self._load_diffusers_pipeline("StableAudioPipeline")
        else:
            raise ValueError(f"Unknown pipeline_type: {self.spec.pipeline_type!r}")
        with _registry_lock:
            _model_registry[self._cache_key] = (self._pipe, self._processor)

    @property
    def pipeline(self) -> Any:
        """The loaded pipeline (triggers weight load on first access)."""
        self._ensure_loaded()
        return self._pipe

    # ------------------------------------------------------------------
    # Runner methods (one per pipeline type)
    # ------------------------------------------------------------------

    def _run_musicgen(
        self, prompt: str, *, duration_s: float, num_audio: int, seed: Optional[int]
    ) -> list[tuple[int, Any]]:
        """Run MusicGen inference. Returns list of (sample_rate, ndarray)."""
        self._ensure_loaded()
        inputs = self._processor(text=[prompt] * num_audio, padding=True, return_tensors="pt")

        # Move inputs to the same device as the model.
        try:
            import torch

            device = next(self._pipe.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            gen_kwargs: dict[str, Any] = {}
            if seed is not None:
                gen_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
        except (ImportError, StopIteration):
            gen_kwargs = {}

        max_new_tokens = int(duration_s * _MUSICGEN_TOKENS_PER_SECOND)
        audio_values = self._pipe.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)

        # audio_values: (batch, channels, samples) — squeeze channel dim for mono.
        results = []
        for i in range(audio_values.shape[0]):
            arr = audio_values[i].cpu().numpy()
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]  # mono: (1, samples) → (samples,)
            results.append((_MUSICGEN_SAMPLE_RATE, arr))
        return results

    def _run_audioldm2(
        self,
        prompt: str,
        *,
        duration_s: float,
        num_inference_steps: int,
        num_audio: int,
        seed: Optional[int],
    ) -> list[tuple[int, Any]]:
        """Run AudioLDM2 inference. Returns list of (sample_rate, ndarray)."""
        import numpy as np

        self._ensure_loaded()
        kwargs: dict[str, Any] = {
            "prompt": [prompt] * num_audio,
            "num_inference_steps": num_inference_steps,
            "audio_length_in_s": duration_s,
            "num_waveforms_per_prompt": 1,
        }
        if seed is not None:
            import torch

            device = getattr(self._pipe, "device", None)
            device_str = device.type if device is not None else "cpu"
            kwargs["generator"] = torch.Generator(device=device_str).manual_seed(seed)

        result = self._pipe(**kwargs)
        audios = result.audios  # shape: (batch, samples)
        sample_rate = self._pipe.vocoder.config.sampling_rate
        return [(sample_rate, np.array(a)) for a in audios]

    def _run_stable_audio(
        self,
        prompt: str,
        *,
        duration_s: float,
        num_inference_steps: int,
        num_audio: int,
        seed: Optional[int],
    ) -> list[tuple[int, Any]]:
        """Run Stable Audio Open inference. Returns list of (sample_rate, ndarray).

        Stable Audio Open produces stereo output: shape (2, samples). We preserve the
        stereo layout so callers receive full-fidelity audio; soundfile handles multi-
        channel WAV natively.
        """
        import numpy as np

        self._ensure_loaded()
        kwargs: dict[str, Any] = {
            "prompt": [prompt] * num_audio,
            "num_inference_steps": num_inference_steps,
            "audio_end_in_s": duration_s,
        }
        if seed is not None:
            import torch

            device = getattr(self._pipe, "device", None)
            device_str = device.type if device is not None else "cpu"
            kwargs["generator"] = torch.Generator(device=device_str).manual_seed(seed)

        result = self._pipe(**kwargs)
        audios = result.audios  # shape: (batch, channels, samples)
        sample_rate = self._pipe.vae.config.sampling_rate
        return [(sample_rate, np.array(a)) for a in audios]

    # ------------------------------------------------------------------
    # BaseAudioClient interface
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        *,
        duration_s: float,
        num_inference_steps: Optional[int] = None,
        num_audio: int = 1,
        seed: Optional[int] = None,
        **_ignored: Any,
    ) -> list:
        """Dispatch to the pipeline-type-specific runner."""
        ptype = self.spec.pipeline_type
        steps = num_inference_steps if num_inference_steps is not None else self.spec.default_steps

        if ptype == "musicgen":
            return self._run_musicgen(prompt, duration_s=duration_s, num_audio=num_audio, seed=seed)
        elif ptype == "audioldm2":
            return self._run_audioldm2(
                prompt, duration_s=duration_s, num_inference_steps=steps, num_audio=num_audio, seed=seed
            )
        elif ptype == "stable_audio":
            return self._run_stable_audio(
                prompt, duration_s=duration_s, num_inference_steps=steps, num_audio=num_audio, seed=seed
            )
        else:
            raise ValueError(f"Unknown pipeline_type: {ptype!r}")

    def _generate_streamed(
        self,
        prompt: str,
        *,
        duration_s: float,
        num_inference_steps: Optional[int] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Any] = None,
        seed: Optional[int] = None,
        **_ignored: Any,
    ):
        """Streaming generation.

        For MusicGen: single final chunk (token-autoregressive, no denoising steps).
        For AudioLDM2 / Stable Audio Open: one AUDIO_GENERATING chunk per denoising
        step via ``callback_on_step_end``, then final chunks carrying encoded results.
        No intermediate audio decoding — intermediate chunks carry step progress only.
        """
        from ..._audio_output import encode_audio

        ptype = self.spec.pipeline_type

        if ptype == "musicgen":
            # MusicGen has no step callbacks — run blocking and emit one final chunk.
            results = self._run_musicgen(prompt, duration_s=duration_s, num_audio=num_audio, seed=seed)
            for sr, audio in results:
                encoded = encode_audio(audio, sr, format=format, prompt=prompt, output_dir=output_dir)
                yield StreamChunk(
                    StreamingContentType.AUDIO_GENERATING,
                    {"step": 1, "total_steps": 1, "final": True, "result": encoded, "duration_s": duration_s},
                )
            return

        # Diffusers-based pipeline: step-callback streaming.
        steps = num_inference_steps if num_inference_steps is not None else self.spec.default_steps
        self._ensure_loaded()

        q: "queue.Queue[Any]" = queue.Queue()
        _DONE = object()
        _EXC = object()

        def _callback(pipe_, step_index, timestep, callback_kwargs):  # noqa: ARG001
            q.put(
                StreamChunk(
                    StreamingContentType.AUDIO_GENERATING,
                    {
                        "step": step_index + 1,
                        "total_steps": steps,
                        "final": False,
                        "result": None,
                        "duration_s": duration_s,
                    },
                )
            )
            return callback_kwargs

        if ptype == "audioldm2":

            def _run():
                try:
                    import numpy as np

                    kwargs: dict[str, Any] = {
                        "prompt": [prompt] * num_audio,
                        "num_inference_steps": steps,
                        "audio_length_in_s": duration_s,
                        "num_waveforms_per_prompt": 1,
                        "callback_on_step_end": _callback,
                    }
                    if seed is not None:
                        import torch

                        device = getattr(self._pipe, "device", None)
                        device_str = device.type if device is not None else "cpu"
                        kwargs["generator"] = torch.Generator(device=device_str).manual_seed(seed)
                    result = self._pipe(**kwargs)
                    sample_rate = self._pipe.vocoder.config.sampling_rate
                    q.put((_DONE, [(sample_rate, np.array(a)) for a in result.audios]))
                except Exception as exc:  # noqa: BLE001
                    q.put((_EXC, exc))

        else:  # stable_audio

            def _run():
                try:
                    import numpy as np

                    kwargs = {
                        "prompt": [prompt] * num_audio,
                        "num_inference_steps": steps,
                        "audio_end_in_s": duration_s,
                        "callback_on_step_end": _callback,
                    }
                    if seed is not None:
                        import torch

                        device = getattr(self._pipe, "device", None)
                        device_str = device.type if device is not None else "cpu"
                        kwargs["generator"] = torch.Generator(device=device_str).manual_seed(seed)
                    result = self._pipe(**kwargs)
                    sample_rate = self._pipe.vae.config.sampling_rate
                    q.put((_DONE, [(sample_rate, np.array(a)) for a in result.audios]))
                except Exception as exc:  # noqa: BLE001
                    q.put((_EXC, exc))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if isinstance(item, tuple) and item[0] is _DONE:
                final_results = item[1]
                break
            if isinstance(item, tuple) and item[0] is _EXC:
                raise item[1]
            yield item  # AUDIO_GENERATING progress chunk

        thread.join()

        for sr, audio in final_results:
            encoded = encode_audio(audio, sr, format=format, prompt=prompt, output_dir=output_dir)
            yield StreamChunk(
                StreamingContentType.AUDIO_GENERATING,
                {
                    "step": steps,
                    "total_steps": steps,
                    "final": True,
                    "result": encoded,
                    "duration_s": duration_s,
                },
            )

    def __repr__(self) -> str:
        loaded = "loaded" if self._pipe is not None else "unloaded"
        return f"HuggingFaceAudioClient(model={self.spec.id!r}, {loaded})"
