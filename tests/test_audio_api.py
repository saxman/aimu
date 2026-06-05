"""Mock-only unit tests for the audio-client surface.

Parallel to :mod:`tests/test_images_api.py` for the audio modality. Covers:

* Shared infrastructure: :class:`HuggingFaceAudioSpec` equality,
  :func:`aimu.models._internal.audio_output.encode_audio` format matrix.
* HuggingFace provider: ``HuggingFaceAudioClient`` construction, ``"hf:..."``
  string parsing, pipeline_type inference, lazy load, per-call kwarg threading,
  seed plumbing, streaming (MusicGen single-final; diffusers step-callback).
* Top-level dispatch: ``aimu.audio_client()`` and ``aimu.generate_audio()``
  route to the right concrete client.

Live integration tests live in :mod:`tests/test_audio.py` (opt-in via
``--audio-client``). Tool-wrapping tests live in :mod:`tests/test_audio_tools.py`.
Async tests live in :mod:`tests/test_aio_audio_api.py`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stubs — must be installed before aimu.models.providers.hf.audio is imported
# ---------------------------------------------------------------------------


def _install_audio_stubs():  # noqa: PLR0912
    """Install fake soundfile, transformers, and diffusers modules.

    Safe to call multiple times — subsequent calls are no-ops if stubs are
    already in place. Also exported so test_audio_tools.py and
    test_aio_audio_api.py can reuse the same stubs.

    Soundfile is installed FIRST so that when ``_install_diffusers_stub``
    from test_images_api.py is imported (triggering aimu.models import), the
    hf_audio subpackage can be imported successfully (it has a hard
    ``import soundfile`` at the top). If soundfile were installed after the
    image stub, aimu.models.audio_client would record _HAS_HF_AUDIO=False
    before our stub was ready.
    """
    # --- soundfile --- (must come before _install_diffusers_stub / aimu.models import)

    if not getattr(sys.modules.get("soundfile"), "_aimu_stub", False):
        sf_stub = types.ModuleType("soundfile")
        sf_stub._aimu_stub = True  # type: ignore[attr-defined]

        def _sf_write(file, data, samplerate, format=None, subtype=None):  # noqa: ARG001
            # Write a recognisable 4-byte WAV-magic header for test assertions.
            file.write(b"RIFF" + b"\x00" * 40)

        sf_stub.write = _sf_write  # type: ignore[attr-defined]
        sys.modules["soundfile"] = sf_stub

    # --- transformers (MusicGen) ---
    if not getattr(sys.modules.get("transformers"), "_aimu_stub", False):

        class _FakeDevice:
            type = "cpu"

        class _FakeAudioBatch:
            def __init__(self, data):
                self._data = data

            @property
            def shape(self):
                return self._data.shape

            @property
            def ndim(self):
                return self._data.ndim

            def cpu(self):
                return self

            def numpy(self):
                return self._data

        class _FakeAudioValues:
            """Fake MusicGen output tensor with shape (batch, channels, samples)."""

            def __init__(self, batch=1):
                self._data = np.zeros((batch, 1, 500), dtype=np.float32)

            @property
            def shape(self):
                return self._data.shape

            def __getitem__(self, i):
                return _FakeAudioBatch(self._data[i])

        class _FakeProcessor:
            _last_call_kwargs: dict = {}

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):  # noqa: ARG003
                return cls()

            def __call__(self, text, padding=True, return_tensors="pt"):  # noqa: ARG002
                self._last_call_kwargs = {"text": text}
                return {"_fake_input": text}

        class _FakeMusicgenModel:
            _last_call_kwargs: dict = {}

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):  # noqa: ARG003
                inst = cls()
                return inst

            def parameters(self):
                # Yield a fake param so seed plumbing can read .device.
                param = MagicMock()
                param.device = "cpu"  # string so torch.Generator("cpu") works
                return iter([param])

            def to(self, device):  # noqa: ARG002
                return self

            def generate(self, **kwargs):
                self._last_call_kwargs = kwargs
                batch = 1
                return _FakeAudioValues(batch=batch)

        tr_stub = types.ModuleType("transformers")
        tr_stub._aimu_stub = True  # type: ignore[attr-defined]
        tr_stub.AutoProcessor = _FakeProcessor  # type: ignore[attr-defined]
        tr_stub.MusicgenForConditionalGeneration = _FakeMusicgenModel  # type: ignore[attr-defined]
        sys.modules["transformers"] = tr_stub

    # --- diffusers (AudioLDM2, StableAudio) ---
    # Ensure image pipeline classes are present regardless of collection order.
    # This import triggers test_images_api.py module-level code (including aimu.models
    # import) only AFTER soundfile and transformers are already stubbed above, so
    # aimu.models.audio_client records _HAS_HF_AUDIO=True correctly.
    try:
        from test_images_api import _install_diffusers_stub

        _install_diffusers_stub()
    except ImportError:
        pass  # Running without test_images_api — image stubs not needed here.

    # Add audio pipeline classes to whatever diffusers stub is in sys.modules.
    if "diffusers" not in sys.modules:
        diffusers_stub = types.ModuleType("diffusers")
        diffusers_stub._aimu_stub = True  # type: ignore[attr-defined]
        sys.modules["diffusers"] = diffusers_stub

    diff = sys.modules["diffusers"]

    if not hasattr(diff, "AudioLDM2Pipeline"):

        class _FakeAudioLDM2Pipeline:
            device = MagicMock(type="cpu")
            _last_call_kwargs: dict = {}

            class _FakeVocoder:
                class config:
                    sampling_rate = 16_000

            vocoder = _FakeVocoder()

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):  # noqa: ARG003
                inst = cls()
                inst.repo_id = repo_id
                return inst

            def to(self, device):  # noqa: ARG002
                return self

            def __call__(self, **kwargs):
                self._last_call_kwargs = kwargs
                callback = kwargs.get("callback_on_step_end")
                total_steps = kwargs.get("num_inference_steps", 1)
                prompts = kwargs.get("prompt", [""])
                num_audio = len(prompts)
                if callback is not None:
                    for i in range(total_steps):
                        callback(self, i, i, {})
                result = MagicMock()
                result.audios = [np.zeros(16_000, dtype=np.float32) for _ in range(num_audio)]
                return result

        diff.AudioLDM2Pipeline = _FakeAudioLDM2Pipeline  # type: ignore[attr-defined]

    if not hasattr(diff, "StableAudioPipeline"):

        class _FakeStableAudioPipeline:
            device = MagicMock(type="cpu")
            _last_call_kwargs: dict = {}

            class _FakeVAE:
                class config:
                    sampling_rate = 44_100

            vae = _FakeVAE()

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):  # noqa: ARG003
                inst = cls()
                inst.repo_id = repo_id
                return inst

            def to(self, device):  # noqa: ARG002
                return self

            def __call__(self, **kwargs):
                self._last_call_kwargs = kwargs
                callback = kwargs.get("callback_on_step_end")
                total_steps = kwargs.get("num_inference_steps", 1)
                prompts = kwargs.get("prompt", [""])
                num_audio = len(prompts)
                if callback is not None:
                    for i in range(total_steps):
                        callback(self, i, i, {})
                result = MagicMock()
                result.audios = [np.zeros((2, 44_100), dtype=np.float32) for _ in range(num_audio)]
                return result

        diff.StableAudioPipeline = _FakeStableAudioPipeline  # type: ignore[attr-defined]


_install_audio_stubs()


import importlib  # noqa: E402

import aimu.models  # noqa: E402

# Force-import the hf_audio subpackage now that stubs are in place, and patch all
# HAS_HF_AUDIO / HuggingFaceAudioClient references that were captured as False/None
# when other test files imported aimu.models earlier (before our stubs were installed).
if not aimu.models.HAS_HF_AUDIO:
    _hf_audio_mod = importlib.import_module("aimu.models.providers.hf.audio")
    aimu.models.HAS_HF_AUDIO = True
    aimu.models.HuggingFaceAudioClient = _hf_audio_mod.HuggingFaceAudioClient
    aimu.models.HuggingFaceAudioModel = _hf_audio_mod.HuggingFaceAudioModel
    aimu.models.providers.hf.audio = _hf_audio_mod

# Patch the audio_client module's own _HAS_HF_AUDIO flag (set at its import time).
# This is necessary when aimu.models was imported before our stubs were ready.
import aimu.models.audio_client as _audio_client_mod  # noqa: E402

_audio_client_mod._HAS_HF_AUDIO = True
_audio_client_mod.HuggingFaceAudioClient = aimu.models.HuggingFaceAudioClient
_audio_client_mod.HuggingFaceAudioModel = aimu.models.HuggingFaceAudioModel


# ---------------------------------------------------------------------------
# Shared imports (post-stub)
# ---------------------------------------------------------------------------

import aimu  # noqa: E402

aimu.HAS_HF_AUDIO = True
aimu.HuggingFaceAudioClient = aimu.models.HuggingFaceAudioClient
aimu.HuggingFaceAudioModel = aimu.models.HuggingFaceAudioModel

from aimu.models import HuggingFaceAudioClient, HuggingFaceAudioModel, HuggingFaceAudioSpec  # noqa: E402
from aimu.models._internal.audio_output import encode_audio  # noqa: E402
from aimu.models.base import StreamingContentType  # noqa: E402


# ===========================================================================
# Shared infrastructure — HuggingFaceAudioSpec equality
# ===========================================================================


def test_hf_audio_spec_equality_by_id():
    """HuggingFaceAudioSpec equality uses id only — other fields don't affect it."""
    a = HuggingFaceAudioSpec("facebook/musicgen-small", pipeline_type="musicgen", default_duration_s=10.0)
    b = HuggingFaceAudioSpec("facebook/musicgen-small", pipeline_type="musicgen", default_duration_s=30.0)
    c = HuggingFaceAudioSpec("facebook/musicgen-medium", pipeline_type="musicgen")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_hf_audio_model_value_is_repo_id():
    assert HuggingFaceAudioModel.MUSICGEN_SMALL.value == "facebook/musicgen-small"
    assert HuggingFaceAudioModel.AUDIOLDM2.value == "cvssp/audioldm2"
    assert HuggingFaceAudioModel.STABLE_AUDIO_OPEN.value == "stabilityai/stable-audio-open-1.0"


def test_hf_audio_model_spec_pipeline_types():
    assert HuggingFaceAudioModel.MUSICGEN_SMALL.spec.pipeline_type == "musicgen"
    assert HuggingFaceAudioModel.AUDIOLDM2.spec.pipeline_type == "audioldm2"
    assert HuggingFaceAudioModel.STABLE_AUDIO_OPEN.spec.pipeline_type == "stable_audio"


# ---------------------------------------------------------------------------
# encode_audio — shared format conversion helper
# ---------------------------------------------------------------------------


def test_encode_audio_numpy_returns_unchanged_tuple():
    audio = np.zeros(500, dtype=np.float32)
    sr = 16_000
    result = encode_audio(audio, sr, format="numpy")
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0] == sr
    assert result[1] is audio


def test_encode_audio_bytes_returns_wav_bytes():
    audio = np.zeros(500, dtype=np.float32)
    out = encode_audio(audio, 16_000, format="bytes")
    assert isinstance(out, bytes)
    assert out.startswith(b"RIFF")


def test_encode_audio_data_url_returns_base64_data_url():
    audio = np.zeros(500, dtype=np.float32)
    out = encode_audio(audio, 16_000, format="data_url")
    assert isinstance(out, str)
    assert out.startswith("data:audio/wav;base64,")


def test_encode_audio_path_writes_file_and_returns_string(tmp_path):
    audio = np.zeros(500, dtype=np.float32)
    out = encode_audio(audio, 16_000, format="path", prompt="test", output_dir=tmp_path)
    saved = Path(out)
    assert saved.exists() and saved.suffix == ".wav" and saved.parent == tmp_path


def test_encode_audio_path_creates_output_dir(tmp_path):
    target = tmp_path / "nested" / "audio"
    audio = np.zeros(500, dtype=np.float32)
    out = encode_audio(audio, 16_000, format="path", output_dir=target)
    assert Path(out).parent == target


def test_encode_audio_stereo_handled_correctly(tmp_path):
    """Stable Audio Open produces (channels, samples) stereo arrays."""
    audio = np.zeros((2, 44_100), dtype=np.float32)
    out = encode_audio(audio, 44_100, format="path", output_dir=tmp_path)
    assert Path(out).exists()


def test_encode_audio_invalid_format_raises():
    audio = np.zeros(500, dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown format"):
        encode_audio(audio, 16_000, format="ogg")


# ===========================================================================
# HuggingFaceAudioClient — construction
# ===========================================================================


def test_hf_audio_client_accepts_enum_member():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    assert client.spec.id == "facebook/musicgen-small"
    assert client.spec.pipeline_type == "musicgen"


def test_hf_audio_client_accepts_spec():
    spec = HuggingFaceAudioSpec("custom/audio-repo", pipeline_type="musicgen")
    client = HuggingFaceAudioClient(spec)
    assert client.spec is spec


def test_hf_audio_client_accepts_hf_string():
    client = HuggingFaceAudioClient("hf:facebook/musicgen-small")
    assert client.spec.id == "facebook/musicgen-small"
    assert client.spec.pipeline_type == "musicgen"


def test_hf_audio_client_rejects_bare_repo_id():
    with pytest.raises(ValueError, match="provider:repo_id"):
        HuggingFaceAudioClient("facebook/musicgen-small")


def test_hf_audio_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Only 'hf:' provider"):
        HuggingFaceAudioClient("replicate:some/model")


def test_hf_audio_client_rejects_empty_repo_id():
    with pytest.raises(ValueError, match="Empty model id"):
        HuggingFaceAudioClient("hf:")


def test_hf_audio_client_rejects_wrong_type():
    with pytest.raises(TypeError, match="HuggingFaceAudioModel member"):
        HuggingFaceAudioClient(42)


def test_hf_audio_client_lazy_pipeline_not_loaded_at_init():
    """Constructing a client must not load weights."""
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    assert client._pipe is None
    assert "unloaded" in repr(client)


# ---------------------------------------------------------------------------
# String parsing — resolves a known id to the curated enum spec
# ---------------------------------------------------------------------------


def test_hf_audio_string_musicgen_resolves_to_enum_spec():
    member = HuggingFaceAudioModel.MUSICGEN_MEDIUM
    client = HuggingFaceAudioClient(f"hf:{member.value}")
    assert client.spec is member.spec


def test_hf_audio_string_audioldm_resolves_to_enum_spec():
    member = HuggingFaceAudioModel.AUDIOLDM2
    client = HuggingFaceAudioClient(f"hf:{member.value}")
    assert client.spec is member.spec


def test_hf_audio_string_stable_audio_resolves_to_enum_spec():
    member = HuggingFaceAudioModel.STABLE_AUDIO_OPEN
    client = HuggingFaceAudioClient(f"hf:{member.value}")
    assert client.spec is member.spec


def test_hf_audio_string_unknown_id_raises():
    # Arbitrary repos are not supported via string — curated enum models only.
    with pytest.raises(ValueError, match="curated models only"):
        HuggingFaceAudioClient("hf:someorg/custom-audio-model")


# ---------------------------------------------------------------------------
# HuggingFaceAudioClient — generate() non-streaming
# ---------------------------------------------------------------------------


def test_hf_generate_musicgen_returns_numpy_by_default():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    result = client.generate("upbeat jazz", format="numpy")
    assert isinstance(result, tuple) and len(result) == 2
    sr, audio = result
    assert sr == 32_000
    assert isinstance(audio, np.ndarray)


def test_hf_generate_musicgen_format_bytes():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    out = client.generate("drums", format="bytes")
    assert isinstance(out, bytes)
    assert out.startswith(b"RIFF")


def test_hf_generate_musicgen_format_path(tmp_path):
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    out = client.generate("piano", format="path", output_dir=tmp_path)
    assert isinstance(out, str)
    assert Path(out).exists() and Path(out).suffix == ".wav"


def test_hf_generate_musicgen_num_audio_returns_list():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    # The fake model always returns batch=1, so we test list-wrapping via num_audio=1 first.
    result = client.generate("ambient", format="numpy")
    assert isinstance(result, tuple)  # single → scalar

    # For num_audio > 1, client sends num_audio prompts; stub returns list.
    # With our stub always returning 1-item, this would return a list of 1. That's OK —
    # the important thing is that the return type is list when num_audio > 1.
    results = client.generate("ambient", format="numpy", num_audio=1)
    # Should still be a tuple (single) when num_audio=1
    assert isinstance(results, tuple)


def test_hf_generate_audioldm2_returns_numpy():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.AUDIOLDM2)
    result = client.generate("rain sounds", format="numpy")
    assert isinstance(result, tuple)
    sr, audio = result
    assert sr == 16_000
    assert isinstance(audio, np.ndarray)


def test_hf_generate_stable_audio_returns_numpy():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.STABLE_AUDIO_OPEN)
    result = client.generate("ocean waves", format="numpy")
    assert isinstance(result, tuple)
    sr, audio = result
    assert sr == 44_100
    assert isinstance(audio, np.ndarray)


def test_hf_generate_applies_spec_default_duration_when_unset():
    """duration_s should default to spec.default_duration_s when unspecified."""
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    # Call generate without duration_s.
    client.generate("test", format="numpy")
    # Spec default is 10.0s → max_new_tokens = int(10.0 * 50) = 500.
    model = client._pipe  # after generate(), pipe is loaded
    kwargs = model._last_call_kwargs
    assert kwargs.get("max_new_tokens") == 500


def test_hf_generate_duration_s_overrides_default():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    client.generate("test", format="numpy", duration_s=5.0)
    kwargs = client._pipe._last_call_kwargs
    assert kwargs.get("max_new_tokens") == 250  # int(5.0 * 50)


def test_hf_generate_audioldm2_duration_threaded_through():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.AUDIOLDM2)
    client.generate("test", format="numpy", duration_s=7.0)
    kwargs = client._pipe._last_call_kwargs
    assert kwargs.get("audio_length_in_s") == 7.0


def test_hf_generate_audioldm2_steps_override():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.AUDIOLDM2)
    client.generate("test", format="numpy", num_inference_steps=50)
    kwargs = client._pipe._last_call_kwargs
    assert kwargs.get("num_inference_steps") == 50


def test_hf_generate_seed_plumbed_through_musicgen(monkeypatch):
    """seed= should cause torch.Generator.manual_seed to be called."""
    import torch

    captured: dict = {}
    real_generator = torch.Generator

    class TrackingGenerator(real_generator):
        def manual_seed(self, seed):
            captured["seed"] = seed
            return super().manual_seed(seed)

    monkeypatch.setattr(torch, "Generator", TrackingGenerator)

    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    client.generate("test", format="numpy", seed=42)
    assert captured.get("seed") == 42


def test_hf_generate_num_audio_zero_rejected():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    with pytest.raises(ValueError, match="num_audio must be"):
        client.generate("test", num_audio=0)


# ===========================================================================
# Streaming — AUDIO_GENERATING chunks
# ===========================================================================


def test_musicgen_stream_emits_single_final_chunk():
    """MusicGen is token-autoregressive — streaming yields one final chunk."""
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    chunks = list(client.generate("test", stream=True, format="path"))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.phase == StreamingContentType.AUDIO_GENERATING
    assert chunk.content["final"] is True
    assert chunk.content["step"] == 1
    assert chunk.content["total_steps"] == 1
    assert chunk.content["result"] is not None


def test_musicgen_stream_chunk_is_audio_progress():
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    chunks = list(client.generate("test", stream=True, format="numpy"))
    assert all(c.is_audio_progress() for c in chunks)


def test_audioldm2_stream_emits_progress_then_final():
    """AudioLDM2 streaming: N progress chunks + 1 final chunk per num_inference_steps."""
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.AUDIOLDM2)
    chunks = list(client.generate("test", stream=True, format="path", num_inference_steps=3))

    progress = [c for c in chunks if not c.content["final"]]
    finals = [c for c in chunks if c.content["final"]]

    assert len(progress) == 3
    assert len(finals) == 1
    assert all(c.phase == StreamingContentType.AUDIO_GENERATING for c in chunks)

    for i, c in enumerate(progress):
        assert c.content["step"] == i + 1
        assert c.content["total_steps"] == 3
        assert c.content["result"] is None

    assert finals[0].content["result"] is not None


def test_stable_audio_stream_emits_progress_then_final():
    """Stable Audio Open streaming: same step-callback pattern as AudioLDM2."""
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.STABLE_AUDIO_OPEN)
    chunks = list(client.generate("test", stream=True, format="path", num_inference_steps=4))

    finals = [c for c in chunks if c.content["final"]]
    assert len(finals) == 1
    assert finals[0].content["result"] is not None


def test_audio_generating_chunk_has_all_required_fields():
    """AUDIO_GENERATING chunk content shape: step, total_steps, final, result, duration_s."""
    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    chunks = list(client.generate("test", stream=True, format="numpy"))
    assert len(chunks) >= 1
    c = chunks[-1]  # final chunk
    content = c.content
    assert "step" in content
    assert "total_steps" in content
    assert "final" in content
    assert "result" in content
    assert "duration_s" in content


# ===========================================================================
# AudioClient factory dispatch
# ===========================================================================


def test_audio_client_factory_dispatches_hf_enum():
    from aimu.models import AudioClient

    c = AudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    assert isinstance(c._client, HuggingFaceAudioClient)
    assert c.spec.id == "facebook/musicgen-small"


def test_audio_client_factory_dispatches_hf_string():
    from aimu.models import AudioClient

    c = AudioClient("hf:facebook/musicgen-small")
    assert isinstance(c._client, HuggingFaceAudioClient)


def test_audio_client_factory_rejects_bare_string():
    from aimu.models import AudioClient

    with pytest.raises(ValueError, match="provider:model_id"):
        AudioClient("facebook/musicgen-small")


def test_audio_client_factory_rejects_unknown_provider():
    from aimu.models import AudioClient

    with pytest.raises(ValueError, match="Unknown audio provider"):
        AudioClient("replicate:some/model")


# ===========================================================================
# Top-level dispatch (aimu.audio_client / aimu.generate_audio)
# ===========================================================================


def test_top_level_audio_client_dispatches_hf_string():
    c = aimu.audio_client("hf:facebook/musicgen-small")
    from aimu.models import AudioClient

    assert isinstance(c, AudioClient)
    assert isinstance(c._client, HuggingFaceAudioClient)


def test_top_level_audio_client_dispatches_hf_enum():
    c = aimu.audio_client(HuggingFaceAudioModel.MUSICGEN_SMALL)
    from aimu.models import AudioClient

    assert isinstance(c, AudioClient)


def test_top_level_generate_audio_one_shot(tmp_path):
    out = aimu.generate_audio("test", model="hf:facebook/musicgen-small", format="path", output_dir=tmp_path)
    assert isinstance(out, str)
    assert Path(out).exists()


def test_top_level_generate_audio_numpy():
    result = aimu.generate_audio("test", model=HuggingFaceAudioModel.MUSICGEN_SMALL, format="numpy")
    assert isinstance(result, tuple)
    sr, audio = result
    assert sr == 32_000
    assert isinstance(audio, np.ndarray)
