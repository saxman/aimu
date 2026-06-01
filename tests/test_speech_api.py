# tests/test_speech_api.py
"""Mock-only unit tests for the speech-client surface.

Parallel to tests/test_audio_api.py for the speech (TTS) modality.
"""

from __future__ import annotations
import sys
import types
from unittest.mock import MagicMock
import numpy as np
import pytest


def _make_stub_module(name: str) -> types.ModuleType:
    """Create a stub module with a valid __spec__ so importlib.util.find_spec works."""
    import importlib.machinery

    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod._aimu_stub = True  # type: ignore[attr-defined]
    return mod


def _install_speech_stubs(monkeypatch=None, force=False):
    """Install minimal stubs for soundfile, transformers, openai before importing speech clients.

    ``test_audio_api`` and ``test_speech_api`` both stub ``transformers`` but with different
    fakes (MusicGen vs. SpeechT5/Bark/pipeline), and they share the ``_aimu_stub`` marker — so
    whichever is imported first during collection wins and the other's polite guard skips, leaving
    the wrong stub in ``sys.modules``. Pass ``force=True`` (from the autouse fixture below) to
    overwrite whatever is present; pass ``monkeypatch`` so the override is scoped to one test and
    restored afterwards (no cross-file pollution). Called once at import time with neither argument
    purely so module-level ``from aimu.models.hf_speech import ...`` can import (it hard-imports
    ``soundfile``)."""

    def _set(name, mod):
        if monkeypatch is not None:
            monkeypatch.setitem(sys.modules, name, mod)
        else:
            sys.modules[name] = mod

    # soundfile — needed by encode_audio (imported by hf_speech_client)
    if force or not getattr(sys.modules.get("soundfile"), "_aimu_stub", False):
        sf_stub = _make_stub_module("soundfile")

        def _sf_write(file, data, samplerate, format=None, subtype=None):
            file.write(b"RIFF" + b"\x00" * 40)

        sf_stub.write = _sf_write
        _set("soundfile", sf_stub)

    # diffusers — stub before transformers to prevent the real diffusers from
    # calling importlib.util.find_spec("transformers") on our stub (which has
    # __spec__=None and triggers a ValueError).
    try:
        from test_images_api import _install_diffusers_stub

        _install_diffusers_stub()
    except ImportError:
        if "diffusers" not in sys.modules:
            sys.modules["diffusers"] = _make_stub_module("diffusers")

    # transformers
    if force or not getattr(sys.modules.get("transformers"), "_aimu_stub", False):
        tr_stub = _make_stub_module("transformers")

        class _FakeTTSPipeline:
            _last_text = None

            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def __call__(self, text, **kw):
                _FakeTTSPipeline._last_text = text
                return {"audio": np.zeros(1000, dtype=np.float32), "sampling_rate": 16000}

        class _FakeBarkProcessor:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def __call__(self, text, voice_preset=None, return_tensors=None):
                return {"input_ids": np.array([[1, 2, 3]])}

        class _FakeBarkModel:
            class generation_config:
                sample_rate = 24000

            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def parameters(self):
                p = MagicMock()
                p.device = MagicMock()
                p.device.type = "cpu"
                yield p

            def to(self, device):
                return self

            def generate(self, **kw):
                arr = MagicMock()
                arr.cpu.return_value = arr
                arr.numpy.return_value = np.zeros((1, 1, 1000), dtype=np.float32)
                arr.squeeze.return_value = np.zeros(1000, dtype=np.float32)
                return arr

        class _FakeSpeechT5Processor:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def __call__(self, text=None, return_tensors=None, **kw):
                return {"input_ids": MagicMock(to=lambda d: MagicMock())}

        class _FakeSpeechT5Model:
            _device = "cpu"

            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def to(self, device):
                return self

            def parameters(self):
                p = MagicMock()
                p.device = "cpu"
                yield p

            def generate_speech(self, input_ids, speaker_embeddings, vocoder=None):
                arr = MagicMock()
                arr.cpu.return_value = arr
                arr.numpy.return_value = np.zeros(16000, dtype=np.float32)
                arr.ndim = 1
                return arr

        class _FakeSpeechT5HifiGan:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def to(self, device):
                return self

        def _pipeline(task, model=None, **kw):
            return _FakeTTSPipeline()

        tr_stub.pipeline = _pipeline
        tr_stub.AutoProcessor = _FakeBarkProcessor
        tr_stub.BarkModel = _FakeBarkModel
        tr_stub.SpeechT5Processor = _FakeSpeechT5Processor
        tr_stub.SpeechT5ForTextToSpeech = _FakeSpeechT5Model
        tr_stub.SpeechT5HifiGan = _FakeSpeechT5HifiGan
        _set("transformers", tr_stub)

    # datasets — needed by SpeechT5 for speaker embeddings
    if force or not getattr(sys.modules.get("datasets"), "_aimu_stub", False):
        ds_stub = _make_stub_module("datasets")

        class _FakeXvectorsDataset:
            def __getitem__(self, idx):
                return {"xvector": np.zeros(512, dtype=np.float32).tolist()}

        def _load_dataset(name, split=None, **kw):
            return _FakeXvectorsDataset()

        ds_stub.load_dataset = _load_dataset
        _set("datasets", ds_stub)

    # openai
    if force or not getattr(sys.modules.get("openai"), "_aimu_stub", False):
        openai_stub = _make_stub_module("openai")

        class _FakeSpeechResponse:
            content = np.zeros(1000, dtype=np.int16).tobytes()

        class _FakeSpeechCreate:
            def create(self, model, voice, input, speed, response_format):
                return _FakeSpeechResponse()

        class _FakeSpeechStreamCtx:
            def __enter__(self):
                self._chunks = [b"\x00\x01" * 512, b"\x00\x02" * 512]
                return self

            def __exit__(self, *a):
                pass

            def iter_bytes(self, chunk_size=4096):
                yield from self._chunks

        class _FakeWithStreaming:
            def create(self, model, voice, input, speed, response_format):
                return _FakeSpeechStreamCtx()

        class _FakeAudioSpeech:
            speech = _FakeSpeechCreate()

            class with_streaming_response:
                speech = _FakeWithStreaming()

        class _FakeOpenAI:
            def __init__(self, **kw):
                pass

            audio = _FakeAudioSpeech()

        openai_stub.OpenAI = _FakeOpenAI
        _set("openai", openai_stub)
        _set("openai.resources", _make_stub_module("openai.resources"))
        _set("openai._client", _make_stub_module("openai._client"))


# Permanent install so module-level ``from aimu.models.hf_speech import ...`` can import at
# collection time (hf_speech_client hard-imports soundfile).
_install_speech_stubs()


@pytest.fixture(autouse=True)
def _force_speech_stubs(monkeypatch):
    """Re-assert the speech stubs before each test, scoped + restored via monkeypatch.

    Guarantees this module's tests see the SpeechT5/Bark/pipeline transformers stub even when
    another test file (e.g. test_audio_api) installed its own transformers stub earlier in the
    session. monkeypatch restores the prior sys.modules entries afterwards, so this never leaks
    into other files."""
    _install_speech_stubs(monkeypatch=monkeypatch, force=True)


# ---------------------------------------------------------------------------
# SpeechSpec equality and StreamChunk helper
# ---------------------------------------------------------------------------

from aimu.models.base import (  # noqa: E402
    BaseSpeechClient,
    SpeechSpec,
    HuggingFaceSpeechSpec,
    OpenAISpeechSpec,
    StreamChunk,
    StreamingContentType,
)


def test_speech_spec_equality_by_id():
    a = SpeechSpec("tts-1")
    b = SpeechSpec("tts-1")
    c = SpeechSpec("tts-2")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_hf_speech_spec_inherits_id_equality():
    a = HuggingFaceSpeechSpec("suno/bark", pipeline_type="bark")
    b = HuggingFaceSpeechSpec("suno/bark", pipeline_type="bark")
    c = HuggingFaceSpeechSpec("facebook/mms-tts-eng", pipeline_type="tts_pipeline")
    assert a == b
    assert a != c


def test_openai_speech_spec_inherits_id_equality():
    a = OpenAISpeechSpec("tts-1", default_voice="alloy")
    b = OpenAISpeechSpec("tts-1", default_voice="nova")  # voice differs, id same
    assert a == b  # equality is by id only


def test_speech_generating_in_streaming_content_type():
    assert StreamingContentType.SPEECH_GENERATING == "speech_generating"


def test_stream_chunk_is_speech_progress():
    chunk = StreamChunk(
        StreamingContentType.SPEECH_GENERATING,
        {"chunk_index": 1, "total_chunks": 1, "final": True, "result": "path.wav"},
    )
    assert chunk.is_speech_progress() is True
    assert chunk.is_audio_progress() is False
    assert chunk.is_image_progress() is False


# ---------------------------------------------------------------------------
# BaseSpeechClient public interface
# ---------------------------------------------------------------------------


class _MinimalSpeechClient(BaseSpeechClient):
    """Concrete minimal subclass for testing BaseSpeechClient directly."""

    def __init__(self, spec, model_kwargs=None):
        super().__init__(model=spec.id, model_kwargs=model_kwargs)
        self.spec = spec

    def _generate(self, text, *, voice=None, speed=None, num_audio=1, **kwargs):
        sr = 16000
        audio = np.zeros(100, dtype=np.float32)
        return [(sr, audio)] * num_audio


def test_base_client_generate_numpy():
    spec = SpeechSpec("test-model", default_voice="alloy", default_speed=1.0)
    client = _MinimalSpeechClient(spec)
    result = client.generate("hello", format="numpy")
    sr, audio = result
    assert sr == 16000
    assert hasattr(audio, "shape")


def test_base_client_generate_bytes():
    spec = SpeechSpec("test-model")
    client = _MinimalSpeechClient(spec)
    result = client.generate("hello", format="bytes")
    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"


def test_base_client_generate_num_audio_returns_list():
    spec = SpeechSpec("test-model")
    client = _MinimalSpeechClient(spec)
    result = client.generate("hello", format="numpy", num_audio=3)
    assert isinstance(result, list)
    assert len(result) == 3


def test_base_client_generate_num_audio_one_returns_scalar():
    spec = SpeechSpec("test-model")
    client = _MinimalSpeechClient(spec)
    result = client.generate("hello", format="numpy", num_audio=1)
    sr, audio = result  # not a list
    assert sr == 16000


def test_base_client_generate_num_audio_zero_raises():
    spec = SpeechSpec("test-model")
    client = _MinimalSpeechClient(spec)
    with pytest.raises(ValueError, match="num_audio"):
        client.generate("hello", num_audio=0)


def test_base_client_generate_resolves_voice_from_spec():
    spec = SpeechSpec("test-model", default_voice="nova")
    captured = {}

    class _CapturingClient(_MinimalSpeechClient):
        def _generate(self, text, *, voice=None, speed=None, num_audio=1, **kwargs):
            captured["voice"] = voice
            captured["speed"] = speed
            return super()._generate(text, voice=voice, speed=speed, num_audio=num_audio, **kwargs)

    client = _CapturingClient(spec)
    client.generate("hello", format="numpy")
    assert captured["voice"] == "nova"
    assert captured["speed"] == 1.0


def test_base_client_generate_caller_voice_overrides_spec():
    spec = SpeechSpec("test-model", default_voice="nova")
    captured = {}

    class _CapturingClient(_MinimalSpeechClient):
        def _generate(self, text, *, voice=None, speed=None, num_audio=1, **kwargs):
            captured["voice"] = voice
            return super()._generate(text, voice=voice, speed=speed, num_audio=num_audio, **kwargs)

    client = _CapturingClient(spec)
    client.generate("hello", voice="alloy", format="numpy")
    assert captured["voice"] == "alloy"


def test_base_client_stream_yields_final_chunks():
    spec = SpeechSpec("test-model")
    client = _MinimalSpeechClient(spec)
    chunks = list(client.generate("hello", format="numpy", stream=True))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.phase == StreamingContentType.SPEECH_GENERATING
    assert c.content["final"] is True
    assert c.content["chunk_index"] == 1
    assert c.content["total_chunks"] == 1


def test_base_client_stream_num_audio_correct_indices():
    spec = SpeechSpec("test-model")
    client = _MinimalSpeechClient(spec)
    chunks = list(client.generate("hello", format="numpy", stream=True, num_audio=3))
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.content["chunk_index"] == i
        assert chunk.content["total_chunks"] == 3
        assert chunk.content["final"] is True


# ---------------------------------------------------------------------------
# HuggingFace speech client
# ---------------------------------------------------------------------------

import importlib  # noqa: E402
import aimu.models  # noqa: E402


def _register_hf_speech():
    if getattr(aimu.models, "HAS_HF_SPEECH", False):
        return
    mod = importlib.import_module("aimu.models.hf_speech")
    aimu.models.HAS_HF_SPEECH = True
    aimu.models.HuggingFaceSpeechClient = mod.HuggingFaceSpeechClient
    aimu.models.HuggingFaceSpeechModel = mod.HuggingFaceSpeechModel


_register_hf_speech()

from aimu.models.hf_speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel  # noqa: E402


def test_hf_speech_model_enum_values():
    assert HuggingFaceSpeechModel.MMS_TTS_ENG.value == "facebook/mms-tts-eng"
    assert HuggingFaceSpeechModel.SPEECHT5.value == "microsoft/speecht5_tts"
    assert HuggingFaceSpeechModel.BARK.value == "suno/bark"
    assert HuggingFaceSpeechModel.MMS_TTS_ENG.spec.pipeline_type == "tts_pipeline"
    assert HuggingFaceSpeechModel.SPEECHT5.spec.pipeline_type == "speecht5"
    assert HuggingFaceSpeechModel.BARK.spec.pipeline_type == "bark"
    assert HuggingFaceSpeechModel.BARK.spec.default_voice == "v2/en_speaker_6"
    assert HuggingFaceSpeechModel.SPEECHT5.spec.default_voice is None


def test_hf_speech_client_from_enum():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    assert client.spec.id == "facebook/mms-tts-eng"
    assert client._pipe is None  # lazy


def test_hf_speech_client_from_string():
    client = HuggingFaceSpeechClient("hf:facebook/mms-tts-eng")
    assert client.spec.id == "facebook/mms-tts-eng"
    assert client.spec.pipeline_type == "tts_pipeline"


def test_hf_speech_client_speecht5_string_resolves_to_enum_spec():
    member = HuggingFaceSpeechModel.SPEECHT5
    client = HuggingFaceSpeechClient(f"hf:{member.value}")
    assert client.spec is member.spec


def test_hf_speech_client_bark_string_resolves_to_enum_spec():
    # Regression: the string form must yield BARK's curated default_voice, not a default None.
    client = HuggingFaceSpeechClient("hf:suno/bark")
    assert client.spec is HuggingFaceSpeechModel.BARK.spec
    assert client.spec.default_voice == "v2/en_speaker_6"


def test_hf_speech_client_unknown_id_raises():
    # Arbitrary repos are not supported via string — curated enum models only.
    with pytest.raises(ValueError, match="curated models only"):
        HuggingFaceSpeechClient("hf:someorg/custom-tts")


def test_hf_speech_client_invalid_string():
    with pytest.raises(ValueError, match="provider:repo_id"):
        HuggingFaceSpeechClient("no-colon")


def test_hf_speech_client_wrong_type():
    with pytest.raises(TypeError):
        HuggingFaceSpeechClient(42)


def test_hf_speech_client_generate_tts_pipeline_returns_path(tmp_path):
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    result = client.generate("Hello world", format="path", output_dir=tmp_path)
    assert result.endswith(".wav")


def test_hf_speech_client_generate_numpy():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    result = client.generate("Hello world", format="numpy")
    sr, audio = result
    assert isinstance(sr, int)
    assert hasattr(audio, "shape")


def test_hf_speech_client_generate_speecht5_numpy():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.SPEECHT5)
    result = client.generate("Hello world", format="numpy")
    sr, audio = result
    assert sr == 16000
    assert hasattr(audio, "shape")


def test_hf_speech_client_generate_speecht5_lazy_load():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.SPEECHT5)
    assert client._pipe is None
    client.generate("Hello", format="numpy")
    assert client._pipe is not None
    assert client._vocoder is not None
    assert client._default_speaker_embedding is not None


def test_hf_speech_client_generate_speecht5_voice_index():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.SPEECHT5)
    # voice="100" should look up index 100 from xvectors (stubbed to return zeros)
    result = client.generate("Hello", format="numpy", voice="100")
    sr, audio = result
    assert sr == 16000


def test_hf_speech_client_generate_bark_numpy():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.BARK)
    result = client.generate("Hello world", format="numpy")
    sr, audio = result
    assert sr == 24000


def test_hf_speech_client_streaming_yields_final_chunk():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    chunks = list(client.generate("Hello", stream=True, format="numpy"))
    assert len(chunks) == 1
    assert chunks[0].is_speech_progress()
    assert chunks[0].content["final"] is True
    assert chunks[0].content["chunk_index"] == 1


def test_hf_speech_client_num_audio():
    client = HuggingFaceSpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    result = client.generate("Hello", format="numpy", num_audio=2)
    assert isinstance(result, list)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# OpenAI speech client
# ---------------------------------------------------------------------------


def _register_openai_speech():
    if getattr(aimu.models, "HAS_OPENAI_SPEECH", False):
        return
    # Piggyback on HAS_OPENAI_COMPAT — openai SDK already present
    mod = importlib.import_module("aimu.models.openai_speech")
    aimu.models.OpenAISpeechClient = mod.OpenAISpeechClient
    aimu.models.OpenAISpeechModel = mod.OpenAISpeechModel
    aimu.models.HAS_OPENAI_SPEECH = True


_register_openai_speech()

from aimu.models.openai_speech import OpenAISpeechClient, OpenAISpeechModel  # noqa: E402


def test_openai_speech_model_enum_values():
    assert OpenAISpeechModel.TTS_1.value == "tts-1"
    assert OpenAISpeechModel.TTS_1_HD.value == "tts-1-hd"
    assert OpenAISpeechModel.TTS_1.spec.default_voice == "alloy"


def test_openai_speech_client_from_enum(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    assert client.spec.id == "tts-1"


def test_openai_speech_client_generate_path(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    result = client.generate("Hello world", format="path", output_dir=tmp_path)
    assert result.endswith(".wav")


def test_openai_speech_client_generate_numpy(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    result = client.generate("Hello world", format="numpy")
    sr, audio = result
    assert sr == 24000
    assert hasattr(audio, "shape")


def test_openai_speech_client_pcm_decode_produces_float32(monkeypatch):
    """PCM bytes must be decoded to float32 in [-1, 1]."""
    import numpy as np

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    sr, audio = client.generate("Test", format="numpy")
    assert audio.dtype == np.float32
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_openai_speech_client_streaming_yields_progress_then_final(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    chunks = list(client.generate("Hello", stream=True, format="numpy"))
    # At least: some progress chunks (final=False) + 1 final chunk
    assert len(chunks) >= 1
    assert chunks[-1].is_speech_progress()
    assert chunks[-1].content["final"] is True
    # All non-final chunks have result=None
    for c in chunks[:-1]:
        assert c.content["result"] is None


def test_openai_speech_client_voice_param_forwarded(monkeypatch):
    """voice= kwarg should be used; spec default is 'alloy'."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    # Just verify it doesn't error — fake client ignores the voice value
    sr, audio = client.generate("Hello", voice="nova", format="numpy")
    assert sr == 24000


# ---------------------------------------------------------------------------
# SpeechClient factory + top-level API
# ---------------------------------------------------------------------------

import aimu  # noqa: E402


def test_speech_client_factory_hf_string():
    from aimu.models.speech_client import SpeechClient

    client = SpeechClient("hf:facebook/mms-tts-eng")
    assert client.spec.id == "facebook/mms-tts-eng"


def test_speech_client_factory_openai_string(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from aimu.models.speech_client import SpeechClient

    client = SpeechClient("openai:tts-1")
    assert client.spec.id == "tts-1"


def test_speech_client_factory_hf_enum():
    from aimu.models.speech_client import SpeechClient

    client = SpeechClient(HuggingFaceSpeechModel.MMS_TTS_ENG)
    assert client.spec.id == "facebook/mms-tts-eng"


def test_speech_client_factory_openai_enum(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from aimu.models.speech_client import SpeechClient

    client = SpeechClient(OpenAISpeechModel.TTS_1)
    assert client.spec.id == "tts-1"


def test_speech_client_factory_spec_raises():
    from aimu.models.speech_client import SpeechClient
    from aimu.models.base import HuggingFaceSpeechSpec

    with pytest.raises(TypeError, match="enum member"):
        SpeechClient(HuggingFaceSpeechSpec("facebook/mms-tts-eng"))


def test_speech_client_factory_unknown_provider():
    from aimu.models.speech_client import SpeechClient

    with pytest.raises(ValueError, match="Unknown speech provider"):
        SpeechClient("gcp:wavenet-en")


def test_speech_client_generate_delegates(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from aimu.models.speech_client import SpeechClient

    client = SpeechClient("openai:tts-1")
    result = client.generate("Hello", format="path", output_dir=tmp_path)
    assert result.endswith(".wav")


def test_top_level_speech_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import aimu

    assert callable(aimu.speech_client)
    client = aimu.speech_client("openai:tts-1")
    assert client.spec.id == "tts-1"


def test_top_level_generate_speech(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import aimu

    result = aimu.generate_speech("Hello", model="openai:tts-1", format="path", output_dir=tmp_path)
    assert result.endswith(".wav")


def test_available_speech_clients_returns_list():
    from aimu.models import available_speech_clients

    clients = available_speech_clients()
    assert isinstance(clients, list)
    assert len(clients) >= 1  # at least one provider installed
