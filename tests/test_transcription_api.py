"""Mock-only unit tests for the transcription surface."""

from __future__ import annotations
import pytest


def test_transcription_spec_equality_by_id():
    from aimu.models._base.transcription import TranscriptionSpec

    a = TranscriptionSpec("model-a")
    b = TranscriptionSpec("model-a")
    c = TranscriptionSpec("model-b")
    assert a == b
    assert a != c


def test_transcription_spec_hash_by_id():
    from aimu.models._base.transcription import TranscriptionSpec

    a = TranscriptionSpec("model-a")
    b = TranscriptionSpec("model-a")
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_transcription_spec_defaults():
    from aimu.models._base.transcription import TranscriptionSpec

    spec = TranscriptionSpec("test-model")
    assert spec.default_language is None
    assert spec.supports_timestamps is False
    assert spec.supports_translation is False


def test_transcription_spec_subclasses_inherit_equality():
    from aimu.models._base.transcription import HuggingFaceTranscriptionSpec, OpenAITranscriptionSpec

    a = HuggingFaceTranscriptionSpec("openai/whisper-tiny")
    b = OpenAITranscriptionSpec("openai/whisper-tiny")
    assert a == b


def test_transcription_model_value_is_spec_id():
    from aimu.models._base.transcription import TranscriptionModel, TranscriptionSpec

    class _DummyModel(TranscriptionModel):
        TEST = TranscriptionSpec("test-model-id")

    assert _DummyModel.TEST.value == "test-model-id"
    assert _DummyModel.TEST.spec.id == "test-model-id"


def test_transcribe_returns_string_for_text_format():
    client = _make_base_client()
    result = client.transcribe(b"fake-audio")
    assert result == "hello world"


def test_transcribe_passes_response_format_to_impl():
    client = _make_base_client()
    result = client.transcribe(b"fake-audio", response_format="json")
    assert result == {"text": "hello world"}
    assert client._calls[0]["response_format"] == "json"


def test_transcribe_rejects_invalid_response_format():
    client = _make_base_client()
    with pytest.raises(ValueError, match="response_format"):
        client.transcribe(b"fake-audio", response_format="xml")


def test_transcribe_rejects_verbose_json_on_model_without_timestamps():
    client = _make_base_client(supports_timestamps=False)
    with pytest.raises(ValueError, match="supports_timestamps"):
        client.transcribe(b"fake-audio", response_format="verbose_json")


def test_transcribe_verbose_json_allowed_when_model_supports_timestamps():
    client = _make_base_client(supports_timestamps=True)
    result = client.transcribe(b"fake-audio", response_format="verbose_json")
    assert isinstance(result, dict)


def _make_base_client(supports_timestamps=True):
    from aimu.models._base.transcription import BaseTranscriptionClient, TranscriptionSpec

    class _FakeClient(BaseTranscriptionClient):
        def __init__(self):
            self.model = None
            self.model_kwargs = None
            self.spec = TranscriptionSpec("test-model", supports_timestamps=supports_timestamps)
            self._calls = []

        def _transcribe(self, audio, *, language=None, response_format="text", prompt=None, temperature=None):
            self._calls.append({"audio": audio, "response_format": response_format})
            return "hello world" if response_format == "text" else {"text": "hello world"}

    return _FakeClient()


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _make_fake_openai():
    """Return a stub openai module and a call log."""
    from unittest.mock import MagicMock

    fake_openai = MagicMock()
    fake_openai.__version__ = "1.0.0"
    call_log = []

    def _create(**kwargs):
        call_log.append(kwargs)
        fmt = kwargs.get("response_format", "text")
        if fmt == "text":
            return "hello world"
        obj = MagicMock()
        obj.text = "hello world"
        if fmt == "json":
            obj.model_dump.return_value = {"text": "hello world"}
        else:  # verbose_json
            obj.model_dump.return_value = {
                "text": "hello world",
                "language": "en",
                "duration": 2.5,
                "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "hello world"}],
            }
        return obj

    fake_openai.OpenAI.return_value.audio.transcriptions.create.side_effect = _create
    return fake_openai, call_log


def test_openai_transcription_model_members():
    pytest.importorskip("openai")
    from aimu.models.providers.openai.transcription import OpenAITranscriptionModel

    names = {m.name for m in OpenAITranscriptionModel}
    assert "WHISPER_1" in names
    assert "GPT_4O_TRANSCRIBE" in names
    assert "GPT_4O_MINI_TRANSCRIBE" in names


def test_openai_transcription_model_supports_timestamps():
    pytest.importorskip("openai")
    from aimu.models.providers.openai.transcription import OpenAITranscriptionModel

    assert OpenAITranscriptionModel.WHISPER_1.spec.supports_timestamps is True
    assert OpenAITranscriptionModel.GPT_4O_TRANSCRIBE.spec.supports_timestamps is True


def test_openai_client_transcribe_text_format(monkeypatch):
    import sys

    fake_openai, call_log = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as mod

    reload(mod)

    client = mod.OpenAITranscriptionClient(mod.OpenAITranscriptionModel.WHISPER_1)
    result = client.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40)
    assert result == "hello world"
    assert call_log[-1]["model"] == "whisper-1"
    assert call_log[-1]["response_format"] == "text"


def test_openai_client_transcribe_verbose_json(monkeypatch):
    import sys

    fake_openai, call_log = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as mod

    reload(mod)

    client = mod.OpenAITranscriptionClient(mod.OpenAITranscriptionModel.WHISPER_1)
    result = client.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40, response_format="verbose_json")
    assert isinstance(result, dict)
    assert "text" in result
    assert "segments" in result


def test_openai_client_passes_language(monkeypatch):
    import sys

    fake_openai, call_log = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as mod

    reload(mod)

    client = mod.OpenAITranscriptionClient(mod.OpenAITranscriptionModel.WHISPER_1)
    client.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40, language="fr")
    assert call_log[-1].get("language") == "fr"


def test_openai_client_omits_language_when_none(monkeypatch):
    import sys

    fake_openai, call_log = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as mod

    reload(mod)

    client = mod.OpenAITranscriptionClient(mod.OpenAITranscriptionModel.WHISPER_1)
    client.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40)
    assert "language" not in call_log[-1]


def test_openai_client_string_construction(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as mod

    reload(mod)

    client = mod.OpenAITranscriptionClient("openai:whisper-1")
    assert client.spec.id == "whisper-1"


def test_openai_client_unknown_string_raises(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as mod

    reload(mod)

    with pytest.raises(ValueError, match="Unknown"):
        mod.OpenAITranscriptionClient("openai:gpt-4-turbo")


# ---------------------------------------------------------------------------
# HuggingFace provider
# ---------------------------------------------------------------------------


def _install_hf_transcription_stubs(monkeypatch):
    """Stub soundfile and transformers for HF transcription tests."""
    import sys
    from unittest.mock import MagicMock
    import numpy as np

    fake_sf = MagicMock()
    fake_sf._aimu_stub = True
    fake_sf.read.return_value = (np.zeros(16000, dtype="float32"), 16000)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    fake_tf = MagicMock()
    _pipe_output = {"text": "hello world"}

    def _fake_pipeline(task, model, **kwargs):
        pipe = MagicMock()
        pipe.return_value = _pipe_output
        return pipe

    fake_tf.pipeline.side_effect = _fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    return fake_tf


def test_hf_transcription_model_members():
    pytest.importorskip("soundfile")
    pytest.importorskip("transformers")
    from aimu.models.providers.hf.transcription import HuggingFaceTranscriptionModel

    names = {m.name for m in HuggingFaceTranscriptionModel}
    assert "WHISPER_TINY" in names
    assert "WHISPER_LARGE_V3" in names
    assert "DISTIL_WHISPER_LARGE_V3" in names


def test_hf_transcription_model_supports_timestamps():
    pytest.importorskip("soundfile")
    pytest.importorskip("transformers")
    from aimu.models.providers.hf.transcription import HuggingFaceTranscriptionModel

    assert HuggingFaceTranscriptionModel.WHISPER_LARGE_V3.spec.supports_timestamps is True


def test_hf_client_transcribe_text_format(monkeypatch):
    _install_hf_transcription_stubs(monkeypatch)
    from importlib import reload
    import aimu.models.providers.hf.transcription as mod

    reload(mod)

    client = mod.HuggingFaceTranscriptionClient(mod.HuggingFaceTranscriptionModel.WHISPER_TINY)
    result = client.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40)
    assert result == "hello world"


def test_hf_client_transcribe_json_format(monkeypatch):
    _install_hf_transcription_stubs(monkeypatch)
    from importlib import reload
    import aimu.models.providers.hf.transcription as mod

    reload(mod)

    client = mod.HuggingFaceTranscriptionClient(mod.HuggingFaceTranscriptionModel.WHISPER_TINY)
    result = client.transcribe(b"\x00" * 4, response_format="json")
    assert result == {"text": "hello world"}


def test_hf_client_transcribe_verbose_json_format(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    import numpy as np

    fake_sf = MagicMock()
    fake_sf._aimu_stub = True
    fake_sf.read.return_value = (np.zeros(16000, dtype="float32"), 16000)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    fake_tf = MagicMock()
    _chunks_output = {
        "text": "hello world",
        "chunks": [
            {"text": "hello", "timestamp": [0.0, 0.5]},
            {"text": " world", "timestamp": [0.5, 1.0]},
        ],
    }

    def _fake_pipeline(task, model, **kwargs):
        pipe = MagicMock()
        pipe.return_value = _chunks_output
        return pipe

    fake_tf.pipeline.side_effect = _fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    from importlib import reload
    import aimu.models.providers.hf.transcription as mod

    reload(mod)

    client = mod.HuggingFaceTranscriptionClient(mod.HuggingFaceTranscriptionModel.WHISPER_TINY)
    result = client.transcribe(b"\x00" * 4, response_format="verbose_json")
    assert isinstance(result, dict)
    assert result["text"] == "hello world"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 0.5


def test_hf_client_passes_language_to_pipeline(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    import numpy as np

    fake_sf = MagicMock()
    fake_sf._aimu_stub = True
    fake_sf.read.return_value = (np.zeros(16000, dtype="float32"), 16000)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    captured_kwargs = {}

    fake_tf = MagicMock()

    def _fake_pipeline(task, model, **kwargs):
        pipe = MagicMock()

        def _call(inputs, **kw):
            captured_kwargs.update(kw)
            return {"text": "bonjour"}

        pipe.side_effect = _call
        return pipe

    fake_tf.pipeline.side_effect = _fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    from importlib import reload
    import aimu.models.providers.hf.transcription as mod

    reload(mod)

    client = mod.HuggingFaceTranscriptionClient(mod.HuggingFaceTranscriptionModel.WHISPER_TINY)
    client.transcribe(b"\x00" * 4, language="fr")
    assert captured_kwargs.get("generate_kwargs", {}).get("language") == "fr"


# ---------------------------------------------------------------------------
# TranscriptionClient factory
# ---------------------------------------------------------------------------


def test_transcription_client_factory_with_openai_enum(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as op_mod

    reload(op_mod)
    import aimu.models.transcription_client as tc_mod

    reload(tc_mod)

    client = tc_mod.TranscriptionClient(op_mod.OpenAITranscriptionModel.WHISPER_1)
    assert hasattr(client, "transcribe")


def test_transcription_client_factory_with_openai_string(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.transcription_client as tc_mod

    reload(tc_mod)

    client = tc_mod.TranscriptionClient("openai:whisper-1")
    assert client.spec.id == "whisper-1"


def test_resolve_transcription_model_string_openai(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from importlib import reload
    import aimu.models.transcription_client as tc_mod

    reload(tc_mod)

    member = tc_mod.resolve_transcription_model_string("openai:whisper-1")
    assert member.value == "whisper-1"


def test_resolve_transcription_model_string_unknown_provider():
    from aimu.models.transcription_client import resolve_transcription_model_string

    with pytest.raises(ValueError, match="Unknown"):
        resolve_transcription_model_string("anthropic:whisper-1")


def test_transcription_model_env_constant():
    from aimu.models._internal.model_defaults import TRANSCRIPTION_MODEL_ENV

    assert TRANSCRIPTION_MODEL_ENV == "AIMU_TRANSCRIPTION_MODEL"


# ---------------------------------------------------------------------------
# Top-level aimu.transcription_client / aimu.transcribe
# ---------------------------------------------------------------------------


def test_aimu_transcription_client_constructs_openai(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    import aimu
    from importlib import reload

    reload(aimu)

    client = aimu.transcription_client("openai:whisper-1")
    assert client.spec.id == "whisper-1"


def test_aimu_transcribe_one_shot(monkeypatch):
    import sys

    fake_openai, call_log = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    import aimu
    from importlib import reload

    reload(aimu)

    result = aimu.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40, model="openai:whisper-1")
    assert isinstance(result, str)


def test_aimu_transcription_model_env_var(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AIMU_TRANSCRIPTION_MODEL", "openai:whisper-1")

    import aimu
    from importlib import reload

    reload(aimu)

    client = aimu.transcription_client()
    assert client.spec.id == "whisper-1"


def test_aimu_transcription_client_raises_without_model_and_env(monkeypatch):
    monkeypatch.delenv("AIMU_TRANSCRIPTION_MODEL", raising=False)

    import aimu

    with pytest.raises(ValueError, match="AIMU_TRANSCRIPTION_MODEL"):
        aimu.transcription_client()


# ---------------------------------------------------------------------------
# Async surface
# ---------------------------------------------------------------------------


def test_async_transcription_client_wrap_refusal():
    """aio.transcription_client() refuses non-sync-client inputs."""
    from aimu.aio.transcription import AsyncTranscriptionClient
    from aimu.models._base.transcription import TranscriptionSpec

    spec = TranscriptionSpec("test-model")
    with pytest.raises((ValueError, TypeError)):
        AsyncTranscriptionClient(spec)


async def test_async_transcription_client_wraps_openai(monkeypatch):
    import sys

    fake_openai, call_log = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as op_mod

    reload(op_mod)

    sync_client = op_mod.OpenAITranscriptionClient(op_mod.OpenAITranscriptionModel.WHISPER_1)

    from aimu.aio.transcription import AsyncTranscriptionClient

    async_client = AsyncTranscriptionClient(sync_client)
    result = await async_client.transcribe(b"\x52\x49\x46\x46" + b"\x00" * 40)
    assert result == "hello world"


async def test_aio_transcription_client_convenience(monkeypatch):
    import sys

    fake_openai, _ = _make_fake_openai()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from importlib import reload
    import aimu.models.providers.openai.transcription as op_mod

    reload(op_mod)

    sync_client = op_mod.OpenAITranscriptionClient(op_mod.OpenAITranscriptionModel.WHISPER_1)

    from aimu.aio import transcription_client

    async_client = transcription_client(sync_client)
    assert hasattr(async_client, "transcribe")


# ---------------------------------------------------------------------------
# Built-in transcribe_audio tool
# ---------------------------------------------------------------------------


def test_transcribe_audio_tool_spec():
    from aimu.tools import builtin

    assert hasattr(builtin, "transcription")
    tool = next(t for t in builtin.transcription if t.__name__ == "transcribe_audio")
    spec = tool.__tool_spec__
    assert spec["function"]["name"] == "transcribe_audio"
    assert "audio_path" in spec["function"]["parameters"]["properties"]
    assert tool.__tool_is_streaming__ is False


def test_make_transcription_tool_returns_bound_tool():
    class _MockTranscriptionClient:
        def transcribe(self, audio, **kwargs):
            return "hello world"

    from aimu.tools.builtin import make_transcription_tool

    tool = make_transcription_tool(_MockTranscriptionClient())
    assert callable(tool)
    assert tool.__tool_spec__["function"]["name"] == "transcribe_audio"


def test_transcribe_audio_tool_in_all_tools():
    from aimu.tools import builtin

    names = {t.__name__ for t in builtin.ALL_TOOLS}
    assert "transcribe_audio" in names
