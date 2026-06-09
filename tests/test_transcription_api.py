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

        def _transcribe(self, audio, *, language=None, response_format="text",
                        prompt=None, temperature=None):
            self._calls.append({"audio": audio, "response_format": response_format})
            return "hello world" if response_format == "text" else {"text": "hello world"}

    return _FakeClient()


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _make_fake_openai():
    """Return a stub openai module and a call log."""
    import sys
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
