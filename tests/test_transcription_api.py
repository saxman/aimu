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
