"""
Audio input tests.

Mock-only by default (``pytest tests/test_audio_input.py``); pass ``--client=<provider>``
to additionally run real-provider smoke tests against audio-capable models.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

from helpers import MockModelClient, create_real_model_client, resolve_model_params


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #


def _make_wav_bytes(sample_rate: int = 16000, num_samples: int = 1) -> bytes:
    """Build a minimal valid 16-bit mono WAV in memory."""
    data = b"\x00\x00" * num_samples
    data_size = len(data)
    fmt_size = 16
    riff_size = 4 + 8 + fmt_size + 8 + data_size
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", fmt_size, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_chunk = struct.pack("<4sI", b"data", data_size) + data
    return header + fmt + data_chunk


_TINY_WAV_BYTES = _make_wav_bytes()
_TINY_WAV_B64 = base64.b64encode(_TINY_WAV_BYTES).decode("ascii")
_TINY_WAV_DATA_URL = f"data:audio/wav;base64,{_TINY_WAV_B64}"
_TINY_MP3_BYTES = b"ID3" + b"\x00" * 7  # not a real mp3, but enough to test format detection


# --------------------------------------------------------------------------- #
# Helper: an audio-capable mock model client                                   #
# --------------------------------------------------------------------------- #


class _AudioMockClient(MockModelClient):
    """MockModelClient with supports_audio=True so _chat_setup accepts audio."""

    def __init__(self, responses):
        super().__init__(responses)
        self.model.supports_audio = True
        self.model.supports_vision = False

    def chat(
        self,
        user_message,
        generate_kwargs=None,
        use_tools=True,
        stream=False,
        images=None,
        audio=None,
        tools=None,
        include=None,
    ):
        if audio:
            self._chat_setup(user_message, generate_kwargs, use_tools, audio=audio)
            response = self._responses[self._call_count]
            self._call_count += 1
            self.messages.append({"role": "assistant", "content": response})
            return response
        return super().chat(user_message, generate_kwargs, use_tools, stream)


# --------------------------------------------------------------------------- #
# _normalize_audio                                                              #
# --------------------------------------------------------------------------- #


def test_normalize_audio_encodes_bytes_as_wav():
    from aimu.models._internal.audio_input import _normalize_audio

    b64, fmt = _normalize_audio(_TINY_WAV_BYTES)
    assert fmt == "wav"
    assert base64.b64decode(b64) == _TINY_WAV_BYTES


def test_normalize_audio_parses_data_url():
    from aimu.models._internal.audio_input import _normalize_audio

    b64, fmt = _normalize_audio(_TINY_WAV_DATA_URL)
    assert fmt == "wav"
    assert b64 == _TINY_WAV_B64


def test_normalize_audio_reads_wav_path(tmp_path: Path):
    from aimu.models._internal.audio_input import _normalize_audio

    fp = tmp_path / "clip.wav"
    fp.write_bytes(_TINY_WAV_BYTES)
    b64, fmt = _normalize_audio(str(fp))
    assert fmt == "wav"
    assert base64.b64decode(b64) == _TINY_WAV_BYTES


def test_normalize_audio_reads_pathlib_path(tmp_path: Path):
    from aimu.models._internal.audio_input import _normalize_audio

    fp = tmp_path / "clip.mp3"
    fp.write_bytes(_TINY_MP3_BYTES)
    b64, fmt = _normalize_audio(fp)
    assert fmt == "mp3"
    assert base64.b64decode(b64) == _TINY_MP3_BYTES


def test_normalize_audio_infers_format_from_extension(tmp_path: Path):
    from aimu.models._internal.audio_input import _normalize_audio

    for ext, expected_fmt in [("flac", "flac"), ("ogg", "ogg"), ("m4a", "m4a"), ("webm", "webm")]:
        fp = tmp_path / f"clip.{ext}"
        fp.write_bytes(b"fake")
        _, fmt = _normalize_audio(fp)
        assert fmt == expected_fmt


def test_normalize_audio_defaults_bytes_to_wav():
    from aimu.models._internal.audio_input import _normalize_audio

    _, fmt = _normalize_audio(b"\x00\x01\x02")
    assert fmt == "wav"


def test_normalize_audio_rejects_unknown_type():
    from aimu.models._internal.audio_input import _normalize_audio

    with pytest.raises(TypeError):
        _normalize_audio(12345)  # type: ignore[arg-type]


def test_normalize_audio_fetches_http_url(monkeypatch):
    """HTTP(S) URLs are fetched eagerly since input_audio has no remote-URL field."""
    from aimu.models._internal.audio_input import _normalize_audio
    import urllib.request

    class _FakeResponse:
        def read(self):
            return _TINY_WAV_BYTES

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: _FakeResponse())
    b64, fmt = _normalize_audio("https://example.com/clip.wav")
    assert fmt == "wav"
    assert base64.b64decode(b64) == _TINY_WAV_BYTES


# --------------------------------------------------------------------------- #
# _build_audio_content_blocks                                                   #
# --------------------------------------------------------------------------- #


def test_build_audio_content_blocks_shape():
    from aimu.models._internal.audio_input import _build_audio_content_blocks

    blocks = _build_audio_content_blocks("transcribe this", [_TINY_WAV_BYTES])
    assert blocks[0] == {"type": "text", "text": "transcribe this"}
    assert blocks[1]["type"] == "input_audio"
    assert blocks[1]["input_audio"]["format"] == "wav"
    assert blocks[1]["input_audio"]["data"] == _TINY_WAV_B64


def test_build_audio_content_blocks_multiple():
    from aimu.models._internal.audio_input import _build_audio_content_blocks

    blocks = _build_audio_content_blocks("x", [_TINY_WAV_BYTES, _TINY_WAV_BYTES])
    assert len(blocks) == 3  # 1 text + 2 audio
    assert blocks[1]["type"] == "input_audio"
    assert blocks[2]["type"] == "input_audio"


def test_build_audio_content_blocks_data_url_input():
    from aimu.models._internal.audio_input import _build_audio_content_blocks

    blocks = _build_audio_content_blocks("x", [_TINY_WAV_DATA_URL])
    assert blocks[1]["input_audio"]["data"] == _TINY_WAV_B64
    assert blocks[1]["input_audio"]["format"] == "wav"


# --------------------------------------------------------------------------- #
# _chat_setup with audio= / _require_audio                                      #
# --------------------------------------------------------------------------- #


def test_chat_setup_with_audio_builds_input_audio_blocks():
    client = _AudioMockClient(["ok"])
    client._chat_setup("transcribe", audio=[_TINY_WAV_BYTES])
    user_msg = client.messages[-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "input_audio"


def test_chat_setup_rejects_audio_on_non_audio_model():
    client = MockModelClient(["unused"])
    client.model.supports_audio = False
    with pytest.raises(ValueError, match="does not support audio"):
        client._chat_setup("transcribe", audio=[_TINY_WAV_BYTES])


def test_chat_setup_without_audio_keeps_string_content():
    client = _AudioMockClient(["ok"])
    client._chat_setup("hello")
    assert client.messages[-1]["content"] == "hello"


def test_chat_setup_rejects_images_and_audio_together():
    client = _AudioMockClient(["ok"])
    client.model.supports_vision = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        client._chat_setup("x", images=[b"\x89PNG"], audio=[_TINY_WAV_BYTES])


def test_is_audio_model_property():
    client = _AudioMockClient(["ok"])
    assert client.is_audio_model is True

    non_audio = MockModelClient(["ok"])
    non_audio.model.supports_audio = False
    assert non_audio.is_audio_model is False


# --------------------------------------------------------------------------- #
# Anthropic adapter                                                             #
# --------------------------------------------------------------------------- #


def test_openai_audio_blocks_to_anthropic():
    from aimu.models._internal.audio_input import _openai_audio_blocks_to_anthropic

    blocks = [
        {"type": "text", "text": "transcribe"},
        {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
    ]
    out = _openai_audio_blocks_to_anthropic(blocks)
    assert out[0] == {"type": "text", "text": "transcribe"}
    assert out[1]["type"] == "audio"
    assert out[1]["source"]["type"] == "base64"
    assert out[1]["source"]["media_type"] == "audio/wav"
    assert out[1]["source"]["data"] == _TINY_WAV_B64


def test_anthropic_client_message_conversion_with_audio():
    pytest.importorskip("anthropic")
    from aimu.models.providers.anthropic import AnthropicClient

    client = AnthropicClient.__new__(AnthropicClient)
    messages = [
        {"role": "system", "content": "be helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
            ],
        },
    ]
    system_str, ant = client._openai_messages_to_anthropic(messages)
    assert system_str == "be helpful"
    assert ant[0]["role"] == "user"
    assert ant[0]["content"][0]["type"] == "text"
    assert ant[0]["content"][1]["type"] == "audio"
    assert ant[0]["content"][1]["source"]["type"] == "base64"


# --------------------------------------------------------------------------- #
# Ollama: rejects input_audio blocks                                            #
# --------------------------------------------------------------------------- #


def test_adapt_messages_for_ollama_rejects_audio():
    from aimu.models._internal.audio_input import _adapt_messages_for_ollama

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="Ollama"):
        _adapt_messages_for_ollama(messages)


# --------------------------------------------------------------------------- #
# HuggingFace: extract audio arrays + placeholder replacement                  #
# --------------------------------------------------------------------------- #


def test_extract_audio_arrays_returns_numpy_arrays():
    soundfile = pytest.importorskip("soundfile")
    if getattr(soundfile, "_aimu_stub", False):
        pytest.skip("soundfile is stubbed; requires real soundfile package")
    from aimu.models._internal.audio_input import _extract_audio_arrays

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
            ],
        }
    ]
    arrays = _extract_audio_arrays(messages)
    assert len(arrays) == 1
    import numpy as np

    assert isinstance(arrays[0], np.ndarray)


def test_replace_audio_with_placeholder():
    from aimu.models._internal.audio_input import _replace_audio_with_placeholder

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
            ],
        }
    ]
    out = _replace_audio_with_placeholder(messages)
    assert out[0]["content"][0] == {"type": "text", "text": "transcribe"}
    assert out[0]["content"][1] == {"type": "audio"}
    # original not mutated
    assert messages[0]["content"][1]["type"] == "input_audio"


def test_replace_audio_with_placeholder_leaves_non_audio_messages():
    from aimu.models._internal.audio_input import _replace_audio_with_placeholder

    messages = [{"role": "user", "content": "plain text"}]
    out = _replace_audio_with_placeholder(messages)
    assert out == messages


# --------------------------------------------------------------------------- #
# ModelSpec audio flag                                                          #
# --------------------------------------------------------------------------- #


def test_modelspec_audio_flag_defaults_false():
    from aimu.models._base.text import ModelSpec

    spec = ModelSpec("test-model")
    assert spec.audio is False


def test_modelspec_audio_flag_settable():
    from aimu.models._base.text import ModelSpec

    spec = ModelSpec("test-model", audio=True)
    assert spec.audio is True


# --------------------------------------------------------------------------- #
# Live provider smoke tests (opt-in via --client flag)                         #
# --------------------------------------------------------------------------- #


def pytest_generate_tests(metafunc):
    if "audio_model_client" in metafunc.fixturenames:
        params = resolve_model_params(metafunc.config, default_params=[])
        metafunc.parametrize("audio_model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def audio_model_client(request):
    model = request.param
    if not getattr(model, "supports_audio", False):
        pytest.skip(f"Model {model.name} does not support audio input")
    yield from create_real_model_client(request)


@pytest.mark.live
def test_generate_with_audio(audio_model_client):
    response = audio_model_client.generate("Transcribe this audio clip.", audio=[_TINY_WAV_BYTES])
    assert isinstance(response, str)
    assert len(response.strip()) > 0


@pytest.mark.live
def test_chat_with_audio(audio_model_client):
    response = audio_model_client.chat("What sound is in this clip?", audio=[_TINY_WAV_BYTES])
    assert isinstance(response, str)
    assert len(response.strip()) > 0
