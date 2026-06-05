"""Mock-only unit tests for the built-in ``generate_speech`` tool and ``make_speech_tool``.

Verifies the lazy singleton wiring, the tool spec shape, env-var override, and
the per-agent factory escape hatch. No real speech weights required.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_speech_api import _install_speech_stubs, _register_hf_speech, _register_openai_speech

_install_speech_stubs()
_register_hf_speech()
_register_openai_speech()

import aimu.models  # noqa: E402

# Patch speech clients into aimu namespace so speech_client() works
aimu.models.HAS_OPENAI_SPEECH = True
import aimu  # noqa: E402

aimu.HAS_OPENAI_SPEECH = True
aimu.OpenAISpeechClient = aimu.models.OpenAISpeechClient
aimu.OpenAISpeechModel = aimu.models.OpenAISpeechModel

from aimu.tools import builtin  # noqa: E402


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_generate_speech_has_tool_spec():
    spec = builtin.generate_speech.__tool_spec__
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "generate_speech"
    assert "text" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["required"] == ["text"]


def test_generate_speech_is_sync():
    assert builtin.generate_speech.__tool_is_async__ is False


def test_generate_speech_is_streaming():
    assert builtin.generate_speech.__tool_is_streaming__ is True


def test_generate_speech_in_speech_subgroup():
    assert builtin.generate_speech in builtin.speech
    assert builtin.generate_speech in builtin.ALL_TOOLS


# ---------------------------------------------------------------------------
# Singleton + env var
# ---------------------------------------------------------------------------


def test_lazy_singleton_constructed_once(monkeypatch):
    monkeypatch.setattr(builtin, "_speech_client", None)
    monkeypatch.setenv("AIMU_SPEECH_MODEL", "openai:tts-1")

    constructed: list[str] = []

    def fake_speech_client(model=None):
        import os

        constructed.append(model or os.environ.get("AIMU_SPEECH_MODEL"))
        from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

        return OpenAISpeechClient(OpenAISpeechModel.TTS_1)

    monkeypatch.setattr("aimu.speech_client", fake_speech_client)

    c1 = builtin._get_speech_client()
    c2 = builtin._get_speech_client()
    assert c1 is c2
    assert constructed == ["openai:tts-1"]


def test_singleton_honours_env_var(monkeypatch):
    monkeypatch.setattr(builtin, "_speech_client", None)
    monkeypatch.setenv("AIMU_SPEECH_MODEL", "openai:tts-1")

    c = builtin._get_speech_client()
    assert c.spec.id == "tts-1"


def test_singleton_raises_when_env_unset(monkeypatch):
    """With AIMU_SPEECH_MODEL unset, the tool raises and never downloads a default."""
    import pytest

    monkeypatch.setattr(builtin, "_speech_client", None)
    monkeypatch.delenv("AIMU_SPEECH_MODEL", raising=False)

    with pytest.raises(ValueError, match="AIMU_SPEECH_MODEL"):
        builtin._get_speech_client()
    assert builtin._speech_client is None


def test_tool_drains_generator_and_returns_path(monkeypatch, tmp_path):
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    final_path = str(tmp_path / "fake.wav")

    def fake_stream(text, **kwargs):
        yield StreamChunk(
            StreamingContentType.SPEECH_GENERATING,
            {"chunk_index": 1, "total_chunks": None, "final": False, "result": None},
        )
        yield StreamChunk(
            StreamingContentType.SPEECH_GENERATING,
            {"chunk_index": 2, "total_chunks": 2, "final": True, "result": final_path},
        )

    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    monkeypatch.setattr(builtin, "_speech_client", client)
    monkeypatch.setattr(client, "generate", fake_stream)

    gen = builtin.generate_speech("Hello")
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        result = stop.value

    assert len(chunks) == 2
    assert all(c.phase == StreamingContentType.SPEECH_GENERATING for c in chunks)
    assert chunks[0].content["final"] is False
    assert chunks[1].content["final"] is True
    assert chunks[1].content["result"] == final_path
    assert result == final_path


# ---------------------------------------------------------------------------
# make_speech_tool
# ---------------------------------------------------------------------------


def test_make_speech_tool_returns_new_streaming_tool():
    from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    bound_tool = builtin.make_speech_tool(client)

    assert bound_tool is not builtin.generate_speech
    assert bound_tool.__tool_spec__["function"]["name"] == "generate_speech"
    assert bound_tool.__tool_is_streaming__ is True


def test_make_speech_tool_uses_its_client_not_singleton(monkeypatch, tmp_path):
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    monkeypatch.setattr(builtin, "_speech_client", "singleton-sentinel")

    custom = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    bound_tool = builtin.make_speech_tool(custom)
    final_path = str(tmp_path / "custom.wav")

    def fake_stream(text, **kwargs):
        yield StreamChunk(
            StreamingContentType.SPEECH_GENERATING,
            {"chunk_index": 1, "total_chunks": 1, "final": True, "result": final_path},
        )

    monkeypatch.setattr(custom, "generate", fake_stream)
    list(bound_tool("Hello world"))
    assert builtin._speech_client == "singleton-sentinel"


def test_make_speech_tool_threads_voice_and_speed(monkeypatch, tmp_path):
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    received = {}
    final_path = str(tmp_path / "voice.wav")

    def fake_stream(text, **kwargs):
        received.update(kwargs)
        yield StreamChunk(
            StreamingContentType.SPEECH_GENERATING,
            {"chunk_index": 1, "total_chunks": 1, "final": True, "result": final_path},
        )

    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    monkeypatch.setattr(client, "generate", fake_stream)
    bound_tool = builtin.make_speech_tool(client, voice="nova", speed=1.25)
    list(bound_tool("Hello"))

    assert received["voice"] == "nova"
    assert received["speed"] == 1.25


# ---------------------------------------------------------------------------
# make_tools — speech_client parameter
# ---------------------------------------------------------------------------


def _fake_base_client(supports_vision=False):
    from unittest.mock import MagicMock

    client = MagicMock()
    client.model = MagicMock()
    client.model.supports_vision = supports_vision
    return client


def test_make_tools_with_speech_client_replaces_generate_speech():
    from aimu.models.providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    base_client = _fake_base_client()
    speech_client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    tools = builtin.make_tools(base_client, speech_client=speech_client)

    assert builtin.generate_speech not in tools
    names = [t.__tool_spec__["function"]["name"] for t in tools]
    assert "generate_speech" in names


def test_make_tools_without_speech_client_keeps_default():
    base_client = _fake_base_client()
    tools = builtin.make_tools(base_client)
    assert builtin.generate_speech in tools
