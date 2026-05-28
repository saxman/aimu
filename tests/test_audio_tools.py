"""Mock-only unit tests for the built-in ``generate_audio`` tool and ``make_audio_tool``.

Verifies the lazy singleton wiring, the tool spec shape, env-var override, and
the per-agent factory escape hatch. No real audio weights required.
"""

from __future__ import annotations

# Install audio stubs before importing aimu.tools.builtin's audio bits.
from test_audio_api import _install_audio_stubs  # noqa: F401 — side effect

_install_audio_stubs()

import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_AUDIO:
    aimu.models.hf_audio = importlib.import_module("aimu.models.hf_audio")
    aimu.models.HAS_HF_AUDIO = True
    aimu.models.HuggingFaceAudioClient = aimu.models.hf_audio.HuggingFaceAudioClient
    aimu.models.HuggingFaceAudioModel = aimu.models.hf_audio.HuggingFaceAudioModel

import aimu  # noqa: E402

aimu.HAS_HF_AUDIO = True
aimu.HuggingFaceAudioClient = aimu.models.HuggingFaceAudioClient
aimu.HuggingFaceAudioModel = aimu.models.HuggingFaceAudioModel

from aimu.tools import builtin  # noqa: E402


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_generate_audio_has_tool_spec():
    spec = builtin.generate_audio.__tool_spec__
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "generate_audio"
    assert "prompt" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["properties"]["prompt"]["type"] == "string"
    assert spec["function"]["parameters"]["required"] == ["prompt"]


def test_generate_audio_is_sync():
    """The sync built-in must be a regular def, not async."""
    assert builtin.generate_audio.__tool_is_async__ is False


def test_generate_audio_is_streaming():
    """The sync built-in is a generator (streaming) tool."""
    assert builtin.generate_audio.__tool_is_streaming__ is True


def test_generate_audio_in_audio_subgroup():
    assert builtin.generate_audio in builtin.audio
    assert builtin.generate_audio in builtin.ALL_TOOLS


# ---------------------------------------------------------------------------
# Singleton + env var
# ---------------------------------------------------------------------------


def test_lazy_singleton_constructed_once(monkeypatch):
    """_get_audio_client should cache a single audio client and reuse it."""
    monkeypatch.setattr(builtin, "_audio_client", None)
    monkeypatch.setenv("AIMU_AUDIO_MODEL", "hf:facebook/musicgen-small")

    constructed: list[str] = []

    def fake_audio_client(model_str):
        constructed.append(model_str)
        from aimu.models.hf_audio import HuggingFaceAudioClient

        return HuggingFaceAudioClient(model_str)

    monkeypatch.setattr("aimu.audio_client", fake_audio_client)

    c1 = builtin._get_audio_client()
    c2 = builtin._get_audio_client()
    assert c1 is c2
    assert constructed == ["hf:facebook/musicgen-small"]


def test_singleton_honours_env_var(monkeypatch):
    """AIMU_AUDIO_MODEL env var should determine the singleton's model."""
    monkeypatch.setattr(builtin, "_audio_client", None)
    monkeypatch.setenv("AIMU_AUDIO_MODEL", "hf:facebook/musicgen-small")

    c = builtin._get_audio_client()
    assert c.spec.id == "facebook/musicgen-small"


def test_tool_drains_generator_and_returns_path(monkeypatch, tmp_path):
    """Calling generate_audio() yields AUDIO_GENERATING chunks and returns the final path."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.hf_audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    final_path = str(tmp_path / "fake.wav")

    def fake_stream(prompt, format, stream):  # noqa: ARG001
        yield StreamChunk(
            StreamingContentType.AUDIO_GENERATING,
            {"step": 1, "total_steps": 1, "final": False, "result": None, "duration_s": 10.0},
        )
        yield StreamChunk(
            StreamingContentType.AUDIO_GENERATING,
            {"step": 1, "total_steps": 1, "final": True, "result": final_path, "duration_s": 10.0},
        )

    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    monkeypatch.setattr(builtin, "_audio_client", client)
    monkeypatch.setattr(client, "generate", fake_stream)

    gen = builtin.generate_audio("upbeat jazz")
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        result = stop.value

    assert len(chunks) == 2
    assert all(c.phase == StreamingContentType.AUDIO_GENERATING for c in chunks)
    assert result == final_path


# ---------------------------------------------------------------------------
# make_audio_tool — per-agent factory escape hatch
# ---------------------------------------------------------------------------


def test_make_audio_tool_returns_new_streaming_tool_bound_to_supplied_client():
    from aimu.models.hf_audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    bound_tool = builtin.make_audio_tool(client)

    assert bound_tool is not builtin.generate_audio
    assert bound_tool.__tool_spec__["function"]["name"] == "generate_audio"
    assert bound_tool.__tool_is_async__ is False
    assert bound_tool.__tool_is_streaming__ is True


def test_make_audio_tool_threads_duration_s_through(monkeypatch, tmp_path):
    """make_audio_tool(duration_s=N) should pass N to client.generate(...)."""
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.hf_audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    captured: dict = {}

    def fake_stream(prompt, format, stream, duration_s=None):  # noqa: ARG001
        captured["duration_s"] = duration_s
        yield StreamChunk(
            StreamingContentType.AUDIO_GENERATING,
            {"step": 1, "total_steps": 1, "final": True, "result": str(tmp_path / "x.wav"), "duration_s": duration_s},
        )

    custom = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    monkeypatch.setattr(custom, "generate", fake_stream)
    bound = builtin.make_audio_tool(custom, duration_s=15)

    list(bound("ambient music"))
    assert captured["duration_s"] == 15


def test_make_audio_tool_uses_its_client_not_singleton(monkeypatch, tmp_path):
    from aimu.models.base import StreamChunk, StreamingContentType
    from aimu.models.hf_audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    sentinel = "singleton-should-not-be-touched"
    monkeypatch.setattr(builtin, "_audio_client", sentinel)

    custom = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    bound_tool = builtin.make_audio_tool(custom)

    final_path = str(tmp_path / "custom.wav")

    def fake_stream(prompt, format, stream):  # noqa: ARG001
        yield StreamChunk(
            StreamingContentType.AUDIO_GENERATING,
            {"step": 1, "total_steps": 1, "final": True, "result": final_path, "duration_s": 10.0},
        )

    monkeypatch.setattr(custom, "generate", fake_stream)

    list(bound_tool("rain sounds"))
    assert builtin._audio_client is sentinel  # singleton was never constructed


# ---------------------------------------------------------------------------
# make_tools — audio_client parameter
# ---------------------------------------------------------------------------


def _fake_base_client(supports_vision=False):
    from unittest.mock import MagicMock

    client = MagicMock()
    client.model = MagicMock()
    client.model.supports_vision = supports_vision
    return client


def test_make_tools_with_audio_client_replaces_generate_audio():
    """When an audio client is supplied, the default generate_audio singleton is replaced."""
    from aimu.models.hf_audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    base_client = _fake_base_client()
    audio_client = HuggingFaceAudioClient(HuggingFaceAudioModel.MUSICGEN_SMALL)
    tools = builtin.make_tools(base_client, audio_client=audio_client)

    assert len(tools) == len(builtin.ALL_TOOLS)
    assert builtin.generate_audio not in tools
    names = [t.__tool_spec__["function"]["name"] for t in tools]
    assert "generate_audio" in names


def test_make_tools_without_audio_client_keeps_default():
    """Without an audio client, make_tools returns a copy of ALL_TOOLS unchanged."""
    base_client = _fake_base_client()
    tools = builtin.make_tools(base_client)
    assert builtin.generate_audio in tools
