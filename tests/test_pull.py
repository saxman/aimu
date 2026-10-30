"""``aimu.pull()``: what it resolves, what it downloads, and what it refuses.

No network: the Ollama pull helper and the HuggingFace ``fetch`` are replaced with recorders, so these
check dispatch and resolution only. What the helpers themselves do is pinned in
``tests/test_ollama_pull.py`` and ``tests/test_hf_downloads.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ollama")

import aimu  # noqa: E402
import aimu.models.model_pull as pull_module  # noqa: E402
from aimu.models import OllamaModel  # noqa: E402


@pytest.fixture
def ollama_pulls(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(pull_module, "_ollama_sdk_client", lambda host: {"host": host})
    monkeypatch.setattr(pull_module, "pull_ollama_model", lambda sdk, model_id: calls.append((model_id, sdk)))
    return calls


@pytest.fixture
def hf_fetches(monkeypatch):
    calls = []
    monkeypatch.setattr(pull_module, "fetch", calls.append)
    return calls


def test_pull_ollama_text_and_embedding(ollama_pulls):
    aimu.pull(OllamaModel.QWEN_3_8B, host="http://gpu-box:11434")
    aimu.pull("ollama:qwen3:8b")
    aimu.pull("ollama:nomic-embed-text")
    assert [model_id for model_id, _ in ollama_pulls] == ["qwen3:8b", "qwen3:8b", "nomic-embed-text"]
    assert ollama_pulls[0][1] == {"host": "http://gpu-box:11434"}
    assert ollama_pulls[1][1] == {"host": None}


def test_pull_hf_runs_every_declared_download(hf_fetches):
    pytest.importorskip("transformers")
    from aimu.models.providers.hf.speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    aimu.pull(HuggingFaceSpeechModel.SPEECHT5)
    aimu.pull("hf:microsoft/speecht5_tts")
    declared = HuggingFaceSpeechClient._hub_downloads(HuggingFaceSpeechModel.SPEECHT5.spec)
    assert len(declared) == 3
    assert hf_fetches == declared + declared


def test_pull_refuses_bare_name():
    with pytest.raises(ValueError, match='"provider:model_id"'):
        aimu.pull("QWEN_3_8B")


@pytest.mark.parametrize("model", ["ollama:qwen3:8b@gpu-box:11434", "ollama:qwen3:8b;tools"])
def test_pull_refuses_endpoint_and_flags_suffix(model):
    with pytest.raises(ValueError, match="host="):
        aimu.pull(model)


def test_pull_refuses_host_for_hf():
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="host="):
        aimu.pull("hf:microsoft/speecht5_tts", host="http://gpu-box:11434")


def test_pull_refuses_ollama_openai():
    with pytest.raises(ValueError, match='aimu.pull\\("ollama:qwen3:8b"\\)'):
        aimu.pull("ollama-openai:qwen3:8b")


@pytest.mark.parametrize(
    ("model", "mechanism"),
    [
        ("anthropic:claude-sonnet-4-6", "nothing to download"),
        ("openai:gpt-4o", "nothing to download"),
        ("llamacpp:qwen3-8b.gguf", "model_path="),
        ("lmstudio:qwen/qwen3-8b", "lms get"),
        ("vllm:Qwen/Qwen3-8B", "launch"),
    ],
)
def test_pull_refuses_other_providers(model, mechanism):
    with pytest.raises(ValueError, match=mechanism):
        aimu.pull(model)


def test_pull_refuses_cloud_enum_member():
    gemini_image = pytest.importorskip("aimu.models.providers.gemini.image")
    with pytest.raises(ValueError, match="nothing to download"):
        aimu.pull(next(iter(gemini_image.GeminiImageModel)))


def test_pull_unknown_id_lists_what_was_tried():
    pytest.importorskip("transformers")
    with pytest.raises(ValueError) as info:
        aimu.pull("hf:not/a-model")
    message = str(info.value)
    for kind in ("text", "image", "audio", "speech", "transcription", "embedding"):
        assert kind in message


def test_pull_missing_extra_raises_import_error(monkeypatch):
    monkeypatch.setattr(pull_module, "_installed", lambda provider: False)
    with pytest.raises(ImportError, match=r"aimu\[ollama\]"):
        aimu.pull("ollama:qwen3:8b")
