"""Each HuggingFace client declares what its loader downloads, and the declaration cannot drift.

``aimu.pull`` promises offline readiness: after it, the client builds and runs with no network. That
holds only if the declaration (``_hub_downloads``) names every repo the loader fetches, so the first
test runs each loader against recording stand-ins (no weights, no network) and checks the
declaration covers what it asked for. A loader that gains a ``from_pretrained`` call without a
matching declaration fails here.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytest.importorskip("transformers")
pytest.importorskip("diffusers")
pytest.importorskip("sentence_transformers")

import huggingface_hub  # noqa: E402
import sentence_transformers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import aimu.models.providers.hf._download as download  # noqa: E402
from aimu.models.providers.hf._download import HubDownload  # noqa: E402


class _Loaded:
    """What a recorded loader returns: absorbs any attribute access or placement call."""

    tokenizer = None

    def to(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return _Loaded()


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def loader_class(self):
        recorder = self

        class _Class:
            @classmethod
            def from_pretrained(cls, repo_id, *args, **kwargs):
                recorder.calls.append((repo_id, kwargs))
                return _Loaded()

        return _Class

    def pipeline(self, task, model=None, **kwargs):
        self.calls.append((model, kwargs))
        return _Loaded()

    def sentence_transformer(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return _Loaded()

    def load_dataset(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return {7306: {"xvector": [0.0] * 512}, 0: {"xvector": [0.0] * 512}}


# (module, client class). Looked up when each test runs, not at import: other test files reload these
# provider modules, and a class captured at collection would no longer match its own enum.
_CLIENTS = [
    ("text", "HuggingFaceClient"),
    ("image", "HuggingFaceImageClient"),
    ("audio", "HuggingFaceAudioClient"),
    ("speech", "HuggingFaceSpeechClient"),
    ("transcription", "HuggingFaceTranscriptionClient"),
    ("embedding", "HuggingFaceEmbeddingClient"),
]


def _module(name: str):
    return importlib.import_module(f"aimu.models.providers.hf.{name}")


def _client_class(module_name: str, class_name: str):
    return getattr(_module(module_name), class_name)


@pytest.fixture
def recorder(monkeypatch):
    text, image = _module("text"), _module("image")
    rec = _Recorder()
    cls = rec.loader_class()
    for name in (
        "AutoTokenizer",
        "AutoProcessor",
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "Mistral3ForConditionalGeneration",
        "MusicgenForConditionalGeneration",
        "SpeechT5ForTextToSpeech",
        "SpeechT5HifiGan",
        "SpeechT5Processor",
        "BarkModel",
    ):
        monkeypatch.setattr(transformers, name, cls)
        if hasattr(text, name):
            monkeypatch.setattr(text, name, cls)
    monkeypatch.setattr(text, "BitsAndBytesConfig", lambda **kw: None)
    monkeypatch.setattr(transformers, "pipeline", rec.pipeline)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", rec.sentence_transformer)
    # `datasets` is not in any extra (SpeechT5 needs it installed separately), so it is stubbed whole.
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=rec.load_dataset))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.__getattr__ = lambda name: cls  # every pipeline class name resolves to the recorder
    monkeypatch.setattr(image, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    for module in (_module(name) for name, _ in _CLIENTS):
        monkeypatch.setattr(module, "_model_registry", {})
        for helper in ("auto_place_pipeline", "move_to_device"):
            if hasattr(module, helper):
                monkeypatch.setattr(module, helper, lambda obj, *a, **k: obj)
        if hasattr(module, "resolve_device"):
            monkeypatch.setattr(module, "resolve_device", lambda hint: "cpu")
    return rec


def _run_loader(client_cls, member):
    client = client_cls(member)  # the text client loads in its constructor
    if client_cls.__name__ == "HuggingFaceClient":
        return
    for name in ("_ensure_loaded", "_load_pipeline", "_load_model"):
        if hasattr(client, name):
            getattr(client, name)()
            return
    raise AssertionError(f"no loader found on {client_cls.__name__}")


_CASES = [
    pytest.param(module_name, class_name, member.name, id=f"{class_name}-{member.name}")
    for module_name, class_name in _CLIENTS
    for member in _client_class(module_name, class_name).MODELS
]


@pytest.mark.parametrize(("module_name", "class_name", "member_name"), _CASES)
def test_declaration_covers_every_repo_the_loader_requests(recorder, module_name, class_name, member_name):
    client_cls = _client_class(module_name, class_name)
    member = client_cls.MODELS[member_name]
    _run_loader(client_cls, member)
    declared = {d.repo_id: d for d in client_cls._hub_downloads(member.spec)}
    requested = {repo_id for repo_id, _ in recorder.calls}
    assert requested, "the loader requested nothing; the recording stand-ins missed a code path"
    assert requested <= set(declared), f"undeclared: {requested - set(declared)}"
    for repo_id, kwargs in recorder.calls:
        if "variant" in kwargs:
            assert declared[repo_id].options.get("variant") == kwargs["variant"]


def test_speecht5_declares_its_vocoder_and_dataset():
    speech = _module("speech")
    declared = speech.HuggingFaceSpeechClient._hub_downloads(speech.HuggingFaceSpeechModel.SPEECHT5.spec)
    assert HubDownload("microsoft/speecht5_hifigan") in declared
    assert HubDownload("Matthijs/cmu-arctic-xvectors", "dataset", {"split": "validation"}) in declared


def _siblings(*names):
    return types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=n) for n in names])


def test_weight_patterns_allows_exactly_the_selected_files(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda self, repo_id: _siblings("config.json", "model.safetensors", "pytorch_model.bin", "onnx/model.onnx"),
    )
    assert download.weight_patterns("org/both") == (["config.json", "model.safetensors"], None)

    monkeypatch.setattr(
        huggingface_hub.HfApi, "model_info", lambda self, repo_id: _siblings("pytorch_model.bin", "config.json")
    )
    assert download.weight_patterns("org/bin-only") == (["pytorch_model.bin", "config.json"], None)


# Real Hub listings (trimmed of nothing that matters to weight selection), captured 2026-10-30. Each repo
# carries a second copy of its weights that transformers never reads; a snapshot of the whole repo would
# download 2-5x what the loader needs.
_GPT_OSS = [
    "config.json",
    "metal/model.bin",
    "model-00000-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "original/config.json",
    "original/model.safetensors",
    "tokenizer.json",
]
_WHISPER_LARGE_V3 = [
    "config.json",
    "flax_model.msgpack",
    "model.fp32-00001-of-00002.safetensors",
    "model.fp32-00002-of-00002.safetensors",
    "model.safetensors",
    "model.safetensors.index.fp32.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.fp32.json",
    "pytorch_model.fp32-00001-of-00002.bin",
    "tokenizer.json",
]
_DISTIL_WHISPER = ["config.json", "model.fp32.safetensors", "model.safetensors", "onnx/encoder_model.onnx"]
_MUSICGEN_LARGE = [
    "compression_state_dict.bin",
    "config.json",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
    "pytorch_model.bin.index.json",
    "spiece.model",
    "state_dict.bin",
]
_BARK = [
    "coarse.pt",
    "coarse_2.pt",
    "config.json",
    "fine_2.pt",
    "pytorch_model.bin",
    "speaker_embeddings/v2/en_speaker_0_coarse_prompt.npy",
    "speaker_embeddings_path.json",
    "text_2.pt",
]
_EMBEDDING_GEMMA_2 = [
    "1_Pooling/config.json",
    "2_Normalize/config.json",
    "config.json",
    "model.safetensors",
    "modules.json",
]


def _weights(selected):
    return sorted(f for f in selected if f.rsplit(".", 1)[-1] in {"safetensors", "bin", "pt", "msgpack", "onnx"})


@pytest.mark.parametrize(
    ("files", "expected_weights"),
    [
        pytest.param(_GPT_OSS, [f"model-0000{i}-of-00002.safetensors" for i in range(3)], id="gpt-oss-20b"),
        pytest.param(_WHISPER_LARGE_V3, ["model.safetensors"], id="whisper-large-v3"),
        pytest.param(_DISTIL_WHISPER, ["model.safetensors"], id="distil-large-v3"),
        pytest.param(
            _MUSICGEN_LARGE,
            ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"],
            id="musicgen-large",
        ),
        pytest.param(_BARK, ["pytorch_model.bin"], id="bark"),
        pytest.param(_EMBEDDING_GEMMA_2, ["model.safetensors"], id="embeddinggemma-2"),
    ],
)
def test_select_files_keeps_only_the_weights_transformers_reads(files, expected_weights):
    selected = download.select_files(files)
    assert _weights(selected) == expected_weights
    # Everything that is not a weight file stays: configs, tokenizers, sentence-transformers modules,
    # Bark's speaker embeddings.
    non_weights = [f for f in files if f not in _weights(files) and not f.startswith(("original/", "metal/", "onnx/"))]
    assert set(non_weights) <= set(selected)


def test_select_files_keeps_every_weight_in_an_unrecognized_layout():
    # A directory with weights under no standard name is kept whole: over-fetching beats a pull that
    # leaves the loader something to download.
    assert download.select_files(["config.json", "weights/encoder.safetensors"]) == [
        "config.json",
        "weights/encoder.safetensors",
    ]


def test_fetch_dispatches_by_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(download, "weight_patterns", lambda repo_id: (None, ["*.onnx"]))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kw: calls.append(("snapshot", repo_id, kw))
    )
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.DiffusionPipeline = types.SimpleNamespace(
        download=lambda repo_id, **kw: calls.append(("diffusers", repo_id, kw))
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda repo_id, **kw: calls.append(("dataset", repo_id, kw))),
    )

    download.fetch(HubDownload("org/model"))
    download.fetch(HubDownload("org/pipe", "diffusers", {"variant": "fp16", "use_safetensors": True}))
    download.fetch(HubDownload("org/data", "dataset", {"split": "validation"}))

    assert calls == [
        ("snapshot", "org/model", {"allow_patterns": None, "ignore_patterns": ["*.onnx"]}),
        ("diffusers", "org/pipe", {"variant": "fp16", "use_safetensors": True}),
        ("dataset", "org/data", {"split": "validation"}),
    ]


def test_fetch_dataset_without_datasets_installed_names_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="pip install datasets"):
        download.fetch(HubDownload("org/data", "dataset", {"split": "validation"}))
