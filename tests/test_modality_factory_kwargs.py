"""The five modality factories forward keyword arguments to the client unchanged.

They used to bundle every keyword they did not recognize into the client's ``model_kwargs``.
That made ``model_kwargs=`` mean the loader's dict on ``aimu.client()`` but one level deeper on
``aimu.image_client()`` and the rest, so the documented
``image_client(m, model_kwargs={"device": "cuda:1"})`` arrived as
``{"model_kwargs": {"device": ...}}`` and the device hint never reached the loader, while a
misspelled keyword was handed to a loader that might ignore it.

Every HuggingFace client here loads weights lazily, so constructing one touches no weights.
"""

from __future__ import annotations

import pytest

import aimu
import aimu.models as models

FACTORIES = [
    ("image_client", "HuggingFaceImageModel", "SD_1_5"),
    ("audio_client", "HuggingFaceAudioModel", "MUSICGEN_SMALL"),
    ("speech_client", "HuggingFaceSpeechModel", "MMS_TTS_ENG"),
    ("transcription_client", "HuggingFaceTranscriptionModel", "WHISPER_TINY"),
    ("embedding_client", "HuggingFaceEmbeddingModel", "ALL_MINILM_L6_V2"),
]


def _model(enum_name, member):
    enum_cls = getattr(models, enum_name)
    if enum_cls is None:
        pytest.skip(f"{enum_name} needs an optional dependency that is not installed")
    return enum_cls[member]


@pytest.mark.parametrize("factory, enum_name, member", FACTORIES)
def test_model_kwargs_reaches_the_client_unchanged(factory, enum_name, member):
    client = getattr(aimu, factory)(_model(enum_name, member), model_kwargs={"device": "cpu"})
    assert client.model_kwargs == {"device": "cpu"}


@pytest.mark.parametrize("factory, enum_name, member", FACTORIES)
def test_a_keyword_the_client_does_not_declare_raises_naming_it(factory, enum_name, member):
    with pytest.raises(TypeError, match="device"):
        getattr(aimu, factory)(_model(enum_name, member), device="cpu")
