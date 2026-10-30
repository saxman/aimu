"""``aimu.pull()``: download a model ahead of use, outside client construction.

AIMU does not download Ollama models implicitly (pass ``pull=True`` to a client, or call this). What
"pull" means depends on the provider:

- **Ollama** (``OllamaModel``, ``OllamaEmbeddingModel``): Ollama's own pull on the server at
  ``host`` (else ``OLLAMA_HOST``, else ``127.0.0.1:11434``). Only missing or changed layers
  download; a WARNING is logged when bytes actually start moving.
- **HuggingFace** (every HF modality): everything the client's loader fetches, as the client
  declares it, so afterwards the client builds and runs with no network access.
- **Everything else** is refused with that provider's own mechanism: a cloud model has nothing to
  download, llama.cpp loads a GGUF you supply, and each local OpenAI-compatible server has its own
  loader.

Takes an enum member or a ``"provider:model_id"`` string. A bare enum-member name is refused: the
same name exists under several providers, and the usual tiebreak (prefer the provider where the
model is already available) is meaningless for a model you are about to download.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from .audio_client import resolve_audio_model_string
from .embedding_client import resolve_embedding_model_string
from .image_client import resolve_image_model_string
from .model_client import resolve_model_string
from .speech_client import resolve_speech_model_string
from .transcription_client import resolve_transcription_model_string

PullBackend = Literal["ollama", "hf"]

# Resolvers tried for each pullable prefix, in order, until one recognizes the id.
_RESOLVERS = {
    "ollama": [("text", resolve_model_string), ("embedding", resolve_embedding_model_string)],
    "hf": [
        ("text", resolve_model_string),
        ("image", resolve_image_model_string),
        ("audio", resolve_audio_model_string),
        ("speech", resolve_speech_model_string),
        ("transcription", resolve_transcription_model_string),
        ("embedding", resolve_embedding_model_string),
    ],
}

_NOTHING_TO_DOWNLOAD = "it is a cloud model, so there is nothing to download"
_SERVER_LOADER = "load the model through that server's own launch arguments or model manager"

# Why each non-pullable text provider is refused, and the mechanism that does the job there.
_REFUSALS = {
    "anthropic": _NOTHING_TO_DOWNLOAD,
    "openai": _NOTHING_TO_DOWNLOAD,
    "gemini": _NOTHING_TO_DOWNLOAD,
    "llamacpp": "llama.cpp loads a GGUF file you supply; download it yourself and pass model_path=",
    "lmstudio": "LM Studio manages its own models; download it in the app or with `lms get`",
    "vllm": f"vLLM serves the model named in its launch arguments; {_SERVER_LOADER}",
    "sglang": f"SGLang serves the model named in its launch arguments; {_SERVER_LOADER}",
    "llamaserver": f"llama-server serves the GGUF named in its launch arguments; {_SERVER_LOADER}",
    "hf-openai": f"HF Transformers Serve loads models itself; {_SERVER_LOADER}",
    "omlx": "oMLX serves the directories under its --model-dir; place the model there",
}


def pull(model: Union[str, Enum], *, host: Optional[str] = None) -> None:
    """Download ``model`` so a client for it can be built without downloading anything.

    ``host`` targets a remote Ollama server and is refused for any other provider. See the module
    docstring for what is downloaded per provider and what is refused.
    """
    backend, member = _resolve_pull_target(model, host)
    if backend == "ollama":
        pull_ollama_model(_ollama_sdk_client(host), member.value)
    else:
        for download in _hf_client_class(member)._hub_downloads(member.spec):
            fetch(download)


def _resolve_pull_target(model: Union[str, Enum], host: Optional[str]) -> tuple[PullBackend, Any]:
    """Map ``model`` to the backend that pulls it and its enum member, or raise naming why not."""
    if isinstance(model, str):
        backend, member = _resolve_string(model)
    else:
        backend, member = _classify_member(model), model
    if backend == "hf" and host is not None:
        raise ValueError(
            f"host= targets a remote Ollama server; {member!r} is a HuggingFace model, which downloads "
            "into the local HuggingFace cache. Omit host=."
        )
    return backend, member


def _resolve_string(model: str) -> tuple[PullBackend, Any]:
    if ":" not in model:
        raise ValueError(
            f'aimu.pull() takes an enum member or a "provider:model_id" string, got {model!r}. A bare name '
            'exists under several providers; name the one to download from, e.g. "ollama:qwen3:8b".'
        )
    provider, model_id = model.split(":", 1)
    if "@" in model_id or ";" in model_id:
        raise ValueError(
            f"aimu.pull() does not take an @endpoint or ;flags suffix ({model!r}). Pass the server as "
            "host=, and the bare catalogued id."
        )
    if provider == "ollama-openai":
        raise ValueError(
            f"'ollama-openai' reaches the same Ollama server through its OpenAI-compatible endpoint; pull "
            f'with the native provider instead: aimu.pull("ollama:{model_id}").'
        )
    if provider in _REFUSALS:
        raise ValueError(f"aimu.pull() cannot download {model!r}: {_REFUSALS[provider]}.")
    if provider not in _RESOLVERS:
        raise ValueError(f"aimu.pull() knows no provider {provider!r}; it downloads 'ollama:' and 'hf:' models.")
    if not _installed(provider):
        extra = "ollama" if provider == "ollama" else "hf"
        raise ImportError(f"Pulling {model!r} needs the {provider} dependencies: pip install 'aimu[{extra}]'.")
    tried = []
    for kind, resolver in _RESOLVERS[provider]:
        try:
            return provider, resolver(model)
        except ValueError:
            tried.append(kind)
    raise ValueError(
        f"{model!r} is not a catalogued model: it matched none of the {provider!r} {', '.join(tried)} "
        "catalogs. Check the id against that provider's Model enum."
    )


def _classify_member(member: Enum) -> PullBackend:
    from aimu import models

    def matches(*names: str) -> bool:
        return any(
            getattr(models, name, None) is not None and isinstance(member, getattr(models, name)) for name in names
        )

    if matches("OllamaModel", "OllamaEmbeddingModel"):
        return "ollama"
    if matches(*_HF_CLIENTS.values()):
        return "hf"
    if isinstance(member, models.Model):
        from ._internal.model_defaults import _provider_prefix

        provider = _provider_prefix(member)
        if provider == "ollama-openai":
            raise ValueError(f'Pull {member!r} with the native provider instead: aimu.pull("ollama:{member.value}").')
        raise ValueError(f"aimu.pull() cannot download {member!r}: {_REFUSALS.get(provider, _NOTHING_TO_DOWNLOAD)}.")
    raise ValueError(f"aimu.pull() cannot download {member!r}: {_NOTHING_TO_DOWNLOAD}.")


# HuggingFace client class name -> the enum it serves.
_HF_CLIENTS = {
    "HuggingFaceClient": "HuggingFaceModel",
    "HuggingFaceImageClient": "HuggingFaceImageModel",
    "HuggingFaceAudioClient": "HuggingFaceAudioModel",
    "HuggingFaceSpeechClient": "HuggingFaceSpeechModel",
    "HuggingFaceTranscriptionClient": "HuggingFaceTranscriptionModel",
    "HuggingFaceEmbeddingClient": "HuggingFaceEmbeddingModel",
}


def _hf_client_class(member: Enum) -> Any:
    from aimu import models

    for client_name, enum_name in _HF_CLIENTS.items():
        enum_cls = getattr(models, enum_name, None)
        if enum_cls is not None and isinstance(member, enum_cls):
            return getattr(models, client_name)
    raise ValueError(f"No HuggingFace client serves {member!r}.")


def _installed(provider: str) -> bool:
    from aimu import models

    if provider == "ollama":
        return models.HAS_OLLAMA
    return any((models.HAS_HF, models.HAS_HF_IMAGE, models.HAS_HF_AUDIO, models.HAS_HF_SPEECH, models.HAS_HF_EMBEDDING))


# The provider helpers are imported on call, not at module load: importing aimu.models must not import
# a provider SDK. Module-level names so tests can replace them.


def _ollama_sdk_client(host: Optional[str]) -> Any:
    import ollama

    from .providers.ollama import _ollama_client_kwargs

    return ollama.Client(**_ollama_client_kwargs(host, None))


def pull_ollama_model(sdk_client: Any, model_id: str) -> None:
    from .providers.ollama import pull_ollama_model as pull_with_logging

    pull_with_logging(sdk_client, model_id)


def fetch(download: Any) -> None:
    from .providers.hf._download import fetch as fetch_from_hub

    fetch_from_hub(download)
