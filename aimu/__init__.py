"""
aimu - AI Modeling Utilities

Lightweight Python library for building LLM-powered apps. Provider-agnostic model
clients (Ollama, HuggingFace, Anthropic, OpenAI, Gemini, llama-cpp, any OpenAI-compatible
server), text-to-image clients (HuggingFace ``diffusers`` + Google Nano Banana),
text-to-audio clients (HuggingFace MusicGen, AudioLDM2, Stable Audio Open), in-process
tools via the ``@tool`` decorator, and code-controlled workflows (Chain, Router, Parallel,
EvaluatorOptimizer) plus autonomous Agents.

Quick start::

    import aimu

    text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

    client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
    client.chat("Hi there")

    agent = aimu.agent("anthropic:claude-sonnet-4-6", tools=[my_tool])
    print(agent.run("How many r's in strawberry?"))

    image = aimu.generate_image("a watercolor of a fox", model="hf:runwayml/stable-diffusion-v1-5")
    image = aimu.generate_image("a watercolor of a fox", model="gemini:nano-banana")

    path = aimu.generate_audio("upbeat lo-fi jazz loop", model="hf:facebook/musicgen-small")
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Union

try:
    __version__ = _pkg_version("aimu")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout without an install)
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from .agents import Agent

from . import aio
from .models import (
    HAS_GEMINI_IMAGE,
    extract_tool_calls,
    generate_json,
    parse_json_response,
    HAS_HF_AUDIO,
    HAS_HF_EMBEDDING,
    HAS_HF_IMAGE,
    HAS_HF_SPEECH,
    HAS_OLLAMA_EMBEDDING,
    HAS_OPENAI_EMBEDDING,
    HAS_OPENAI_SPEECH,
    AudioClient,
    AudioModel,
    AudioSpec,
    BaseAudioClient,
    BaseEmbeddingClient,
    BaseImageClient,
    BaseModelClient,
    BaseSpeechClient,
    EmbeddingClient,
    EmbeddingModel,
    EmbeddingSpec,
    GeminiImageClient,
    GeminiImageModel,
    GeminiImageSpec,
    HuggingFaceAudioClient,
    HuggingFaceAudioModel,
    HuggingFaceAudioSpec,
    HuggingFaceImageClient,
    HuggingFaceImageModel,
    HuggingFaceImageSpec,
    HuggingFaceSpeechClient,
    HuggingFaceSpeechModel,
    HuggingFaceSpeechSpec,
    ImageClient,
    ImageModel,
    ImageSpec,
    Model,
    ModelClient,
    ModelSpec,
    OpenAISpeechClient,
    OpenAISpeechModel,
    OpenAISpeechSpec,
    SpeechClient,
    SpeechModel,
    SpeechSpec,
    StreamChunk,
    StreamingContentType,
    TranscriptionClient,
    TranscriptionModel,
    TranscriptionSpec,
    available_audio_models,
    available_embedding_models,
    available_image_models,
    available_speech_clients,
    available_speech_models,
    available_text_models,
    available_transcription_models,
    resolve_audio_model_string,
    resolve_embedding_model_string,
    resolve_default_text_model_enum,
    resolve_image_model_enum,
    resolve_image_model_string,
    resolve_model_enum,
    resolve_model_string,
    resolve_speech_model_string,
    resolve_transcription_model_string,
)
from .tools import ToolContext, tool
from .display import pretty_print


def client(model: Union[str, Model, None] = None, *, system: Optional[str] = None, **kwargs: Any) -> ModelClient:
    """Construct a :class:`ModelClient` from a model string or enum member.

    ``model`` may be a ``"provider:model_id"`` string (``"anthropic:claude-sonnet-4-6"``,
    ``"ollama:qwen3.5:9b"``) or any provider's ``Model`` enum member. Extra ``**kwargs``
    are forwarded to the underlying provider client (e.g. ``model_path=`` for llama-cpp).

    When ``model`` is omitted, a default is resolved: the ``AIMU_LANGUAGE_MODEL`` env var
    if set, otherwise an already-available local model (a running Ollama server, a cached
    HuggingFace model, or a local OpenAI-compatible server). A cloud provider is never
    auto-selected and weights are never downloaded implicitly; if nothing resolves a
    ``ValueError`` is raised naming the remedies.

    Use this as the one-line construction helper. For full control over the provider
    client constructor, use :class:`ModelClient` directly.
    """
    if model is None:
        from .models._internal.model_defaults import resolve_default_text_model

        model = resolve_default_text_model()
    if system is not None:
        kwargs["system_message"] = system
    return ModelClient(model, **kwargs)


def chat(
    user_message: str,
    *,
    model: Union[str, Model, None] = None,
    system: Optional[str] = None,
    generate_kwargs: Optional[dict] = None,
    stream: bool = False,
    images: Optional[list] = None,
    include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
) -> Union[str, Iterator[StreamChunk]]:
    """One-shot chat: builds a fresh client, sends one message, returns the response.

    For multi-turn conversations construct a :class:`ModelClient` with :func:`client` and
    call its ``chat()`` repeatedly.

    ``model`` may be omitted to use the default resolved by :func:`client` (the
    ``AIMU_LANGUAGE_MODEL`` env var or an already-available local model).

    Example::

        text = aimu.chat("Summarize this", model="anthropic:claude-sonnet-4-6")
        text = aimu.chat("Hello")  # uses AIMU_LANGUAGE_MODEL or a local model

        for chunk in aimu.chat("Tell me a story", model="ollama:qwen3.5:9b", stream=True):
            if chunk.is_text():
                print(chunk.content, end="")
    """
    c = client(model, system=system)
    return c.chat(
        user_message,
        generate_kwargs=generate_kwargs,
        stream=stream,
        images=images,
        include=include,
    )


def agent(
    model: Union[str, Model, None] = None,
    *,
    system: Optional[str] = None,
    tools: Optional[list] = None,
    **kwargs: Any,
) -> "Agent":
    """Construct an :class:`~aimu.agents.Agent` from a model string or enum member.

    Shortcut for the common case of ``Agent(aimu.client(model), ...)``. For full
    control (``mcp_client``, ``max_iterations``, ``name``, etc.) construct
    :class:`~aimu.agents.Agent` directly.

    ``model`` may be omitted to use the default resolved by :func:`client`.

    ``**kwargs`` are forwarded to :class:`~aimu.agents.Agent` (e.g.
    ``max_iterations=5``, ``name="my-agent"``).

    Example::

        import aimu

        @aimu.tool
        def letter_counter(word: str, letter: str) -> int:
            \"\"\"Count occurrences of a letter in a word.\"\"\"
            return word.lower().count(letter.lower())

        agent = aimu.agent("ollama:qwen3.5:9b", tools=[letter_counter])
        print(agent.run("How many r's in strawberry?"))
    """
    from .agents import Agent

    return Agent(client(model), system, tools=tools or [], **kwargs)


def audio_client(model: Union[str, AudioModel, AudioSpec, None] = None, **kwargs: Any) -> AudioClient:
    """Construct an :class:`AudioClient` for text-to-audio (music/sound) generation.

    ``model`` may be a :class:`HuggingFaceAudioModel` member, a :class:`HuggingFaceAudioSpec`,
    or a ``"hf:repo_id"`` string. Extra ``**kwargs`` are forwarded as ``model_kwargs`` to
    the underlying provider client.

    When ``model`` is omitted, the ``AIMU_AUDIO_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).

    Example::

        client = aimu.audio_client(aimu.HuggingFaceAudioModel.MUSICGEN_SMALL)
        client = aimu.audio_client("hf:facebook/musicgen-medium")
    """
    if not HAS_HF_AUDIO:
        raise ImportError(
            "Audio generation requires the [hf] extra (soundfile, torch, transformers, diffusers): "
            "pip install -e '.[hf]'"
        )
    if model is None:
        from .models._internal.model_defaults import AUDIO_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(AUDIO_MODEL_ENV)
    return AudioClient(model, model_kwargs=kwargs or None)


def generate_audio(
    prompt: str,
    *,
    model: Union[str, AudioModel, AudioSpec, None] = None,
    format: str = "path",
    **kwargs: Any,
) -> Any:
    """One-shot audio generation: builds a fresh audio client and returns one audio clip.

    For multiple generations, construct a client with :func:`audio_client` and reuse it
    so weights are not reloaded per call.

    Example::

        path = aimu.generate_audio(
            "upbeat lo-fi jazz loop",
            model="hf:facebook/musicgen-small",
            duration_s=10,
        )

        sr, audio = aimu.generate_audio(
            "ambient forest soundscape",
            model=aimu.HuggingFaceAudioModel.AUDIOLDM2,
            format="numpy",
        )
    """
    if not HAS_HF_AUDIO:
        raise ImportError(
            "Audio generation requires the [hf] extra (soundfile, torch, transformers, diffusers): "
            "pip install -e '.[hf]'"
        )
    c = audio_client(model)
    return c.generate(prompt, format=format, **kwargs)


def speech_client(model: Union[str, SpeechModel, SpeechSpec, None] = None, **kwargs: Any) -> SpeechClient:
    """Construct a :class:`SpeechClient` for text-to-speech generation.

    ``model`` may be a :class:`HuggingFaceSpeechModel` / :class:`OpenAISpeechModel`
    member, a :class:`SpeechSpec`, or a ``"provider:model_id"`` string
    (``"hf:..."`` for HuggingFace; ``"openai:..."`` for OpenAI TTS).
    Extra ``**kwargs`` are forwarded as ``model_kwargs`` to the underlying client.

    When ``model`` is omitted, the ``AIMU_SPEECH_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).

    Example::

        client = aimu.speech_client("openai:tts-1")
        client = aimu.speech_client(aimu.HuggingFaceSpeechModel.MMS_TTS_ENG)
    """
    if model is None:
        from .models._internal.model_defaults import SPEECH_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(SPEECH_MODEL_ENV)
    return SpeechClient(model, model_kwargs=kwargs or None)


def generate_speech(
    text: str,
    *,
    model: Union[str, SpeechModel, SpeechSpec, None] = None,
    format: str = "path",
    **kwargs: Any,
) -> Any:
    """One-shot speech synthesis: builds a fresh client and returns one audio clip.

    For repeated synthesis, construct a client with :func:`speech_client` and
    reuse it so API clients aren't rebuilt per call.

    Example::

        path = aimu.generate_speech("Hello, world!", model="openai:tts-1")
        sr, audio = aimu.generate_speech("Hello", model="hf:facebook/mms-tts-eng", format="numpy")
    """
    c = speech_client(model)
    return c.generate(text, format=format, **kwargs)


def transcription_client(
    model: Union[str, "TranscriptionModel", "TranscriptionSpec", None] = None,
    **kwargs: Any,
) -> "TranscriptionClient":
    """Construct a :class:`TranscriptionClient` for speech-to-text (ASR) transcription.

    ``model`` accepts a ``TranscriptionModel`` enum member, a ``"provider:model_id"``
    string (e.g. ``"openai:whisper-1"``, ``"hf:openai/whisper-large-v3"``), or a
    :class:`TranscriptionSpec`.

    When ``model`` is omitted, the ``AIMU_TRANSCRIPTION_MODEL`` env var is used; if
    it is unset a ``ValueError`` is raised (no model is downloaded implicitly).
    """
    if model is None:
        from .models._internal.model_defaults import (
            TRANSCRIPTION_MODEL_ENV,
            resolve_default_modality_model,
        )

        model = resolve_default_modality_model(TRANSCRIPTION_MODEL_ENV)
    return TranscriptionClient(model, model_kwargs=kwargs or None)


def transcribe(
    audio: Any, *, model: Union[str, "TranscriptionModel", "TranscriptionSpec", None] = None, **kwargs: Any
) -> str:
    """One-shot transcription -- builds a fresh client and transcribes one audio clip.

    ``audio`` accepts a file path, raw bytes, an ``https://`` URL, or a
    ``data:audio/...;base64,...`` URL (same forms as ``audio=`` on :func:`chat`).

    ``model`` may be a ``"provider:model_id"`` string, a ``TranscriptionModel`` enum
    member, or omitted (reads ``AIMU_TRANSCRIPTION_MODEL`` env var).
    """
    return transcription_client(model).transcribe(audio, **kwargs)


def embedding_client(
    model: Union[str, "EmbeddingModel", "EmbeddingSpec", None] = None,
    **kwargs: Any,
) -> "EmbeddingClient":
    """Construct an :class:`EmbeddingClient` for text-embedding generation.

    ``model`` accepts an :class:`EmbeddingModel` enum member, a ``"provider:model_id"``
    string (e.g. ``"openai:text-embedding-3-small"``, ``"ollama:nomic-embed-text"``), or
    an :class:`EmbeddingSpec`. Extra ``**kwargs`` are forwarded as ``model_kwargs`` to the
    underlying client (e.g. ``api_key=`` for OpenAI).

    When ``model`` is omitted, the ``AIMU_EMBEDDING_MODEL`` env var is used; if it is unset
    a ``ValueError`` is raised (no model is downloaded implicitly).

    Example::

        client = aimu.embedding_client("openai:text-embedding-3-small")
        vectors = client.embed(["hello", "world"])
    """
    if not (HAS_OPENAI_EMBEDDING or HAS_OLLAMA_EMBEDDING or HAS_HF_EMBEDDING):
        raise ImportError(
            "Embedding generation requires one of: the [openai_compat] extra (OpenAI), the [ollama] "
            "extra (Ollama), or the [hf] extra (local sentence-transformers): "
            "pip install -e '.[openai_compat]' | '.[ollama]' | '.[hf]'"
        )
    if model is None:
        from .models._internal.model_defaults import EMBEDDING_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(EMBEDDING_MODEL_ENV)
    return EmbeddingClient(model, model_kwargs=kwargs or None)


def embed(
    texts: Union[str, list],
    *,
    model: Union[str, "EmbeddingModel", "EmbeddingSpec", None] = None,
    **kwargs: Any,
) -> Any:
    """One-shot embedding: builds a fresh client and embeds one string or a list.

    A single ``str`` returns one vector (``list[float]``); a list returns a list of
    vectors (``list[list[float]]``). For repeated embedding, construct a client with
    :func:`embedding_client` and reuse it.

    ``model`` may be a ``"provider:model_id"`` string, an :class:`EmbeddingModel` member,
    or omitted (reads ``AIMU_EMBEDDING_MODEL`` env var).

    Example::

        vector = aimu.embed("hello", model="openai:text-embedding-3-small")
        vectors = aimu.embed(["a", "b"], model="ollama:nomic-embed-text")
    """
    return embedding_client(model).embed(texts, **kwargs)


def clear_hf_cache(model: Any = None) -> None:
    """Release cached HuggingFace model weights and free GPU memory.

    All four HuggingFace modality clients (text, image, audio, speech) share
    module-level weight registries so that multiple client instances for the
    same model don't load weights twice. Call this when you are done with a
    model to reclaim VRAM. Without calling this, weights remain in the registry
    for the lifetime of the process even after all client instances are deleted.

    If *model* is provided (any HuggingFace model enum member), only that
    model's entry is cleared. Pass ``None`` to clear all cached models.

    Example::

        c1 = aimu.client("hf:Qwen/Qwen3-8B")
        c2 = aimu.client("hf:Qwen/Qwen3-8B")
        assert c1._hf_model is c2._hf_model  # shared weights

        aimu.clear_hf_cache()  # free all HF weights
    """
    import gc
    import importlib

    suffixes = [
        "models.providers.hf.text",
        "models.providers.hf.image",
        "models.providers.hf.audio",
        "models.providers.hf.speech",
        "models.providers.hf.embedding",
    ]
    for suffix in suffixes:
        try:
            mod = importlib.import_module(f"aimu.{suffix}")
            registry = mod._model_registry
            lock = mod._registry_lock
            with lock:
                if model is None:
                    registry.clear()
                elif hasattr(model, "value"):
                    for key in [k for k in registry if k[0] == model.value]:
                        del registry[key]
        except (ImportError, AttributeError):
            pass

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def clear_llamacpp_cache(model: Any = None) -> None:
    """Release cached llama-cpp-python model weights.

    LlamaCppClient instances share a module-level registry so that multiple
    instances with the same ``model_path`` and construction parameters don't
    load the GGUF file twice. Call this to free the cached Llama instance.

    If *model* is provided (a :class:`~aimu.models.providers.llamacpp.LlamaCppModel`
    member or path string), only that entry is cleared.
    """
    import gc
    import importlib

    try:
        mod = importlib.import_module("aimu.models.providers.llamacpp")
        registry = mod._model_registry
        lock = mod._registry_lock
        with lock:
            if model is None:
                registry.clear()
            elif isinstance(model, str):
                for key in [k for k in registry if k[0] == model]:
                    del registry[key]
            elif hasattr(model, "value"):
                for key in [k for k in registry if k[0] == model.value]:
                    del registry[key]
    except (ImportError, AttributeError):
        pass

    gc.collect()


def image_client(model: Union[str, ImageModel, ImageSpec, None] = None, **kwargs: Any) -> ImageClient:
    """Construct an :class:`ImageClient` for text-to-image generation.

    ``model`` may be a :class:`HuggingFaceImageModel` / :class:`GeminiImageModel` member,
    an :class:`ImageSpec` subclass, or a ``"provider:model_id"`` string
    (``"hf:..."`` for HuggingFace ``diffusers``; ``"gemini:..."`` for Google Nano Banana).
    Extra ``**kwargs`` are forwarded as ``model_kwargs`` to the underlying provider client
    (e.g. ``api_key=`` for Gemini, ``variant="fp16"`` for diffusers pipelines).

    When ``model`` is omitted, the ``AIMU_IMAGE_MODEL`` env var is used; if it is unset a
    ``ValueError`` is raised (no model is downloaded implicitly).

    Example::

        client = aimu.image_client(aimu.HuggingFaceImageModel.SD_1_5)
        client = aimu.image_client("gemini:nano-banana")
    """
    if model is None:
        from .models._internal.model_defaults import IMAGE_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(IMAGE_MODEL_ENV)
    return ImageClient(model, model_kwargs=kwargs or None)


def generate_image(
    prompt: str,
    *,
    model: Union[str, ImageModel, ImageSpec, None] = None,
    format: str = "pil",
    **kwargs: Any,
) -> Any:
    """One-shot image generation: builds a fresh image client and returns one image.

    For multiple generations, construct a client with :func:`image_client` and reuse it
    so weights / API clients aren't rebuilt per call.

    Example::

        # Local diffusers
        path = aimu.generate_image(
            "a watercolor of a fox",
            model="hf:runwayml/stable-diffusion-v1-5",
            format="path",
        )

        # Google Nano Banana
        img = aimu.generate_image(
            "a watercolor of a fox",
            model="gemini:nano-banana",
            aspect_ratio="1:1",
        )
    """
    c = image_client(model)
    return c.generate(prompt, format=format, **kwargs)


__all__ = [
    "__version__",
    "Agent",
    "AudioClient",
    "AudioModel",
    "AudioSpec",
    "BaseAudioClient",
    "BaseEmbeddingClient",
    "BaseImageClient",
    "BaseModelClient",
    "BaseSpeechClient",
    "EmbeddingClient",
    "EmbeddingModel",
    "EmbeddingSpec",
    "HAS_GEMINI_IMAGE",
    "HAS_HF_AUDIO",
    "HAS_HF_EMBEDDING",
    "HAS_HF_IMAGE",
    "HAS_HF_SPEECH",
    "HAS_OLLAMA_EMBEDDING",
    "HAS_OPENAI_EMBEDDING",
    "HAS_OPENAI_SPEECH",
    "GeminiImageClient",
    "GeminiImageModel",
    "GeminiImageSpec",
    "HuggingFaceAudioClient",
    "HuggingFaceAudioModel",
    "HuggingFaceAudioSpec",
    "HuggingFaceImageClient",
    "HuggingFaceImageModel",
    "HuggingFaceImageSpec",
    "HuggingFaceSpeechClient",
    "HuggingFaceSpeechModel",
    "HuggingFaceSpeechSpec",
    "ImageClient",
    "ImageModel",
    "ImageSpec",
    "Model",
    "ModelClient",
    "ModelSpec",
    "OpenAISpeechClient",
    "OpenAISpeechModel",
    "OpenAISpeechSpec",
    "SpeechClient",
    "SpeechModel",
    "SpeechSpec",
    "StreamChunk",
    "StreamingContentType",
    "ToolContext",
    "TranscriptionClient",
    "TranscriptionModel",
    "TranscriptionSpec",
    "agent",
    "aio",
    "audio_client",
    "available_speech_clients",
    "chat",
    "clear_hf_cache",
    "clear_llamacpp_cache",
    "client",
    "embed",
    "embedding_client",
    "extract_tool_calls",
    "generate_json",
    "parse_json_response",
    "generate_audio",
    "generate_image",
    "generate_speech",
    "image_client",
    "pretty_print",
    "available_audio_models",
    "available_embedding_models",
    "available_image_models",
    "available_speech_models",
    "available_text_models",
    "available_transcription_models",
    "resolve_audio_model_string",
    "resolve_embedding_model_string",
    "resolve_default_text_model_enum",
    "resolve_image_model_enum",
    "resolve_image_model_string",
    "resolve_model_enum",
    "resolve_model_string",
    "resolve_speech_model_string",
    "resolve_transcription_model_string",
    "speech_client",
    "tool",
    "transcribe",
    "transcription_client",
]
