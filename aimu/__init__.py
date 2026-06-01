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
    HAS_HF_AUDIO,
    HAS_HF_IMAGE,
    HAS_HF_SPEECH,
    HAS_OPENAI_SPEECH,
    AudioClient,
    AudioModel,
    AudioSpec,
    BaseAudioClient,
    BaseImageClient,
    BaseModelClient,
    BaseSpeechClient,
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
    available_speech_clients,
    resolve_audio_model_string,
    resolve_image_model_string,
    resolve_model_string,
    resolve_speech_model_string,
)


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
        from .models._defaults import resolve_default_text_model

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
    """One-shot chat — builds a fresh client, sends one message, returns the response.

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

        from aimu.tools import tool

        @tool
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
        from .models._defaults import AUDIO_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(AUDIO_MODEL_ENV)
    return AudioClient(model, model_kwargs=kwargs or None)


def generate_audio(
    prompt: str,
    *,
    model: Union[str, AudioModel, AudioSpec, None] = None,
    format: str = "path",
    **kwargs: Any,
) -> Any:
    """One-shot audio generation — builds a fresh audio client and returns one audio clip.

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
        from .models._defaults import SPEECH_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(SPEECH_MODEL_ENV)
    return SpeechClient(model, model_kwargs=kwargs or None)


def generate_speech(
    text: str,
    *,
    model: Union[str, SpeechModel, SpeechSpec, None] = None,
    format: str = "path",
    **kwargs: Any,
) -> Any:
    """One-shot speech synthesis — builds a fresh client and returns one audio clip.

    For repeated synthesis, construct a client with :func:`speech_client` and
    reuse it so API clients aren't rebuilt per call.

    Example::

        path = aimu.generate_speech("Hello, world!", model="openai:tts-1")
        sr, audio = aimu.generate_speech("Hello", model="hf:facebook/mms-tts-eng", format="numpy")
    """
    c = speech_client(model)
    return c.generate(text, format=format, **kwargs)


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
        from .models._defaults import IMAGE_MODEL_ENV, resolve_default_modality_model

        model = resolve_default_modality_model(IMAGE_MODEL_ENV)
    return ImageClient(model, model_kwargs=kwargs or None)


def generate_image(
    prompt: str,
    *,
    model: Union[str, ImageModel, ImageSpec, None] = None,
    format: str = "pil",
    **kwargs: Any,
) -> Any:
    """One-shot image generation — builds a fresh image client and returns one image.

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
    "BaseImageClient",
    "BaseModelClient",
    "BaseSpeechClient",
    "HAS_GEMINI_IMAGE",
    "HAS_HF_AUDIO",
    "HAS_HF_IMAGE",
    "HAS_HF_SPEECH",
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
    "agent",
    "aio",
    "audio_client",
    "available_speech_clients",
    "chat",
    "client",
    "generate_audio",
    "generate_image",
    "generate_speech",
    "image_client",
    "resolve_audio_model_string",
    "resolve_image_model_string",
    "resolve_model_string",
    "resolve_speech_model_string",
    "speech_client",
]
