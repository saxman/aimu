"""
aimu - AI Modeling Utilities

Lightweight Python library for building LLM-powered apps. Provider-agnostic model
clients (Ollama, HuggingFace, Anthropic, OpenAI, Gemini, llama-cpp, any OpenAI-compatible
server), text-to-image clients (HuggingFace ``diffusers`` + Google Nano Banana), in-process
tools via the ``@tool`` decorator, and code-controlled workflows (Chain, Router, Parallel,
EvaluatorOptimizer) plus autonomous Agents.

Quick start::

    import aimu

    text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

    client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
    client.chat("Hi there")

    image = aimu.generate_image("a watercolor of a fox", model="hf:runwayml/stable-diffusion-v1-5")
    image = aimu.generate_image("a watercolor of a fox", model="gemini:nano-banana")
"""

from typing import Any, Iterable, Iterator, Optional, Union

from . import aio
from .models import (
    HAS_GEMINI_IMAGE,
    HAS_HF_IMAGE,
    BaseImageClient,
    BaseModelClient,
    GeminiImageClient,
    GeminiImageModel,
    GeminiImageSpec,
    HuggingFaceImageClient,
    HuggingFaceImageModel,
    HuggingFaceImageSpec,
    ImageClient,
    ImageModel,
    ImageSpec,
    Model,
    ModelClient,
    ModelSpec,
    StreamChunk,
    StreamingContentType,
    resolve_image_model_string,
    resolve_model_string,
)


def client(model: Union[str, Model], *, system: Optional[str] = None, **kwargs: Any) -> ModelClient:
    """Construct a :class:`ModelClient` from a model string or enum member.

    ``model`` may be a ``"provider:model_id"`` string (``"anthropic:claude-sonnet-4-6"``,
    ``"ollama:qwen3.5:9b"``) or any provider's ``Model`` enum member. Extra ``**kwargs``
    are forwarded to the underlying provider client (e.g. ``model_path=`` for llama-cpp).

    Use this as the one-line construction helper. For full control over the provider
    client constructor, use :class:`ModelClient` directly.
    """
    if system is not None:
        kwargs["system_message"] = system
    return ModelClient(model, **kwargs)


def chat(
    user_message: str,
    *,
    model: Union[str, Model],
    system: Optional[str] = None,
    generate_kwargs: Optional[dict] = None,
    stream: bool = False,
    images: Optional[list] = None,
    include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
) -> Union[str, Iterator[StreamChunk]]:
    """One-shot chat — builds a fresh client, sends one message, returns the response.

    For multi-turn conversations construct a :class:`ModelClient` with :func:`client` and
    call its ``chat()`` repeatedly.

    Example::

        text = aimu.chat("Summarize this", model="anthropic:claude-sonnet-4-6")

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


def image_client(model: Union[str, ImageModel, ImageSpec], **kwargs: Any) -> ImageClient:
    """Construct an :class:`ImageClient` for text-to-image generation.

    ``model`` may be a :class:`HuggingFaceImageModel` / :class:`GeminiImageModel` member,
    an :class:`ImageSpec` subclass, or a ``"provider:model_id"`` string
    (``"hf:..."`` for HuggingFace ``diffusers``; ``"gemini:..."`` for Google Nano Banana).
    Extra ``**kwargs`` are forwarded as ``model_kwargs`` to the underlying provider client
    (e.g. ``api_key=`` for Gemini, ``variant="fp16"`` for diffusers pipelines).

    Example::

        client = aimu.image_client(aimu.HuggingFaceImageModel.SD_1_5)
        client = aimu.image_client("gemini:nano-banana")
    """
    return ImageClient(model, model_kwargs=kwargs or None)


def generate_image(
    prompt: str,
    *,
    model: Union[str, ImageModel, ImageSpec],
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
    "BaseImageClient",
    "BaseModelClient",
    "HAS_GEMINI_IMAGE",
    "HAS_HF_IMAGE",
    "GeminiImageClient",
    "GeminiImageModel",
    "GeminiImageSpec",
    "HuggingFaceImageClient",
    "HuggingFaceImageModel",
    "HuggingFaceImageSpec",
    "ImageClient",
    "ImageModel",
    "ImageSpec",
    "Model",
    "ModelClient",
    "ModelSpec",
    "StreamChunk",
    "StreamingContentType",
    "aio",
    "chat",
    "client",
    "generate_image",
    "image_client",
    "resolve_image_model_string",
    "resolve_model_string",
]
