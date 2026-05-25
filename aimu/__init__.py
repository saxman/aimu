"""
aimu - AI Model Utilities

Lightweight Python library for building LLM-powered apps. Provider-agnostic model
clients (Ollama, HuggingFace, Anthropic, OpenAI, Gemini, llama-cpp, any OpenAI-compatible
server), text-to-image diffusion clients (HuggingFace ``diffusers``), in-process tools
via the ``@tool`` decorator, and code-controlled workflows (Chain, Router, Parallel,
EvaluatorOptimizer) plus autonomous Agents.

Quick start::

    import aimu

    text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

    client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
    client.chat("Hi there")

    image = aimu.generate_image("a watercolor of a fox", model="hf:runwayml/stable-diffusion-v1-5")
"""

from typing import Any, Iterable, Iterator, Optional, Union

from . import aio
from .models import (
    HAS_DIFFUSION,
    BaseModelClient,
    DiffusionClient,
    DiffusionModel,
    DiffusionSpec,
    Model,
    ModelClient,
    ModelSpec,
    StreamChunk,
    StreamingContentType,
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


def image_client(model: Union[str, "DiffusionModel", "DiffusionSpec"], **kwargs: Any) -> "DiffusionClient":
    """Construct a :class:`DiffusionClient` for text-to-image generation.

    ``model`` may be a :class:`DiffusionModel` enum member (e.g.
    ``DiffusionModel.SDXL_BASE``), a :class:`DiffusionSpec`, or a
    ``"hf:<repo_id>"`` string. Extra ``**kwargs`` are forwarded as
    ``model_kwargs`` to the diffusers pipeline loader.

    Example::

        client = aimu.image_client(aimu.DiffusionModel.SD_1_5)
        image = client.generate("a watercolor of a fox in a snowy forest")
    """
    if not HAS_DIFFUSION:
        raise ImportError(
            "Diffusion support requires the [diffusion] extra: "
            "pip install -e '.[diffusion]'"
        )
    return DiffusionClient(model, model_kwargs=kwargs or None)


def generate_image(
    prompt: str,
    *,
    model: Union[str, "DiffusionModel", "DiffusionSpec"],
    format: str = "pil",
    **kwargs: Any,
) -> Any:
    """One-shot image generation — builds a fresh :class:`DiffusionClient` and returns one image.

    For multiple generations, construct a :class:`DiffusionClient` with :func:`image_client`
    and reuse it so weights are loaded only once.

    Example::

        path = aimu.generate_image(
            "a watercolor of a fox",
            model="hf:runwayml/stable-diffusion-v1-5",
            format="path",
        )
    """
    c = image_client(model)
    return c.generate(prompt, format=format, **kwargs)


__all__ = [
    "BaseModelClient",
    "DiffusionClient",
    "DiffusionModel",
    "DiffusionSpec",
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
    "resolve_model_string",
]
