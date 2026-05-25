"""
aimu - AI Model Utilities

Lightweight Python library for building LLM-powered apps. Provider-agnostic model
clients (Ollama, HuggingFace, Anthropic, OpenAI, Gemini, llama-cpp, any OpenAI-compatible
server), in-process tools via the ``@tool`` decorator, and code-controlled workflows
(Chain, Router, Parallel, EvaluatorOptimizer) plus autonomous Agents.

Quick start::

    import aimu

    text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

    client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
    client.chat("Hi there")
"""

from typing import Any, Iterable, Iterator, Optional, Union

from .models import (
    BaseModelClient,
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


__all__ = [
    "BaseModelClient",
    "Model",
    "ModelClient",
    "ModelSpec",
    "StreamChunk",
    "StreamingContentType",
    "chat",
    "client",
    "resolve_model_string",
]
