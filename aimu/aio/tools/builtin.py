"""Async equivalents of :mod:`aimu.tools.builtin`.

Re-exports every sync tool from the sync module (the async client dispatches them
through :func:`asyncio.to_thread`) and provides an async-native ``generate_image``
streaming tool that yields :class:`~aimu.models.StreamChunk` progress chunks during
generation.
"""

from __future__ import annotations

from typing import Optional

from aimu.tools.builtin import (  # noqa: F401 (re-exports)
    calculate,
    compute,
    echo,
    fs,
    get_current_date_and_time,
    get_weather,
    get_webpage,
    list_directory,
    make_document_tools,
    make_memory_tools,
    make_retrieval_tool,
    misc,
    read_file,
    web_search,
    web,
    wikipedia,
)
from aimu.tools.decorator import tool

_async_image_client = None


def _get_async_image_client():
    """Lazy singleton :class:`AsyncImageClient` for the async built-in tool.

    Reads ``AIMU_IMAGE_MODEL`` from the environment. Raises ``ValueError`` if it is
    unset (no model is downloaded implicitly). Accepts any model string supported by
    :func:`aimu.aio.image_client`: ``"hf:..."`` / ``"gemini:..."``.
    """
    global _async_image_client
    if _async_image_client is None:
        import aimu
        from aimu.aio import image_client as _aio_image_client

        _async_image_client = _aio_image_client(aimu.image_client())  # resolves AIMU_IMAGE_MODEL or raises
    return _async_image_client


@tool
async def generate_image(prompt: str):
    """Generate an image from a text prompt and return the saved file path.

    **Streaming async tool**: async generator yielding
    :attr:`~aimu.models.StreamingContentType.IMAGE_GENERATING` chunks during
    denoising. When dispatched by ``aio.Agent.run(stream=True)``, chunks flow
    through the agent's own stream and into the UI live.

    Uses an :class:`aimu.aio.AsyncImageClient`. The model is controlled by
    ``AIMU_IMAGE_MODEL`` (required; the tool raises if it is unset). Use
    :func:`make_async_image_tool` to override the client or opt into
    ``preview_every=N`` intermediate previews.

    Args:
        prompt: A description of the desired image.
    """
    client = _get_async_image_client()
    final_result: Optional[str] = None
    async for chunk in await client.generate(prompt, format="path", stream=True):
        yield chunk
        content = chunk.content
        if isinstance(content, dict) and content.get("final"):
            final_result = content.get("result")
    # Final chunk's content["result"] is picked up by _handle_tool_calls_streamed
    # as the canonical tool response; no return-value needed (PEP 525 async
    # generators don't carry return values anyway).
    del final_result


def make_async_image_tool(client, *, preview_every: Optional[int] = None):
    """Build an async streaming ``generate_image`` tool bound to a specific client.

    Pass a sync :class:`aimu.BaseImageClient` (e.g.
    :class:`HuggingFaceImageClient`, :class:`GeminiImageClient`), which will be
    wrapped automatically, or an existing :class:`aimu.aio.AsyncImageClient`.
    ``preview_every=N`` opts into intermediate denoised-image previews (HF only;
    Gemini ignores it).
    """
    from aimu.aio import image_client as _aio_image_client
    from aimu.aio.image import AsyncImageClient
    from aimu.models.base import BaseImageClient

    if isinstance(client, BaseImageClient):
        client = _aio_image_client(client)
    elif not isinstance(client, AsyncImageClient):
        # Permit the per-provider async classes (AsyncHuggingFaceImageClient,
        # AsyncGeminiImageClient) directly; they expose .generate() too.
        pass

    @tool
    async def generate_image(prompt: str):
        """Generate an image from a text prompt and return the saved file path.

        Async streaming tool: yields progress chunks during generation.

        Args:
            prompt: A description of the desired image.
        """
        async for chunk in await client.generate(prompt, format="path", stream=True, preview_every=preview_every):
            yield chunk

    return generate_image


image = [generate_image]
