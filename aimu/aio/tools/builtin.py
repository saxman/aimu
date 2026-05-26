"""Async equivalents of :mod:`aimu.tools.builtin`.

Re-exports every sync tool from the sync module (the async client dispatches them
through :func:`asyncio.to_thread`) and provides an async-native ``generate_image``
that awaits the lazy :class:`AsyncDiffusionClient` singleton.
"""

from __future__ import annotations

import os

from aimu.tools.builtin import (  # noqa: F401 — re-exports
    calculate,
    compute,
    echo,
    fs,
    get_current_date_and_time,
    get_weather,
    get_webpage,
    list_directory,
    misc,
    read_file,
    search,
    web,
    wikipedia,
)
from aimu.tools.decorator import tool

_AIMU_IMAGE_DEFAULT = "hf:stabilityai/stable-diffusion-xl-base-1.0"
_async_image_client = None


def _get_async_image_client():
    """Lazy singleton :class:`AsyncImageClient` for the async built-in tool.

    Reads ``AIMU_IMAGE_MODEL`` from the environment (default: SDXL base). Accepts
    any model string supported by :func:`aimu.aio.image_client` —
    ``"hf:..."`` / ``"gemini:..."``.
    """
    global _async_image_client
    if _async_image_client is None:
        import aimu
        from aimu.aio import image_client as _aio_image_client

        model_str = os.environ.get("AIMU_IMAGE_MODEL", _AIMU_IMAGE_DEFAULT)
        _async_image_client = _aio_image_client(aimu.image_client(model_str))
    return _async_image_client


@tool
async def generate_image(prompt: str) -> str:
    """Generate an image from a text prompt and return the saved file path.

    Uses an :class:`aimu.aio.AsyncImageClient`. The default model is controlled by
    the ``AIMU_IMAGE_MODEL`` env var (default: SDXL base). Override per-agent by
    constructing your own tool with :func:`make_async_image_tool`.

    Args:
        prompt: A description of the desired image.
    """
    client = _get_async_image_client()
    return await client.generate(prompt, format="path")


def make_async_image_tool(client):
    """Build an async ``generate_image`` tool bound to a specific async image client.

    Pass either a sync :class:`aimu.BaseImageClient` (e.g. :class:`HuggingFaceImageClient`,
    :class:`GeminiImageClient`) or an existing :class:`aimu.aio.AsyncImageClient`;
    sync clients are wrapped automatically.
    """
    from aimu.aio import image_client as _aio_image_client
    from aimu.aio.image import AsyncImageClient
    from aimu.models.base import BaseImageClient

    if isinstance(client, BaseImageClient):
        client = _aio_image_client(client)
    elif not isinstance(client, AsyncImageClient):
        # Permit the per-provider async classes (AsyncHuggingFaceImageClient,
        # AsyncGeminiImageClient) directly — they already expose `.generate()`.
        pass

    @tool
    async def generate_image(prompt: str) -> str:
        """Generate an image from a text prompt and return the saved file path.

        Args:
            prompt: A description of the desired image.
        """
        return await client.generate(prompt, format="path")

    return generate_image


image = [generate_image]
