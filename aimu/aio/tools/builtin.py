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

_AIMU_DIFFUSION_DEFAULT = "hf:stabilityai/stable-diffusion-xl-base-1.0"
_async_image_client = None


def _get_async_image_client():
    """Lazy singleton :class:`AsyncDiffusionClient` for the async built-in tool."""
    global _async_image_client
    if _async_image_client is None:
        import aimu
        from aimu.aio.providers.diffusion import AsyncDiffusionClient

        model_str = os.environ.get("AIMU_DIFFUSION_MODEL", _AIMU_DIFFUSION_DEFAULT)
        _async_image_client = AsyncDiffusionClient(aimu.image_client(model_str))
    return _async_image_client


@tool
async def generate_image(prompt: str) -> str:
    """Generate an image from a text prompt and return the saved file path.

    Uses a HuggingFace diffusion pipeline. The default model is controlled by the
    ``AIMU_DIFFUSION_MODEL`` env var (default: SDXL base). Override per-agent by
    constructing your own tool with :func:`make_async_image_tool`.

    Args:
        prompt: A description of the desired image.
    """
    client = _get_async_image_client()
    return await client.generate(prompt, format="path")


def make_async_image_tool(client):
    """Build an async ``generate_image`` tool bound to a specific :class:`AsyncDiffusionClient`.

    Pass either a sync :class:`DiffusionClient` or an existing
    :class:`AsyncDiffusionClient`; sync clients are wrapped automatically.
    """
    from aimu.aio.providers.diffusion import AsyncDiffusionClient
    from aimu.models.diffusion import DiffusionClient as SyncDiffusionClient

    if isinstance(client, SyncDiffusionClient):
        client = AsyncDiffusionClient(client)

    @tool
    async def generate_image(prompt: str) -> str:
        """Generate an image from a text prompt and return the saved file path.

        Args:
            prompt: A description of the desired image.
        """
        return await client.generate(prompt, format="path")

    return generate_image


image = [generate_image]
