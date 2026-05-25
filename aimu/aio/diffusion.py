"""Async diffusion surface: :func:`image_client` / :func:`generate_image`.

Mirrors the sync :func:`aimu.image_client` / :func:`aimu.generate_image` shape.
Because diffusion is an in-process provider, the factory follows the wrap pattern
established for HuggingFace / LlamaCpp: pass an existing sync :class:`DiffusionClient`
to wrap. Direct enum / string construction is refused with a helpful error.
"""

from __future__ import annotations

from typing import Any, Union

try:
    from aimu.models.diffusion import DiffusionClient, DiffusionModel
    from aimu.models.base import DiffusionSpec

    from .providers.diffusion import AsyncDiffusionClient

    _HAS_DIFFUSION = True
except ImportError:
    _HAS_DIFFUSION = False
    DiffusionClient = None  # type: ignore[assignment,misc]
    DiffusionModel = None  # type: ignore[assignment,misc]
    DiffusionSpec = None  # type: ignore[assignment,misc]
    AsyncDiffusionClient = None  # type: ignore[assignment,misc]


_WRAP_GUIDANCE = (
    "Diffusion is an in-process provider — model weights live in memory; "
    "constructing both a sync and async client for the same model would load weights twice. "
    "Build a sync client first and pass it to aio.image_client():\n"
    "    sync_client = aimu.image_client({model})\n"
    "    async_client = aio.image_client(sync_client)"
)


def image_client(model: Any) -> "AsyncDiffusionClient":
    """Construct an :class:`AsyncDiffusionClient`.

    Accepts an existing sync :class:`DiffusionClient` (the recommended path —
    shares weights). Passing a :class:`DiffusionModel` enum member, a
    :class:`DiffusionSpec`, or a ``"hf:<repo_id>"`` string raises ``ValueError``
    pointing at the wrap pattern.
    """
    if not _HAS_DIFFUSION:
        raise ImportError(
            "Diffusion support requires the [diffusion] extra: "
            "pip install -e '.[diffusion]'"
        )

    if isinstance(model, DiffusionClient):
        return AsyncDiffusionClient(model)

    if isinstance(model, DiffusionModel):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"DiffusionModel.{model.name}"))
    if isinstance(model, DiffusionSpec):
        raise ValueError(_WRAP_GUIDANCE.format(model=f"DiffusionSpec({model.id!r})"))
    if isinstance(model, str):
        raise ValueError(_WRAP_GUIDANCE.format(model=repr(model)))

    raise TypeError(
        f"aio.image_client expects a sync DiffusionClient. Got: {type(model).__name__}"
    )


async def generate_image(
    prompt: str,
    *,
    model: Union[str, "DiffusionModel", "DiffusionSpec", "DiffusionClient"],
    format: str = "pil",
    **kwargs: Any,
) -> Any:
    """One-shot async image generation.

    Accepts either an existing sync :class:`DiffusionClient` (preferred — weights
    reused across calls) or a model spec/string (constructs a fresh sync client
    inside, which loads weights each call — convenient for ad-hoc use).
    """
    if not _HAS_DIFFUSION:
        raise ImportError(
            "Diffusion support requires the [diffusion] extra: "
            "pip install -e '.[diffusion]'"
        )

    if isinstance(model, DiffusionClient):
        sync = model
    else:
        sync = DiffusionClient(model)
    async_client = AsyncDiffusionClient(sync)
    return await async_client.generate(prompt, format=format, **kwargs)
