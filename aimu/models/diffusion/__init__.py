"""HuggingFace ``diffusers``-backed text-to-image client.

Parallel to the text :class:`~aimu.models.BaseModelClient` surface — diffusion is
a separate modality with its own interface (``generate(prompt) -> image``) rather
than a forced fit into ``chat()`` / ``messages``. See ``DiffusionClient`` for the
direct API and :func:`aimu.image_client` / :func:`aimu.generate_image` for the
one-line entry points.
"""

from .diffusion_client import DiffusionClient, DiffusionModel

__all__ = ["DiffusionClient", "DiffusionModel"]
