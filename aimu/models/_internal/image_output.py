"""Image output encoding for diffusion clients.

Sibling to :mod:`aimu.models._internal.image_input` (which decodes images for vision *input*);
this module encodes images for diffusion *output*. The split is directional: input
helpers turn arbitrary user-supplied images into the canonical OpenAI ``image_url``
block; output helpers turn PIL Images produced by a diffusion pipeline into one of
``"pil"`` / ``"path"`` / ``"bytes"`` / ``"data_url"``.

The single helper :func:`encode_image` is shared by the sync and async diffusion
clients so format-conversion logic lives in one place and is unit-testable without
loading model weights.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, Union

OutputFormat = Literal["pil", "path", "bytes", "data_url"]
VALID_FORMATS = ("pil", "path", "bytes", "data_url")


def encode_image(
    image: Any,
    format: str = "pil",
    *,
    prompt: str = "",
    output_dir: Optional[Path] = None,
) -> Union[Any, str, bytes]:
    """Encode a single PIL Image into the requested output format.

    Args:
        image: A ``PIL.Image.Image``. Type is annotated as ``Any`` to keep this
            module importable without Pillow installed (diffusion clients have
            already imported Pillow by the time they call this).
        format: One of ``"pil"`` (return image unchanged), ``"bytes"`` (PNG bytes),
            ``"data_url"`` (``data:image/png;base64,...``), or ``"path"`` (save to
            ``output_dir`` and return the file path string).
        prompt: The originating prompt, used only to seed the filename hash when
            ``format="path"``. Not stored anywhere else.
        output_dir: Directory to save into when ``format="path"``. Defaults to
            ``<repo>/output/images/``; created on demand.

    Returns:
        The encoded image: PIL Image, ``bytes``, or path/data-url ``str``.

    Raises:
        ValueError: If ``format`` is not one of :data:`VALID_FORMATS`.
    """
    if format == "pil":
        return image

    if format not in VALID_FORMATS:
        raise ValueError(f"Unknown format: {format!r}. Expected one of {VALID_FORMATS}.")

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    if format == "bytes":
        return data
    if format == "data_url":
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    # format == "path"
    if output_dir is None:
        from aimu.paths import output as default_output_root

        output_dir = default_output_root / "images"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_hash = hashlib.sha1(f"{prompt}{ts}".encode("utf-8")).hexdigest()[:8]
    path = output_dir / f"{ts}-{short_hash}.png"
    path.write_bytes(data)
    return str(path)
