"""Image normalization and per-provider conversion helpers for vision input.

The unified canonical format inside ``self.messages`` mirrors OpenAI's content
blocks, e.g.::

    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ],
    }

Providers that don't speak OpenAI's format natively (Anthropic, Ollama) call the
adapter helpers below at request time without mutating ``self.messages``.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Union

ImageInput = Union[str, bytes, Path]


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _normalize_image(image: ImageInput) -> str:
    """Normalize a single image input to a URL string (http(s):// or data:image/...).

    Accepts:
      - http:// or https:// URL string  → returned unchanged
      - data:image/<mime>;base64,<b64> string → returned unchanged
      - file path string or pathlib.Path → read and base64-encoded as a data URL
        (mime inferred from the file suffix)
      - raw bytes → base64-encoded as image/png by default
    """
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode("ascii")
        return f"data:image/png;base64,{b64}"

    if isinstance(image, Path):
        return _path_to_data_url(image)

    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:image/")):
            return image
        return _path_to_data_url(Path(image))

    raise TypeError(f"Unsupported image input type: {type(image).__name__}")


def _path_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        mime = "image/png"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _parse_data_url(url: str) -> tuple[str, str]:
    """Parse a data:image/<mime>;base64,<b64> URL into (mime, base64_data)."""
    m = _DATA_URL_RE.match(url)
    if not m:
        raise ValueError(f"Not a base64 image data URL: {url[:64]!r}")
    return m.group("mime"), m.group("data")


def _build_user_content_blocks(text: str, images: list[ImageInput]) -> list[dict]:
    """Build a multi-modal user message content list in OpenAI block format."""
    blocks: list[dict] = [{"type": "text", "text": text}]
    for img in images:
        blocks.append({"type": "image_url", "image_url": {"url": _normalize_image(img)}})
    return blocks


def _openai_blocks_to_anthropic(blocks: list[dict]) -> list[dict]:
    """Convert an OpenAI-format content block list to Anthropic content blocks.

    Handles ``image_url`` (vision) and ``input_audio`` (audio input) blocks alongside
    plain ``text`` blocks. All other block types are passed through unchanged.
    """
    out: list[dict] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image_url":
            url = block["image_url"]["url"]
            if url.startswith("data:"):
                mime, data = _parse_data_url(url)
                out.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data},
                    }
                )
            else:
                out.append({"type": "image", "source": {"type": "url", "url": url}})
        elif btype == "input_audio":
            ia = block["input_audio"]
            out.append(
                {
                    "type": "audio",
                    "source": {
                        "type": "base64",
                        "media_type": f"audio/{ia['format']}",
                        "data": ia["data"],
                    },
                }
            )
        else:
            out.append(block)
    return out


def _ollama_split_message(message: dict) -> dict:
    """Convert a user message with OpenAI image_url content blocks to Ollama's format.

    Ollama's native API takes images as a top-level ``images=[<bare base64>]``
    field on the message, with ``content`` as a plain string. http(s) image URLs
    are not supported by Ollama; they are dropped with a ValueError.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message

    text_parts: list[str] = []
    images: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image_url":
            url = block["image_url"]["url"]
            if not url.startswith("data:"):
                raise ValueError(
                    "Ollama only accepts inline base64 images; pass a file path or bytes rather than an http(s) URL."
                )
            _, data = _parse_data_url(url)
            images.append(data)

    out = {**message, "content": "\n".join(text_parts)}
    if images:
        out["images"] = images
    return out


def _adapt_messages_for_ollama(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with vision blocks rewritten for Ollama."""
    return [_ollama_split_message(msg) for msg in messages]


def _decode_image_url_to_pil(url: str):
    """Decode an OpenAI image_url URL string to a PIL.Image. Used by HF processor path."""
    from io import BytesIO

    from PIL import Image

    if url.startswith("data:"):
        _, data = _parse_data_url(url)
        return Image.open(BytesIO(base64.b64decode(data)))
    if url.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url) as resp:
            return Image.open(BytesIO(resp.read()))
    raise ValueError(f"Unrecognized image URL: {url[:64]!r}")


def _reference_image_to_pil(image):
    """Decode any reference_image input to a PIL.Image.

    Accepts: PIL.Image (passthrough), file path string/Path, raw bytes, data URL, http(s) URL.
    """
    from PIL import Image as _PIL

    if isinstance(image, _PIL.Image):
        return image
    return _decode_image_url_to_pil(_normalize_image(image))


def _extract_pil_images(messages: list[dict]) -> list:
    """Walk messages and return all images decoded to PIL.Image (in order)."""
    images: list = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    images.append(_decode_image_url_to_pil(block["image_url"]["url"]))
    return images


def _replace_image_url_with_image_placeholder(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with image_url blocks rewritten to ``{"type": "image"}``.

    HuggingFace VL chat templates use ``{"type": "image"}`` placeholder blocks; the
    actual pixel data is passed separately as ``images=`` to the processor.
    """
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks = []
            for block in content:
                if block.get("type") == "image_url":
                    new_blocks.append({"type": "image"})
                else:
                    new_blocks.append(block)
            out.append({**msg, "content": new_blocks})
        else:
            out.append(msg)
    return out
