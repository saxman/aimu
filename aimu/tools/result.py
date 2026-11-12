"""``ToolResult``: a tool's answer that carries images as well as text.

A tool's answer reaches the model as the content of a ``role: "tool"`` message, and the tool loops
build that content with ``str(response)``. That is enough for every tool that answers in words, and
it is why a tool that *makes* or *fetches* an image could only ever tell the model where the image
was: the model, which could look at an image the user attached, could not look at one its own tool
produced. Returning ``ToolResult(text, images=[...])`` lets the pixels through.

What the loop does with it:

- The tool message's ``content`` becomes the canonical content-block list, the same OpenAI shape a
  user message with an image already has (``[{"type": "text", ...}, {"type": "image_url", ...}]``),
  so ``self.messages`` stays plain OpenAI data and each provider adapts at request time. Anthropic
  puts the images inside the ``tool_result``; Ollama puts them on the tool message's ``images``;
  OpenAI-format servers accept only text in a tool message, so their adapter moves the images into a
  user message straight after the run of tool results.
- When the client's model does not support vision, the images are dropped and only the text is sent,
  so a tool can return images unconditionally and a text-only model still gets a valid request.
- ``ToolCalled.result`` and the streamed ``TOOL_CALLING`` chunk's ``response`` stay the text, since
  every consumer of those reads a string.

``str(ToolResult(...))`` is the text, so code that stringifies a result (a host's own wrapper, a log
line) keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aimu.models._internal.image_input import ImageInput, _normalize_image


@dataclass(frozen=True)
class ToolResult:
    """A tool's answer: *text* for the model to read, and *images* for it to look at.

    Each image is anything ``chat(images=...)`` accepts: a file path, raw bytes, a ``data:image/...``
    URL, or an http(s) URL. They are normalized at construction (a path is read and base64-encoded
    there), so a missing file raises inside the tool, where the loop reports it as the tool's error,
    rather than later while a request is being built.

    Return it from a plain tool, or put it in a streaming tool's final chunk as ``content["result"]``.
    """

    text: str
    images: Sequence[ImageInput] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(_normalize_image(image) for image in self.images))

    def __str__(self) -> str:
        return self.text

    def content_blocks(self) -> list[dict]:
        """The tool message content for a vision model: the text block, then one block per image."""
        return [{"type": "text", "text": self.text}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in self.images
        ]


def tool_message_text(content) -> str:
    """The text of a tool message's *content*, whether a plain string or a content-block list."""
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
    return content


__all__ = ["ToolResult", "tool_message_text"]
