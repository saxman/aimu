import re
from collections.abc import Mapping
from typing import Any, Optional

from ..base import StreamingContentType


def _reasoning_text(message_or_delta: Any) -> Optional[str]:
    """The server-side-stripped reasoning on an OpenAI-shaped message or delta, if any.

    Servers that parse reasoning out of the content disagree on what to call the field they
    put it in: llama-server, vLLM and SGLang use ``reasoning_content`` (the DeepSeek spelling),
    while mlx-lm and OpenRouter use ``reasoning``. Both are extra fields on an otherwise
    standard message, so neither can be detected any way but by name, and reading only one of
    them drops a model's entire reasoning block with nothing raised.

    ``reasoning_content`` wins when a server sends both, so a gateway that echoes the same text
    into an alias cannot have it counted twice. A non-text value (a summary object, a list of
    parts) is not reasoning this can surface: callers concatenate the result, so returning one
    would raise mid-stream.

    Takes a mapping as well as an attribute-bearing object, because llama.cpp hands back plain
    dicts where the OpenAI SDK hands back models, and the field names are the same either way.
    """
    if isinstance(message_or_delta, Mapping):
        text = message_or_delta.get("reasoning_content") or message_or_delta.get("reasoning")
    else:
        text = getattr(message_or_delta, "reasoning_content", None) or getattr(message_or_delta, "reasoning", None)
    return text if isinstance(text, str) else None


def _split_thinking(content: str) -> tuple[str, str]:
    """Extract <think>...</think> block from content. Returns (thinking, clean_content)."""
    match = re.match(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Unclosed think tag
    match = re.match(r"<think>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), ""
    return "", content


class _ThinkingParser:
    """Stateful streaming parser that separates <think>...</think> from content across chunk boundaries."""

    def __init__(self):
        self._in_thinking = False
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[StreamingContentType, str]]:
        results = []
        self._buffer += text
        while True:
            tag = "</think>" if self._in_thinking else "<think>"
            phase = StreamingContentType.THINKING if self._in_thinking else StreamingContentType.GENERATING
            idx = self._buffer.find(tag)

            if idx == -1:
                safe_len = self._safe_emit_length(self._buffer, tag)
                if safe_len > 0:
                    results.append((phase, self._buffer[:safe_len]))
                    self._buffer = self._buffer[safe_len:]
                break
            else:
                if idx > 0:
                    results.append((phase, self._buffer[:idx]))
                self._buffer = self._buffer[idx + len(tag) :]
                self._in_thinking = not self._in_thinking

        return results

    @staticmethod
    def _safe_emit_length(buffer: str, tag: str) -> int:
        """Return how many leading characters can be safely emitted without risking a partial tag at the end."""
        for i in range(1, min(len(tag), len(buffer)) + 1):
            if buffer.endswith(tag[:i]):
                return len(buffer) - i
        return len(buffer)
