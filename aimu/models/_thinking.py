import re

from .base_client import StreamingContentType


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
                self._buffer = self._buffer[idx + len(tag):]
                self._in_thinking = not self._in_thinking

        return results

    @staticmethod
    def _safe_emit_length(buffer: str, tag: str) -> int:
        """Return how many leading characters can be safely emitted without risking a partial tag at the end."""
        for i in range(1, min(len(tag), len(buffer)) + 1):
            if buffer.endswith(tag[:i]):
                return len(buffer) - i
        return len(buffer)
