"""Console rendering for streamed runs.

``pretty_print`` consumes the :class:`~aimu.models.StreamChunk` iterator from
``client.chat(stream=True)``, ``Agent.run(stream=True)``, or any workflow run, and writes a
readable transcript: tool calls flagged, generated text streamed inline, thinking optional. It
saves every caller from re-implementing the same ``chunk.is_tool_call()`` / ``chunk.is_text()``
dispatch loop.
"""

from __future__ import annotations

import sys
from typing import Iterator, Optional, TextIO

from .models import StreamChunk, StreamingContentType


def pretty_print(
    stream: Iterator[StreamChunk],
    *,
    file: Optional[TextIO] = None,
    show_thinking: bool = False,
    show_tools: bool = True,
) -> str:
    """Render a stream of :class:`StreamChunk` to ``file`` and return the generated text.

    Args:
        stream: The iterator returned by a ``stream=True`` run.
        file: Where to write (defaults to ``sys.stdout``).
        show_thinking: If True, also print THINKING tokens (model reasoning).
        show_tools: If True, print a ``[tool] <name>`` marker for each tool call.

    Returns:
        The concatenated GENERATING text, for callers that want the final answer as well as
        the printed transcript.
    """
    out = file if file is not None else sys.stdout
    generated: list[str] = []
    for chunk in stream:
        if chunk.phase == StreamingContentType.TOOL_CALLING:
            if show_tools:
                name = chunk.content.get("name") if isinstance(chunk.content, dict) else chunk.content
                print(f"\n  [tool] {name}", file=out)
        elif chunk.phase == StreamingContentType.THINKING:
            if show_thinking:
                print(chunk.content, end="", file=out)
        elif chunk.phase == StreamingContentType.GENERATING:
            print(chunk.content, end="", file=out)
            generated.append(chunk.content)
    print(file=out)
    return "".join(generated)
