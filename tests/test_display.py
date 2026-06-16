"""Tests for aimu.pretty_print stream rendering."""

import io

from aimu import pretty_print
from aimu.models import StreamChunk, StreamingContentType


def _stream():
    return iter(
        [
            StreamChunk(StreamingContentType.THINKING, "deliberating"),
            StreamChunk(StreamingContentType.TOOL_CALLING, {"name": "search", "arguments": {}, "response": "r"}),
            StreamChunk(StreamingContentType.GENERATING, "Hello "),
            StreamChunk(StreamingContentType.GENERATING, "world"),
        ]
    )


def test_pretty_print_returns_generated_text_and_marks_tools():
    buf = io.StringIO()
    text = pretty_print(_stream(), file=buf)

    assert text == "Hello world"
    out = buf.getvalue()
    assert "[tool] search" in out
    assert "Hello world" in out
    assert "deliberating" not in out  # thinking hidden by default


def test_pretty_print_show_thinking_and_hide_tools():
    buf = io.StringIO()
    pretty_print(_stream(), file=buf, show_thinking=True, show_tools=False)

    out = buf.getvalue()
    assert "deliberating" in out
    assert "[tool]" not in out
