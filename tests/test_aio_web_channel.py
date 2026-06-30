"""Mock-only unit tests for the library WebChannel (the WebSocket Channel adapter).

The websocket is duck-typed, so a fake captures the JSON frames without any real socket or starlette.
Server round-trip coverage lives with the app that wires a Starlette route (e.g. the personal-assistant
example's tests), since the library ships the adapter only.
"""

from __future__ import annotations

from aimu.aio.channels.base import ChannelMessage
from aimu.aio.channels.web import WebChannel
from aimu.models import StreamChunk, StreamingContentType


class _FakeWS:
    """Captures send_json frames; stands in for a Starlette WebSocket in unit tests."""

    def __init__(self, fail: bool = False):
        self.frames: list[dict] = []
        self.closed = 0
        self._fail = fail

    async def send_json(self, frame):
        if self._fail:
            raise RuntimeError("socket gone")
        self.frames.append(frame)

    async def close(self):
        self.closed += 1


async def test_send_str_flags_proactive():
    ws = _FakeWS()
    channel = WebChannel(ws)
    await channel.send("hello")  # no reply_to -> proactive
    await channel.send("hi back", reply_to=ChannelMessage(text="x"))
    assert ws.frames[0] == {"type": "message", "text": "hello", "proactive": True}
    assert ws.frames[1] == {"type": "message", "text": "hi back", "proactive": False}


async def test_send_stream_emits_tokens_then_done():
    ws = _FakeWS()
    channel = WebChannel(ws)

    async def gen():
        yield StreamChunk(StreamingContentType.GENERATING, "a")
        yield StreamChunk(StreamingContentType.GENERATING, "b")

    await channel.send(gen())
    assert ws.frames == [
        {"type": "token", "text": "a"},
        {"type": "token", "text": "b"},
        {"type": "done"},
    ]


async def test_emits_thinking_and_tool_frames_when_enabled():
    ws = _FakeWS()
    channel = WebChannel(ws, show_thinking=True, show_tools=True)

    async def gen():
        yield StreamChunk(StreamingContentType.THINKING, "hmm")
        yield StreamChunk(StreamingContentType.TOOL_CALLING, {"name": "calc", "arguments": {"x": 2}})
        yield StreamChunk(StreamingContentType.GENERATING, "4")

    await channel.send(gen())
    assert ws.frames == [
        {"type": "thinking", "text": "hmm"},
        {"type": "tool", "name": "calc", "arguments": {"x": 2}},
        {"type": "token", "text": "4"},
        {"type": "done"},
    ]


async def test_skips_empty_generating_chunks():
    # An empty GENERATING chunk (e.g. Ollama's trailing `done` part) must not become a token frame.
    ws = _FakeWS()
    channel = WebChannel(ws)

    async def gen():
        yield StreamChunk(StreamingContentType.GENERATING, "")
        yield StreamChunk(StreamingContentType.GENERATING, "hi")
        yield StreamChunk(StreamingContentType.GENERATING, "")

    await channel.send(gen())
    assert ws.frames == [{"type": "token", "text": "hi"}, {"type": "done"}]


async def test_omits_thinking_and_tool_frames_by_default():
    ws = _FakeWS()
    channel = WebChannel(ws)  # defaults: show_thinking / show_tools off

    async def gen():
        yield StreamChunk(StreamingContentType.THINKING, "hmm")
        yield StreamChunk(StreamingContentType.TOOL_CALLING, {"name": "calc", "arguments": {}})
        yield StreamChunk(StreamingContentType.GENERATING, "4")

    await channel.send(gen())
    assert ws.frames == [{"type": "token", "text": "4"}, {"type": "done"}]


async def test_receive_ends_on_sentinel():
    channel = WebChannel(_FakeWS())
    await channel.feed("hello")
    await channel.feed(None)
    msgs = [m async for m in channel.receive()]
    assert len(msgs) == 1
    assert msgs[0].text == "hello" and msgs[0].channel == "web" and msgs[0].sender == "web"


async def test_send_frame_is_the_subclass_seam():
    # Subclasses add their own frame types via the public send_frame.
    ws = _FakeWS()
    channel = WebChannel(ws)
    await channel.send_frame({"type": "conversations", "items": [{"id": "1"}]})
    assert ws.frames == [{"type": "conversations", "items": [{"id": "1"}]}]


async def test_send_frame_swallows_after_socket_failure():
    # A send error marks the channel closed; later frames are dropped, not raised.
    ws = _FakeWS(fail=True)
    channel = WebChannel(ws)
    await channel.send_frame({"type": "message", "text": "x"})  # fails internally -> closed
    assert channel._closed is True
    await channel.send_frame({"type": "message", "text": "y"})  # no-op, no raise


async def test_aclose_idempotent():
    ws = _FakeWS()
    channel = WebChannel(ws)
    await channel.aclose()
    await channel.aclose()
    assert ws.closed == 1
