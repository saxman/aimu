"""A WebSocket :class:`~aimu.aio.channels.base.Channel` adapter (the WebSocket twin of ``CLIChannel``).

Bridges one browser WebSocket onto the ``Channel`` ABC: an app-side server pump task feeds inbound text
frames into a queue that :meth:`WebChannel.receive` drains, and :meth:`WebChannel.send` relays a finished
string or a streamed reply as JSON frames a browser page renders.

The websocket is duck-typed (it only needs ``async send_json(dict)`` and ``async close()``), so this module
has no hard ``starlette`` import and can be unit-tested with a fake socket. The Starlette server, route, and
HTML page are deliberately **not** part of the library: an app wires its own (see
``examples/personal-assistant/web_assistant.py``).

JSON frame protocol (server -> browser, the contract with any page):

- ``{"type": "message", "text": str, "proactive": bool}`` -- a finished, non-streamed message. ``proactive``
  is true when there is no ``reply_to`` (e.g. a scheduler push), so the page can flag it.
- ``{"type": "token", "text": str}`` -- one chunk of a streamed answer (``GENERATING``).
- ``{"type": "thinking", "text": str}`` -- one chunk of reasoning (``THINKING``); emitted only when
  ``show_thinking`` is set.
- ``{"type": "tool", "name": str, "arguments": dict}`` -- a tool call (``TOOL_CALLING``); emitted only when
  ``show_tools`` is set.
- ``{"type": "done"}`` -- terminates a streamed reply.

Subclasses add their own frame types (e.g. conversation lists, approval prompts) by calling the public
:meth:`WebChannel.send_frame`, the blessed extension seam.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional, Union

from aimu.aio.channels.base import Channel, ChannelMessage
from aimu.models import StreamChunk, StreamingContentType


class WebChannel(Channel):
    """Bridges one browser WebSocket onto the ``Channel`` ABC.

    The app's server pump task calls :meth:`feed` for each inbound frame and ``feed(None)`` on disconnect;
    :meth:`receive` ends on that sentinel, which lets the agent loop tear down cleanly. Replies are sent as
    JSON frames (see the module docstring for the protocol).
    """

    name = "web"

    def __init__(self, websocket: Any, *, show_thinking: bool = False, show_tools: bool = False):
        # websocket is duck-typed (async send_json / close). Tests pass a fake.
        self._ws = websocket
        self._inbound: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._closed = False
        self.show_thinking = show_thinking
        self.show_tools = show_tools

    async def feed(self, text: Optional[str]) -> None:
        """Enqueue an inbound frame; ``None`` is the end-of-stream sentinel."""
        await self._inbound.put(text)

    async def receive(self) -> AsyncIterator[ChannelMessage]:
        while True:
            text = await self._inbound.get()
            if text is None:  # sentinel: the socket closed
                return
            yield ChannelMessage(text=text, sender="web", channel=self.name)

    async def send(
        self,
        content: Union[str, AsyncIterator[StreamChunk]],
        *,
        reply_to: Optional[ChannelMessage] = None,
    ) -> None:
        # A finished string is a single message frame. A proactive push (scheduler) has no
        # reply_to and arrives as a string, so the page can flag it; reactive replies always stream.
        if isinstance(content, str):
            await self.send_frame({"type": "message", "text": content, "proactive": reply_to is None})
            return
        async for chunk in content:
            if chunk.phase == StreamingContentType.GENERATING and chunk.content:
                await self.send_frame({"type": "token", "text": chunk.content})
            elif chunk.phase == StreamingContentType.THINKING and self.show_thinking and chunk.content:
                await self.send_frame({"type": "thinking", "text": chunk.content})
            elif chunk.phase == StreamingContentType.TOOL_CALLING and self.show_tools:
                call = chunk.content if isinstance(chunk.content, dict) else {}
                await self.send_frame({"type": "tool", "name": call.get("name"), "arguments": call.get("arguments")})
        await self.send_frame({"type": "done"})

    async def send_frame(self, frame: dict) -> None:
        """Send one JSON frame to the browser, swallowing errors once the socket has closed.

        The single point through which every frame reaches the socket, so subclasses add their own frame
        types by calling this rather than touching the socket directly. Once closed (e.g. a proactive push
        racing a disconnect), a late frame is dropped instead of crashing the sending task.
        """
        if self._closed:
            return
        try:
            await self._ws.send_json(frame)
        except Exception:
            self._closed = True

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            pass  # already closed by the client / disconnect
