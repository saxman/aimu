"""A WebSocket Channel adapter for the personal assistant (example-local, not library).

Bridges one browser WebSocket onto AIMU's `Channel` ABC: a server-side pump task feeds inbound
text frames into a queue that `receive()` drains, and `send()` relays a finished string or a
streamed reply as JSON frames the static page renders. Kept separate from the server (web_assistant.py)
so it can be unit-tested in isolation, mirroring `_assistant_common.py` vs `assistant.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional, Union

from aimu.aio.channels.base import Channel, ChannelMessage
from aimu.models import StreamChunk, StreamingContentType


class WebChannel(Channel):
    """Bridges one browser WebSocket onto the Channel ABC.

    The server's pump task calls `feed(text)` for each inbound frame and `feed(None)` on
    disconnect; `receive()` ends on that sentinel, which lets the Assistant loop tear down cleanly.
    Frames sent to the browser are JSON: ``{"type": "message"|"token"|"done", ...}``.
    """

    name = "web"

    def __init__(self, websocket: Any):
        # websocket is a Starlette WebSocket (duck-typed: async send_json / close). Tests pass a fake.
        self._ws = websocket
        self._inbound: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._closed = False

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
            await self._safe_send({"type": "message", "text": content, "proactive": reply_to is None})
            return
        async for chunk in content:
            if chunk.phase == StreamingContentType.GENERATING:
                await self._safe_send({"type": "token", "text": chunk.content})
        await self._safe_send({"type": "done"})

    async def _safe_send(self, frame: dict) -> None:
        # Once closed (e.g. a proactive push racing a disconnect), swallow send errors so a late
        # frame can't crash the scheduler task.
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
