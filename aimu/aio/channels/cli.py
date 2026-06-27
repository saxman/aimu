"""A terminal (stdin/stdout) channel adapter."""

from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator, Optional, Union

from aimu.aio.channels.base import Channel, ChannelMessage
from aimu.models import StreamChunk, StreamingContentType


class CLIChannel(Channel):
    """Read user input from stdin and write replies to stdout.

    Single local user, so ``send(reply_to=...)`` is accepted and ignored. Streaming replies
    are written token-by-token as they arrive.
    """

    name = "cli"

    def __init__(self, *, prompt: str = "> "):
        self.prompt = prompt

    async def receive(self) -> AsyncIterator[ChannelMessage]:
        while True:
            # stdin.readline() blocks; run it off the event loop so the scheduler keeps
            # ticking. A blocked readline cannot be cancelled, so shutdown relies on EOF
            # (Ctrl-D) or process exit -- see the module note in scheduler.py.
            sys.stdout.write(self.prompt)
            sys.stdout.flush()
            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "":  # EOF
                return
            text = line.strip()
            if text:
                yield ChannelMessage(text=text, sender="cli", channel=self.name)

    async def send(
        self,
        content: Union[str, AsyncIterator[StreamChunk]],
        *,
        reply_to: Optional[ChannelMessage] = None,
    ) -> None:
        if isinstance(content, str):
            print(content, flush=True)
            return
        wrote = False
        async for chunk in content:
            if chunk.phase == StreamingContentType.GENERATING:
                sys.stdout.write(chunk.content)
                sys.stdout.flush()
                wrote = True
        if wrote:
            sys.stdout.write("\n")
            sys.stdout.flush()
