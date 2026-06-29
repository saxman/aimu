"""A terminal (stdin/stdout) channel adapter."""

from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator, Optional, Union

from aimu.aio.channels.base import Channel, ChannelMessage
from aimu.models import StreamChunk, StreamingContentType


def _format_tool_call(content) -> str:
    """Render a TOOL_CALLING chunk's ``{"name", "arguments", ...}`` as ``name(arg=value, ...)``."""
    if not isinstance(content, dict):
        return str(content)
    name = content.get("name", "tool")
    args = content.get("arguments")
    if isinstance(args, dict):
        rendered = ", ".join(f"{k}={_truncate(repr(v))}" for k, v in args.items())
    elif args:
        rendered = _truncate(str(args))
    else:
        rendered = ""
    return f"{name}({rendered})"


def _truncate(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


class CLIChannel(Channel):
    """Read user input from stdin and write replies to stdout.

    Single local user, so ``send(reply_to=...)`` is accepted and ignored. Streaming replies
    are written token-by-token as they arrive. ``show_thinking`` / ``show_tools`` (both off by
    default, to keep the bare channel minimal) surface the model's reasoning and tool calls as
    labelled lines alongside the answer.
    """

    name = "cli"

    def __init__(self, *, prompt: str = "> ", show_thinking: bool = False, show_tools: bool = False):
        self.prompt = prompt
        self.show_thinking = show_thinking
        self.show_tools = show_tools

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
        # Track the current block so phase changes (thinking -> tool -> answer) are separated by
        # newlines. `section` is "thinking", "text", or None (nothing open / a tool line just closed).
        section = None
        async for chunk in content:
            if chunk.phase == StreamingContentType.THINKING:
                if not self.show_thinking or not chunk.content:
                    continue
                if section != "thinking":
                    sys.stdout.write("\n[thinking] ")
                    section = "thinking"
                sys.stdout.write(chunk.content)
                sys.stdout.flush()
            elif chunk.phase == StreamingContentType.TOOL_CALLING:
                if not self.show_tools:
                    continue
                sys.stdout.write(("\n" if section else "") + f"[tool] {_format_tool_call(chunk.content)}\n")
                sys.stdout.flush()
                section = None
            elif chunk.phase == StreamingContentType.GENERATING:
                if not chunk.content:
                    continue
                if section == "thinking":
                    sys.stdout.write("\n")
                section = "text"
                sys.stdout.write(chunk.content)
                sys.stdout.flush()
        if section:  # close a trailing thinking/answer block with a newline
            sys.stdout.write("\n")
            sys.stdout.flush()
