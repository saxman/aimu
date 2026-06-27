"""Channel transport ABC for personal assistants.

A :class:`Channel` is the seam between a user-facing transport (a terminal, a chat
platform) and an assistant loop. It is deliberately tiny -- receive inbound messages, send
replies -- so that a network adapter (Telegram, Slack) is a drop-in subclass and heavy
SDK dependencies stay out of core.

It is a sibling of the other AIMU uniform interfaces (``AsyncRunner``,
``AsyncBaseModelClient``, ``MemoryStore``): one ABC, pluggable implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Union

from aimu.models import StreamChunk


@dataclass
class ChannelMessage:
    """A transport-level message in or out of a channel.

    Distinct from LLM conversation state (which stays ``list[dict]`` in OpenAI format):
    ``text`` and ``images`` map directly onto ``agent.run(task, images=...)``.

    Attributes:
        text: The message text.
        sender: Adapter-defined sender id (a chat id, ``"cli"``); used by ``send(reply_to=)``.
        channel: Adapter name, e.g. ``"cli"`` or ``"telegram"``.
        images: Optional images to forward to a vision-capable agent.
        metadata: Raw adapter payload, opaque to the assistant loop.
    """

    text: str
    sender: Optional[str] = None
    channel: Optional[str] = None
    images: Optional[list] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Channel(ABC):
    """Async transport: receive inbound messages, send replies.

    Subclasses that own a live connection (network adapters) should expose an async
    ``connect()`` classmethod factory, mirroring ``aimu.aio.MCPClient.connect``.
    """

    name: str = "channel"

    @abstractmethod
    def receive(self) -> AsyncIterator[ChannelMessage]:
        """Yield inbound messages until the channel closes (an async generator)."""
        ...

    @abstractmethod
    async def send(
        self,
        content: Union[str, AsyncIterator[StreamChunk]],
        *,
        reply_to: Optional[ChannelMessage] = None,
    ) -> None:
        """Send a reply.

        ``content`` is either a finished string or an ``AsyncIterator[StreamChunk]`` to relay
        incrementally. ``reply_to`` carries routing for multi-recipient adapters (which chat
        to answer); single-user adapters like the CLI ignore it.
        """
        ...

    async def aclose(self) -> None:
        """Release any resources. Default no-op so simple adapters need not override."""
        return None
