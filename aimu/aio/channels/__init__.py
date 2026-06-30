"""Channel transports for personal assistants.

``Channel`` is the uniform transport ABC; ``CLIChannel`` is the stdin/stdout adapter and
``WebChannel`` is the browser-WebSocket adapter (the app supplies the Starlette server + HTML page).
Network adapters (e.g. Telegram) would live here behind optional extras with a ``HAS_*``
guard, mirroring the ``a2a`` pattern.
"""

from aimu.aio.channels.base import Channel, ChannelMessage
from aimu.aio.channels.cli import CLIChannel
from aimu.aio.channels.web import WebChannel

__all__ = ["CLIChannel", "Channel", "ChannelMessage", "WebChannel"]
