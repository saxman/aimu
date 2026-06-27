"""Channel transports for personal assistants.

``Channel`` is the uniform transport ABC; ``CLIChannel`` is the stdin/stdout adapter.
Network adapters (e.g. Telegram) would live here behind optional extras with a ``HAS_*``
guard, mirroring the ``a2a`` pattern.
"""

from aimu.aio.channels.base import Channel, ChannelMessage
from aimu.aio.channels.cli import CLIChannel

__all__ = ["CLIChannel", "Channel", "ChannelMessage"]
