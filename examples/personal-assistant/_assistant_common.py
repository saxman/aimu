"""Core wiring for the reference personal assistant.

Ties the AIMU primitives together for a single user:

    Channel.receive()  ->  SkillAgent.run()  ->  Channel.send()
                  Scheduler  ->  proactive SkillAgent.run()  ->  Channel.send()
                  ConversationManager persists history across restarts
                  author_skill tool lets the assistant grow its own skills

Kept in a ``_*_common.py`` module (per the examples convention) so it is importable and
testable independently of the CLI entry point in ``assistant.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aimu import aio, paths
from aimu.aio import Channel, Scheduler
from aimu.aio.channels.base import ChannelMessage
from aimu.history import ConversationManager
from aimu.skills import SkillManager, make_skill_authoring_tool, make_skill_script_tool

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a personal assistant running on the user's own machine. Be concise and helpful. "
    "When the user teaches you a repeatable procedure worth remembering, call `author_skill` to save "
    "it as a reusable skill. When a procedure can be automated, call `add_skill_script` to attach a "
    "runnable Python or shell script to a skill; the script becomes a tool you can run immediately, "
    "even in the same turn. Scripts run with full access to this machine, so only automate what the "
    "user asked for."
)

DEFAULT_REMINDER_TEXT = "Proactively check in with the user with one short, useful suggestion for their day."

# All of the assistant's persistent state (authored skills + conversation history) defaults
# under one subdirectory of the AIMU output directory, so a run leaves nothing scattered in
# the working directory.
DEFAULT_OUTPUT_DIR = paths.output / "personal-assistant"


@dataclass
class AssistantConfig:
    model: Optional[str] = None
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    skills_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "skills")
    history_path: str = field(default_factory=lambda: str(DEFAULT_OUTPUT_DIR / "history.json"))
    reminder_seconds: Optional[float] = None
    reminder_text: str = DEFAULT_REMINDER_TEXT


class Assistant:
    """A single-user personal assistant wired from AIMU primitives."""

    def __init__(
        self,
        agent: aio.SkillAgent,
        channel: Channel,
        scheduler: Scheduler,
        conversation: ConversationManager,
        config: AssistantConfig,
    ):
        self._agent = agent
        self._channel = channel
        self._scheduler = scheduler
        self._conversation = conversation
        self._config = config
        # The reactive turn and a proactive turn share one agent/client; serialize them so
        # a reminder firing mid-conversation can't interleave on shared message state.
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls, config: AssistantConfig, channel: Channel, *, client=None) -> "Assistant":
        if client is None:
            client = aio.client(config.model, system=config.system_message)

        manager = SkillManager(skill_dirs=[str(config.skills_dir)])
        author_skill = make_skill_authoring_tool(manager, config.skills_dir)
        agent = aio.SkillAgent(client, tools=[author_skill], skill_manager=manager, name="assistant")
        # add_skill_script needs the agent (to reload skills so a new script tool is callable this
        # turn), so it is built after the agent and appended to its tool list.
        agent.tools = [author_skill, make_skill_script_tool(agent, manager, config.skills_dir)]

        conversation = ConversationManager(config.history_path, use_last_conversation=True)
        prior = conversation.messages
        if prior:
            agent.restore(prior)

        scheduler = Scheduler()
        assistant = cls(agent, channel, scheduler, conversation, config)
        if config.reminder_seconds is not None:
            scheduler.at(config.reminder_seconds, assistant._proactive, name="reminder")
        return assistant

    async def run(self) -> None:
        """Serve the channel and run the scheduler concurrently until the channel closes."""
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._serve_channel())
            tg.create_task(self._scheduler.run())

    async def _serve_channel(self) -> None:
        try:
            async for msg in self._channel.receive():
                await self._handle(msg)
        finally:
            self._scheduler.stop()  # channel closed -> stop the scheduler so run() returns

    async def _handle(self, msg: ChannelMessage) -> None:
        async with self._lock:
            try:
                stream = await self._agent.run(msg.text, stream=True, images=msg.images)
                await self._channel.send(stream, reply_to=msg)
            except Exception:
                logger.exception("Error handling message")
                await self._channel.send("Sorry, something went wrong handling that.", reply_to=msg)
            self._persist()

    async def _proactive(self) -> None:
        """Scheduled callback: produce a message unprompted and push it to the channel."""
        async with self._lock:
            reply = await self._agent.run(self._config.reminder_text)
            await self._channel.send(reply)
            self._persist()

    def _persist(self) -> None:
        # Copy each message so the manager's timestamp annotation doesn't leak into the live
        # model-client message dicts.
        self._conversation.update_conversation([dict(m) for m in self._agent.model_client.messages])
