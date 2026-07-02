"""Core wiring for the reference personal assistant.

Ties the AIMU primitives together for a single user:

    Channel.receive()  ->  SkillAgent.run()  ->  Channel.send()
                  Scheduler  ->  proactive SkillAgent.run()  ->  Channel.send()
                  ConversationManager persists history across restarts
                  author_skill / add_skill_script let the assistant grow (and run) its own skills

This is a deliberately minimal teaching reference. It wires a fixed, small set of built-in tools
and leaves configurable surfaces (selectable tool groups, remote MCP servers, persistent memory)
to a full application built on AIMU; the library still ships those capabilities, and the how-to
guides show how to add them.

Kept in a ``_*_common.py`` module (per the examples convention) so it is importable and
testable independently of the CLI entry point in ``assistant.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aimu import PROVENANCE_KEY, PROVENANCE_PROACTIVE, aio, paths
from aimu.aio import Channel, RunHandle, Scheduler
from aimu.aio.channels.base import ChannelMessage
from aimu.history import ConversationManager
from aimu.skills import SkillManager, make_skill_authoring_tool, make_skill_script_tool
from aimu.tools import builtin

logger = logging.getLogger(__name__)

# A small, fixed set of AIMU built-in tools so the assistant is useful out of the box: web search /
# weather / wikipedia / fetch (builtin.web) plus date-time and echo (builtin.misc). These are sync;
# the async agent dispatches them via asyncio.to_thread, so no wrapping is needed.
_FIXED_TOOLS = builtin.web + builtin.misc

DEFAULT_SYSTEM_MESSAGE = (
    "You are a personal assistant running on the user's own machine. Be concise and helpful. "
    "When the user teaches you a repeatable procedure worth remembering, call `author_skill` to save "
    "it as a reusable skill; name skills in kebab-case (lowercase words joined by hyphens, e.g. "
    "'weekly-review'), never with underscores or spaces. When a procedure can be automated, call "
    "`add_skill_script` to attach a runnable Python or shell script to a skill; the script becomes a "
    "tool you can run immediately, even in the same turn. If a script fails, fix it by calling "
    "`add_skill_script` again with the SAME filename to overwrite it (a different filename just "
    "creates a duplicate and leaves the broken script). Scripts run with full access to this "
    "machine, so only automate what the user asked for."
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
    # Surface the model's reasoning and tool calls in the channel, not just the final answer.
    show_thinking: bool = True
    show_tools: bool = True


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
        # Each reactive turn runs as a background task (a RunHandle) so the channel reader stays
        # free to receive a `/stop` while a turn is in flight. `_current` is the latest turn (the
        # one `/stop` cancels); `_turns` keeps task refs alive until they finish.
        self._current: Optional[RunHandle] = None
        self._turns: set = set()

    @classmethod
    async def create(cls, config: AssistantConfig, channel: Channel, *, client=None, tool_approval=None) -> "Assistant":
        if client is None:
            client = aio.client(config.model, system=config.system_message)

        manager = SkillManager(skill_dirs=[str(config.skills_dir)])
        author_skill = make_skill_authoring_tool(manager, config.skills_dir)
        # tool_approval is a gate run before each tool call (None -> approve everything). Front ends
        # supply it, since how to confirm with the user is transport-specific (see assistant.py).
        agent = aio.SkillAgent(
            client, tools=[author_skill], skill_manager=manager, name="assistant", tool_approval=tool_approval
        )
        # add_skill_script needs the agent (to reload skills so a new script tool is callable this
        # turn), so it is built after the agent. The fixed built-in tools are appended too; all names
        # are distinct, and the SkillAgent re-appends its skills-server tools each run.
        agent.tools = [
            author_skill,
            make_skill_script_tool(agent, manager, config.skills_dir),
            *_FIXED_TOOLS,
        ]

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
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._serve_channel())
                tg.create_task(self._scheduler.run())
        finally:
            # Cancel any turn still running at shutdown and let the cancellations settle (each turn
            # persists its partial state on stop), so no task is left pending.
            for task in list(self._turns):
                task.cancel()
            if self._turns:
                await asyncio.gather(*self._turns, return_exceptions=True)

    async def _serve_channel(self) -> None:
        try:
            async for msg in self._channel.receive():
                if (msg.text or "").strip().lower() == "/stop":
                    if self._current is not None and not self._current.done:
                        self._current.cancel()
                    continue
                # Start the turn as a background task; the loop keeps reading so a `/stop` can arrive
                # mid-turn. Turns stay serialized by self._lock (a reminder can't interleave).
                handle = RunHandle.start(self._handle(msg))
                self._current = handle
                self._turns.add(handle.task)
                handle.task.add_done_callback(self._turns.discard)
        finally:
            self._scheduler.stop()  # channel closed -> stop the scheduler so run() returns

    async def _handle(self, msg: ChannelMessage) -> None:
        async with self._lock:
            try:
                stream = await self._agent.run(msg.text, stream=True, images=msg.images)
                await self._channel.send(stream, reply_to=msg)
            except asyncio.CancelledError:
                # `/stop` (or shutdown) cancelled this turn. Note it, keep the partial state (the
                # agent snapshots it in a finally), and return so the daemon keeps serving. This
                # deliberate swallow is at the per-turn boundary; the serve loop is unaffected.
                try:
                    await self._channel.send("(stopped)", reply_to=msg)
                except Exception:
                    pass
                self._persist()
                return
            except Exception:
                logger.exception("Error handling message")
                await self._channel.send("Sorry, something went wrong handling that.", reply_to=msg)
            self._persist()

    async def _proactive(self) -> None:
        """Scheduled callback: produce a message unprompted and push it to the channel."""
        async with self._lock:
            # Tag the whole proactive exchange (the framework-injected reminder turn and the
            # assistant's push) so replayed history can distinguish it from a user-driven turn.
            # The agent doesn't reset on run here (system prompt lives on the client), so the
            # pre-run length is a stable start index for this exchange.
            start = len(self._agent.model_client.messages)
            reply = await self._agent.run(self._config.reminder_text)
            for message in self._agent.model_client.messages[start:]:
                message[PROVENANCE_KEY] = PROVENANCE_PROACTIVE
            await self._channel.send(reply)
            self._persist()

    def _persist(self) -> None:
        # Copy each message so the manager's timestamp annotation doesn't leak into the live
        # model-client message dicts.
        self._conversation.update_conversation([dict(m) for m in self._agent.model_client.messages])
