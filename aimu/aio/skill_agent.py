"""Async :class:`SkillAgent` — :class:`Agent` extended with skill discovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from aimu.skills.manager import SkillManager

from ._base import AsyncBaseModelClient
from .agent import DEFAULT_CONTINUATION_PROMPT, Agent

logger = logging.getLogger(__name__)


@dataclass
class SkillAgent(Agent):
    """Async :class:`Agent` with filesystem-discovered skill injection.

    On first run (or after a message reset) the SkillAgent appends the skill catalog
    to its system message and attaches an async skills :class:`MCPClient` so the
    model can call ``activate_skill`` to load full skill instructions on demand.
    """

    skill_manager: SkillManager = field(default_factory=SkillManager, repr=False)
    _skills_setup_done: bool = field(default=False, init=False, repr=False)
    _skills_mcp_client: Optional[Any] = field(default=None, init=False, repr=False)

    def _prepare_run(self) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self._skills_setup_done = False
        super()._prepare_run()
        # NOTE: skill setup is async; defer to the first await point in run().

    async def _setup_skills_async(self) -> None:
        if self._skills_setup_done or not self.skill_manager.skills:
            return
        self._skills_setup_done = True

        catalog = self.skill_manager.catalog_prompt()
        instructions = (
            "\n\n" + catalog + "\n\nWhen a task matches a skill's description, call `activate_skill` "
            "with the skill name to load its full instructions before proceeding."
        )
        # Append the skill catalog to the active system prompt. Assigning system_message
        # rewrites the in-history system entry in place when a conversation is already
        # underway (preserving history), or seeds it before the first chat().
        new_system = (self.model_client.system_message or "") + instructions
        self.model_client.system_message = new_system

        if self._skills_mcp_client is None:
            from aimu.aio._mcp_client import MCPClient
            from aimu.skills.mcp import build_skills_server

            skills_server = build_skills_server(self.skill_manager)
            self._skills_mcp_client = await MCPClient.connect(server=skills_server)
        self.model_client.mcp_client = self._skills_mcp_client

    async def run(self, task, generate_kwargs=None, stream=False, images=None):
        # Setup skills (async) before delegating to the parent loop.
        # _prepare_run() is called inside super().run() too, but skill setup is the
        # one async step that must happen first.
        self._prepare_run()
        await self._setup_skills_async()
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)

        # Reproduce super().run() logic but skip the duplicate _prepare_run() call
        response = await self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images)
        for _ in range(self.max_iterations - 1):
            if not self._last_turn_called_tools():
                break
            logger.debug("SkillAgent '%s' continuing.", self.name)
            response = await self.model_client.chat(self.continuation_prompt, generate_kwargs=generate_kwargs)
        self._last_messages = list(self.model_client.messages)
        return response

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: AsyncBaseModelClient) -> "SkillAgent":
        sm = config.get("system_message")
        skill_dirs = config.get("skill_dirs")
        skill_manager = SkillManager(skill_dirs=skill_dirs) if skill_dirs else SkillManager()
        return cls(
            model_client=model_client,
            system_message=sm,
            name=config.get("name"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
            skill_manager=skill_manager,
        )
