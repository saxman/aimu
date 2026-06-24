"""Async :class:`SkillAgent`: :class:`Agent` extended with skill discovery."""

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
    to its system message and adds the async skills server's tools (via
    ``aio.MCPClient.as_tools()``) to ``model_client.tools`` so the model can call
    ``activate_skill`` to load full skill instructions on demand.
    """

    skill_manager: SkillManager = field(default_factory=SkillManager, repr=False)
    _skills_setup_done: bool = field(default=False, init=False, repr=False)
    _skills_mcp_client: Optional[Any] = field(default=None, init=False, repr=False)
    _skills_tools: Optional[list] = field(default=None, init=False, repr=False)

    def _prepare_run(self, deps: Any = None) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self._skills_setup_done = False
        super()._prepare_run(deps)
        # NOTE: skill setup is async; defer to the first await point in run().

    async def _setup_skills_async(self) -> None:
        if not self.skill_manager.skills:
            return

        # Build the async skills server + tool callables once. The MCPClient is held on
        # the instance so its connection outlives this method (the callables also hold a
        # reference). `as_tools()` snapshots the server's tools as async callables.
        if self._skills_tools is None:
            from aimu.aio._mcp_client import MCPClient
            from aimu.skills.mcp import build_skills_server

            self._skills_mcp_client = await MCPClient.connect(server=build_skills_server(self.skill_manager))
            self._skills_tools = await self._skills_mcp_client.as_tools()

        # Inject the skill catalog into the system prompt once per fresh conversation.
        if not self._skills_setup_done:
            self._skills_setup_done = True
            catalog = self.skill_manager.catalog_prompt()
            instructions = (
                "\n\n" + catalog + "\n\nWhen a task matches a skill's description, call `activate_skill` "
                "with the skill name to load its full instructions before proceeding."
            )
            self.model_client.system_message = (self.model_client.system_message or "") + instructions

        # Ensure the skills tools are present for this run (`_prepare_run` may have reset
        # `model_client.tools` to the configured `self.tools`); dedupe by name.
        existing = {getattr(fn, "__name__", None) for fn in self.model_client.tools}
        self.model_client.tools = list(self.model_client.tools) + [
            t for t in self._skills_tools if t.__name__ not in existing
        ]

    async def run(self, task, generate_kwargs=None, stream=False, images=None, tools=None, deps=None, schema=None):
        # Prepare + async skill setup must complete before the loop, and _prepare_run
        # (which resets model_client.tools) must run exactly once, before skills are added.
        # That ordering is why this can't just call super().run() (which re-prepares); instead
        # it prepares, sets up skills, then delegates to Agent's post-prepare loop helpers.
        # ``deps`` and ``schema`` mirror aio.Agent.run(); see that method for full semantics.
        if schema is not None and stream:
            raise ValueError("schema= and stream=True are mutually exclusive (a typed object can't be streamed).")

        self._prepare_run(deps)
        await self._setup_skills_async()

        if schema is not None:
            result = await self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, schema=schema)
            self._last_messages = list(self.model_client.messages)
            return result

        if stream:
            return self._run_loop_streamed(task, generate_kwargs, images=images, tools=tools)

        return await self._run_loop(task, generate_kwargs, images=images, tools=tools)

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
