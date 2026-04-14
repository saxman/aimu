from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from aimu.agents.simple_agent import SimpleAgent, DEFAULT_CONTINUATION_PROMPT
from aimu.models.base_client import ModelClient
from aimu.skills.manager import SkillManager

logger = logging.getLogger(__name__)


@dataclass
class SkillAgent(SimpleAgent):
    """
    A SimpleAgent extended with skill discovery and injection.

    On the first run (or after a message reset), SkillAgent appends the skill
    catalog to the system message and attaches a skills MCPClient so the model
    can call ``activate_skill`` to load full skill instructions before proceeding.

    By default a ``SkillManager()`` is created, which scans the standard search
    paths (``.agents/skills/``, ``.claude/skills/``, and their ``~/`` equivalents).
    Pass an explicit ``SkillManager`` to use specific directories.

    Usage::

        agent = SkillAgent(client, name="assistant")
        result = agent.run("Use the pdf-processing skill to extract pages.")

    With explicit skill dirs::

        agent = SkillAgent(
            client,
            skill_manager=SkillManager(skill_dirs=["./skills"]),
        )

    From config::

        agent = SkillAgent.from_config(
            {"name": "helper", "skill_dirs": ["./skills"]},
            client,
        )
    """

    skill_manager: SkillManager = field(default_factory=SkillManager, repr=False)
    _skills_setup_done: bool = field(default=False, init=False, repr=False)
    _skills_mcp_client: Optional[Any] = field(default=None, init=False, repr=False)

    def _prepare_run(self) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self._skills_setup_done = False
        super()._prepare_run()
        self._setup_skills()

    def _setup_skills(self) -> None:
        """Inject the skill catalog into the system message and attach a skills MCPClient. Called once per run."""
        if self._skills_setup_done or not self.skill_manager.skills:
            return
        self._skills_setup_done = True

        catalog = self.skill_manager.catalog_prompt()
        instructions = (
            "\n\n" + catalog + "\n\nWhen a task matches a skill's description, call `activate_skill` "
            "with the skill name to load its full instructions before proceeding."
        )
        self.model_client.system_message = (self.model_client.system_message or "") + instructions

        if self._skills_mcp_client is None:
            from aimu.skills.mcp import build_skills_server
            from aimu.tools.client import MCPClient

            skills_server = build_skills_server(self.skill_manager)
            self._skills_mcp_client = MCPClient(server=skills_server)
        self.model_client.mcp_client = self._skills_mcp_client

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: ModelClient) -> SkillAgent:
        """
        Create a SkillAgent from a plain dict config.

        Recognised keys:
            name (str)              — agent identifier
            system_message (str)    — applied before each run
            max_iterations (int)    — max tool-call rounds (default 10)
            continuation_prompt (str)
            skill_dirs (list[str])  — explicit skill search paths; omit to auto-discover
        """
        sm = config.get("system_message")
        if sm is not None:
            model_client.system_message = sm

        skill_dirs = config.get("skill_dirs")
        skill_manager = SkillManager(skill_dirs=skill_dirs) if skill_dirs else SkillManager()

        return cls(
            model_client=model_client,
            name=config.get("name", "agent"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
            system_message=sm,
            skill_manager=skill_manager,
        )
