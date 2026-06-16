from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from aimu.agents.agent import Agent, DEFAULT_CONTINUATION_PROMPT
from aimu.models.base import BaseModelClient
from aimu.skills.manager import SkillManager

logger = logging.getLogger(__name__)


@dataclass
class SkillAgent(Agent):
    """An :class:`Agent` extended with filesystem-discovered skill injection.

    On first run (or after a message reset) the SkillAgent appends the skill catalog
    to its system message and adds the skills server's tools (via ``MCPClient.as_tools()``)
    to ``model_client.tools`` so the model can call ``activate_skill`` to load full skill
    instructions before proceeding.

    By default a fresh :class:`SkillManager` is created, scanning the standard search
    paths (``.agents/skills/``, ``.claude/skills/``, ``~/.agents/skills/``,
    ``~/.claude/skills/``). Pass an explicit ``SkillManager`` to override.

    Usage::

        agent = SkillAgent(client, "Use available skills as needed.")
        result = agent.run("Use the pdf-processing skill to extract pages.")

    With explicit skill dirs::

        agent = SkillAgent(client, skill_manager=SkillManager(skill_dirs=["./skills"]))
    """

    skill_manager: SkillManager = field(default_factory=SkillManager, repr=False)
    _skills_setup_done: bool = field(default=False, init=False, repr=False)
    _skills_mcp_client: Optional[Any] = field(default=None, init=False, repr=False)
    _skills_tools: Optional[list] = field(default=None, init=False, repr=False)

    def _prepare_run(self, deps: Any = None) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self._skills_setup_done = False
        super()._prepare_run(deps)
        self._setup_skills()

    def _setup_skills(self) -> None:
        if not self.skill_manager.skills:
            return

        # Build the skills tool server and its tool callables once. The MCPClient is held
        # on the instance so its connection outlives this method (the callables also keep
        # a reference). `as_tools()` snapshots the server's tools as plain callables.
        if self._skills_tools is None:
            from aimu.skills.mcp import build_skills_server
            from aimu.tools.client import MCPClient

            self._skills_mcp_client = MCPClient(server=build_skills_server(self.skill_manager))
            self._skills_tools = self._skills_mcp_client.as_tools()

        # Inject the skill catalog into the system prompt once per fresh conversation.
        # Assigning system_message rewrites the in-history system entry in place when a
        # conversation is already underway (preserving history), or seeds it before the
        # first chat().
        if not self._skills_setup_done:
            self._skills_setup_done = True
            catalog = self.skill_manager.catalog_prompt()
            instructions = (
                "\n\n" + catalog + "\n\nWhen a task matches a skill's description, call `activate_skill` "
                "with the skill name to load its full instructions before proceeding."
            )
            self.model_client.system_message = (self.model_client.system_message or "") + instructions

        # Ensure the skills tools are present for this run. `super()._prepare_run()` may have
        # just reset `model_client.tools` to the agent's configured `self.tools`, so append
        # the skills tools (deduped by name) every run rather than only on first setup.
        existing = {getattr(fn, "__name__", None) for fn in self.model_client.tools}
        self.model_client.tools = list(self.model_client.tools) + [
            t for t in self._skills_tools if t.__name__ not in existing
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: BaseModelClient) -> SkillAgent:
        """Create a SkillAgent from a plain dict config.

        Recognised keys: ``name``, ``system_message``, ``max_iterations``,
        ``continuation_prompt``, ``skill_dirs`` (omit to auto-discover).
        """
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
