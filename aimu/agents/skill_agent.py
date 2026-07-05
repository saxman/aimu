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
    to its system message and surfaces the skills server's tools (via ``MCPClient.as_tools()``)
    through :meth:`_effective_tools`, so the tool-loop engine advertises and dispatches them
    (the model can call ``activate_skill`` to load full skill instructions before proceeding).

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
    _skills_catalog_injected: Optional[str] = field(default=None, init=False, repr=False)

    def _prepare_run(self, deps: Any = None, tool_approval: Any = None) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self._skills_setup_done = False
        super()._prepare_run(deps, tool_approval)
        self._setup_skills()

    def _effective_tools(self, tools: Optional[list] = None) -> list:
        """The agent's tools (or the ``tools=`` override) plus the discovered skill tools.

        Called each round by the tool-loop engine, so a skill authored mid-run via
        :meth:`reload_skills` is advertised and dispatchable on the next round."""
        base = super()._effective_tools(tools)
        existing = {getattr(fn, "__name__", None) for fn in base}
        skills = [t for t in (self._skills_tools or []) if t.__name__ not in existing]
        return base + skills

    def _catalog_instructions(self) -> str:
        """The skill-catalog block appended to the system prompt (empty when no skills)."""
        catalog = self.skill_manager.catalog_prompt()
        if not catalog:
            return ""
        return (
            "\n\n" + catalog + "\n\nWhen a task matches a skill's description, call `activate_skill` "
            "with the skill name to load its full instructions before proceeding."
        )

    def _reinject_catalog(self) -> None:
        """Replace the injected catalog block in the system message in place (no duplication)."""
        base = self.model_client.system_message or ""
        if self._skills_catalog_injected and base.endswith(self._skills_catalog_injected):
            base = base[: -len(self._skills_catalog_injected)]
        new = self._catalog_instructions()
        self.model_client.system_message = base + new
        self._skills_catalog_injected = new

    def _setup_skills(self) -> None:
        if not self.skill_manager.skills:
            return

        # Build the skills tool server and its tool callables once. The MCPClient is held
        # on the instance so its connection outlives this method (the callables also keep
        # a reference). `as_tools()` snapshots the server's tools as plain callables. The
        # tools reach the model via the Agent's ``_effective_tools`` (dispatched by the engine);
        # this method only builds them and injects the catalog.
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
            self._reinject_catalog()

    def reload_skills(self) -> None:
        """Rebuild the skills server from the (refreshed) manager and surface new tools now.

        Re-snapshots the skills tools and re-injects the catalog. Because the tool-loop engine
        re-reads :meth:`_effective_tools` each round, a skill authored mid-run is advertised and
        dispatchable for the rest of the run (and appears in the system prompt without starting a
        fresh conversation). Call after writing a new skill/script (see
        :func:`aimu.skills.make_skill_script_tool`).
        """
        from aimu.skills.mcp import build_skills_server
        from aimu.tools.client import MCPClient

        self._skills_mcp_client = MCPClient(server=build_skills_server(self.skill_manager))
        self._skills_tools = self._skills_mcp_client.as_tools()
        self._reinject_catalog()

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
