"""Async :class:`SkillAgent`: :class:`Agent` extended with skill discovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from aimu.models.base import StreamChunk
from aimu.skills.manager import SkillManager

from ._base import AsyncBaseModelClient
from .agent import DEFAULT_CONTINUATION_PROMPT, Agent

logger = logging.getLogger(__name__)


@dataclass
class SkillAgent(Agent):
    """Async :class:`Agent` with filesystem-discovered skill injection.

    On first run (or after a message reset) the SkillAgent appends the skill catalog
    to its system message and surfaces the async skills server's tools (via
    ``aio.MCPClient.as_tools()``) through :meth:`_effective_tools`, so the tool-loop engine
    advertises and dispatches them (the model can call ``activate_skill`` on demand).

    ``script_env`` is host context handed to every skill script this agent runs, merged over the
    inherited environment by :func:`aimu.skills.mcp.run_script_file`. Without it the only way to tell a
    script something it cannot discover (where to write output, which account to send from) is a
    process-wide variable, which makes one agent's context every subprocess's context. It is a field
    rather than a ``build_skills_server`` argument the caller passes because this class builds that
    server itself, twice: on first run and again in :meth:`reload_skills`.
    """

    skill_manager: SkillManager = field(default_factory=SkillManager, repr=False)
    script_env: Optional[dict[str, str]] = field(default=None, repr=False)
    _skills_setup_done: bool = field(default=False, init=False, repr=False)
    _skills_mcp_client: Optional[Any] = field(default=None, init=False, repr=False)
    _skills_tools: Optional[list] = field(default=None, init=False, repr=False)
    _skills_catalog_injected: Optional[str] = field(default=None, init=False, repr=False)

    def _prepare_run(self, deps: Any = None, tool_approval: Any = None) -> None:
        if self.reset_messages_on_run or self.system_message is not None:
            self._skills_setup_done = False
        super()._prepare_run(deps, tool_approval)
        # NOTE: skill setup is async; defer to the first await point in run().

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

    def _effective_tools(self, tools: Optional[list] = None) -> list:
        """The agent's tools (or the ``tools=`` override) plus the discovered skill tools.

        Called each round by the tool-loop engine, so a skill authored mid-run via
        :meth:`reload_skills` is advertised and dispatchable on the next round."""
        base = super()._effective_tools(tools)
        existing = {getattr(fn, "__name__", None) for fn in base}
        skills = [t for t in (self._skills_tools or []) if t.__name__ not in existing]
        return base + skills

    async def _setup_skills_async(self) -> None:
        if not self.skill_manager.skills:
            return

        # Build the async skills server + tool callables once. The MCPClient is held on
        # the instance so its connection outlives this method (the callables also hold a
        # reference). `as_tools()` snapshots the server's tools as async callables.
        if self._skills_tools is None:
            from aimu.aio._mcp_client import MCPClient
            from aimu.skills.mcp import build_skills_server

            self._skills_mcp_client = await MCPClient.connect(
                server=build_skills_server(self.skill_manager, env=self.script_env)
            )
            self._skills_tools = await self._skills_mcp_client.as_tools()

        # Inject the skill catalog into the system prompt once per fresh conversation.
        if not self._skills_setup_done:
            self._skills_setup_done = True
            self._reinject_catalog()

    async def reload_skills(self) -> None:
        """Rebuild the skills server from the (refreshed) manager and surface new tools now.

        Re-snapshots the skills tools and re-injects the catalog. Because the tool-loop engine
        re-reads :meth:`_effective_tools` each round, a skill authored mid-run is advertised and
        dispatchable for the rest of the run. Call after writing a new skill/script (see
        :func:`aimu.skills.make_skill_script_tool`).
        """
        from aimu.aio._mcp_client import MCPClient
        from aimu.skills.mcp import build_skills_server

        # Connect the new server first so a failure leaves the old (working) client in place.
        old_client = self._skills_mcp_client
        self._skills_mcp_client = await MCPClient.connect(
            server=build_skills_server(self.skill_manager, env=self.script_env)
        )
        self._skills_tools = await self._skills_mcp_client.as_tools()
        self._reinject_catalog()
        # The old client's callables are no longer returned by _effective_tools; close it. A tool
        # call already in flight from the old snapshot holds its own reference until it completes.
        if old_client is not None:
            await old_client.aclose()

    async def run(
        self,
        task,
        generate_kwargs=None,
        stream=False,
        images=None,
        tools=None,
        deps=None,
        tool_approval=None,
        schema=None,
        thinking=None,
        events=None,
        compaction=None,
    ):
        # Prepare + async skill setup must complete before the loop, and _prepare_run
        # (which resets model_client.tools) must run exactly once, before skills are added.
        # That ordering is why this can't just call super().run() (which re-prepares); instead
        # it prepares, sets up skills, then delegates to Agent's post-prepare loop helpers.
        # ``deps``, ``tool_approval``, ``schema``, ``thinking``, ``events``, and ``compaction``
        # mirror aio.Agent.run(); see that method.
        thinking = thinking if thinking is not None else self.thinking
        events = events if events is not None else self.events
        compaction = compaction if compaction is not None else self.compaction
        self._prepare_run(deps, tool_approval)
        await self._setup_skills_async()

        if schema is not None:
            if stream:
                # Skill setup is already done above; forward the client's structured stream.
                return self._structured_stream_after_setup(task, generate_kwargs, images, schema, thinking, events)
            try:
                # Raw events (unwrapped by _attributing_sink, unlike the tool-loop call
                # sites): this path emits no turn events today (schema= does not currently
                # call _emit_turn_started/_emit_turn_finished), so there is nothing for the
                # wrapper to attribute yet. Swap in self._attributing_sink() here too if that
                # ever changes.
                with self.model_client._events_override(events):
                    return await self.model_client.chat(
                        task, generate_kwargs=generate_kwargs, images=images, schema=schema, thinking=thinking
                    )
            finally:
                self._last_messages = list(self.model_client.messages)

        loop = self._make_tool_loop(tools, deps, tool_approval, thinking, events, compaction)
        if stream:
            return self._run_loop_streamed(loop, task, generate_kwargs, images)

        return await self._run_loop(loop, task, generate_kwargs, images)

    async def _structured_stream_after_setup(self, task, generate_kwargs, images, schema, thinking=None, events=None):
        """Streamed structured-output turn, assuming ``_prepare_run`` + skill setup already ran.

        Mirrors :meth:`aimu.aio.Agent._run_structured_streamed` but does not re-prepare (which
        would reset ``model_client.tools`` and wipe the injected skills)."""
        try:
            # Raw events (unwrapped by _attributing_sink, unlike the tool-loop call sites):
            # this path emits no turn events today (schema= does not currently call
            # _emit_turn_started/_emit_turn_finished), so there is nothing for the wrapper to
            # attribute yet. Swap in self._attributing_sink() here too if that ever changes.
            with self.model_client._events_override(events):
                stream = await self.model_client.chat(
                    task,
                    generate_kwargs=generate_kwargs,
                    stream=True,
                    images=images,
                    schema=schema,
                    thinking=thinking,
                )
                async for chunk in stream:
                    yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=0)
        finally:
            self._last_messages = list(self.model_client.messages)

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
