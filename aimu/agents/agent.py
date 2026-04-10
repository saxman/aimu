from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Optional

from aimu.models.base_client import StreamingContentType, ModelClient

if TYPE_CHECKING:
    from aimu.skills.manager import SkillManager

logger = logging.getLogger(__name__)

DEFAULT_CONTINUATION_PROMPT = "Continue working on the task using available tools as needed."


class AgentChunk(NamedTuple):
    """A chunk tagged with the agent name and loop iteration number."""

    agent_name: str
    iteration: int
    phase: StreamingContentType
    content: Any  # str for THINKING/GENERATING; dict {"name", "response"} for TOOL_CALLING


@dataclass
class Agent:
    """
    Wraps a ModelClient with an agentic loop.

    On each iteration, the agent calls model_client.chat(). If the model invoked
    tools during that turn, the agent sends continuation_prompt and loops again.
    The loop stops when the model produces a response without calling any tools,
    or when max_iterations is reached.

    Usage::

        client = OllamaClient(OllamaModel.QWEN_3_8B)
        client.mcp_client = MCPClient(MCP_SERVERS)
        agent = Agent(client, name="researcher", max_iterations=8)
        result = agent.run("Summarise the files in /tmp.")

    From config::

        agent = Agent.from_config(
            {"name": "helper", "system_message": "Use tools.", "max_iterations": 5},
            client,
        )
    """

    model_client: ModelClient
    name: str = "agent"
    max_iterations: int = 10
    continuation_prompt: str = field(default=DEFAULT_CONTINUATION_PROMPT)
    skill_manager: Optional["SkillManager"] = field(default=None, repr=False)
    _skills_setup_done: bool = field(default=False, init=False, repr=False)

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run the agentic loop synchronously and return the final response."""
        self._setup_skills()
        response = self.model_client.chat(task, generate_kwargs=generate_kwargs)

        for _ in range(self.max_iterations - 1):
            if not self._last_turn_called_tools():
                break
            logger.debug("Agent '%s' continuing — tools were used in last turn.", self.name)
            response = self.model_client.chat(self.continuation_prompt, generate_kwargs=generate_kwargs)

        return response

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        """
        Stream the agentic loop, yielding AgentChunk for every StreamChunk produced
        across all iterations. AgentChunk.iteration indicates which loop round a
        chunk belongs to.
        """
        self._setup_skills()
        iteration = 0
        for chunk in self.model_client.chat_streamed(task, generate_kwargs=generate_kwargs):
            yield AgentChunk(self.name, iteration, chunk.phase, chunk.content)

        iteration += 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            for chunk in self.model_client.chat_streamed(self.continuation_prompt, generate_kwargs=generate_kwargs):
                yield AgentChunk(self.name, iteration, chunk.phase, chunk.content)
            iteration += 1

    def _setup_skills(self) -> None:
        """Inject the skill catalog into the system message and attach a skills MCPClient. Called once per run."""
        if self._skills_setup_done or self.skill_manager is None:
            return
        self._skills_setup_done = True

        if not self.skill_manager.skills:
            return

        catalog = self.skill_manager.catalog_prompt()
        instructions = (
            "\n\n" + catalog + "\n\nWhen a task matches a skill's description, call `activate_skill` "
            "with the skill name to load its full instructions before proceeding."
        )
        self.model_client.system_message = (self.model_client.system_message or "") + instructions

        from aimu.skills.mcp import build_skills_server
        from aimu.tools.client import MCPClient

        skills_server = build_skills_server(self.skill_manager)
        self.model_client.mcp_client = MCPClient(server=skills_server)

    def _last_turn_called_tools(self) -> bool:
        """
        Return True if the most recent agent turn included tool invocations.

        Scans self.model_client.messages in reverse, stopping at the last user
        message. Returns True if any 'tool' role message is found in that turn.
        """
        for msg in reversed(self.model_client.messages):
            if msg.get("role") == "user":
                return False
            if msg.get("role") == "tool":
                return True
        return False

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: ModelClient) -> Agent:
        """
        Create an Agent from a plain dict config.

        Recognised keys:
            name (str)              — agent identifier
            system_message (str)    — applied to model_client.system_message
            max_iterations (int)    — max tool-call rounds (default 10)
            continuation_prompt (str)
            skill_dirs (list[str])  — custom skill search paths; omit to use defaults
            use_skills (bool)       — enable skill discovery with default paths (default False)
        """
        if "system_message" in config:
            model_client.system_message = config["system_message"]

        skill_manager = None
        if "skill_dirs" in config or config.get("use_skills", False):
            from aimu.skills.manager import SkillManager

            skill_manager = SkillManager(skill_dirs=config.get("skill_dirs"))

        return cls(
            model_client=model_client,
            name=config.get("name", "agent"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
            skill_manager=skill_manager,
        )
