from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, NamedTuple, Optional

from aimu.agents.base_agent import Agent, AgentChunk
from aimu.models.base_client import StreamingContentType, ModelClient

logger = logging.getLogger(__name__)


class WorkflowChunk(NamedTuple):
    """An AgentChunk tagged with the sequential step index within the workflow."""

    step: int
    agent_name: str
    iteration: int
    phase: StreamingContentType
    content: str


@dataclass
class WorkflowAgent(Agent):
    """
    Chains agents sequentially. The text output of step N becomes the task
    input for step N+1.

    The simplest way to build a WorkflowAgent is from a list of config dicts and
    a single ModelClient. The client is reused across all steps; before each step
    its messages are cleared and system_message is set from the step's config.

    Usage::

        client = OllamaClient(OllamaModel.QWEN_3_8B)
        client.mcp_client = MCPClient(server=mcp)

        wf = WorkflowAgent.from_config(
            [
                {"name": "planner",  "system_message": "Break the task into steps.", "max_iterations": 3},
                {"name": "executor", "system_message": "Execute each step.",         "max_iterations": 10},
            ],
            client,
        )
        result = wf.run("Research the top Python web frameworks.")
    """

    agents: list[Agent]
    name: str = "workflow"
    # Populated by from_config; None when agents were constructed externally.
    _shared_client: Optional[ModelClient] = field(default=None, init=False, repr=False)
    _step_system_messages: list[Optional[str]] = field(default_factory=list, init=False, repr=False)

    def _prepare_step(self, step: int) -> None:
        """Reset the shared client before a step. No-op when agents own their clients."""
        if self._shared_client is None:
            return
        self._shared_client.messages = []
        msg = self._step_system_messages[step] if step < len(self._step_system_messages) else None
        self._shared_client.system_message = msg

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run all agents sequentially and return the final response."""
        result = task
        for step, agent in enumerate(self.agents):
            self._prepare_step(step)
            logger.debug("Workflow step %d — agent '%s'.", step, agent.name)
            result = agent.run(result, generate_kwargs=generate_kwargs)
        return result

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[WorkflowChunk]:
        """
        Stream all agents sequentially. Yields WorkflowChunk for every chunk
        produced. The text output of each step is accumulated from GENERATING
        chunks and passed as the task to the next step.
        """
        result = task
        for step, agent in enumerate(self.agents):
            self._prepare_step(step)
            logger.debug("Workflow step %d — agent '%s'.", step, agent.name)
            step_chunks: list[AgentChunk] = []
            for chunk in agent.run_streamed(result, generate_kwargs=generate_kwargs):
                step_chunks.append(chunk)
                yield WorkflowChunk(step, chunk.agent_name, chunk.iteration, chunk.phase, chunk.content)
            result = "".join(c.content for c in step_chunks if c.phase == StreamingContentType.GENERATING)

    @classmethod
    def from_config(
        cls,
        configs: list[dict[str, Any]],
        client: ModelClient,
    ) -> WorkflowAgent:
        """
        Build a WorkflowAgent from a list of agent config dicts and a single ModelClient.

        The client is shared across all steps. Before each step runs, its ``messages``
        are cleared and ``system_message`` is set from the step's config, so steps
        remain isolated from one another.

        Example::

            client = OllamaClient(OllamaModel.QWEN_3_8B)
            client.mcp_client = MCPClient(server=mcp)
            WorkflowAgent.from_config(configs, client)
        """
        from aimu.agents.simple_agent import SimpleAgent

        # Extract system_message per step; exclude it from construction so that
        # _prepare_step can apply the correct value at run time.
        system_messages = [cfg.get("system_message") for cfg in configs]
        configs_without_sm = [{k: v for k, v in cfg.items() if k != "system_message"} for cfg in configs]
        agents = [SimpleAgent.from_config(cfg, client) for cfg in configs_without_sm]

        wf = cls(agents=agents)
        wf._shared_client = client
        wf._step_system_messages = system_messages
        return wf
