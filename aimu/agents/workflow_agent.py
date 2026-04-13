from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, NamedTuple, Optional

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

    Usage::

        wf = WorkflowAgent.from_config(
            [
                {"name": "planner",  "system_message": "Break the task into steps.", "max_iterations": 3},
                {"name": "executor", "system_message": "Execute each step.",         "max_iterations": 10},
            ],
            lambda cfg: OllamaClient(OllamaModel.QWEN_3_8B),
        )
        result = wf.run("Research the top Python web frameworks.")
    """

    agents: list[Agent]
    name: str = "workflow"

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run all agents sequentially and return the final response."""
        result = task
        for step, agent in enumerate(self.agents):
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
        client_factory: Callable[[dict[str, Any]], ModelClient],
    ) -> WorkflowAgent:
        """
        Build a WorkflowAgent from a list of agent config dicts.

        client_factory receives each config dict and must return a configured
        ModelClient. This lets the caller choose client type and model per step.

        Example::

            WorkflowAgent.from_config(configs, lambda cfg: OllamaClient(OllamaModel.QWEN_3_8B))
        """
        from aimu.agents.simple_agent import SimpleAgent

        return cls(agents=[SimpleAgent.from_config(cfg, client_factory(cfg)) for cfg in configs])
