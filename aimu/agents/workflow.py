from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, NamedTuple

from aimu.agents.agent import Agent, AgentChunk
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
class Workflow:
    """
    Chains agents sequentially. The text output of step N becomes the task
    input for step N+1.

    Usage::

        wf = Workflow.from_config(
            [
                {"name": "planner",  "system_message": "Break the task into steps.", "max_iterations": 3},
                {"name": "executor", "system_message": "Execute each step.",         "max_iterations": 10},
            ],
            lambda cfg: OllamaClient(OllamaModel.QWEN_3_8B),
        )
        result = wf.run("Research the top Python web frameworks.")
    """

    agents: list[Agent]

    def run(self, task: str) -> str:
        """Run all agents sequentially and return the final response."""
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Workflow step %d — agent '%s'.", step, agent.name)
            result = agent.run(result)
        return result

    def run_streamed(self, task: str) -> Iterator[WorkflowChunk]:
        """
        Stream all agents sequentially. Yields WorkflowChunk for every chunk
        produced. The text output of each step is accumulated from GENERATING
        chunks and passed as the task to the next step.
        """
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Workflow step %d — agent '%s'.", step, agent.name)
            step_chunks: list[AgentChunk] = []
            for chunk in agent.run_streamed(result):
                step_chunks.append(chunk)
                yield WorkflowChunk(step, chunk.agent_name, chunk.iteration, chunk.phase, chunk.content)
            result = "".join(c.content for c in step_chunks if c.phase == StreamingContentType.GENERATING)

    @classmethod
    def from_config(
        cls,
        configs: list[dict[str, Any]],
        client_factory: Callable[[dict[str, Any]], ModelClient],
    ) -> Workflow:
        """
        Build a Workflow from a list of agent config dicts.

        client_factory receives each config dict and must return a configured
        ModelClient. This lets the caller choose client type and model per step.

        Example::

            Workflow.from_config(configs, lambda cfg: OllamaClient(OllamaModel.QWEN_3_8B))
        """
        return cls(agents=[Agent.from_config(cfg, client_factory(cfg)) for cfg in configs])
