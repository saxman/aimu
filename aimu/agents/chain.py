from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple, Optional

from aimu.agents.base_agent import Workflow, AgentChunk
from aimu.models.base_client import StreamingContentType, ModelClient

logger = logging.getLogger(__name__)


class ChainChunk(NamedTuple):
    """An AgentChunk tagged with the sequential step index within the chain."""

    step: int
    agent_name: str
    iteration: int
    phase: StreamingContentType
    content: str


@dataclass
class Chain(Workflow):
    """
    Workflow pattern: chain agents sequentially (Anthropic's Prompt Chaining pattern).

    The text output of step N becomes the task input for step N+1. Unlike a raw
    prompt chain, AIMU's implementation chains full Agent objects, so each step
    can itself run a multi-iteration agentic loop with tool access.

    The simplest way to build a Chain is ``from_config``: pass a list of config
    dicts and a single ModelClient. Each step is a SimpleAgent with
    ``reset_messages_on_run=True``, so the client's messages are cleared and
    ``system_message`` applied before every step — giving each step a clean slate
    even though they share one client.

    For direct construction (each agent owns its own client)::

        chain = Chain(agents=[
            SimpleAgent(client_a, name="planner"),
            SimpleAgent(client_b, name="executor"),
        ])

    Via from_config (shared client)::

        client = OllamaClient(OllamaModel.QWEN_3_8B)
        client.mcp_client = MCPClient(server=mcp)

        chain = Chain.from_config(
            [
                {"name": "planner",  "system_message": "Break the task into steps.", "max_iterations": 3},
                {"name": "executor", "system_message": "Execute each step.",         "max_iterations": 10},
            ],
            client,
        )
        result = chain.run("Research the top Python web frameworks.")
    """

    agents: list
    name: str = "chain"

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run all agents sequentially and return the final response."""
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Chain step %d — agent '%s'.", step, agent.name)
            result = agent.run(result, generate_kwargs=generate_kwargs)
        return result

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[ChainChunk]:
        """
        Stream all agents sequentially. Yields ChainChunk for every chunk
        produced. The text output of each step is accumulated from GENERATING
        chunks and passed as the task to the next step.
        """
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Chain step %d — agent '%s'.", step, agent.name)
            step_chunks: list[AgentChunk] = []
            for chunk in agent.run_streamed(result, generate_kwargs=generate_kwargs):
                step_chunks.append(chunk)
                yield ChainChunk(step, chunk.agent_name, chunk.iteration, chunk.phase, chunk.content)
            result = "".join(c.content for c in step_chunks if c.phase == StreamingContentType.GENERATING)

    @classmethod
    def from_config(
        cls,
        configs: list[dict[str, Any]],
        client: ModelClient,
    ) -> Chain:
        """
        Build a Chain from a list of agent config dicts and a single ModelClient.

        The client is shared across all steps. Each step's SimpleAgent has
        ``reset_messages_on_run=True`` so it clears the client's messages and
        applies its own ``system_message`` before running, keeping steps isolated.

        Example::

            client = OllamaClient(OllamaModel.QWEN_3_8B)
            client.mcp_client = MCPClient(server=mcp)
            Chain.from_config(configs, client)
        """
        from aimu.agents.simple_agent import SimpleAgent

        agents = []
        for cfg in configs:
            agent = SimpleAgent.from_config(cfg, client)
            agent.reset_messages_on_run = True
            agents.append(agent)
        return cls(agents=agents)
