from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import BaseModelClient, StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


# Back-compat alias. ChainChunk used to be a separate NamedTuple with a `step` field;
# that role is now played by StreamChunk.iteration.
ChainChunk = StreamChunk


@dataclass
class Chain(Runner):
    """**Prompt Chaining** pattern: agents run sequentially, output → input.

    Each step's text output (concatenated GENERATING chunks) becomes the next step's
    task input. Steps may be :class:`Agent` instances or nested workflows.

    Quick start::

        chain = Chain.from_client(client, [
            "Break the task into 3 concrete steps.",
            "Execute each step and collect results.",
            "Polish the result into a final report.",
        ])
        result = chain.run("Research top Python web frameworks.")

    Direct construction (each step owns its own client/agent)::

        chain = Chain(agents=[
            Agent(client_a, "Planner", name="planner"),
            Agent(client_b, "Executor", name="executor"),
        ])
    """

    agents: list
    name: str = "chain"

    @classmethod
    def from_client(cls, client: BaseModelClient, prompts: list[str], *, name: str = "chain") -> Chain:
        """Build a Chain from a single client and a list of step system_messages.

        Each step gets its own :class:`Agent` with ``reset_messages_on_run=True`` so it
        clears the shared client's history and applies its own system_message.
        """
        from aimu.agents.agent import Agent

        agents = [
            Agent(client, system_message=prompt, name=f"step-{i}", reset_messages_on_run=True)
            for i, prompt in enumerate(prompts)
        ]
        return cls(agents=agents, name=name)

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Run all agents sequentially. ``images`` are forwarded only to the first step."""
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Chain step %d, agent '%s'.", step, agent.name)
            step_images = images if step == 0 else None
            result = agent.run(result, generate_kwargs=generate_kwargs, images=step_images)
        return result

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        result = task
        for step, agent in enumerate(self.agents):
            logger.debug("Chain step %d, agent '%s'.", step, agent.name)
            step_images = images if step == 0 else None
            step_chunks: list[StreamChunk] = []
            for chunk in agent.run(result, generate_kwargs=generate_kwargs, stream=True, images=step_images):
                step_chunks.append(chunk)
                # Re-tag iteration with the chain step so downstream readers can group.
                yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=step)
            result = "".join(c.content for c in step_chunks if c.phase == StreamingContentType.GENERATING)

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        for agent in self.agents:
            result.update(agent.messages)
        return result

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]], client: BaseModelClient) -> Chain:
        """Build a Chain from a list of agent config dicts and a single client.

        Each step's Agent gets ``reset_messages_on_run=True`` so it clears the client's
        messages and applies its own system_message before running.
        """
        from aimu.agents.agent import Agent

        agents = []
        for cfg in configs:
            agent = Agent.from_config(cfg, client)
            agent.reset_messages_on_run = True
            agents.append(agent)
        return cls(agents=agents)
