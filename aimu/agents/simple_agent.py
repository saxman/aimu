from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from aimu.agents.base_agent import Agent, AgentChunk, MessageHistory
from aimu.models.base_client import StreamingContentType, ModelClient

logger = logging.getLogger(__name__)

DEFAULT_CONTINUATION_PROMPT = "Continue working on the task using available tools as needed. If you have the answer and don't need to use any more tools, just provide the final response."


@dataclass
class SimpleAgent(Agent):
    """
    Wraps a ModelClient with an agentic loop.

    On each iteration, the agent calls model_client.chat(). If the model invoked
    tools during that turn, the agent sends continuation_prompt and loops again.
    The loop stops when the model produces a response without calling any tools,
    or when max_iterations is reached.

    When ``system_message`` is set or ``reset_messages_on_run`` is True, the agent
    clears ``model_client.messages`` and re-applies ``system_message`` before every
    run. This enables clean state isolation when a client is shared across agents
    (e.g. inside an Chain).

    Usage::

        client = OllamaClient(OllamaModel.QWEN_3_8B)
        client.mcp_client = MCPClient(MCP_SERVERS)
        agent = SimpleAgent(client, name="researcher", max_iterations=8)
        result = agent.run("Summarise the files in /tmp.")

    From config::

        agent = SimpleAgent.from_config(
            {"name": "helper", "system_message": "Use tools.", "max_iterations": 5},
            client,
        )
    """

    model_client: ModelClient
    name: str = "agent"
    max_iterations: int = 10
    continuation_prompt: str = field(default=DEFAULT_CONTINUATION_PROMPT)
    system_message: Optional[str] = field(default=None)
    reset_messages_on_run: bool = field(default=False)
    _last_messages: list = field(default_factory=list, init=False, repr=False)

    def _prepare_run(self) -> None:
        """Reset client state and re-apply system_message before a run, when configured."""
        if self.reset_messages_on_run or self.system_message is not None:
            self.model_client.messages = []
        if self.system_message is not None:
            self.model_client.system_message = self.system_message

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run the agentic loop synchronously and return the final response."""
        self._prepare_run()
        response = self.model_client.chat(task, generate_kwargs=generate_kwargs)

        for _ in range(self.max_iterations - 1):
            if not self._last_turn_called_tools():
                break
            logger.debug("Agent '%s' continuing — tools were used in last turn.", self.name)
            response = self.model_client.chat(self.continuation_prompt, generate_kwargs=generate_kwargs)

        self._last_messages = list(self.model_client.messages)
        return response

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        """
        Stream the agentic loop, yielding AgentChunk for every StreamChunk produced
        across all iterations. AgentChunk.iteration indicates which loop round a
        chunk belongs to.
        """
        self._prepare_run()
        iteration = 0
        for chunk in self.model_client.chat_streamed(task, generate_kwargs=generate_kwargs):
            yield AgentChunk(self.name, iteration, chunk.phase, chunk.content)

        iteration += 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            for chunk in self.model_client.chat_streamed(self.continuation_prompt, generate_kwargs=generate_kwargs):
                yield AgentChunk(self.name, iteration, chunk.phase, chunk.content)
            iteration += 1

        self._last_messages = list(self.model_client.messages)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

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
    def from_config(cls, config: dict[str, Any], model_client: ModelClient) -> SimpleAgent:
        """
        Create a SimpleAgent from a plain dict config.

        Recognised keys:
            name (str)              — agent identifier
            system_message (str)    — applied to model_client.system_message at construction
                                      and stored for re-application before each run
            max_iterations (int)    — max tool-call rounds (default 10)
            continuation_prompt (str)
        """
        sm = config.get("system_message")
        if sm is not None:
            model_client.system_message = sm

        return cls(
            model_client=model_client,
            name=config.get("name", "agent"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
            system_message=sm,
        )
