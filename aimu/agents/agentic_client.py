from __future__ import annotations

from typing import Any, Iterator, Optional

from aimu.agents.agent import Agent
from aimu.models.base_client import ModelClient, StreamChunk


class AgenticModelClient(ModelClient):
    """
    A ModelClient whose chat() runs the full Agent agentic loop — looping
    until the model stops calling tools — rather than a single model turn.

    Drop-in replacement anywhere a ModelClient is accepted.

    Usage::

        inner = OllamaClient(OllamaModel.QWEN_3_8B)
        inner.mcp_client = MCPClient(MCP_SERVERS)
        agent = Agent(inner, max_iterations=8)
        client = AgenticModelClient(agent)
        client.chat("Research the top Python web frameworks.")  # loops until done
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._inner_client = agent.model_client
        # Mirror base attributes from inner_client (model caps, generate kwargs).
        # super().__init__() is intentionally not called — it would reset inner_client
        # state (messages, mcp_client, etc.) to defaults.
        self.model = self._inner_client.model
        self.model_kwargs = self._inner_client.model_kwargs
        self.default_generate_kwargs = self._inner_client.default_generate_kwargs

    # --- Delegate mutable state to inner_client so both stay in sync ---

    @property
    def messages(self) -> list[dict]:
        return self._inner_client.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._inner_client.messages = value

    @property
    def mcp_client(self):
        return self._inner_client.mcp_client

    @mcp_client.setter
    def mcp_client(self, value) -> None:
        self._inner_client.mcp_client = value

    @property
    def system_message(self) -> Optional[str]:
        return self._inner_client.system_message

    @system_message.setter
    def system_message(self, message: str) -> None:
        self._inner_client.system_message = message

    @property
    def last_thinking(self) -> str:
        return self._inner_client.last_thinking

    @last_thinking.setter
    def last_thinking(self, value: str) -> None:
        self._inner_client.last_thinking = value

    # --- Agentic overrides ---

    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        """Run the full Agent loop; returns only when the model stops calling tools."""
        return self._agent.run(user_message, generate_kwargs=generate_kwargs)

    def chat_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
    ) -> Iterator[StreamChunk]:
        """Stream the Agent loop, adapting AgentChunk → StreamChunk."""
        for chunk in self._agent.run_streamed(user_message, generate_kwargs=generate_kwargs):
            yield StreamChunk(chunk.phase, chunk.content)

    # --- Pass-through for stateless generation ---

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        return self._inner_client.generate(prompt, generate_kwargs)

    def generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        include_thinking: bool = True,
    ) -> Iterator[StreamChunk]:
        return self._inner_client.generate_streamed(prompt, generate_kwargs, include_thinking)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._inner_client._update_generate_kwargs(generate_kwargs)
