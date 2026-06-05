from __future__ import annotations

from typing import Any, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.models.base import BaseModelClient, StreamChunk


class _AgenticView(BaseModelClient):
    """Internal: a :class:`BaseModelClient` view backed by an :class:`Agent`.

    Each ``chat()`` runs the full agent loop. Returned by ``Agent.as_model_client()``;
    not part of the public API. Use ``Agent.run()`` directly unless you specifically
    need to drop an agent into an API that expects a ``BaseModelClient``.
    """

    def __init__(self, agent: Agent):
        if not isinstance(agent, Agent):
            raise TypeError(
                f"_AgenticView only accepts Agent, got {type(agent).__name__}. "
                "For workflows (Chain, Router, ...) call their run() / run(stream=True) directly."
            )
        self._agent = agent
        self._inner_client = agent.model_client
        # Mirror base attributes from inner_client (model caps, generate kwargs).
        # super().__init__() is intentionally not called; it would reset inner_client state.
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
    def tools(self) -> list:
        return self._inner_client.tools

    @tools.setter
    def tools(self, value: list) -> None:
        self._inner_client.tools = value

    @property
    def system_message(self) -> Optional[str]:
        return self._inner_client.system_message

    @system_message.setter
    def system_message(self, message: Optional[str]) -> None:
        self._inner_client.system_message = message

    @property
    def last_thinking(self) -> str:
        return self._inner_client.last_thinking

    @last_thinking.setter
    def last_thinking(self, value: str) -> None:
        self._inner_client.last_thinking = value

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        self._inner_client.reset(system_message)

    # --- Agentic overrides ---

    def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        if stream:
            return self._agent.run(user_message, generate_kwargs=generate_kwargs, stream=True, images=images)
        return self._agent.run(user_message, generate_kwargs=generate_kwargs, images=images)

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        # generate() bypasses the agent loop (no message history, no tools).
        return self._inner_client._generate(prompt, generate_kwargs, stream=stream, images=images)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._inner_client._update_generate_kwargs(generate_kwargs)
