"""Internal ``_AsyncAgenticView`` for ``Agent.as_model_client()``."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union

from aimu.models.base import StreamChunk

from ._base import AsyncBaseModelClient
from .agent import Agent


class _AsyncAgenticView(AsyncBaseModelClient):
    """Internal: an :class:`AsyncBaseModelClient` view backed by an :class:`Agent`.

    Each ``await chat()`` runs the full async agent loop. Returned by
    ``Agent.as_model_client()``; not part of the public API.
    """

    def __init__(self, agent: Agent):
        if not isinstance(agent, Agent):
            raise TypeError(f"_AsyncAgenticView only accepts aio.Agent, got {type(agent).__name__}.")
        self._agent = agent
        self._inner_client = agent.model_client
        self.model = self._inner_client.model
        self.model_kwargs = self._inner_client.model_kwargs
        self.default_generate_kwargs = self._inner_client.default_generate_kwargs

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

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return await self._agent.run(user_message, generate_kwargs=generate_kwargs, stream=True, images=images)
        return await self._agent.run(user_message, generate_kwargs=generate_kwargs, images=images)

    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        return await self._inner_client._generate(prompt, generate_kwargs, stream=stream, images=images)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._inner_client._update_generate_kwargs(generate_kwargs)
