"""Internal ``_AsyncAgenticView`` for ``Agent.as_model_client()``."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union

from aimu.agents._agentic_view import _AgenticViewMixin
from aimu.models.base import StreamChunk

from ._base import AsyncBaseModelClient
from .agent import Agent


class _AsyncAgenticView(_AgenticViewMixin, AsyncBaseModelClient):
    """Internal: an :class:`AsyncBaseModelClient` view backed by an :class:`Agent`.

    Each ``await chat()`` runs the full async agent loop. Returned by
    ``Agent.as_model_client()``; not part of the public API.
    """

    def __init__(self, agent: Agent):
        if not isinstance(agent, Agent):
            raise TypeError(f"_AsyncAgenticView only accepts aio.Agent, got {type(agent).__name__}.")
        self._bind_agent(agent)

    async def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
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
        audio: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        return await self._inner_client._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio)
