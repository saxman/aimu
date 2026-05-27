"""Async Router workflow (routing pattern)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Union

from aimu.agents.base import MessageHistory
from aimu.models.base import StreamChunk, StreamingContentType

from .._base import AsyncBaseModelClient
from ..agent import Agent, AsyncRunner

logger = logging.getLogger(__name__)


@dataclass
class Router(AsyncRunner):
    """Async routing: classify the task, dispatch to a specialist handler."""

    routing_agent: Agent
    handlers: dict[str, AsyncRunner]
    name: str = "router"
    fallback: Optional[AsyncRunner] = None

    @classmethod
    def from_client(
        cls,
        client: AsyncBaseModelClient,
        classifier_prompt: str,
        handlers: dict[str, AsyncRunner],
        *,
        fallback: Optional[AsyncRunner] = None,
        name: str = "router",
    ) -> Router:
        routing_agent = Agent(
            client,
            system_message=classifier_prompt,
            name="router-classifier",
            reset_messages_on_run=True,
        )
        return cls(routing_agent=routing_agent, handlers=handlers, fallback=fallback, name=name)

    async def _classify(self, task: str, generate_kwargs: Optional[dict[str, Any]]) -> str:
        route = await self.routing_agent.run(task, generate_kwargs=generate_kwargs)
        return route.strip().lower()

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        route = await self._classify(task, generate_kwargs)
        handler = self.handlers.get(route, self.fallback)
        if handler is None:
            raise ValueError(f"No handler for route '{route}' and no fallback set.")
        logger.debug("Router '%s' dispatching to '%s'.", self.name, route)
        return await handler.run(task, generate_kwargs=generate_kwargs, images=images)

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        route_parts: list[str] = []
        route_stream = await self.routing_agent.run(task, generate_kwargs=generate_kwargs, stream=True)
        async for chunk in route_stream:
            yield chunk
            if chunk.phase == StreamingContentType.GENERATING:
                route_parts.append(chunk.content)
        route = "".join(route_parts).strip().lower()
        handler = self.handlers.get(route, self.fallback)
        if handler is None:
            raise ValueError(f"No handler for route '{route}' and no fallback set.")
        logger.debug("Router '%s' dispatching to '%s'.", self.name, route)
        handler_stream = await handler.run(task, generate_kwargs=generate_kwargs, stream=True, images=images)
        async for chunk in handler_stream:
            yield chunk

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        result.update(self.routing_agent.messages)
        for handler in self.handlers.values():
            result.update(handler.messages)
        if self.fallback:
            result.update(self.fallback.messages)
        return result

    @classmethod
    def from_config(
        cls,
        routing_config: dict[str, Any],
        handler_configs: dict[str, dict[str, Any]],
        client: AsyncBaseModelClient,
        fallback_config: Optional[dict[str, Any]] = None,
    ) -> Router:
        routing_agent = Agent.from_config(routing_config, client)
        handlers: dict[str, AsyncRunner] = {
            route: Agent.from_config(cfg, client) for route, cfg in handler_configs.items()
        }
        fallback: Optional[AsyncRunner] = Agent.from_config(fallback_config, client) if fallback_config else None
        return cls(routing_agent=routing_agent, handlers=handlers, fallback=fallback)
