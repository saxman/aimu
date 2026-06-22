from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import BaseModelClient, StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


@dataclass
class Router(Runner):
    """**Routing** pattern: classify the task, dispatch to a specialist.

    The routing_agent receives the task and must respond with a single route name.
    The Router dispatches to the matching handler (case-insensitive, whitespace-stripped).
    Handlers may be any :class:`Runner` (agent or nested workflow).

    Quick start::

        router = Router.from_client(
            client,
            classifier_prompt=(
                "Classify the task as one of: code, writing, math. "
                "Reply with only the category name."
            ),
            handlers={
                "code":    Agent(client, "You are a coder.", name="coder"),
                "writing": Agent(client, "You are a writer.", name="writer"),
                "math":    Agent(client, "You are a mathematician.", name="math"),
            },
            fallback=Agent(client, "Be helpful.", name="general"),
        )
    """

    routing_agent: Agent
    handlers: dict[str, Runner]
    name: str = "router"
    fallback: Optional[Runner] = None

    @classmethod
    def from_client(
        cls,
        client: BaseModelClient,
        classifier_prompt: str,
        handlers: dict[str, Runner],
        *,
        fallback: Optional[Runner] = None,
        name: str = "router",
    ) -> Router:
        """Build a Router using ``client`` as the classifier with the given prompt."""
        routing_agent = Agent(
            client,
            system_message=classifier_prompt,
            name="router-classifier",
            reset_messages_on_run=True,
        )
        return cls(routing_agent=routing_agent, handlers=handlers, fallback=fallback, name=name)

    def _classify(self, task: str, generate_kwargs: Optional[dict[str, Any]]) -> str:
        route = self.routing_agent.run(task, generate_kwargs=generate_kwargs)
        return route.strip().lower()

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Classify the task and dispatch to the matched handler.

        ``images`` are forwarded to the handler; the routing agent classifies on text only.
        """
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        route = self._classify(task, generate_kwargs)
        handler = self.handlers.get(route, self.fallback)
        if handler is None:
            raise ValueError(f"No handler for route '{route}' and no fallback set.")
        logger.debug("Router '%s' dispatching to '%s'.", self.name, route)
        return handler.run(task, generate_kwargs=generate_kwargs, images=images)

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        route_parts: list[str] = []
        for chunk in self.routing_agent.run(task, generate_kwargs=generate_kwargs, stream=True):
            yield chunk
            if chunk.phase == StreamingContentType.GENERATING:
                route_parts.append(chunk.content)
        route = "".join(route_parts).strip().lower()
        handler = self.handlers.get(route, self.fallback)
        if handler is None:
            raise ValueError(f"No handler for route '{route}' and no fallback set.")
        logger.debug("Router '%s' dispatching to '%s'.", self.name, route)
        yield from handler.run(task, generate_kwargs=generate_kwargs, stream=True, images=images)

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        result.update(self.routing_agent.messages)
        for handler in self.handlers.values():
            result.update(handler.messages)
        if self.fallback:
            result.update(self.fallback.messages)
        return result

    def restore(self, messages: list[dict], route: Optional[str] = None) -> None:
        """Restore one sub-runner's state from a saved message list.

        ``route=None`` (default) restores the routing classifier; a route key restores
        that handler (raises ``KeyError`` listing the routes on a miss). Other sub-runners
        start fresh on the next ``run()``. See :meth:`Agent.restore` for the full pattern.
        """
        if route is None:
            self.routing_agent.restore(messages)
            return
        if route not in self.handlers:
            raise KeyError(f"Unknown route '{route}'. Available routes: {sorted(self.handlers)}")
        self.handlers[route].restore(messages)

    @classmethod
    def from_config(
        cls,
        routing_config: dict[str, Any],
        handler_configs: dict[str, dict[str, Any]],
        client: BaseModelClient,
        fallback_config: Optional[dict[str, Any]] = None,
    ) -> Router:
        """Build a Router from config dicts and a single client."""
        routing_agent = Agent.from_config(routing_config, client)
        handlers: dict[str, Runner] = {route: Agent.from_config(cfg, client) for route, cfg in handler_configs.items()}
        fallback: Optional[Runner] = Agent.from_config(fallback_config, client) if fallback_config else None
        return cls(routing_agent=routing_agent, handlers=handlers, fallback=fallback)
