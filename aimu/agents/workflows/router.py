from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.base import Workflow, Runner, AgentChunk, MessageHistory
from aimu.agents.agent import Agent
from aimu.models.base import StreamingContentType, BaseModelClient

logger = logging.getLogger(__name__)


@dataclass
class Router(Workflow):
    """
    Workflow pattern: classify the input then dispatch to a specialist runner.

    The routing_agent receives the task and must return a single route name as
    its response. The Router dispatches to the matching handler. Route names are
    compared case-insensitively after stripping whitespace.

    Handlers may be any Runner subclass (agents or nested workflows), so a
    Router can dispatch to another Router, a Parallel, or a Agent.

    Usage::

        router = Router(
            routing_agent=Agent(client, name="classifier",
                system_message="Classify the task as one of: code, writing, math. Reply with only the category name."),
            handlers={
                "code":    Agent(code_client, name="coder"),
                "writing": Agent(write_client, name="writer"),
                "math":    Agent(math_client, name="mathematician"),
            },
            fallback=Agent(client, name="general"),
        )
        result = router.run("Write a poem about recursion.")

    From config::

        router = Router.from_config(routing_config, handler_configs, client)
    """

    routing_agent: Agent
    handlers: dict[str, Runner]
    name: str = "router"
    fallback: Optional[Runner] = None

    def _classify(self, task: str, generate_kwargs: Optional[dict[str, Any]]) -> str:
        """Run the routing_agent and return the normalised route name."""
        route = self.routing_agent.run(task, generate_kwargs=generate_kwargs)
        return route.strip().lower()

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[AgentChunk]]:
        """Classify the task and dispatch to the matched handler. Returns str or AgentChunk iterator.

        ``images`` are forwarded to the dispatched handler (the routing agent classifies on text only).
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
    ) -> Iterator[AgentChunk]:
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

    @classmethod
    def from_config(
        cls,
        routing_config: dict[str, Any],
        handler_configs: dict[str, dict[str, Any]],
        client: BaseModelClient,
        fallback_config: Optional[dict[str, Any]] = None,
    ) -> Router:
        """
        Build a Router from config dicts and a single BaseModelClient.

        Args:
            routing_config: Config dict for the routing Agent (name, system_message, etc.)
            handler_configs: Mapping of route name → Agent config dict
            client: Shared BaseModelClient for all agents
            fallback_config: Optional config dict for the fallback Agent
        """
        routing_agent = Agent.from_config(routing_config, client)
        handlers: dict[str, Runner] = {
            route: Agent.from_config(cfg, client) for route, cfg in handler_configs.items()
        }
        fallback: Optional[Runner] = Agent.from_config(fallback_config, client) if fallback_config else None
        return cls(routing_agent=routing_agent, handlers=handlers, fallback=fallback)
