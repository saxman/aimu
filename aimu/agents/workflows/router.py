from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from aimu.agents.base import Workflow, Runner, AgentChunk, MessageHistory
from aimu.agents.simple_agent import SimpleAgent
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
    Router can dispatch to another Router, a Parallel, or a SimpleAgent.

    Usage::

        router = Router(
            routing_agent=SimpleAgent(client, name="classifier",
                system_message="Classify the task as one of: code, writing, math. Reply with only the category name."),
            handlers={
                "code":    SimpleAgent(code_client, name="coder"),
                "writing": SimpleAgent(write_client, name="writer"),
                "math":    SimpleAgent(math_client, name="mathematician"),
            },
            fallback=SimpleAgent(client, name="general"),
        )
        result = router.run("Write a poem about recursion.")

    From config::

        router = Router.from_config(routing_config, handler_configs, client)
    """

    routing_agent: SimpleAgent
    handlers: dict[str, Runner]
    name: str = "router"
    fallback: Optional[Runner] = None

    def _classify(self, task: str, generate_kwargs: Optional[dict[str, Any]]) -> str:
        """Run the routing_agent and return the normalised route name."""
        route = self.routing_agent.run(task, generate_kwargs=generate_kwargs)
        return route.strip().lower()

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Classify the task and dispatch to the matched handler."""
        route = self._classify(task, generate_kwargs)
        handler = self.handlers.get(route, self.fallback)
        if handler is None:
            raise ValueError(f"No handler for route '{route}' and no fallback set.")
        logger.debug("Router '%s' dispatching to '%s'.", self.name, route)
        return handler.run(task, generate_kwargs=generate_kwargs)

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        """
        Stream execution. Classifier chunks are yielded first, then the selected
        handler's chunks follow.
        """
        route_parts: list[str] = []
        for chunk in self.routing_agent.run_streamed(task, generate_kwargs=generate_kwargs):
            yield chunk
            if chunk.phase == StreamingContentType.GENERATING:
                route_parts.append(chunk.content)
        route = "".join(route_parts).strip().lower()
        handler = self.handlers.get(route, self.fallback)
        if handler is None:
            raise ValueError(f"No handler for route '{route}' and no fallback set.")
        logger.debug("Router '%s' dispatching to '%s'.", self.name, route)
        yield from handler.run_streamed(task, generate_kwargs=generate_kwargs)

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
            routing_config: Config dict for the routing SimpleAgent (name, system_message, etc.)
            handler_configs: Mapping of route name → SimpleAgent config dict
            client: Shared BaseModelClient for all agents
            fallback_config: Optional config dict for the fallback SimpleAgent
        """
        routing_agent = SimpleAgent.from_config(routing_config, client)
        handlers: dict[str, Runner] = {
            route: SimpleAgent.from_config(cfg, client) for route, cfg in handler_configs.items()
        }
        fallback: Optional[Runner] = SimpleAgent.from_config(fallback_config, client) if fallback_config else None
        return cls(routing_agent=routing_agent, handlers=handlers, fallback=fallback)
