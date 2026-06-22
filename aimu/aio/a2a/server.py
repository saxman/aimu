"""Expose an async :class:`AsyncRunner` as an A2A server.

Async twin of :mod:`aimu.agents.a2a.server`. The executor awaits ``runner.run`` directly
(no thread offload needed). The agent-card construction is shared with the sync surface
via :func:`aimu.agents.a2a._card.build_agent_card`.
"""

from __future__ import annotations

from typing import Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill
from a2a.utils import new_agent_text_message

from aimu.agents.a2a._card import build_agent_card
from aimu.aio.agent import AsyncRunner

DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


class _AsyncRunnerExecutor(AgentExecutor):
    """Adapts an :class:`AsyncRunner` to the A2A ``AgentExecutor`` interface."""

    def __init__(self, runner: AsyncRunner):
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input()
        result = await self._runner.run(task)
        await event_queue.enqueue_event(new_agent_text_message(str(result)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("RemoteAgent execution does not support cancellation.")


def build_a2a_app(
    runner: AsyncRunner,
    *,
    url: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    skills: Optional[list[AgentSkill]] = None,
    agent_card_path: str = DEFAULT_AGENT_CARD_PATH,
):
    """Build the Starlette ASGI app that serves an async ``runner`` over A2A (does not run it)."""
    card = build_agent_card(runner, url=url, name=name, description=description, skills=skills, streaming=True)
    handler = DefaultRequestHandler(agent_executor=_AsyncRunnerExecutor(runner), task_store=InMemoryTaskStore())
    application = A2AStarletteApplication(agent_card=card, http_handler=handler)
    return application.build(agent_card_url=agent_card_path)


def serve_a2a(
    runner: AsyncRunner,
    *,
    host: str = "127.0.0.1",
    port: int = 9000,
    url: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    skills: Optional[list[AgentSkill]] = None,
    **uvicorn_kwargs: Any,
) -> None:
    """Serve an async ``runner`` as an A2A agent over HTTP (blocking)."""
    import uvicorn

    base_url = url or f"http://{host}:{port}/"
    app = build_a2a_app(runner, url=base_url, name=name, description=description, skills=skills)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
