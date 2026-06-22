"""Expose any AIMU :class:`~aimu.agents.base.Runner` as an A2A server.

``serve_a2a(runner)`` is the agent-level analog of running ``python -m aimu.tools.mcp``:
it wraps a runner in an A2A ``AgentExecutor``, advertises it via an ``AgentCard``, and
serves the JSON-RPC + agent-card HTTP routes so any A2A client (AIMU's ``RemoteAgent`` or
another framework) can call it. :func:`build_a2a_app` returns the Starlette app without
serving, for embedding or tests.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill
from a2a.utils import new_agent_text_message

from aimu.agents.base import Runner

from ._card import build_agent_card

DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


class _RunnerExecutor(AgentExecutor):
    """Adapts a sync :class:`Runner` to the A2A ``AgentExecutor`` interface.

    ``runner.run`` is blocking, so it is offloaded to a thread to keep the event loop free.
    """

    def __init__(self, runner: Runner):
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input()
        result = await asyncio.to_thread(self._runner.run, task)
        await event_queue.enqueue_event(new_agent_text_message(str(result)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("RemoteAgent execution does not support cancellation.")


def build_a2a_app(
    runner: Runner,
    *,
    url: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    skills: Optional[list[AgentSkill]] = None,
    agent_card_path: str = DEFAULT_AGENT_CARD_PATH,
):
    """Build the Starlette ASGI app that serves ``runner`` over A2A (does not run it)."""
    card = build_agent_card(runner, url=url, name=name, description=description, skills=skills, streaming=True)
    handler = DefaultRequestHandler(agent_executor=_RunnerExecutor(runner), task_store=InMemoryTaskStore())
    application = A2AStarletteApplication(agent_card=card, http_handler=handler)
    return application.build(agent_card_url=agent_card_path)


def serve_a2a(
    runner: Runner,
    *,
    host: str = "127.0.0.1",
    port: int = 9000,
    url: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    skills: Optional[list[AgentSkill]] = None,
    **uvicorn_kwargs: Any,
) -> None:
    """Serve ``runner`` as an A2A agent over HTTP (blocking).

    ``url`` is the externally reachable base URL advertised in the agent card; it defaults
    to ``http://{host}:{port}/``. Extra kwargs pass through to ``uvicorn.run``.
    """
    import uvicorn

    base_url = url or f"http://{host}:{port}/"
    app = build_a2a_app(runner, url=base_url, name=name, description=description, skills=skills)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
