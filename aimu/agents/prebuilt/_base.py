"""Shared construction helper for the prebuilt orchestrator agents.

The three prebuilt orchestrators build their worker sub-agents the same way: one
fresh ``ModelClient`` per worker (same model as the orchestrator, isolated message
history), wrapped in an ``Agent`` with a name and system message. This factors that
out; each orchestrator keeps its own ``@tool`` dispatch wrappers (which carry
worker-specific names and descriptions).
"""

from __future__ import annotations

from typing import Callable, Optional

from aimu.agents.agent import Agent
from aimu.models.model_client import ModelClient


def make_workers(
    model_client: ModelClient,
    specs: list[tuple[str, str]],
    *,
    tools: Optional[list[Callable]] = None,
) -> list[Agent]:
    """Build worker :class:`Agent`s from ``(name, system_message)`` pairs.

    Each worker gets a fresh ``ModelClient(model_client.model)`` so their histories
    don't collide. When ``tools`` is given, every worker is configured with that tool
    list (e.g. live web search), otherwise workers are text-only.
    """
    workers = []
    for name, system_message in specs:
        client = ModelClient(model_client.model)
        if tools is not None:
            client.tools = list(tools)
        workers.append(Agent(client, name=name, system_message=system_message))
    return workers
