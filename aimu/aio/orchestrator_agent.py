"""Async ``OrchestratorAgent`` + ``assemble()`` factory."""

from __future__ import annotations

import re
from abc import ABC
from typing import Any, AsyncIterator, Callable, Optional, Union

from aimu.agents.base import MessageHistory
from aimu.models.base import StreamChunk

from ._base import AsyncBaseModelClient
from .agent import Agent, AsyncRunner


class OrchestratorAgent(AsyncRunner, ABC):
    """Async base for the orchestrator + worker-tools pattern.

    Subclasses define worker :class:`Agent` instances and ``@tool``-decorated dispatch
    functions in ``__init__``, then call :meth:`_init_orchestrator` to wire everything up.
    Worker dispatch functions should be ``async def`` so the orchestrator's
    ``concurrent_tool_calls=True`` actually overlaps work.

    For the simple case of dispatching to a fixed list of workers, use
    :meth:`assemble` to skip subclassing entirely.
    """

    def _init_orchestrator(
        self,
        model_client: AsyncBaseModelClient,
        *,
        name: str,
        system_message: str,
        tools: list[Callable],
        concurrent_tool_calls: bool = False,
        final_answer_prompt: Optional[str] = None,
    ) -> None:
        model_client.tools = list(tools)
        model_client.concurrent_tool_calls = concurrent_tool_calls
        self._orchestrator = Agent(
            model_client, name=name, system_message=system_message, final_answer_prompt=final_answer_prompt
        )

    @classmethod
    def assemble(
        cls,
        model_client: AsyncBaseModelClient,
        system_message: str,
        *,
        workers: list[Agent],
        name: str = "orchestrator",
        concurrent_tool_calls: bool = True,
        final_answer_prompt: Optional[str] = None,
    ) -> "OrchestratorAgent":
        """Build a ready-to-run async orchestrator from a list of worker agents."""
        from aimu.tools.decorator import tool

        tool_fns: list[Callable] = []
        for worker in workers:
            tool_fns.append(_wrap_worker_as_tool(worker, tool))

        instance = cls.__new__(cls)
        instance._init_orchestrator(
            model_client,
            name=name,
            system_message=system_message,
            tools=tool_fns,
            concurrent_tool_calls=concurrent_tool_calls,
            final_answer_prompt=final_answer_prompt,
        )
        return instance

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        return await self._orchestrator.run(task, generate_kwargs, stream=stream, images=images)

    @property
    def messages(self) -> MessageHistory:
        return self._orchestrator.messages


def _wrap_worker_as_tool(worker: Agent, tool_decorator: Callable) -> Callable:
    """Build an async ``@tool``-decorated function that dispatches to ``await worker.run(task)``."""
    safe_name = re.sub(r"\W+", "_", worker.name or "worker").strip("_") or "worker"
    description = (worker.system_message or f"Dispatch a task to the {safe_name} worker.").splitlines()[0]

    async def _dispatch(task: str) -> str:
        return await worker.run(task)

    _dispatch.__name__ = safe_name
    _dispatch.__doc__ = description
    _dispatch.__annotations__ = {"task": str, "return": str}
    return tool_decorator(_dispatch)
