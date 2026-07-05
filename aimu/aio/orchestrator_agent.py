"""Async ``OrchestratorAgent`` + ``assemble()`` factory."""

from __future__ import annotations

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
        self._orchestrator = Agent(
            model_client,
            name=name,
            system_message=system_message,
            tools=list(tools),
            concurrent_tool_calls=concurrent_tool_calls,
            final_answer_prompt=final_answer_prompt,
        )

    @classmethod
    def assemble(
        cls,
        model_client: AsyncBaseModelClient,
        system_message: str,
        *,
        workers: list[AsyncRunner],
        name: str = "orchestrator",
        concurrent_tool_calls: bool = True,
        final_answer_prompt: Optional[str] = None,
    ) -> "OrchestratorAgent":
        """Build a ready-to-run async orchestrator from a list of worker runners.

        Each worker becomes an async callable tool via :meth:`AsyncRunner.as_tool`. Workers
        may be any :class:`AsyncRunner` (an async ``Agent``, a workflow, or a remote A2A
        agent), not just ``Agent`` instances.
        """
        tool_fns: list[Callable] = [worker.as_tool() for worker in workers]

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

    def restore(self, messages: list[dict]) -> None:
        """Restore the inner orchestrator agent's state from a saved message list.

        Workers are invoked as tools, so their own state is not part of the orchestrator's
        history; restore a worker directly if it needs resuming. See
        :meth:`aimu.aio.Agent.restore` for the full save/restore pattern.
        """
        self._orchestrator.restore(messages)
