from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import BaseModelClient, StreamChunk


class OrchestratorAgent(Runner, ABC):
    """Base class for the orchestrator + worker-tools pattern.

    Subclasses define worker :class:`Agent` instances and ``@tool``-decorated dispatch
    functions in ``__init__``, then call :meth:`_init_orchestrator` to wire everything up::

        from aimu.tools import tool

        class ResearchAgent(OrchestratorAgent):
            def __init__(self, client):
                researcher = Agent(client, "Research the topic.", name="researcher")

                @tool
                def research(topic: str) -> str:
                    \"\"\"Run the researcher on a topic.\"\"\"
                    return researcher.run(topic)

                self._init_orchestrator(
                    client,
                    name="research-orchestrator",
                    system_message="Use the research tool to investigate.",
                    tools=[research],
                )

    For the simple case of dispatching to a fixed list of workers, use
    :meth:`assemble` to skip subclassing entirely.
    """

    def _init_orchestrator(
        self,
        model_client: BaseModelClient,
        *,
        name: str,
        system_message: str,
        tools: list[Callable],
        concurrent_tool_calls: bool = False,
    ) -> None:
        """Wire up the orchestrator's inner :class:`Agent`.

        Call at the end of the subclass ``__init__``, after every ``@tool`` dispatch
        function is defined.
        """
        model_client.tools = list(tools)
        model_client.concurrent_tool_calls = concurrent_tool_calls
        self._orchestrator = Agent(model_client, name=name, system_message=system_message)

    @classmethod
    def assemble(
        cls,
        model_client: BaseModelClient,
        system_message: str,
        *,
        workers: list[Agent],
        name: str = "orchestrator",
        concurrent_tool_calls: bool = True,
    ) -> "OrchestratorAgent":
        """Build a ready-to-run orchestrator from a list of worker agents.

        Each worker becomes a callable tool — the orchestrator dispatches by name.
        Tool descriptions are taken from the worker's ``system_message`` (truncated to
        one line) so callers don't need to write ``@tool`` wrappers manually.

        Example::

            researcher = Agent(client, "Research the topic.", name="researcher")
            critic = Agent(client, "Critique the response.", name="critic")
            orch = OrchestratorAgent.assemble(client, "Use both workers.",
                                              workers=[researcher, critic])
            print(orch.run("Quantum computing"))
        """
        from aimu.tools.decorator import tool

        tool_fns: list[Callable] = []
        for worker in workers:
            tool_fns.append(_wrap_worker_as_tool(worker, tool))

        instance = cls.__new__(cls)  # bypass ABC instantiation if cls is OrchestratorAgent
        instance._init_orchestrator(
            model_client,
            name=name,
            system_message=system_message,
            tools=tool_fns,
            concurrent_tool_calls=concurrent_tool_calls,
        )
        return instance

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        return self._orchestrator.run(task, generate_kwargs, stream=stream, images=images)

    @property
    def messages(self) -> MessageHistory:
        return self._orchestrator.messages


def _wrap_worker_as_tool(worker: Agent, tool_decorator: Callable) -> Callable:
    """Build a ``@tool``-decorated function that dispatches to ``worker.run(task)``.

    The function name is the worker's ``name`` (sanitised); the description is the
    worker's ``system_message`` (truncated to one line).
    """
    import re

    safe_name = re.sub(r"\W+", "_", worker.name or "worker").strip("_") or "worker"
    description = (worker.system_message or f"Dispatch a task to the {safe_name} worker.").splitlines()[0]

    def _dispatch(task: str) -> str:
        return worker.run(task)

    _dispatch.__name__ = safe_name
    _dispatch.__doc__ = description
    _dispatch.__annotations__ = {"task": str, "return": str}
    return tool_decorator(_dispatch)


# Make OrchestratorAgent.assemble usable without a subclass: instantiate the ABC directly
# via __new__ only inside assemble(); regular instantiation still requires a subclass per
# the original pattern, since assemble() is the documented escape hatch.
