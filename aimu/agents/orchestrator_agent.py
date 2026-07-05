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
        final_answer_prompt: Optional[str] = None,
    ) -> None:
        """Wire up the orchestrator's inner :class:`Agent`.

        Call at the end of the subclass ``__init__``, after every ``@tool`` dispatch
        function is defined. ``final_answer_prompt`` (opt-in) is forwarded to the inner
        :class:`Agent` to guarantee a final answer if the orchestrator exhausts its
        iterations while still dispatching to workers (see :class:`Agent`).
        """
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
        model_client: BaseModelClient,
        system_message: str,
        *,
        workers: list[Runner],
        name: str = "orchestrator",
        concurrent_tool_calls: bool = True,
        final_answer_prompt: Optional[str] = None,
    ) -> "OrchestratorAgent":
        """Build a ready-to-run orchestrator from a list of worker runners.

        Each worker becomes a callable tool via :meth:`Runner.as_tool`; the orchestrator
        dispatches by name. Workers may be any :class:`Runner` (an :class:`Agent`, a
        ``Chain``/``Router``/``Parallel`` workflow, or a remote A2A agent), not just
        ``Agent`` instances; tool names/descriptions come from each worker's ``name`` and
        (when present) ``system_message``.

        Example::

            researcher = Agent(client, "Research the topic.", name="researcher")
            critic = Agent(client, "Critique the response.", name="critic")
            orch = OrchestratorAgent.assemble(client, "Use both workers.",
                                              workers=[researcher, critic])
            print(orch.run("Quantum computing"))
        """
        tool_fns: list[Callable] = [worker.as_tool() for worker in workers]

        instance = cls.__new__(cls)  # bypass ABC instantiation if cls is OrchestratorAgent
        instance._init_orchestrator(
            model_client,
            name=name,
            system_message=system_message,
            tools=tool_fns,
            concurrent_tool_calls=concurrent_tool_calls,
            final_answer_prompt=final_answer_prompt,
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

    def restore(self, messages: list[dict]) -> None:
        """Restore the inner orchestrator agent's state from a saved message list.

        Workers are invoked as tools, so their own state is not part of the orchestrator's
        history; restore a worker directly if it needs resuming. See :meth:`Agent.restore`
        for the full save/restore pattern.
        """
        self._orchestrator.restore(messages)
