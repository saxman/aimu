from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.agents.base import AgentChunk, BaseAgent, MessageHistory
from aimu.models.model_client import ModelClient


class OrchestratorAgent(BaseAgent, ABC):
    """
    Base class for orchestrator agents that dispatch to worker sub-agents via tools.

    Subclasses define workers and decorate dispatch functions with ``@aimu.tools.tool``
    in ``__init__``, then call ``_setup_orchestrator()`` with the list of dispatch
    tools. ``run()`` and ``messages`` are provided by this class and need not be
    re-implemented.

    Pattern::

        from aimu.tools import tool

        class MyAgent(OrchestratorAgent):
            def __init__(self, model_client: ModelClient) -> None:
                worker = Agent(ModelClient(model_client.model),
                               name="worker", system_message="...")

                @tool
                def do_work(task: str) -> str:
                    \"\"\"Do the work.\"\"\"
                    return worker.run(task)

                self._setup_orchestrator(
                    model_client,
                    name="my-agent",
                    system_message="Use do_work to complete tasks.",
                    tools=[do_work],
                )
    """

    def _setup_orchestrator(
        self,
        model_client: ModelClient,
        *,
        name: str,
        system_message: str,
        tools: list[Callable],
        concurrent_tool_calls: bool = False,
    ) -> None:
        """Wire up the orchestrator Agent with its dispatch tools.

        Call this at the end of the subclass ``__init__`` once all ``@tool``-decorated
        dispatch functions are defined.
        """
        model_client.tools = list(tools)
        model_client.concurrent_tool_calls = concurrent_tool_calls
        self._orchestrator = Agent(model_client, name=name, system_message=system_message)

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[AgentChunk]]:
        return self._orchestrator.run(task, generate_kwargs, stream=stream)

    @property
    def messages(self) -> MessageHistory:
        return self._orchestrator.messages
