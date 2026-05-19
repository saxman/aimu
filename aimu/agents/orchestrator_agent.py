from __future__ import annotations

from abc import ABC
from typing import Any, Iterator, Optional, Union

from fastmcp import FastMCP

from aimu.agents.base import Agent, AgentChunk, MessageHistory
from aimu.agents.simple_agent import SimpleAgent
from aimu.models.model_client import ModelClient
from aimu.tools.client import MCPClient


class OrchestratorAgent(Agent, ABC):
    """
    Base class for orchestrator agents that dispatch to worker sub-agents via MCP tools.

    Subclasses define workers and MCP tools in ``__init__``, then call
    ``_setup_orchestrator()`` to wire everything together. ``run()`` and
    ``messages`` are provided by this class and need not be re-implemented.

    Pattern::

        class MyAgent(OrchestratorAgent):
            def __init__(self, model_client: ModelClient) -> None:
                worker = SimpleAgent(ModelClient(model_client.model),
                                     name="worker", system_message="...")

                mcp = FastMCP("My Workers")

                @mcp.tool()
                def do_work(task: str) -> str:
                    \"\"\"Do the work.\"\"\"
                    return worker.run(task)

                self._setup_orchestrator(
                    model_client, mcp,
                    name="my-agent",
                    system_message="Use do_work to complete tasks.",
                )
    """

    def _setup_orchestrator(
        self,
        model_client: ModelClient,
        mcp: FastMCP,
        name: str,
        system_message: str,
        concurrent_tool_calls: bool = False,
    ) -> None:
        """Wire up the MCP server and create the orchestrator SimpleAgent.

        Call this at the end of the subclass ``__init__`` after all tools are
        registered on ``mcp``.
        """
        model_client.mcp_client = MCPClient(server=mcp)
        model_client.concurrent_tool_calls = concurrent_tool_calls
        self._orchestrator = SimpleAgent(model_client, name=name, system_message=system_message)

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
