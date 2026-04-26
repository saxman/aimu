from __future__ import annotations

from typing import Any, Iterator, Optional

from fastmcp import FastMCP

from aimu.agents.base import AgentChunk, MessageHistory
from aimu.agents.simple_agent import SimpleAgent
from aimu.models.model_client import ModelClient
from aimu.tools.client import MCPClient


class ResearchReportAgent:
    """
    Orchestrator agent that coordinates three worker sub-agents to produce a
    structured research report.

    The orchestrator autonomously decides which workers to call (and how many
    times) via tool use:

    - ``research_overview``  — broad background on the topic
    - ``find_examples``      — concrete real-world examples
    - ``find_counterpoints`` — counterarguments and limitations

    The model_client is used for the orchestrator; fresh client instances
    (same model, isolated message histories) are created for each worker.

    Usage::

        from aimu.models import ModelClient
        from aimu.models.ollama.ollama_client import OllamaModel
        from aimu.agents.examples import ResearchReportAgent

        client = ModelClient(OllamaModel.QWEN_3_8B)
        agent = ResearchReportAgent(client)
        report = agent.run("What is retrieval-augmented generation?")
    """

    def __init__(self, model_client: ModelClient) -> None:
        overview_agent = SimpleAgent(
            ModelClient(model_client.model),
            name="overview-worker",
            system_message=(
                "Provide a thorough 2-3 paragraph overview of the given topic. "
                "Cover what it is, why it matters, and its key components or mechanisms."
            ),
        )
        examples_agent = SimpleAgent(
            ModelClient(model_client.model),
            name="examples-worker",
            system_message=(
                "Provide 3-5 concrete, real-world examples or applications related to the given topic. "
                "For each example, briefly explain its relevance and significance."
            ),
        )
        counterpoints_agent = SimpleAgent(
            ModelClient(model_client.model),
            name="counterpoints-worker",
            system_message=(
                "Identify 3-5 counterarguments, limitations, criticisms, or alternative perspectives "
                "on the given topic. Be specific, balanced, and evidence-based."
            ),
        )

        mcp = FastMCP("Research Workers")

        @mcp.tool()
        def research_overview(topic: str) -> str:
            """Research and return a broad background overview of the topic."""
            return overview_agent.run(topic)

        @mcp.tool()
        def find_examples(topic: str) -> str:
            """Find concrete real-world examples and applications related to the topic."""
            return examples_agent.run(topic)

        @mcp.tool()
        def find_counterpoints(topic: str) -> str:
            """Identify counterarguments, limitations, and alternative perspectives on the topic."""
            return counterpoints_agent.run(topic)

        model_client.mcp_client = MCPClient(server=mcp)
        self._orchestrator = SimpleAgent(
            model_client,
            name="research-report-agent",
            system_message=(
                "You are a research report writer. Use the available tools to gather material: "
                "call research_overview for background, find_examples for concrete illustrations, "
                "and find_counterpoints for critical analysis. "
                "Then synthesize all gathered material into a structured report with clearly labeled sections."
            ),
        )

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        return self._orchestrator.run(task, generate_kwargs)

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        return self._orchestrator.run_streamed(task, generate_kwargs)

    @property
    def messages(self) -> MessageHistory:
        return self._orchestrator.messages
