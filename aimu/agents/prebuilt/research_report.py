from __future__ import annotations

from typing import Callable, Optional

from aimu.agents.agent import Agent
from aimu.agents.orchestrator_agent import OrchestratorAgent
from aimu.models.model_client import ModelClient
from aimu.tools import tool


class ResearchReportAgent(OrchestratorAgent):
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

    Parameters
    ----------
    model_client:
        Used for both orchestrator and worker agents.
    worker_tools:
        Optional list of ``@tool``-decorated functions injected into each worker so
        they can perform live lookups (e.g. ``aimu.tools.builtin.web_search`` and
        ``get_webpage``). When provided, the orchestrator also enables concurrent
        tool calls so all three workers can be dispatched in a single round-trip
        when the model supports it.

    Usage — text-only workers (no live search)::

        from aimu.models import ModelClient, OllamaModel
        from aimu.agents.prebuilt import ResearchReportAgent

        client = ModelClient(OllamaModel.QWEN_3_8B)
        agent = ResearchReportAgent(client)
        report = agent.run("What is retrieval-augmented generation?")

    Usage — workers with live web search::

        from aimu.tools import builtin

        agent = ResearchReportAgent(client, worker_tools=[builtin.web_search, builtin.get_webpage])
        report = agent.run("What is retrieval-augmented generation?")
    """

    def __init__(self, model_client: ModelClient, worker_tools: Optional[list[Callable]] = None) -> None:
        def _make_worker(name: str, system_message: str) -> Agent:
            worker_client = ModelClient(model_client.model)
            if worker_tools is not None:
                worker_client.tools = list(worker_tools)
            return Agent(worker_client, name=name, system_message=system_message)

        live_tools_note = (
            " Use your search and browsing tools to find current, accurate information." if worker_tools else ""
        )

        overview_agent = _make_worker(
            "overview-worker",
            "Provide a thorough 2-3 paragraph overview of the given topic. "
            "Cover what it is, why it matters, and its key components or mechanisms." + live_tools_note,
        )
        examples_agent = _make_worker(
            "examples-worker",
            "Provide 3-5 concrete, real-world examples or applications related to the given topic. "
            "For each example, briefly explain its relevance and significance." + live_tools_note,
        )
        counterpoints_agent = _make_worker(
            "counterpoints-worker",
            "Identify 3-5 counterarguments, limitations, criticisms, or alternative perspectives "
            "on the given topic. Be specific, balanced, and evidence-based." + live_tools_note,
        )

        @tool
        def research_overview(topic: str) -> str:
            """Research and return a broad background overview of the topic."""
            return overview_agent.run(topic)

        @tool
        def find_examples(topic: str) -> str:
            """Find concrete real-world examples and applications related to the topic."""
            return examples_agent.run(topic)

        @tool
        def find_counterpoints(topic: str) -> str:
            """Identify counterarguments, limitations, and alternative perspectives on the topic."""
            return counterpoints_agent.run(topic)

        self._init_orchestrator(
            model_client,
            name="research-report-agent",
            system_message=(
                "You are a research report writer. Use the available tools to gather material: "
                "call research_overview for background, find_examples for concrete illustrations, "
                "and find_counterpoints for critical analysis. "
                "Then synthesize all gathered material into a structured report with clearly labeled sections."
            ),
            tools=[research_overview, find_examples, find_counterpoints],
            concurrent_tool_calls=worker_tools is not None,
        )
