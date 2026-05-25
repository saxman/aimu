from __future__ import annotations

from aimu.agents.agent import Agent
from aimu.agents.orchestrator_agent import OrchestratorAgent
from aimu.models.model_client import ModelClient
from aimu.tools import tool


class ContentCreationAgent(OrchestratorAgent):
    """
    Orchestrator agent that coordinates three worker sub-agents to build
    content step by step.

    The orchestrator autonomously decides the order and number of tool calls:

    - ``research_topic``  — extract key facts and angles from the brief
    - ``create_outline``  — build a structured outline from the research
    - ``write_section``   — draft an individual content section

    The model_client is used for the orchestrator; fresh client instances
    (same model, isolated message histories) are created for each worker.

    Usage::

        from aimu.models import ModelClient, OllamaModel
        from aimu.agents.prebuilt import ContentCreationAgent

        client = ModelClient(OllamaModel.QWEN_3_8B)
        agent = ContentCreationAgent(client)
        content = agent.run("Write about the benefits of test-driven development")
    """

    def __init__(self, model_client: ModelClient) -> None:
        research_agent = Agent(
            ModelClient(model_client.model),
            name="research-worker",
            system_message=(
                "Extract and organize the key facts, statistics, angles, and talking points "
                "for the given content topic or brief. Return a structured bullet list of "
                "8-12 research points ready for use in writing."
            ),
        )
        outline_agent = Agent(
            ModelClient(model_client.model),
            name="outline-worker",
            system_message=(
                "Create a detailed content outline based on the provided research facts. "
                "Include an introduction hook, 3-5 main sections with sub-points, and a conclusion. "
                "Structure it for the implied target audience and format."
            ),
        )
        section_agent = Agent(
            ModelClient(model_client.model),
            name="section-writer",
            system_message=(
                "Write a complete, polished draft of the content section described in the input. "
                "The input specifies a section title and contextual notes. "
                "Return only the section prose, ready to be assembled into the final piece."
            ),
        )

        @tool
        def research_topic(brief: str) -> str:
            """Extract key facts, angles, and talking points for the given content brief."""
            return research_agent.run(brief)

        @tool
        def create_outline(facts: str) -> str:
            """Build a structured content outline from the provided research facts."""
            return outline_agent.run(facts)

        @tool
        def write_section(section_title_and_context: str) -> str:
            """Write a complete draft of one content section given its title and context notes."""
            return section_agent.run(section_title_and_context)

        self._init_orchestrator(
            model_client,
            name="content-creation-agent",
            system_message=(
                "You are a content creator. Build content step by step using the available tools: "
                "call research_topic to gather facts from the brief, create_outline to structure "
                "the piece, then write_section for each section in the outline. "
                "Assemble and return the final content with all sections joined together."
            ),
            tools=[research_topic, create_outline, write_section],
        )
