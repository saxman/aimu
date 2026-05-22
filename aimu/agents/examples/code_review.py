from __future__ import annotations

from aimu.agents.agent import Agent
from aimu.agents.orchestrator_agent import OrchestratorAgent
from aimu.models.model_client import ModelClient
from aimu.tools import tool


class CodeReviewAgent(OrchestratorAgent):
    """
    Orchestrator agent that coordinates three specialist reviewer sub-agents to
    produce a comprehensive code review.

    The orchestrator autonomously decides which reviewers to call via tool use:

    - ``review_security``     — vulnerabilities and security issues
    - ``review_performance``  — bottlenecks and algorithmic inefficiencies
    - ``review_readability``  — naming, clarity, and maintainability

    The model_client is used for the orchestrator; fresh client instances
    (same model, isolated message histories) are created for each reviewer.

    Usage::

        from aimu.models import ModelClient, OllamaModel
        from aimu.agents.examples import CodeReviewAgent

        client = ModelClient(OllamaModel.QWEN_3_8B)
        agent = CodeReviewAgent(client)
        review = agent.run("Review this code:\\n\\n" + code_snippet)
    """

    def __init__(self, model_client: ModelClient) -> None:
        security_agent = Agent(
            ModelClient(model_client.model),
            name="security-reviewer",
            system_message=(
                "Analyze the provided code for security vulnerabilities. Check for injection flaws, "
                "improper authentication, insecure data handling, exposed secrets, and other "
                "OWASP-related issues. For each issue found, state the location, severity "
                "(Critical/High/Medium/Low), and a specific recommended fix."
            ),
        )
        performance_agent = Agent(
            ModelClient(model_client.model),
            name="performance-reviewer",
            system_message=(
                "Analyze the provided code for performance issues. Look for algorithmic inefficiencies "
                "(e.g. O(n²) loops, repeated work), unnecessary I/O, memory leaks, and missed "
                "optimization opportunities. For each issue, state the location and a concrete "
                "improvement suggestion."
            ),
        )
        readability_agent = Agent(
            ModelClient(model_client.model),
            name="readability-reviewer",
            system_message=(
                "Analyze the provided code for readability and maintainability. Look for unclear naming, "
                "missing documentation, overly complex logic, code duplication, and style violations. "
                "For each issue, state the location and a suggested improvement."
            ),
        )

        @tool
        def review_security(code: str) -> str:
            """Analyze code for security vulnerabilities and return findings with severity ratings."""
            return security_agent.run(code)

        @tool
        def review_performance(code: str) -> str:
            """Analyze code for performance bottlenecks and inefficiencies."""
            return performance_agent.run(code)

        @tool
        def review_readability(code: str) -> str:
            """Analyze code for readability and maintainability issues."""
            return readability_agent.run(code)

        self._setup_orchestrator(
            model_client,
            name="code-review-agent",
            system_message=(
                "You are a senior code reviewer. Use the specialist review tools to analyze the "
                "submitted code from multiple angles: call review_security, review_performance, "
                "and review_readability. Then synthesize all findings into a prioritized review "
                "report with actionable fix recommendations, ordered by severity."
            ),
            tools=[review_security, review_performance, review_readability],
        )
