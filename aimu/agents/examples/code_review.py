from __future__ import annotations

from typing import Any, Iterator, Optional

from fastmcp import FastMCP

from aimu.agents.base import AgentChunk, MessageHistory
from aimu.agents.simple_agent import SimpleAgent
from aimu.models.model_client import ModelClient
from aimu.tools.client import MCPClient


class CodeReviewAgent:
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

        from aimu.models import ModelClient
        from aimu.models.ollama.ollama_client import OllamaModel
        from aimu.agents.examples import CodeReviewAgent

        client = ModelClient(OllamaModel.QWEN_3_8B)
        agent = CodeReviewAgent(client)
        review = agent.run("Review this code:\\n\\n" + code_snippet)
    """

    def __init__(self, model_client: ModelClient) -> None:
        security_agent = SimpleAgent(
            ModelClient(model_client.model),
            name="security-reviewer",
            system_message=(
                "Analyze the provided code for security vulnerabilities. Check for injection flaws, "
                "improper authentication, insecure data handling, exposed secrets, and other "
                "OWASP-related issues. For each issue found, state the location, severity "
                "(Critical/High/Medium/Low), and a specific recommended fix."
            ),
        )
        performance_agent = SimpleAgent(
            ModelClient(model_client.model),
            name="performance-reviewer",
            system_message=(
                "Analyze the provided code for performance issues. Look for algorithmic inefficiencies "
                "(e.g. O(n²) loops, repeated work), unnecessary I/O, memory leaks, and missed "
                "optimization opportunities. For each issue, state the location and a concrete "
                "improvement suggestion."
            ),
        )
        readability_agent = SimpleAgent(
            ModelClient(model_client.model),
            name="readability-reviewer",
            system_message=(
                "Analyze the provided code for readability and maintainability. Look for unclear naming, "
                "missing documentation, overly complex logic, code duplication, and style violations. "
                "For each issue, state the location and a suggested improvement."
            ),
        )

        mcp = FastMCP("Code Review Workers")

        @mcp.tool()
        def review_security(code: str) -> str:
            """Analyze code for security vulnerabilities and return findings with severity ratings."""
            return security_agent.run(code)

        @mcp.tool()
        def review_performance(code: str) -> str:
            """Analyze code for performance bottlenecks and inefficiencies."""
            return performance_agent.run(code)

        @mcp.tool()
        def review_readability(code: str) -> str:
            """Analyze code for readability and maintainability issues."""
            return readability_agent.run(code)

        model_client.mcp_client = MCPClient(server=mcp)
        self._orchestrator = SimpleAgent(
            model_client,
            name="code-review-agent",
            system_message=(
                "You are a senior code reviewer. Use the specialist review tools to analyze the "
                "submitted code from multiple angles: call review_security, review_performance, "
                "and review_readability. Then synthesize all findings into a prioritized review "
                "report with actionable fix recommendations, ordered by severity."
            ),
        )

    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        return self._orchestrator.run(task, generate_kwargs)

    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        return self._orchestrator.run_streamed(task, generate_kwargs)

    @property
    def messages(self) -> MessageHistory:
        return self._orchestrator.messages
