from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, NamedTuple, Optional

from aimu.models.base import StreamingContentType

# Type alias used by the messages property across the runner hierarchy.
MessageHistory = dict[str, list[dict]]


class AgentChunk(NamedTuple):
    """A chunk tagged with the runner name and loop iteration number."""

    agent_name: str
    iteration: int
    phase: StreamingContentType
    content: Any  # str for THINKING/GENERATING; dict {"name", "response"} for TOOL_CALLING


class Runner(ABC):
    """
    Abstract base class for all executable units — agents and workflows alike.

    Both autonomous agents and workflow patterns share this interface:
    ``run()`` for synchronous execution and ``run_streamed()`` for streaming.
    """

    @abstractmethod
    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run synchronously and return the final response string."""
        ...

    @abstractmethod
    def run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        """Stream execution, yielding AgentChunk for every chunk produced."""
        ...

    @property
    @abstractmethod
    def messages(self) -> MessageHistory:
        """
        Return the message histories of all agents in this runner, keyed by agent name.

        For a leaf agent (SimpleAgent / SkillAgent) this is ``{agent.name: model_client.messages}``.
        For composite workflows the dicts from every constituent runner are merged, so the
        result spans the full sub-tree regardless of nesting depth.

        Note: when agents share a single ModelClient (e.g. a Chain built via
        ``Chain.from_config``), all agents reference the same messages list. After the
        run, every key in the returned dict will point at the last step's messages.
        """
        ...


class Agent(Runner):
    """
    Marker ABC for autonomous agents that direct their own process.

    The LLM decides which tools to call, in which order, and when to stop.
    Per Anthropic's "Building Effective Agents" taxonomy: the model autonomously
    directs the process and tool selection rather than following a predetermined
    code path.

    Concrete subclasses: SimpleAgent, SkillAgent.
    """


class Workflow(Runner):
    """
    Marker ABC for workflow patterns with predetermined code paths.

    The code controls routing, sequencing, and aggregation; the LLM is invoked
    at defined points but does not direct the overall control flow.
    Per Anthropic's "Building Effective Agents" taxonomy: LLMs and tools operate
    through predetermined steps.

    Concrete subclasses: Chain, Router, Parallel, EvaluatorOptimizer.
    """
