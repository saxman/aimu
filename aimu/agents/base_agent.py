from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, NamedTuple, Optional

from aimu.models.base_client import StreamingContentType


class AgentChunk(NamedTuple):
    """A chunk tagged with the agent name and loop iteration number."""

    agent_name: str
    iteration: int
    phase: StreamingContentType
    content: Any  # str for THINKING/GENERATING; dict {"name", "response"} for TOOL_CALLING


class Agent(ABC):
    """
    Abstract base class for all agent runners.

    Concrete subclasses:
      - SimpleAgent  — single model loop (wraps a ModelClient)
      - WorkflowAgent — sequential pipeline of Agent instances
    """

    @abstractmethod
    def run(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        """Run the agent synchronously and return the final response string."""
        ...

    @abstractmethod
    def run_streamed(
        self, task: str, generate_kwargs: Optional[dict[str, Any]] = None
    ) -> Iterator[AgentChunk]:
        """Stream the agent, yielding AgentChunk for every chunk produced."""
        ...
