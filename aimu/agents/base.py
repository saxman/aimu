"""Agent / workflow base classes.

Decision tree for picking the right class:

* **Single ``chat()`` call, no loop** → :class:`aimu.models.ModelClient`
* **Tool-using loop until model stops** → :class:`Agent`
* **Skills auto-discovered from disk** → :class:`SkillAgent`
* **Fixed sequence / classifier / parallel / generate-critique** →
  :class:`Chain` / :class:`Router` / :class:`Parallel` / :class:`EvaluatorOptimizer`
* **Drop an agent into an API that wants a ``ModelClient``** → ``agent.as_model_client()``

The hierarchy is two marker ABCs (``BaseAgent`` for autonomous agents, ``Workflow``
for code-controlled patterns) under a common ``Runner`` interface. Both ``run()``
synchronously or with ``stream=True`` for streaming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Union

from aimu.models.base import StreamChunk

# Backwards-compatible alias. New code should use StreamChunk directly.
AgentChunk = StreamChunk


# Type alias used by the messages property across the runner hierarchy.
MessageHistory = dict[str, list[dict]]


class Runner(ABC):
    """Abstract base for all executable units, agents and workflows alike.

    Concrete subclasses implement :meth:`run`. The legacy ``run_streamed()`` alias is
    kept for back-compat.
    """

    @abstractmethod
    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Run synchronously (stream=False) or streaming (stream=True)."""
        ...

    def run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        """Compatibility alias for ``run(..., stream=True)``."""
        return self.run(task, generate_kwargs=generate_kwargs, stream=True, images=images)

    @property
    @abstractmethod
    def messages(self) -> MessageHistory:
        """Message histories of all agents in this runner, keyed by agent name.

        For a leaf agent: ``{agent.name: model_client.messages}``. For composite
        workflows: dicts from every constituent runner merged into one, so the result
        spans the full sub-tree regardless of nesting depth.

        Note: when agents share a single ``ModelClient`` (e.g. via ``Chain.from_config``),
        all agents reference the same messages list. After the run every key in the
        returned dict points at the last step's messages.
        """
        ...


class BaseAgent(Runner):
    """Marker ABC for autonomous agents — the LLM directs tool use and stop condition.

    Per Anthropic's "Building Effective Agents" taxonomy: the model autonomously
    directs the process and tool selection rather than following a predetermined
    code path. Concrete subclasses: :class:`Agent`, :class:`SkillAgent`,
    :class:`OrchestratorAgent`.
    """


class Workflow(Runner):
    """Marker ABC for code-controlled patterns — routing, sequencing, aggregation.

    Per Anthropic's "Building Effective Agents" taxonomy: LLMs and tools operate
    through predetermined steps. Concrete subclasses: :class:`Chain`,
    :class:`Router`, :class:`Parallel`, :class:`EvaluatorOptimizer`.
    """
