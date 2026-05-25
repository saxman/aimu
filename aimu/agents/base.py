"""The `Runner` hierarchy.

Decision tree for picking the right class:

* **Single ``chat()`` call, no loop** → :class:`aimu.models.ModelClient`
* **Tool-using loop until model stops** → :class:`Agent`
* **Skills auto-discovered from disk** → :class:`SkillAgent`
* **Orchestrator dispatching to worker agents via tool calls** → :class:`OrchestratorAgent`
* **Fixed sequence / classifier / parallel / generate-critique** →
  :class:`Chain` / :class:`Router` / :class:`Parallel` / :class:`EvaluatorOptimizer`
* **Plan → execute with tools → score → replan on fail** → :class:`PlanExecuteEvaluator`
* **Drop an agent into an API that wants a ``ModelClient``** → ``agent.as_model_client()``

All concrete classes implement the single :class:`Runner` interface
(``run()`` + ``messages``). The "autonomous vs code-controlled" split is a
conceptual categorisation of those concrete classes — autonomous agents
(``Agent``, ``SkillAgent``, ``OrchestratorAgent``) let the LLM direct the
flow; workflow patterns (``Chain``, ``Router``, ``Parallel``,
``EvaluatorOptimizer``, ``PlanExecuteEvaluator``) keep the flow fixed in
code. The distinction lives in the docs, not in the type hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Union

from aimu.models.base import StreamChunk

# Type alias used by the messages property across the runner hierarchy.
MessageHistory = dict[str, list[dict]]


class Runner(ABC):
    """Abstract base for every concrete agent and workflow in AIMU.

    Concrete subclasses implement :meth:`run` and :attr:`messages`.
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

    @property
    @abstractmethod
    def messages(self) -> MessageHistory:
        """Message histories of all sub-runners, keyed by runner name.

        For a leaf agent: ``{agent.name: model_client.messages}``. For composite
        workflows: dicts from every constituent runner merged into one, so the result
        spans the full sub-tree regardless of nesting depth.

        Note: when agents share a single ``ModelClient`` (e.g. via ``Chain.from_config``),
        all agents reference the same messages list. After the run every key in the
        returned dict points at the last step's messages.
        """
        ...
