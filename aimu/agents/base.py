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
conceptual categorisation of those concrete classes: autonomous agents
(``Agent``, ``SkillAgent``, ``OrchestratorAgent``) let the LLM direct the
flow; workflow patterns (``Chain``, ``Router``, ``Parallel``,
``EvaluatorOptimizer``, ``PlanExecuteEvaluator``) keep the flow fixed in
code. The distinction lives in the docs, not in the type hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Optional, Union

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
        workflows: the per-runner dicts merged into one (via ``dict.update``), so the
        result spans the full sub-tree regardless of nesting depth. Example::

            router.messages
            # {"router-classifier": [...], "code-handler": [...], "prose-handler": [...]}

        Merge is **by runner name**: two sub-runners sharing a name collide and the one
        merged later wins (last-write). Give sub-runners distinct ``name=``s if you need
        every history addressable.

        Separately, when sub-runners share a single ``ModelClient`` (e.g. via
        ``Chain.from_config``), they all reference the *same* messages list, so after a run
        every key points at that one shared (last step's) history. Use distinct clients per
        sub-runner to keep histories separate.
        """
        ...

    def as_tool(self, *, name: Optional[str] = None, description: Optional[str] = None) -> Callable:
        """Wrap this runner as a ``@tool``-style callable: ``tool(task: str) -> str``.

        The returned callable runs :meth:`run` and returns its string result, so any agent
        or workflow can call this runner as a tool: drop it into ``Agent(tools=[...])`` or
        an :class:`OrchestratorAgent`'s worker list. Works for every concrete ``Runner``
        (``Agent``, ``Chain``, ``Router``, a remote A2A agent, ...) since it only relies on
        ``run()``.

        ``name`` defaults to ``self.name`` (sanitised to a valid identifier) or ``"runner"``.
        ``description`` defaults to the first line of ``self.system_message`` when present
        (``Agent`` / ``SkillAgent``), else a generic delegation string (workflows have no
        ``system_message``).
        """
        from aimu.agents._as_tool import build_as_tool

        def _dispatch(task: str) -> str:
            return self.run(task)

        return build_as_tool(self, _dispatch, name=name, description=description)
