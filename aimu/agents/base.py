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

import re
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
        workflows: dicts from every constituent runner merged into one, so the result
        spans the full sub-tree regardless of nesting depth.

        Note: when agents share a single ``ModelClient`` (e.g. via ``Chain.from_config``),
        all agents reference the same messages list. After the run every key in the
        returned dict points at the last step's messages.
        """
        ...

    def as_tool(self, *, name: Optional[str] = None, description: Optional[str] = None) -> Callable:
        """Wrap this runner as a ``@tool``-style callable: ``tool(task: str) -> str``.

        The returned callable runs :meth:`run` and returns its string result, so any agent
        or workflow can call this runner as a tool — drop it into ``Agent(tools=[...])`` or
        an :class:`OrchestratorAgent`'s worker list. Works for every concrete ``Runner``
        (``Agent``, ``Chain``, ``Router``, a remote A2A agent, ...) since it only relies on
        ``run()``.

        ``name`` defaults to ``self.name`` (sanitised to a valid identifier) or ``"runner"``.
        ``description`` defaults to the first line of ``self.system_message`` when present
        (``Agent`` / ``SkillAgent``), else a generic delegation string (workflows have no
        ``system_message``).
        """
        from aimu.tools.decorator import tool

        resolved_name = name or getattr(self, "name", None) or "runner"
        safe_name = re.sub(r"\W+", "_", resolved_name).strip("_") or "runner"

        if description is None:
            system_message = getattr(self, "system_message", None)
            description = (
                system_message.splitlines()[0]
                if system_message
                else f"Delegate a task to the {safe_name} runner."
            )

        def _dispatch(task: str) -> str:
            return self.run(task)

        _dispatch.__name__ = safe_name
        _dispatch.__qualname__ = safe_name
        _dispatch.__doc__ = description
        _dispatch.__annotations__ = {"task": str, "return": str}
        return tool(_dispatch)
