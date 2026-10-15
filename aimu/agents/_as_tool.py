"""Shared builder for ``Runner.as_tool()`` / ``AsyncRunner.as_tool()``.

Both surfaces resolve the tool name and description and stamp the dispatch callable
identically; only the dispatch itself differs (sync ``return self.run(task)`` vs async
``return await self.run(task)``). The caller builds that callable; this helper does the
rest and applies the ``@tool`` decorator.
"""

from __future__ import annotations

import re
import inspect
from typing import Callable, Optional

from aimu.tools.context import ToolContext


def accepts_run_identity(runner) -> bool:
    """Whether ``runner.run`` takes ``run_id`` and ``parent_run_id``.

    Inspected once, when the tool is built. ``Agent`` and ``OrchestratorAgent`` do; workflows
    (``Chain``, ``Router``, ``Parallel``) and ``RemoteAgent`` do not, and are called with the
    task alone.
    """
    parameters = inspect.signature(runner.run).parameters
    return "run_id" in parameters and "parent_run_id" in parameters


def run_identity(runner, ctx: Optional[ToolContext]) -> dict:
    """The ``run_id`` / ``parent_run_id`` keywords for one delegated run: a fresh id, parented to the caller."""
    from aimu.agents._tool_loop import new_run_id

    return {
        "run_id": new_run_id(getattr(runner, "name", None)),
        "parent_run_id": ctx.run_id if ctx is not None else None,
    }


def build_as_tool(runner, dispatch: Callable, *, name: Optional[str], description: Optional[str]) -> Callable:
    """Stamp *dispatch* with a resolved name/description and return it as a ``@tool``.

    ``name`` defaults to ``runner.name`` (sanitised to a valid identifier) or ``"runner"``.
    ``description`` defaults to the first line of ``runner.system_message`` when present,
    else a generic delegation string.
    """
    from aimu.tools.decorator import tool

    resolved_name = name or getattr(runner, "name", None) or "runner"
    safe_name = re.sub(r"\W+", "_", resolved_name).strip("_") or "runner"

    if description is None:
        system_message = getattr(runner, "system_message", None)
        description = (
            system_message.splitlines()[0] if system_message else f"Delegate a task to the {safe_name} runner."
        )

    dispatch.__name__ = safe_name
    dispatch.__qualname__ = safe_name
    dispatch.__doc__ = description
    dispatch.__annotations__ = {"task": str, "ctx": Optional[ToolContext], "return": str}
    return tool(dispatch)
