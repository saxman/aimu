"""Shared builder for ``Runner.as_tool()`` / ``AsyncRunner.as_tool()``.

Both surfaces resolve the tool name and description and stamp the dispatch callable
identically; only the dispatch itself differs (sync ``return self.run(task)`` vs async
``return await self.run(task)``). The caller builds that callable; this helper does the
rest and applies the ``@tool`` decorator.
"""

from __future__ import annotations

import re
from typing import Callable, Optional


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
    dispatch.__annotations__ = {"task": str, "return": str}
    return tool(dispatch)
