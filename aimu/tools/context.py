"""Tool dependency injection: ``ToolContext``.

A tool parameter annotated as :class:`ToolContext` (or ``ToolContext[Deps]``) is *injected*
by the agent at call time and excluded from the tool's JSON schema, so the model never sees
it. This lets tools reach shared state (a document store, a cache, configuration) without
module-level globals.

The value supplied as ``ctx.deps`` comes from the ``Agent``: either its ``deps=`` field or a
per-run ``run(..., deps=...)`` override.

Usage::

    from dataclasses import dataclass, field

    import aimu
    from aimu import ToolContext
    from aimu.agents import Agent

    @dataclass
    class Deps:
        seen: dict = field(default_factory=dict)

    @aimu.tool
    def remember(ctx: ToolContext[Deps], key: str, value: str) -> str:
        "Store a value."  # the model only sees `key` and `value`
        ctx.deps.seen[key] = value
        return "ok"

    agent = Agent(client, tools=[remember], deps=Deps())
    agent.run("remember that the sky is blue")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class ToolContext(Generic[T]):
    """Framework-supplied context handed to a tool that declares it.

    ``deps`` is whatever the ``Agent`` was given (its ``deps=`` field, or a ``run(deps=)``
    override); it is ``None`` when no deps were provided (e.g. a bare ``client.chat()`` call).
    """

    deps: Optional[T] = None
