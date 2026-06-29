"""Tool-call approval: an optional gate run right before each tool invocation.

A `ToolApproval` policy receives a tool's name and the model-supplied arguments and returns
whether the call may proceed. It lets a host require confirmation for risky tools (skill shell
scripts, ``execute_python``, filesystem writes, remote MCP tools) without changing the model
client or the tools themselves. The default policy, :func:`approve_all`, approves everything, so
the gate is inert until a caller sets one.

On the async surface a policy may be a coroutine function (the dispatcher awaits it); on the sync
surface it must be a plain function. Set it on a client (``client.tool_approval = policy``) for
bare ``chat()``, or on an ``Agent`` (``Agent(tool_approval=policy)`` / ``run(tool_approval=...)``),
mirroring how ``deps`` / ``ToolContext`` injection is plumbed.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Union

# A policy: given (tool_name, arguments) -> may this call proceed? Sync returns bool; async may
# return an awaitable bool (awaited only on the async dispatch path).
ToolApproval = Callable[[str, dict], Union[bool, Awaitable[bool]]]


def approve_all(tool_name: str, arguments: dict) -> bool:
    """The default policy: approve every tool call (no behavior change)."""
    return True
