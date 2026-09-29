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

from dataclasses import dataclass
from typing import Awaitable, Callable, Union


@dataclass(frozen=True)
class Denied:
    """A refusal that says why, returned by a policy in place of ``False``.

    The engine renders ``reason`` into the tool message the model sees, so a refused call can be
    corrected rather than merely retried. A bare ``False`` says only that something was
    disallowed, and a model's obvious next move is the same call again.

    ``reason`` is written for the model: name the constraint and, where there is one, the thing
    that would satisfy it ("only api.example.com is allowed"). Note that it lands in the
    conversation, so it is readable by the model and by anything replaying the transcript --
    a host that considers its own policy sensitive should keep the reason vague and return a
    plain ``False`` instead. An empty ``reason`` renders as the unadorned refusal.

    A type rather than a bare string because a policy's return value passes through ``bool()``:
    a non-empty string already means *approved*, so repurposing one as a refusal would silently
    invert any policy returning one. Mirrors ``Unsupported(remedy)`` in
    ``aimu.models._internal.generate_kwargs``, the same verdict-with-a-remedy shape.
    """

    reason: str = ""

    def __bool__(self) -> bool:
        """Always falsy, so ``Denied`` substitutes for ``False`` at any truthiness check.

        Without this a plain object is truthy, and every ``if not approved:`` in the engine --
        including a host's own policy composing two verdicts -- would read a refusal as an
        approval. That is the one failure mode this type must not have.
        """
        return False


# A policy: given (tool_name, arguments) -> may this call proceed? Return ``True`` to allow,
# ``False`` to refuse, or ``Denied(reason)`` to refuse and tell the model why. Async may return an
# awaitable of any of those (awaited only on the async dispatch path).
ToolApproval = Callable[[str, dict], Union[bool, "Denied", Awaitable[Union[bool, "Denied"]]]]


def approve_all(tool_name: str, arguments: dict) -> bool:
    """The default policy: approve every tool call (no behavior change)."""
    return True
