"""AIMU tool integration.

There are two routes for exposing tools to an agent:

* **In-process**: decorate a Python function with ``@tool`` and pass it to
  ``Agent(client, tools=[fn])``. The Agent executes tools (via its tool-loop engine);
  a bare ``client.chat("q", tools=[fn])`` only *advertises* and parses tool calls, it
  does not run them. This is the default and recommended path for code you control.

* **Cross-process**: wrap an MCP server (or external command) with :class:`MCPClient`.
  Use this only when you need to integrate a tool server you don't control. For
  *sharing* AIMU tools across processes, register them on a FastMCP server yourself;
  there's no second framework to learn.

Built-in tools live in :mod:`aimu.tools.builtin` and are grouped by domain
(``builtin.web``, ``builtin.fs``, ``builtin.compute``, ``builtin.time``,
``builtin.misc``). Pass a group directly: ``Agent(client, tools=builtin.web)``.
"""

from importlib import import_module

from . import builtin
from .approval import ToolApproval, approve_all
from .context import ToolContext
from .decorator import ToolArgumentError, ToolSignatureError, coerce_tool_arguments, tool

# MCPClient / MCPConnectionError live in .client, which imports fastmcp at module level.
# fastmcp is a required (not optional) dependency, so this isn't the HAS_*-gated pattern
# used for provider SDKs elsewhere -- it's a plain lazy import, deferring the cost of
# loading fastmcp (and its own dependency tree) until a caller actually touches MCP.
_LAZY_CLIENT_SYMBOLS = frozenset({"MCPClient", "MCPConnectionError"})


def __getattr__(name: str):
    if name in _LAZY_CLIENT_SYMBOLS:
        return getattr(import_module(".client", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_CLIENT_SYMBOLS})


__all__ = [
    "MCPClient",
    "MCPConnectionError",
    "ToolApproval",
    "ToolArgumentError",
    "ToolContext",
    "ToolSignatureError",
    "approve_all",
    "builtin",
    "coerce_tool_arguments",
    "tool",
]
