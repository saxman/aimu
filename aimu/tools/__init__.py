"""AIMU tool integration.

There are two routes for exposing tools to an agent:

* **In-process**: decorate a Python function with ``@tool``. Pass the function to
  ``Agent(client, tools=[fn])`` or set ``model_client.tools = [fn]``. This is the
  default and recommended path for code you control.

* **Cross-process**: wrap an MCP server (or external command) with :class:`MCPClient`.
  Use this only when you need to integrate a tool server you don't control. For
  *sharing* AIMU tools across processes, register them on a FastMCP server yourself;
  there's no second framework to learn.

Built-in tools live in :mod:`aimu.tools.builtin` and are grouped by domain
(``builtin.web``, ``builtin.fs``, ``builtin.compute``, ``builtin.misc``). Pass a
group directly: ``Agent(client, tools=builtin.web)``.
"""

from . import builtin
from .approval import ToolApproval, approve_all
from .client import MCPClient, MCPConnectionError
from .context import ToolContext
from .decorator import ToolArgumentError, ToolSignatureError, coerce_tool_arguments, tool

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
