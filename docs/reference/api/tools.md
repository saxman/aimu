# `aimu.tools`

In-process `@tool` decorator and cross-process `MCPClient`.

## Decorator

::: aimu.tools.tool

::: aimu.tools.ToolSignatureError

## Argument validation

::: aimu.tools.coerce_tool_arguments

::: aimu.tools.ToolArgumentError

## MCP client

::: aimu.tools.MCPClient

::: aimu.tools.MCPConnectionError

## Built-in tools

The `aimu.tools.builtin` module ships ready-made `@tool` functions grouped by domain:

| Group | Tools |
|---|---|
| `builtin.web` | `get_weather`, `get_webpage`, `get_webpage_html`, `web_search`, `wikipedia` |
| `builtin.fs` | `list_directory`, `read_file` |
| `builtin.compute` | `calculate`, `execute_python` |
| `builtin.misc` | `echo`, `get_current_date_and_time` |
| `builtin.ALL_TOOLS` | All of the above **except** `execute_python` (sandboxed REPL; opt in via `builtin.compute`) |

::: aimu.tools.builtin.echo

::: aimu.tools.builtin.get_current_date_and_time

::: aimu.tools.builtin.get_weather

::: aimu.tools.builtin.calculate

::: aimu.tools.builtin.execute_python

::: aimu.tools.builtin.get_webpage

::: aimu.tools.builtin.get_webpage_html

::: aimu.tools.builtin.web_search

::: aimu.tools.builtin.wikipedia

::: aimu.tools.builtin.list_directory

::: aimu.tools.builtin.read_file

## Tool factories

Bind a tool to a specific resource (a memory store, a knowledge base) instead of a
process-wide singleton.

::: aimu.tools.builtin.make_memory_tools

::: aimu.tools.builtin.make_retrieval_tool

::: aimu.tools.builtin.make_subagent_tool

::: aimu.tools.builtin.make_web_tools
