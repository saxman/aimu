# `aimu.tools`

In-process `@tool` decorator and cross-process `MCPClient`.

## Decorator

::: aimu.tools.tool

::: aimu.tools.ToolSignatureError

## MCP client

::: aimu.tools.MCPClient

::: aimu.tools.MCPConnectionError

## Built-in tools

The `aimu.tools.builtin` module ships ready-made `@tool` functions grouped by domain:

| Group | Tools |
|---|---|
| `builtin.web` | `get_weather`, `get_webpage`, `search`, `wikipedia` |
| `builtin.fs` | `list_directory`, `read_file` |
| `builtin.compute` | `calculate` |
| `builtin.misc` | `echo`, `get_current_date_and_time` |
| `builtin.ALL_TOOLS` | All of the above |

::: aimu.tools.builtin.echo

::: aimu.tools.builtin.get_current_date_and_time

::: aimu.tools.builtin.get_weather

::: aimu.tools.builtin.calculate

::: aimu.tools.builtin.get_webpage

::: aimu.tools.builtin.search

::: aimu.tools.builtin.wikipedia

::: aimu.tools.builtin.list_directory

::: aimu.tools.builtin.read_file
