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
| `builtin.compute` | `calculate`, `execute_python`, `run_command` |
| `builtin.time` | `get_current_date_and_time`, `convert_time` |
| `builtin.misc` | `echo` |
| `builtin.ALL_TOOLS` | All of the above **except** `execute_python` and `run_command` (isolation, not containment; opt in via `builtin.compute`) |

`execute_python` runs code in a fresh subprocess (its own hard timeout, crash isolation,
and no access to this process's environment variables -- but it does not confine the
filesystem or the network; see its docstring). `execute_python_in_process` is the
explicit opt-in for trusted code where subprocess startup cost matters; it is not
included in `builtin.compute` or `builtin.ALL_TOOLS`.

`run_command` runs a command line through `/bin/sh -c` (`COMSPEC /c` on Windows), sharing
`execute_python`'s subprocess supervisor rather than a separate implementation:

```python
run_command(command, cwd="", timeout=30)
```

`timeout` is seconds, clamped to 600. It returns the exit code plus stdout and stderr labelled
separately; a nonzero exit returns that output rather than an error string, since `pytest` exits 1
with the answer on stdout and `git diff --exit-code` exits 1 to mean "yes, there is a diff." Unlike
`execute_python`, there is no memory cap: a 512 MB address-space limit breaks compilers and test
suites, and imposing one on a shell child needs `preexec_fn`, which is neither portable nor safe
alongside threads.

**Not a security boundary.** This is isolation, not containment, one step sharper than
`execute_python`: the command reaches credentials sitting in files (a `.env`, `~/.aws/credentials`)
as the calling user, and process signalling is unconfined, so `kill -9` against the host process is
one command away. Gate it with `tool_approval` for untrusted callers and reach for a container when
you need real containment.

Unlike `execute_python`, `run_command` is not added by `make_tools(allow_code_execution=True)`:
that flag names *code* execution, and widening it would hand a shell to every caller already
passing it, so `builtin.compute` is the only route in. `make_command_tool(env_passthrough=...)`
builds a variant whose child also sees the named environment variables, for callers that need `gh`
or `ssh` to work; `run_command` itself is that factory called with no arguments, so no extra
variable reaches the child beyond its default allowlist.

::: aimu.tools.builtin.echo

::: aimu.tools.builtin.get_current_date_and_time

::: aimu.tools.builtin.convert_time

::: aimu.tools.builtin.get_weather

::: aimu.tools.builtin.calculate

::: aimu.tools.builtin.execute_python

::: aimu.tools.builtin.execute_python_in_process

::: aimu.tools.builtin.get_webpage

::: aimu.tools.builtin.get_webpage_html

::: aimu.tools.builtin.web_search

::: aimu.tools.builtin.wikipedia

::: aimu.tools.builtin.list_directory

::: aimu.tools.builtin.read_file

## Tool factories

Bind a tool to a specific resource (a memory store, a knowledge base) or policy (a command's
environment allowlist) instead of a process-wide singleton.

::: aimu.tools.builtin.make_command_tool

::: aimu.tools.builtin.make_memory_tools

::: aimu.tools.builtin.make_retrieval_tool

::: aimu.tools.builtin.make_subagent_tool

::: aimu.tools.builtin.make_web_tools
