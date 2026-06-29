# Gate tool calls (approval hook)

An agent with tools can do real things: run skill scripts, `execute_python`, write files, call a
remote MCP service. The **tool-approval hook** lets you decide, per call, whether a tool may run.
It's optional and off by default (everything is approved), so existing code is unchanged until you
set a policy.

## The policy

A policy is a callable `(tool_name, arguments) -> bool` run right before each tool invocation.
Return `False` to block the call: the tool does not run, and a refusal tool message
(`"Tool '<name>' was not approved."`) is appended so the model sees it and can react. The type alias
`ToolApproval` and the default `approve_all` are exported from `aimu`.

```python
import aimu

RISKY = {"execute_python", "write_file"}

def confirm_risky(name: str, arguments: dict) -> bool:
    if name not in RISKY:
        return True
    answer = input(f"Allow {name}({arguments})? [y/N] ")
    return answer.strip().lower() == "y"
```

## Use it

Set it on a client for a bare `chat()` loop, or on an agent (a constructor field and a per-run
override, mirroring `deps=`):

```python
client = aimu.client("ollama:qwen3:8b")
client.tools = [my_tool]
client.tool_approval = confirm_risky          # bare client

agent = aimu.agents.Agent(client, tools=[my_tool], tool_approval=confirm_risky)
agent.run("do the thing")                      # agent-level default
agent.run("do it", tool_approval=confirm_risky)  # per-run override
```

The policy sees the tool name and the model-supplied arguments, so you can gate only the risky
tools and approve the rest. It covers every dispatch path: non-streaming, streaming, and concurrent
(`concurrent_tool_calls=True`).

## Async

On `aimu.aio` the policy may be a coroutine function (it is awaited), so it can do async I/O such as
asking the user over a chat channel:

```python
from aimu import aio

async def confirm(name, arguments):
    if name not in RISKY:
        return True
    return await ask_user_yes_no(f"Allow {name}?")

agent = aio.Agent(aio.client("ollama:qwen3:8b"), tools=[my_tool], tool_approval=confirm)
```

A sync client requires a sync policy; handing it a coroutine raises a clear error pointing at the
`aimu.aio` surface.

## Worked example

The [personal-assistant example](build-personal-assistant.md) gates its full-access
`add_skill_script` tool with a terminal y/n prompt by default (see its `CONFIRM_BEFORE` set), so the
user confirms before the assistant writes and runs code.

## See also

- [Add a custom tool](add-custom-tool.md) and [Use MCP tools](use-mcp-tools.md): the tools a policy gates
- [Build a personal assistant](build-personal-assistant.md): the approval demo in context
