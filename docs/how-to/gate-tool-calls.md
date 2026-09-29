# Gate tool calls (approval hook)

An agent with tools can do real things: run skill scripts, `execute_python`, write files, call a
remote MCP service. The **tool-approval hook** lets you decide, per call, whether a tool may run.
It's optional and off by default (everything is approved), so existing code is unchanged until you
set a policy.

## The policy

A policy is a callable `(tool_name, arguments) -> bool | Denied` run right before each tool
invocation. Return `False` to block the call: the tool does not run, and a refusal tool message
(`"Tool '<name>' was not approved."`) is appended so the model sees it and can react. The type alias
`ToolApproval`, the `Denied` verdict below, and the default `approve_all` are exported from `aimu`.

```python
import aimu

RISKY = {"execute_python", "run_command", "write_file", "edit_file"}

def confirm_risky(name: str, arguments: dict) -> bool:
    if name not in RISKY:
        return True
    answer = input(f"Allow {name}({arguments})? [y/N] ")
    return answer.strip().lower() == "y"
```

## Say why you refused

A bare `False` tells the model only that *something* was disallowed. It does not say what, so its
most likely next move is the identical call, and it burns iterations discovering nothing. Return
`Denied(reason)` instead and the reason reaches the model in the tool message, which is what lets it
correct the argument rather than repeat it:

```python
from urllib.parse import urlparse

import aimu

ALLOWED_HOSTS = {"api.example.com", "docs.example.com"}

def allowlist_hosts(name: str, arguments: dict) -> bool | aimu.Denied:
    if name not in ("submit_json", "submit_form", "get_web_content"):
        return True
    host = urlparse(arguments.get("url", "")).hostname or ""
    if host in ALLOWED_HOSTS:
        return True
    return aimu.Denied(f"{host!r} is not allowed; permitted hosts are {sorted(ALLOWED_HOSTS)}")
```

The model then sees `Tool 'submit_json' was not approved: 'evil.example' is not allowed; permitted
hosts are ['api.example.com', 'docs.example.com']`.

This is the shape to reach for when you want an **argument-scoped** rule rather than a per-tool one.
Because the gate runs at dispatch, it covers every tool uniformly: built-ins, `@tool` functions, and
MCP tools alike, including tools that expose no factory to configure.

Three properties worth knowing:

- `Denied` is **falsy**, so it substitutes for `False` anywhere you compose verdicts
  (`return cheap_check() and expensive_check()` behaves).
- `Denied("")` renders as the plain refusal, so an empty reason is the same as `False`.
- The reason **lands in the conversation** and is visible to the model and to anything replaying the
  transcript. If your policy itself is sensitive (internal hostnames, say), keep the reason vague or
  return a plain `False`.

A reason is *not* a security boundary: it constrains the arguments the model chose, and a tool that
follows a redirect elsewhere is still doing that on its own. Gate on what you can check here, and
keep the tool's own limits in the tool.

## Use it

Approval is part of the agent's tool-loop, so it lives on the `Agent` — a constructor field and a
per-run override, mirroring `deps=`. (Tools are only ever executed by an agent, so there is no
bare-client approval path.)

```python
client = aimu.client("ollama:qwen3:8b")

agent = aimu.agents.Agent(client, tools=[my_tool], tool_approval=confirm_risky)
agent.run("do the thing")                      # agent-level default
agent.run("do it", tool_approval=confirm_risky)  # per-run override
```

The policy sees the tool name and the model-supplied arguments, so you can gate only the risky
tools and approve the rest. It covers every dispatch path the engine runs: non-streaming, streaming,
and concurrent (when the agent sets `concurrent_tool_calls=True`).

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

A sync agent requires a sync policy; handing it a coroutine raises a clear error pointing at the
`aimu.aio` surface. An async policy may return `Denied(reason)` too — the awaited result is the
verdict, so it works the same as on the sync surface.

## Worked example

The [personal-assistant example](build-personal-assistant.md) gates its full-access
`add_skill_script` tool with a terminal y/n prompt by default (see its `CONFIRM_BEFORE` set), so the
user confirms before the assistant writes and runs code.

## See also

- [Add a custom tool](add-custom-tool.md) and [Use MCP tools](use-mcp-tools.md): the tools a policy gates
- [Build a personal assistant](build-personal-assistant.md): the approval demo in context
