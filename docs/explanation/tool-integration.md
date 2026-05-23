# Tool integration

AIMU exposes tools to a model in two ways. They look different but end up in the same place — a list of OpenAI-format tool specs that the model client passes to the provider. This page explains the two routes, the dispatch order, and when to pick which.

## The two routes

### In-process: `@tool` decorator

Decorate a Python function:

```python
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())
```

The decorator inspects the signature and docstring at decoration time and attaches an OpenAI-format spec to `func.__tool_spec__`. The function itself is unchanged.

You hand the function to a client:

```python
model_client.tools = [letter_counter]
# or
agent = Agent(model_client, tools=[letter_counter])
```

Dispatch is a direct function call. No serialization, no transport, no separate process. Best when you own the code, the dependencies are already in your agent's environment, and you want minimal overhead.

### Cross-process: `MCPClient`

For tools that run elsewhere — a subprocess, a shared server, a remote machine — wrap them with `MCPClient`:

```python
from aimu.tools import MCPClient

model_client.mcp_client = MCPClient({
    "mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}},
})
```

Arguments and results cross the process boundary as JSON via the [Model Context Protocol](https://modelcontextprotocol.io/). Best when the tool is written in another language, ships as a binary, is shared across agents or users, has conflicting dependencies, or needs sandboxing.

## How they coexist

Both routes can be active on the same client. The base class assembles the request `tools` list in `_chat_setup()`:

```python
# from aimu/models/base.py
tools = []
if self.model.supports_tools and use_tools:
    if self.mcp_client:
        tools.extend(self.mcp_client.get_tools())
    for fn in self.tools:
        tools.append(fn.__tool_spec__)
```

Note the order: MCP tools first, then Python `@tool` functions. The model sees both groups as a flat list, picks whichever it wants.

## Dispatch order

When the model returns a tool call, `_handle_tool_calls()` resolves the name in this order:

1. **Python `@tool` functions first.** A lookup table `{fn.__name__: fn for fn in self.tools}` is built; if the called name matches, it's invoked in-process.
2. **MCP tools second.** If no Python tool matches, the request goes to `self.mcp_client.call_tool(...)`.
3. **Otherwise, a "tool not found" message is appended.** The model sees this and can decide what to do.

So Python `@tool` wins on name collision. This is intentional — if you've defined a local override of an MCP tool, you almost certainly want the local version.

## The single source of truth

`aimu.tools.builtin` defines every built-in tool exactly once, as a `@tool`-decorated Python function. The same callables back both routes:

- For in-process use: import `from aimu.tools import builtin` and pass `builtin.web` (etc.) to your agent.
- For cross-process use: run `python -m aimu.tools.mcp`, which is a FastMCP server that registers `builtin.ALL_TOOLS`.

There is no second copy. If you add a new built-in tool, it appears on both routes automatically.

## When to pick which

| Situation | Pick |
|---|---|
| You own the code, in Python, with the same dependencies as your agent | **`@tool`** |
| The tool is small (no DB, no heavy state) | **`@tool`** |
| The tool is in another language or ships as a binary | **`MCPClient`** |
| Many agents share the tool catalogue | **`MCPClient`** to a shared server |
| You want sandboxing or process isolation | **`MCPClient`** |
| Conflicting dependencies (e.g. your tool needs PyTorch but your agent shouldn't) | **`MCPClient`** to an isolated subprocess |

If you can't decide, start with `@tool`. Switching to `MCPClient` later is a one-line change: lift the function to a separate file, run it as an MCP server, set `model_client.mcp_client`.

## Failure modes that are now loud

- **Bad `@tool` signature** — `ToolSignatureError` raised at decoration time. Variadic params (`*args` / `**kwargs`) and required params without type hints are the two cases.
- **MCP connection error** — `MCPConnectionError` raised at `MCPClient` construction or on `call_tool()` failure. `MCPClient.ping()` lets you verify a connection upfront.
- **Tool not found at dispatch** — appended as a `tool` role message with the text `"Tool 'X' not found."`. The model sees this and can adjust.

## See also

- [How-to: add a custom tool](../how-to/add-custom-tool.md) — `@tool` signature rules and patterns.
- [How-to: use MCP tools](../how-to/use-mcp-tools.md) — cross-process tool setup.
- [Architecture](architecture.md) — how `_chat_setup` and `_handle_tool_calls` fit into `BaseModelClient`.
