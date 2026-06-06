# Tool integration

AIMU exposes tools to a model in two ways. They look different but end up in the same place — a list of OpenAI-format tool specs that the model client passes to the provider. This page explains the two routes, the dispatch order, and when to pick which.

## The two routes

### In-process: `@tool` decorator

Decorate a Python function:

```python
import aimu

@aimu.tool
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

For tools that run elsewhere — a subprocess, a shared server, a remote machine — wrap them with `MCPClient`, then call `.as_tools()` to turn the server's tools into callables and add them to `client.tools`:

```python
from aimu.tools import MCPClient

mcp = MCPClient({
    "mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}},
})
model_client.tools = mcp.as_tools()        # or: builtin.web + mcp.as_tools()
```

Each callable returned by `as_tools()` closes over the client, invokes `call_tool()` cross-process, and returns the result's text — but to the model client it looks like any other `@tool`. Arguments and results cross the process boundary as JSON via the [Model Context Protocol](https://modelcontextprotocol.io/). Best when the tool is written in another language, ships as a binary, is shared across agents or users, has conflicting dependencies, or needs sandboxing.

Keep a reference to the `MCPClient` (the `as_tools()` callables hold one, so the connection stays alive) for its lifetime — and to call `.ping()` or refresh the tool list. The list `as_tools()` returns is a snapshot; call it again to pick up server-side changes.

## How they coexist

There is only one tool registry: `client.tools`. Both routes produce plain callables that live there — `@tool` functions directly, MCP tools via `as_tools()`. The base class assembles the request `tools` list in `_chat_setup()` from that one place:

```python
# from aimu/models/_base/text.py
tools = []
if self.model.supports_tools and use_tools:
    tools.extend(self._collect_python_tool_specs())   # __tool_spec__ from every callable in self.tools
```

The model sees one flat list and picks whichever it wants. There's no separate MCP reference on the client and no second assembly path.

## Per-call tool override

`self.tools` is the client's *configured* set of Python tools. Sometimes you want a different set for a single call without mutating that state — a quick lookup that should only see `search`, a turn where tools should be off entirely. Both `chat()` and `Agent.run()` take a `tools=` keyword for exactly this:

```python
client.tools = [search, calculate]      # configured set

client.chat("normal question")           # sees search + calculate
client.chat("just look this up", tools=[search])  # sees only search, this call
client.chat("answer from memory", tools=[])       # no Python tools, this call

agent = Agent(client, tools=[search, calculate])
agent.run("research task")               # uses configured tools
agent.run("quick lookup", tools=[search])  # override for the whole run (every loop turn)
```

`tools=None` (the default) uses the configured `self.tools` — unchanged behaviour. Any other value, *including `[]`* to disable tools, replaces them for the duration of that call and is restored afterward. Since MCP tools also live in `self.tools` (via `as_tools()`), the override covers them too. On an `Agent`, the override applies to every `chat()` in the agentic loop (the initial turn and all continuations), then the agent's configured tools are back in place.

Mechanically it's a scoped swap of `self.tools` that covers both request-spec building and dispatch (both read `self.tools`). Like `self.messages`, it isn't safe across *concurrent* `chat()` calls on a shared client — which matches the existing single-conversation contract.

## Dispatch

When the model returns a tool call, `_call_plain_tool()` resolves the name against one lookup table built from the single registry, `{fn.__name__: fn for fn in self.tools}`:

1. **Match → invoke the callable in-process.** For a `@tool` function that's a direct call; for an `as_tools()` wrapper it's a cross-process `call_tool()` behind the same calling convention.
2. **No match → a "tool not found" message is appended.** The model sees this and can decide what to do.

Because it's one dict keyed by name, a **name collision resolves to the last entry in the list**. So to shadow an MCP tool with a local Python implementation, append the Python tool after `mcp.as_tools()`: `tools = mcp.as_tools() + [my_override]`.

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

If you can't decide, start with `@tool`. Switching to `MCPClient` later is a small change: lift the function to a separate file, run it as an MCP server, and swap `tools=[fn]` for `tools=mcp.as_tools()`.

## Failure modes that are now loud

- **Bad `@tool` signature** — `ToolSignatureError` raised at decoration time. Variadic params (`*args` / `**kwargs`) and required params without type hints are the two cases.
- **MCP connection error** — `MCPConnectionError` raised at `MCPClient` construction or on `call_tool()` failure. `MCPClient.ping()` lets you verify a connection upfront.
- **Tool not found at dispatch** — appended as a `tool` role message with the text `"Tool 'X' not found."`. The model sees this and can adjust.
- **Async tool on the sync surface** — the sync `_handle_tool_calls` raises `ValueError` if a tool's `__tool_is_async__` flag is set. The message points the caller at `aimu.aio`.

## Async tools

`@tool` records `func.__tool_is_async__ = inspect.iscoroutinefunction(func)` at decoration time. The same decorated function is usable on either surface:

- On the **sync** surface, async tools raise at dispatch (loud, immediate).
- On the **async** surface, async tools are awaited directly, and sync (CPU-bound) tools are routed through `asyncio.to_thread` so the event loop stays free.

```python
@aimu.tool
async def fetch(url: str) -> str:
    """Fetch a URL."""
    ...

@aimu.tool
def normalize(text: str) -> str:
    """Trim and lowercase."""
    ...

# Both work on aio.Agent — async awaited, sync runs in a worker thread
agent = aio.Agent(client, tools=[fetch, normalize])
```

`concurrent_tool_calls=True` on the async surface uses `asyncio.TaskGroup` (structured concurrency, sibling cancellation on first failure) instead of `ThreadPoolExecutor`.

## See also

- [How-to: add a custom tool](../how-to/add-custom-tool.md) — `@tool` signature rules and patterns.
- [How-to: use MCP tools](../how-to/use-mcp-tools.md) — cross-process tool setup.
- [Architecture](architecture.md) — how `_chat_setup` and `_handle_tool_calls` fit into `BaseModelClient`.
