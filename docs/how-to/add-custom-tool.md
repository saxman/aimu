# Add a custom tool

Decorate any Python function with `@aimu.tool`. The decorator inspects the signature and docstring at decoration time and attaches an OpenAI-format spec to `func.__tool_spec__`. The function itself is unchanged and remains directly callable.

## Minimal example

```python
import aimu
from aimu.agents import Agent

@aimu.tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

## Signature rules

The decorator inspects parameters and types. Each parameter must satisfy:

- It has either a **type hint** or a **default value** (or both). Unhinted required params raise `ToolSignatureError`.
- It is *not* variadic. `*args` and `**kwargs` raise `ToolSignatureError`; declare each argument explicitly.

Supported types map to JSON Schema like this:

| Python | JSON Schema |
|---|---|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list`, `list[T]` | `array` |
| `dict`, `dict[K, V]` | `object` |
| `Optional[T]`, `T \| None` | unwrapped to `T` |

Unknown types fall back to `string`.

The first paragraph of the docstring becomes the tool description.

## Errors you'll see

```python
@aimu.tool
def bad(*args):              # ❌ variadic
    return args
# ToolSignatureError: @aimu.tool: function 'bad' uses variadic parameter '*args'...

@aimu.tool
def bad(x):                  # ❌ no hint, no default
    return x
# ToolSignatureError: @aimu.tool: parameter 'x' on 'bad' has no type hint and no default...
```

## Argument validation

Before a tool runs, AIMU validates and coerces the arguments the model supplied against your function's type hints. This happens automatically for every `@tool` function, on both the sync and async surfaces. Three things can go wrong, and each is reported back to the model as a tool result (not raised as a Python exception) so the model can correct itself and retry:

- **Coercible mismatch → coerced.** A model that returns `{"count": "5"}` for `count: int` gets `5`; `"true"` for a `bool` becomes `True`. Coercion is lax (Pydantic's `validate_python`), which absorbs the common case of a model stringifying a scalar.
- **Uncoercible value → rejected.** `{"count": "abc"}` for `count: int` produces `invalid arguments for tool 'f': 'count' input should be a valid integer...`.
- **Missing required / unknown argument → rejected.** Omitting a required parameter, or passing one not in the signature, is reported the same way (`missing required argument 'x'` / `unexpected argument 'y'`).

```python
@aimu.tool
def set_volume(level: int) -> str:
    """Set the volume (0-100)."""
    return f"volume={level}"

# Model calls set_volume(level="40")  ->  the tool runs with level == 40 (int)
# Model calls set_volume(level="loud") ->  tool body never runs; the model is told 'level' must be an integer
```

The validation metadata (a Pydantic `TypeAdapter` per parameter) is built once at decoration time, so dispatch stays cheap. Tools that don't go through `@tool`, notably MCP `as_tools()` wrappers, carry no type hints locally and pass through unchanged; the MCP server validates them. To validate or coerce arguments yourself outside the dispatch loop, call `aimu.tools.coerce_tool_arguments(fn, arguments)`, which returns the coerced dict or raises `ToolArgumentError`.

## Optional arguments

A parameter with a default value is optional in the generated spec:

```python
@aimu.tool
def search(query: str, num_results: int = 5) -> str:
    """Search for a query."""
    ...
```

The model can call `search(query="...")` without `num_results`.

`Optional[T]` / `T | None` parameters unwrap to the inner type: the spec shows `string` rather than `string | null`:

```python
from typing import Optional

@aimu.tool
def lookup(name: Optional[str] = None) -> str:
    """Look up a record by name."""
    ...
```

## Combine with MCP tools

In-process `@tool` functions and cross-process MCP tools both end up as callables in one `tools` list; `MCPClient.as_tools()` turns a server's tools into callables you concatenate. On a name collision the **last** entry wins, so append a local tool after the MCP tools to shadow one:

```python
from aimu.tools import MCPClient

client = aimu.client("ollama:qwen3.5:9b")
mcp = MCPClient(server=my_mcp_server)

agent = Agent(client, tools=mcp.as_tools() + [letter_counter])   # both routes, one list
```

## Override tools for a single call

`tools=` on the constructor / `client.tools` sets the *configured* tools. To use a different set for just one call (without mutating that state), pass `tools=` to `chat()` or `Agent.run()`:

```python
client.tools = [search, calculate]

client.chat("just look this up", tools=[search])  # only search, this call
client.chat("answer from memory", tools=[])       # no Python tools, this call

agent = Agent(client, tools=[search, calculate])
agent.run("quick lookup", tools=[search])          # override for the whole run
```

`tools=None` (default) uses the configured tools. Any other value (including `[]` to disable tools) replaces them for that call only and is restored afterward (MCP tools, being callables in `tools` via `as_tools()`, are included in the swap). On `Agent.run()` the override applies to every turn of the agentic loop.

## Share state across tool calls (`ToolContext`)

Tools often need shared state (a store, a cache, a config object) that persists across calls within a run but shouldn't be a module-level global. Declare a parameter annotated as `ToolContext` (or `ToolContext[Deps]`) and the agent **injects** it at call time. That parameter is excluded from the tool's JSON schema, so the model never sees it and only fills in the remaining arguments.

```python
from dataclasses import dataclass, field

import aimu
from aimu import ToolContext
from aimu.agents import Agent

@dataclass
class Deps:
    seen: dict = field(default_factory=dict)

@aimu.tool
def remember(ctx: ToolContext[Deps], key: str, value: str) -> str:
    """Store a value under a key."""   # the model only sees key + value; ctx is injected
    ctx.deps.seen[key] = value
    return "ok"

deps = Deps()
agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[remember], deps=deps)
agent.run("Remember that the sky is blue.")
print(deps.seen)   # {"sky": "blue"}
```

The value of `ctx.deps` comes from the `Agent`: its `deps=` field, or a per-run `agent.run(..., deps=...)` override (the override wins). When no deps are provided (for example a bare `client.chat()` call), `ctx.deps` is `None`.

```python
agent.run("Remember the meeting is Friday.", deps=other_deps)   # per-run override
```

## Built-in tools

`aimu.tools.builtin` ships ready-made tools grouped by domain. Pass a group directly:

```python
from aimu.tools import builtin

agent = Agent(client, tools=builtin.web + builtin.fs)
```

| Group | Functions |
|---|---|
| `builtin.web` | `get_weather`, `get_webpage`, `web_search`, `wikipedia` |
| `builtin.fs` | `list_directory`, `read_file` |
| `builtin.compute` | `calculate` |
| `builtin.misc` | `echo`, `get_current_date_and_time` |
| `builtin.ALL_TOOLS` | All of the above |

See the [`aimu.tools` API reference](../reference/api/tools.md) for the full list with descriptions.

## Streaming tools (generators)

For long-running tools, make the function a generator and yield `StreamChunk` objects during execution. The agent's `Agent.run(stream=True)` forwards each yielded chunk through its own stream so callers see progress live, with no side channels and no callbacks.

```python
import aimu
from aimu.models import StreamChunk, StreamingContentType

@aimu.tool
def long_search(query: str):
    """Search the web with live progress updates."""
    yield StreamChunk(StreamingContentType.GENERATING, f"Searching {query!r}...")
    results = _hit_search_api(query)         # imaginary slow call
    yield StreamChunk(StreamingContentType.GENERATING, f"Got {len(results)} results, fetching pages...")
    pages = [_fetch(r.url) for r in results]
    return "\n".join(p.title for p in pages)  # this is the canonical tool response
```

Three rules cover the contract:

1. **The decorator detects it automatically.** `@aimu.tool` sets `func.__tool_is_streaming__ = True` when the function is a generator (`inspect.isgeneratorfunction`) or async generator (`inspect.isasyncgenfunction`). No opt-in flag.
2. **Yield any phase that fits.** `GENERATING` for text progress, `IMAGE_GENERATING` for image-gen progress, future custom phases as needed. The agent forwards each chunk untouched (it only adds the `agent` and `iteration` metadata fields).
3. **The result comes from one of three places**, in priority order: the generator's `return` value (sync only, via `StopIteration.value`); the last yielded chunk's `content["result"]` if it's a dict with that key (matches the `IMAGE_GENERATING` final-chunk convention); or `str(last_chunk.content)`.

Async streaming tools are async generators (`async def + yield`); the async agent's `_handle_tool_calls_streamed` drains them with `async for`. Sync generator tools also work in the async surface: each `next()` is routed through `asyncio.to_thread` so the event loop stays free between yields.

Streaming tools require `stream=True` on the calling `chat()` / `agent.run()`. The non-streaming dispatch path raises `ValueError` with a clear message pointing at `stream=True`.

## See also

- [Explanation: tool integration](../explanation/tool-integration.md): dispatch order, precedence, when to pick in-process vs MCP
- [Explanation: StreamChunk model](../explanation/streamchunk-model.md): phases, content shapes, why one chunk type
- [Use MCP tools](use-mcp-tools.md): cross-process tool servers
- [Generate images](generate-images.md): the built-in `generate_image` streaming tool
