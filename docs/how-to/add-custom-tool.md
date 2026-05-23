# Add a custom tool

Decorate any Python function with `@tool`. The decorator inspects the signature and docstring at decoration time and attaches an OpenAI-format spec to `func.__tool_spec__`. The function itself is unchanged and remains directly callable.

## Minimal example

```python
from aimu.tools import tool
from aimu.agents import Agent
import aimu

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

## Signature rules

The decorator inspects parameters and types. Each parameter must satisfy:

- It has either a **type hint** or a **default value** (or both). Unhinted required params raise `ToolSignatureError`.
- It is *not* variadic. `*args` and `**kwargs` raise `ToolSignatureError` — declare each argument explicitly.

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
@tool
def bad(*args):              # ❌ variadic
    return args
# ToolSignatureError: @tool: function 'bad' uses variadic parameter '*args'...

@tool
def bad(x):                  # ❌ no hint, no default
    return x
# ToolSignatureError: @tool: parameter 'x' on 'bad' has no type hint and no default...
```

## Optional arguments

A parameter with a default value is optional in the generated spec:

```python
@tool
def search(query: str, num_results: int = 5) -> str:
    """Search for a query."""
    ...
```

The model can call `search(query="...")` without `num_results`.

`Optional[T]` / `T | None` parameters unwrap to the inner type — the spec shows `string` rather than `string | null`:

```python
from typing import Optional

@tool
def lookup(name: Optional[str] = None) -> str:
    """Look up a record by name."""
    ...
```

## Combine with MCP tools

`tools=` (in-process) and `model_client.mcp_client` (cross-process) coexist. Python `@tool` functions take precedence over MCP tools with the same name:

```python
from aimu.tools import MCPClient

client = aimu.client("ollama:qwen3.5:9b")
client.mcp_client = MCPClient(server=my_mcp_server)

agent = Agent(client, tools=[letter_counter])   # both routes active
```

## Built-in tools

`aimu.tools.builtin` ships ready-made tools grouped by domain. Pass a group directly:

```python
from aimu.tools import builtin

agent = Agent(client, tools=builtin.web + builtin.fs)
```

| Group | Functions |
|---|---|
| `builtin.web` | `get_weather`, `get_webpage`, `search`, `wikipedia` |
| `builtin.fs` | `list_directory`, `read_file` |
| `builtin.compute` | `calculate` |
| `builtin.misc` | `echo`, `get_current_date_and_time` |
| `builtin.ALL_TOOLS` | All of the above |

See the [`aimu.tools` API reference](../reference/api/tools.md) for the full list with descriptions.

## See also

- [Explanation: tool integration](../explanation/tool-integration.md) — dispatch order, precedence, when to pick in-process vs MCP
- [Use MCP tools](use-mcp-tools.md) — cross-process tool servers
