# First agent with tools

In ~10 minutes you'll build an agent that uses three tools — built-in and custom — to answer a multi-step question.

If you haven't done [Getting started](01-getting-started.md), do that first.

## What we're building

An assistant agent that can answer questions involving the current date, basic math, and counting characters. The agent decides which tools to call.

```python
agent.run("If today is the start of a 30-day trial, when does it end? Also, how many 'a's in 'Madagascar'?")
```

The model will call `get_current_date_and_time`, `calculate`, and `letter_counter` in whatever order it decides, then synthesise the answer.

## 1. Set up the model client

```python
import aimu

client = aimu.client("ollama:qwen3.5:9b")
```

Any tool-capable model works. To check: `client.is_tool_using_model`.

## 2. Use built-in tools

`aimu.tools.builtin` ships a set of ready-made tools. They're grouped by domain so you don't have to remember individual names:

```python
from aimu.tools import builtin

builtin.web       # [get_weather, get_webpage, search, wikipedia]
builtin.fs        # [list_directory, read_file]
builtin.compute   # [calculate]
builtin.misc      # [echo, get_current_date_and_time]
```

For this tutorial we want `compute` (for math) and `misc` (for the date):

```python
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(client, "You are a helpful assistant.", tools=builtin.compute + builtin.misc)
print(agent.run("What is 17 * 23?"))
# 17 * 23 = 391.
```

The agent's loop dispatched `calculate("17 * 23")` and reported the result.

## 3. Add a custom tool

Now declare a tool the built-ins don't cover:

```python
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())
```

Three things to notice:

1. **Type hints are required** for every parameter (or a default value). Missing both raises `ToolSignatureError`.
2. **The first paragraph of the docstring becomes the tool description.** Be specific — this is what the model reads to decide whether to call it.
3. **The function is unchanged** — `letter_counter("hi", "i")` still works directly. The decorator only attaches an OpenAI-format spec at `func.__tool_spec__`.

Add it to the agent's tools:

```python
agent = Agent(
    client,
    "You are a helpful assistant.",
    tools=builtin.compute + builtin.misc + [letter_counter],
)

print(agent.run(
    "If today is the start of a 30-day trial, when does it end? "
    "Also, how many 'a's are in 'Madagascar'?"
))
```

The model will:

1. Call `get_current_date_and_time()` to learn today's date.
2. Call `calculate("...")` to add 30 days.
3. Call `letter_counter(word="Madagascar", letter="a")` → 4.
4. Return a synthesised answer.

## 4. Inspect what happened

The agent records the full conversation in `client.messages`. Each tool call appears as an `assistant` message with `tool_calls`, followed by a `tool` message with the result.

```python
for msg in client.messages:
    role = msg["role"]
    if "tool_calls" in msg:
        for tc in msg["tool_calls"]:
            print(f"  → {tc['function']['name']}({tc['function']['arguments']})")
    elif role == "tool":
        print(f"  ← {msg['name']}: {msg['content']}")
    elif msg.get("content"):
        print(f"[{role}] {msg['content'][:80]}")
```

This is OpenAI's message format — there's no proprietary wrapper.

## 5. Stream the agent

Same iterator API as `client.chat(stream=True)`, but each `StreamChunk` carries the agent name and loop iteration:

```python
for chunk in agent.run("count r's in strawberry and 2+2", stream=True):
    if chunk.is_tool_call():
        name = chunk.content["name"]
        args = chunk.content["arguments"]
        print(f"\n[tool: {name}({args!r})]")
    elif chunk.is_text():
        print(chunk.content, end="")
```

See [how-to: stream output](../how-to/stream-output.md) for full streaming patterns.

## What's next

You now know:

- The three tool routes: `builtin.<group>` for ready-made tools, `@tool` for custom Python functions, `MCPClient` for cross-process tools ([how-to: use MCP tools](../how-to/use-mcp-tools.md)).
- The agent loop: keep calling `chat()` until the model stops calling tools.
- How to inspect the message trail.

Up next: **[Workflows](03-workflows.md)** — when you want the orchestration *fixed in code* rather than directed by the LLM.

### See also

- [How-to: add a custom tool](../how-to/add-custom-tool.md) — full `@tool` signature rules
- [How-to: build an orchestrator](../how-to/build-orchestrator.md) — multi-agent pattern when one model dispatches to others
