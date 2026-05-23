# Stream output

Pass `stream=True` to `chat()`, `generate()`, or any agent / workflow `run()`. You get an iterator of `StreamChunk` named tuples, each tagged by phase.

## Basic streaming

```python
import aimu

for chunk in aimu.chat("Tell me a story", model="ollama:qwen3.5:9b", stream=True):
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

## StreamChunk fields

| Field | Type | Meaning |
|---|---|---|
| `phase` | `StreamingContentType` | `THINKING`, `TOOL_CALLING`, `GENERATING`, or `DONE` |
| `content` | `str` or `dict` | `str` for text phases; `{"name", "response"}` for tool calls |
| `agent` | `str \| None` | Agent name when emitted by an `Agent` or workflow; `None` for plain `client.chat()` |
| `iteration` | `int` | Loop index inside the agent (or chain step); `0` for plain chats |

Use `chunk.is_text()` and `chunk.is_tool_call()` to dispatch without writing out the phase comparison.

## Filter phases

Pass `include=[...]` to drop unwanted phases:

```python
# Only response tokens — skip thinking and tool calls
for chunk in client.chat("hi", stream=True, include=["generating"]):
    print(chunk.content, end="")

# Text only — thinking + generating, drop tool-call dicts
for chunk in client.chat("hi", stream=True, include=["thinking", "generating"]):
    print(chunk.content, end="")
```

Accepts `StreamingContentType` members or their string equivalents. Defaults to all phases.

The legacy `include_thinking=False` flag on `generate()` is kept as an alias for backward compatibility.

## Distinguish phases

```python
from aimu.models import StreamingContentType as Phase

for chunk in client.chat("Plan a trip and call weather tool", stream=True):
    match chunk.phase:
        case Phase.THINKING:
            print(f"[thinking] {chunk.content}", end="")
        case Phase.GENERATING:
            print(chunk.content, end="")
        case Phase.TOOL_CALLING:
            name = chunk.content["name"]
            result = chunk.content["response"]
            print(f"\n[tool: {name}] → {result}\n")
```

## Streaming from agents and workflows

`Agent.run(stream=True)` and every workflow `run(stream=True)` yield the same `StreamChunk` type. Agent context shows up in `chunk.agent` and `chunk.iteration`:

```python
from aimu.agents import Agent
agent = Agent(client, "Be helpful.", name="assistant")

for chunk in agent.run("count r's in strawberry", stream=True):
    if chunk.agent != current_agent:
        print(f"\n--- [{chunk.agent}] iter {chunk.iteration} ---")
        current_agent = chunk.agent
    if chunk.is_text():
        print(chunk.content, end="")
```

For chains, `chunk.iteration` is the chain step (0 for step 0, 1 for step 1, …).

## See also

- [Explanation: StreamChunk model](../explanation/streamchunk-model.md) — one chunk type for all contexts
- [Stream phases reference](../reference/stream-phases.md) — full enum
