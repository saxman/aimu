# Streaming

In ~10 minutes you'll learn the full streaming API — one `StreamChunk` type that flows through plain
chats, agents, and workflows alike.

If you've done the first three tutorials you've seen `stream=True` in passing; this is the one place
that teaches it end-to-end.

## Setup

Any model works. We'll use Ollama:

```python
import aimu

client = aimu.client("ollama:qwen3.5:9b")
```

## 1. Stream a response

Pass `stream=True` to `chat()` (or `aimu.chat()`) and you get a `StreamChunk` iterator instead of a
string. Consume it with a `for` loop and print tokens as they arrive:

```python
for chunk in client.chat("Tell me a short story.", stream=True):
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

`chunk.is_text()` is `True` for both `THINKING` and `GENERATING` phases, so this prints reasoning and
answer alike. The next sections show how to tell them apart and filter.

## 2. The StreamChunk API

Every streaming source yields the same named tuple:

```python
StreamChunk(phase, content, agent=None, iteration=0)
```

| Field | Type | Meaning |
|---|---|---|
| `phase` | `StreamingContentType` | `THINKING` / `TOOL_CALLING` / `GENERATING` / `DONE` |
| `content` | `str` or `dict` | `str` for text phases; `{"name", "arguments", "response"}` for a tool call |
| `agent` | `str \| None` | Agent/workflow name when emitted by one; `None` for a plain `chat()` |
| `iteration` | `int` | Agent loop index or workflow step; `0` for a plain chat |

Helpers cut the phase-checking boilerplate:

```python
chunk.is_text()        # True for THINKING, GENERATING
chunk.is_tool_call()   # True for TOOL_CALLING
chunk.is_done()        # True for the terminal DONE chunk
```

See the [stream phases reference](../reference/stream-phases.md) for the exact phase sequence per
call type.

## 3. Filter phases with `include=`

Thinking models emit `THINKING` chunks before the answer. To show only the final response, pass
`include=[...]`:

```python
for chunk in client.chat("Solve this puzzle step by step...", stream=True, include=["generating"]):
    print(chunk.content, end="", flush=True)
```

Or, to watch *only* the reasoning (handy when debugging a reasoning model):

```python
for chunk in client.chat("Reason about this...", stream=True, include=["thinking"]):
    print(chunk.content, end="", flush=True)
```

`include=` accepts string names (`"thinking"`, `"tool_calling"`, `"generating"`, `"done"`) or
`StreamingContentType` members. It defaults to all phases.

## 4. Stream through an agent

The same chunk type flows through an `Agent` — now the `agent` and `iteration` fields carry context,
and you'll see `TOOL_CALLING` chunks as the loop dispatches tools:

```python
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(client, "You are a helpful assistant.", tools=builtin.compute, name="assistant")

for chunk in agent.run("What is 17 * 23, and then that plus 100?", stream=True):
    if chunk.is_tool_call():
        print(f"\n[tool: {chunk.content['name']}({chunk.content['arguments']!r})]")
    elif chunk.is_text():
        print(chunk.content, end="", flush=True)
```

`chunk.iteration` increments each time the loop calls the model again after running tools, so you can
group output by round:

```python
current = -1
for chunk in agent.run("...", stream=True):
    if chunk.iteration != current:
        current = chunk.iteration
        print(f"\n--- iteration {current} ---")
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

## 5. Stream through workflows

Workflows yield the same chunks. For a `Chain`, `chunk.iteration` is the chain step:

```python
from aimu.agents import Chain

chain = Chain.from_client(client, [
    "List three facts about the topic.",
    "Summarise them in one sentence.",
])

for chunk in chain.run("The Python GIL", stream=True):
    if chunk.iteration == 0 and chunk.is_text():   # step 0's output
        print(chunk.content, end="", flush=True)
```

Same field, contextually meaningful: agent loop index for an `Agent`, step index for a `Chain`.
`Router` streams the classifier then the matched handler; `Parallel` streams the aggregator.

## What's next

You now know the whole streaming surface: `stream=True`, the `StreamChunk` fields and helpers, the
`include=` filter, and how the same chunks flow through agents and workflows.

- **[Vision & audio input](05-vision-and-audio.md)**: the last tutorial — send images and audio to a
  model.
- **[How-to: stream output](../how-to/stream-output.md)**: task-focused streaming recipes.
- **[Explanation: StreamChunk model](../explanation/streamchunk-model.md)**: why there's one chunk
  type for everything.
