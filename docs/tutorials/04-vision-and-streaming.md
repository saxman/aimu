# Vision and streaming

In ~10 minutes you'll send an image to a vision-capable model and learn the full `StreamChunk` API along the way.

If you've done the first three tutorials, this one consolidates what you've seen and fills in the remaining I/O patterns.

## Setup

You need a vision-capable model. The fastest free option is **Ollama with Gemma 4**:

```bash
ollama pull gemma4:e4b
```

Or use any cloud model that supports vision (GPT-4o, Claude 4.x, Gemini 1.5+/2.x).

```python
import aimu

client = aimu.client("ollama:gemma4:e4b")
client.is_vision_model   # True
```

## 1. Send an image

Pass `images=[...]` to `chat()`. Each item can be a file path, raw bytes, an http(s) URL, or a `data:image/...` URL:

```python
response = client.chat("What's in this image?", images=["./cat.jpg"])
print(response)
```

Multiple images? Just add to the list:

```python
client.chat("Compare these.", images=["a.png", "b.png"])
```

`client.messages` keeps everything in OpenAI content-block format internally. Each provider adapts at request time — see [how-to: handle vision](../how-to/handle-vision.md) for the per-provider details.

## 2. Stream the response

`stream=True` returns a `StreamChunk` iterator instead of a string:

```python
for chunk in client.chat("Describe the scene in detail.", images=["./photo.jpg"], stream=True):
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

## 3. The StreamChunk API

Each `StreamChunk` is a named tuple:

```python
StreamChunk(phase, content, agent=None, iteration=0)
```

| Field | Type | Meaning |
|---|---|---|
| `phase` | `StreamingContentType` | `THINKING` / `TOOL_CALLING` / `GENERATING` / `DONE` |
| `content` | `str` or `dict` | `str` for text phases; `{"name", "response"}` for tool calls |
| `agent` | `str \| None` | Agent name when emitted by an agent or workflow; `None` for plain `chat()` |
| `iteration` | `int` | Agent loop index or chain step; `0` for plain chat |

Helpers cut the boilerplate:

```python
chunk.is_text()        # True for THINKING, GENERATING
chunk.is_tool_call()   # True for TOOL_CALLING
```

## 4. Filter phases

To skip thinking output (e.g. show only the final response), pass `include=[...]`:

```python
for chunk in client.chat("Solve this puzzle...", stream=True, include=["generating"]):
    print(chunk.content, end="")
```

Or to *only* show thinking (useful for debugging reasoning models):

```python
for chunk in client.chat("Reason about this...", stream=True, include=["thinking"]):
    print(chunk.content, end="")
```

Accepts string names or `StreamingContentType` members.

## 5. Streaming through agents

The same chunk type flows through agents and workflows. Agent context shows up in `chunk.agent` and `chunk.iteration`:

```python
from aimu.agents import Agent

agent = Agent(client, "You are a helpful image analyst.", name="vision-bot")

current_agent = None
for chunk in agent.run("Describe the scene", images=["./photo.jpg"], stream=True):
    if chunk.agent != current_agent:
        print(f"\n--- [{chunk.agent}] iteration {chunk.iteration} ---")
        current_agent = chunk.agent
    if chunk.is_text():
        print(chunk.content, end="")
    elif chunk.is_tool_call():
        print(f"\n[tool: {chunk.content['name']}]")
```

For chains, `chunk.iteration` is the chain step. For agent loops, it's the loop iteration. Same field, contextually meaningful.

## 6. Images through workflows

`images=` is threaded through every workflow's `run()`:

- `Chain.run(task, images=[...])` — forwarded only to step 0
- `Router.run(task, images=[...])` — forwarded to the dispatched handler
- `Parallel.run(task, images=[...])` — forwarded to every worker
- `EvaluatorOptimizer.run(task, images=[...])` — forwarded only to the initial generator turn

```python
from aimu.agents import Chain

chain = Chain.of(client, [
    "Describe what you see in detail.",
    "Summarise the description in one sentence.",
])
print(chain.run("Tell me about this photo.", images=["./photo.jpg"]))
```

Step 0 sees the image; step 1 sees only step 0's text output.

## What's next

You've completed the tutorials. You now know:

- The top-level API: `aimu.chat()`, `aimu.client()`, model strings
- The agent loop and the `@tool` decorator
- Four workflow patterns: Chain, Router, Parallel, EvaluatorOptimizer
- Vision input and the `StreamChunk` API

For specific tasks, browse the [how-to guides](../how-to/index.md). For the design intent behind any of this, see [explanation](../explanation/index.md).
