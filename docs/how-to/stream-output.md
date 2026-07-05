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
| `content` | `str` or `dict` | `str` for text phases; `{"name", "arguments", "response"}` for tool calls (`arguments` is the dict the model passed to the tool) |
| `agent` | `str \| None` | Agent name when emitted by an `Agent` or workflow; `None` for plain `client.chat()` |
| `iteration` | `int` | Loop index inside the agent (or chain step); `0` for plain chats |

Use `chunk.is_text()` and `chunk.is_tool_call()` to dispatch without writing out the phase comparison.

## Filter phases

Pass `include=[...]` to drop unwanted phases:

```python
# Only response tokens: skip thinking and tool calls
for chunk in client.chat("hi", stream=True, include=["generating"]):
    print(chunk.content, end="")

# Text only: thinking + generating, drop tool-call dicts
for chunk in client.chat("hi", stream=True, include=["thinking", "generating"]):
    print(chunk.content, end="")
```

Accepts `StreamingContentType` members or their string equivalents. Defaults to all phases.

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
            args = chunk.content["arguments"]
            result = chunk.content["response"]
            print(f"\n[tool: {name}({args!r})] → {result}\n")
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

## Streaming from the async surface

Every async surface call that returns a stream produces an `AsyncIterator[StreamChunk]` instead of an `Iterator[StreamChunk]`. The chunk type is identical; only the consumption protocol differs. Use `async for`:

```python
from aimu import aio

async def main():
    client = aio.client("ollama:qwen3.5:9b")
    async for chunk in await client.chat("Tell me a story", stream=True):
        if chunk.is_text():
            print(chunk.content, end="", flush=True)
```

`include=` works identically. The same applies to `await aio.Agent.run(stream=True)` and every async workflow's `run(stream=True)`. See [how-to: use async](use-async.md) for the full async surface walkthrough.

## Trace a run

Two helpers turn a run into something you can read or inspect, instead of re-implementing the phase-dispatch loop or reconstructing tool calls by hand.

### `pretty_print`: render a stream to the console

`pretty_print(stream)` consumes any `StreamChunk` iterator (`client.chat(stream=True)`, `Agent.run(stream=True)`, or any workflow run) and writes a readable transcript: tool calls flagged, generated text streamed inline, reasoning optional. It returns the concatenated `GENERATING` text, so you get the final answer as well as the printed transcript.

```python
import aimu
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=builtin.web)
answer = aimu.pretty_print(agent.run("Search the web and summarize today's AI news", stream=True))
```

| Argument | Default | Effect |
|---|---|---|
| `file` | `sys.stdout` | Where to write |
| `show_thinking` | `False` | Also print `THINKING` (reasoning) tokens |
| `show_tools` | `True` | Print a `[tool] <name>` marker for each tool call |

### `extract_tool_calls`: pull tool calls from a message history

`extract_tool_calls(messages)` reconstructs the tool call/result pairs from an OpenAI-format message list **after** a run, with no streaming. Pass `client.messages`, or an entry from `agent.messages` (which is keyed by agent name). It returns a list of `{iteration, tool, arguments, result}` dicts and does not mutate the input.

```python
from aimu import extract_tool_calls

agent.run("How many r's in strawberry?")
for call in extract_tool_calls(agent.messages[agent.name]):
    print(call["iteration"], call["tool"], call["arguments"], "->", call["result"])
# 1 letter_counter {'word': 'strawberry', 'letter': 'r'} -> 3
```

`iteration` increments once per assistant turn that contains tool calls, so you can see how many rounds the agent took. `arguments` is the parsed argument dict (it handles models that emit `parameters` instead of `arguments`); `result` is the matching tool-role content, or `""` if none.

### Tell agent-loop turns from user input

The agent continues a tool-using run by calling `chat()` with no user message (a turn on the current messages), so it injects **no** synthetic user turn between tool rounds. The one exception is the opt-in `final_answer_prompt` at the iteration cap. A replayed history can't tell an injected turn from real input unless it's marked, so AIMU tags that one with an inert `provenance` key (`message.get(aimu.PROVENANCE_KEY)` is `PROVENANCE_FINAL_ANSWER`, or `PROVENANCE_PROACTIVE` for scheduler pushes; genuine input is untagged. `PROVENANCE_CONTINUATION` is legacy — no longer produced). See [Persist conversations](persist-conversations.md#distinguish-agent-loop-turns-from-user-input) for the display pattern.

## Streaming structured output

`stream=True` combines with `schema=`: thinking / generation chunks stream live, then a terminal
`DONE` chunk carries `{"result": <validated object>}` (also on `client.last_structured`). See
[Get structured output](use-structured-output.md#streaming-structured-output) for the full contract
and the Anthropic caveat (it streams the JSON, not thinking).

## See also

- [Explanation: StreamChunk model](../explanation/streamchunk-model.md): one chunk type for all contexts
- [Stream phases reference](../reference/stream-phases.md): full enum
- [Get structured output](use-structured-output.md): `schema=` typed objects, incl. streaming
