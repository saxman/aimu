# StreamChunk model

AIMU has exactly one chunk type for streaming output. Every streaming path — `client.chat()`, `Agent.run()`, every workflow `run()` — yields `StreamChunk` objects. This page explains why.

## The shape

```python
class StreamChunk(NamedTuple):
    phase: StreamingContentType
    content: str | dict
    agent: str | None = None
    iteration: int = 0
```

| Field | Purpose |
|---|---|
| `phase` | Which kind of content: `THINKING` / `TOOL_CALLING` / `GENERATING` / `DONE` |
| `content` | The payload: `str` for text phases, `dict` for tool calls |
| `agent` | Which agent emitted this (workflow / agent contexts); `None` for plain `chat()` |
| `iteration` | Loop iteration or chain step; `0` for plain `chat()` |

Helpers:

```python
chunk.is_text()        # True for THINKING and GENERATING
chunk.is_tool_call()   # True for TOOL_CALLING
```

## Why one type, not three

Earlier versions had three named tuples: `StreamChunk` for model clients, `AgentChunk` for agents, `ChainChunk` for chains. Each was nearly identical, just with different optional fields. The differences leaked:

- Code that processed `Agent.run(stream=True)` output couldn't be reused for `client.chat(stream=True)`.
- `AgenticModelClient` had to convert `AgentChunk → StreamChunk` mid-stream, dropping the agent name and iteration.
- Users had to remember which class came out of which call site.

The fix was to make the optional context fields (`agent`, `iteration`) part of the single type, defaulting to `None` and `0` for plain chats. The result: one type to learn, no conversion code, full context preserved through every layer. The historical `AgentChunk` and `ChainChunk` names were removed; `StreamChunk` is the only name.

## Why `content` is a union, not three classes

`content` is `str` for `THINKING` / `GENERATING` and `dict {"name", "arguments", "response"}` for `TOOL_CALLING` (where `arguments` is the parameter dict the model passed to the tool). A typed approach would model these as a discriminated union of three NamedTuples or Pydantic classes.

We chose the union because:

- It matches what users want to write. `if chunk.is_text(): print(chunk.content)` is one line.
- A discriminated union forces every reader to handle every variant or write a match statement. For UIs that want to print text and skip tool calls, this is noise.
- Pydantic discriminated unions would violate the "OpenAI message dicts as the only data model" principle.

The cost: `chunk.content["name"]` works only when `phase == TOOL_CALLING`. The `is_tool_call()` helper makes this readable; type checkers can narrow with `assert chunk.is_tool_call()` if needed.

## Why phases instead of separate events

Some libraries (LangChain, OpenLLMetry) model streaming as discrete events with their own subscribed handlers. We chose phases-on-chunks because:

- The phase is *inherent* to the chunk, not a separate event. A thinking token is text; a tool-call chunk is structured. Tagging the chunk with its phase is data, not control flow.
- Iteration over a single iterator beats subscribing to multiple handlers when the consumer is a UI render loop or a logging pipeline.
- One mental model — "consume an iterator of chunks" — covers everything from a CLI print loop to a Streamlit app.

The `include=[...]` parameter on `chat()` and `generate()` is the only filtering primitive we need; it covers "skip thinking", "skip tool calls", and any combination.

## Phase semantics

| Phase | When | `content` |
|---|---|---|
| `THINKING` | Reasoning model emits internal-monologue tokens | `str` (token) |
| `TOOL_CALLING` | A tool call has just been dispatched and its result is back | `dict {"name": str, "arguments": dict, "response": str}` |
| `GENERATING` | The final response stream | `str` (token) |
| `DONE` | Terminal marker (rarely yielded by providers; reserved) | `str` (usually empty) |

For thinking models, you typically see THINKING chunks first, then GENERATING. For tool-using models, you may see THINKING → TOOL_CALLING (one per tool used) → GENERATING.

## See also

- [How-to: stream output](../how-to/stream-output.md) — the practical patterns.
- [Stream phases reference](../reference/stream-phases.md) — the full enum.
