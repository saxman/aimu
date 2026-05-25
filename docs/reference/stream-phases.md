# Stream phases

`StreamingContentType` is a string enum with four values, attached to every `StreamChunk` via `chunk.phase`.

| Phase | Value | Emitted when | `chunk.content` type |
|---|---|---|---|
| `THINKING` | `"thinking"` | Reasoning model emits internal-monologue tokens | `str` (token) |
| `TOOL_CALLING` | `"tool_calling"` | A tool call has just been dispatched and its result is available | `dict {"name": str, "response": str}` |
| `GENERATING` | `"generating"` | The final response stream | `str` (token) |
| `DONE` | `"done"` | Terminal marker (rarely yielded; reserved) | `str` (usually empty) |

The enum inherits from `str`, so members compare equal to their string values: `chunk.phase == "thinking"` is fine.

## Helpers on `StreamChunk`

```python
chunk.is_text()        # True for THINKING and GENERATING
chunk.is_tool_call()   # True for TOOL_CALLING
```

These avoid sprinkling phase comparisons through your code.

## Typical sequences

| Model behaviour | Phase sequence |
|---|---|
| Plain text generation | `GENERATING*` |
| Thinking model, no tools | `THINKING*`, then `GENERATING*` |
| Tool-using model | `THINKING*` (if thinking), `TOOL_CALLING` per call, `GENERATING*` |
| Tool-using model, multi-round | `TOOL_CALLING* → THINKING*? → GENERATING*` repeated; `chunk.iteration` increments per round (agent context) |

`*` = "one or more chunks".

## Async surface

`aimu.aio` yields the same `StreamChunk` type — `phase`, `content`, `agent`, `iteration` are unchanged. The consumption protocol differs: `Iterator[StreamChunk]` on the sync surface vs `AsyncIterator[StreamChunk]` on the async surface. Phase semantics and filtering are identical on both. See [how-to: use async](../how-to/use-async.md).

## Filtering

Pass `include=[...]` to `chat()` or `generate()` to drop phases:

```python
client.chat("hi", stream=True, include=["generating"])
client.chat("hi", stream=True, include=["thinking", "generating"])
client.chat("hi", stream=True, include=[StreamingContentType.GENERATING])  # enum form
```

Accepts strings or enum members. Omitting `include=` yields all phases.

## See also

- [Explanation: StreamChunk model](../explanation/streamchunk-model.md) — why one chunk type instead of three
- [How-to: stream output](../how-to/stream-output.md) — practical patterns
- [`aimu.models.StreamingContentType`](api/models.md#aimu.models.StreamingContentType) — API reference
