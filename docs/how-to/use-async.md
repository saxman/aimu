# Use async (`aimu.aio`)

The sync surface is the default. If you're embedding AIMU in an async app (FastAPI, Starlette, Discord bot, anything with a running event loop), import `aimu.aio` instead. Same class names, one import away.

The only differences from the sync surface are at the **call site** (you `await`) and the **concurrency primitive** (`Parallel` and `concurrent_tool_calls` use `asyncio.TaskGroup` instead of `ThreadPoolExecutor`). Everything else is identical.

## Quick start

```python
import asyncio
from aimu import aio

async def main():
    client = aio.client("anthropic:claude-sonnet-4-6")
    reply = await client.chat("Hi there")
    print(reply)

asyncio.run(main())
```

## One-shot vs multi-turn

```python
from aimu import aio

# One-shot
text = await aio.chat("Hello", model="anthropic:claude-sonnet-4-6")

# Multi-turn
client = aio.client("ollama:qwen3.5:9b", system="You are concise.")
await client.chat("Hi there")
await client.chat("What did I just say?")     # history preserved
```

## Streaming

`aio.chat(stream=True)` and every `aio.Agent.run(stream=True)` / `aio.Chain.run(stream=True)` return an **`AsyncIterator[StreamChunk]`**. Consume with `async for`:

```python
async for chunk in await client.chat("Tell me a story", stream=True):
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

The `include=` filter works identically:

```python
stream = await client.chat("hi", stream=True, include=["generating"])
async for chunk in stream:
    print(chunk.content, end="")
```

## Agents with async tools

The `@aimu.tool` decorator auto-detects `async def` and tags the function. The async surface awaits async tools and routes sync (CPU-bound) tools through `asyncio.to_thread` so the event loop stays free.

```python
import httpx
import aimu
from aimu import aio

@aimu.tool
async def fetch(url: str) -> str:
    """Fetch the contents of a URL."""
    async with httpx.AsyncClient() as c:
        return (await c.get(url)).text[:500]

@aimu.tool
def normalize(text: str) -> str:
    """Strip whitespace and lower-case the input."""
    return text.strip().lower()

client = aio.client("anthropic:claude-sonnet-4-6")
agent = aio.Agent(client, "Use the tools.", tools=[fetch, normalize])
reply = await agent.run("Fetch example.com and normalize the title.")
```

The same `@aimu.tool` function can be used on either surface — the sync surface rejects async tools with a clear error, and the async surface accepts both.

## Parallel — the headline win

`aio.Parallel` uses `asyncio.TaskGroup` instead of a thread pool. You get true coroutine concurrency, structured cancellation, and `ExceptionGroup` aggregation when multiple workers fail.

```python
from aimu import aio

parallel = aio.Parallel.from_client(
    client,
    worker_prompts=[
        "Analyze this from a security perspective.",
        "Analyze this from a performance perspective.",
        "Analyze this from a readability perspective.",
    ],
    aggregator_prompt="Synthesize into one concise review.",
)
result = await parallel.run("def login(user, pw): return user == 'admin' and pw == 'pw'")
```

Behaviour change vs sync `Parallel`: if one worker raises, in-flight siblings are cancelled and an `ExceptionGroup` surfaces. The sync `ThreadPoolExecutor` version lets all workers run to completion.

## Per-call timeouts

Use `asyncio.timeout()` — no AIMU plumbing needed:

```python
import asyncio
async with asyncio.timeout(30):
    result = await agent.run("Long task")     # raises TimeoutError if exceeded
```

## MCP tools

Async users get a parallel `aio.MCPClient` with the same construction signature as the sync wrapper. Connect via the `connect()` classmethod (since you can't `await` in `__init__`):

```python
from aimu import aio

async def main():
    mcp = await aio.MCPClient.connect(server=my_fastmcp_server)
    try:
        c = aio.client("anthropic:claude-sonnet-4-6")
        agent = aio.Agent(c, tools=await mcp.as_tools())   # async @tool-style callables
        reply = await agent.run("Use the MCP tools to ...")
    finally:
        await mcp.aclose()
```

The sync `aimu.tools.MCPClient` is unchanged and still supported — use it when you want MCP tools without writing async code at all.

## In-process providers (HuggingFace, LlamaCpp)

These providers load model weights into memory. Constructing both a sync and async client for the same model would load weights twice — likely OOM'ing the host for any non-trivial model. The async surface enforces a wrapping pattern instead:

```python
import aimu
from aimu import aio
from aimu.models import HuggingFaceModel

sync_client = aimu.client(HuggingFaceModel.QWEN_3_8B)   # loads weights once
async_client = aio.client(sync_client)                   # wraps; shares weights
```

State (`messages`, `system_message`, `tools`) is shared with the wrapped sync client. There's conceptually one client; the async wrapper just adds an awaitable interface.

Calling `aio.client(HuggingFaceModel.X)` directly raises a `ValueError` pointing you at the pattern above.

**Note:** for in-process providers, "async" only buys event-loop integration (your handler doesn't block on inference). It does *not* unlock coroutine-level concurrency — the GIL and CUDA stream serialize execution anyway. If you need batch throughput from one HF model, use `model.generate(input_ids=batched)` directly, not concurrent `await` calls.

## Mixing with sync code

The two surfaces don't share class instances — `client.messages` isn't concurrency-safe, so a single client can't be used both ways simultaneously. If you have a sync handler that needs to spawn async work:

```python
import asyncio
import aimu
from aimu import aio

# Sync conversation
client = aimu.client("anthropic:claude-sonnet-4-6")
reply = client.chat("Hello")

# Background async batch (separate client; fresh conversation)
async def batch():
    workers = aio.Parallel.from_client(
        aio.client("anthropic:claude-sonnet-4-6"),
        worker_prompts=["...", "...", "..."],
    )
    return await workers.run("topic")

result = asyncio.run(batch())
```

## When *not* to use the async surface

- **You're writing a notebook or a CLI script.** Stick with the sync surface; it's one less concept.
- **You don't have a running event loop.** Wrapping with `asyncio.run` is fine for one-shots, but if every call needs a wrap, you're not gaining anything over sync.
- **You're using HuggingFace / LlamaCpp and want concurrency.** The async wrappers exist for event-loop integration, not parallelism — in-process inference is GIL-bound regardless of paradigm.

## See also

- [Explanation: async design](../explanation/async-design.md) — why `aimu.aio` is shaped the way it is, with rejected alternatives.
- [API reference: `aimu.aio`](../reference/api/aio.md) — the full async surface.
- [Stream phases](../reference/stream-phases.md) — the same `StreamChunk` type yielded on both surfaces.
