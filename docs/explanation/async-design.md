# Async design

AIMU has two surfaces: the sync ladder (`aimu.chat()` → `aimu.client()` → `Agent` → workflows) and a one-import-away async mirror under `aimu.aio`. The sync surface is the default. Async is strictly opt-in.

```python
# Sync (default)
import aimu
reply = aimu.chat("Hi", model="anthropic:claude-sonnet-4-6")

# Async — same names, different namespace
from aimu import aio
async def main():
    client = aio.client("anthropic:claude-sonnet-4-6")
    agent = aio.Agent(client, tools=[my_async_tool])
    reply = await agent.run("Hello")
```

This page records *why* the async surface looks like it does. Each decision was weighed against the [six design principles](design-principles.md); each names the alternatives that were considered and rejected.

## Decision 1: `aimu.aio` submodule (twin classes)

The async surface mirrors the public sync surface inside `aimu.aio` using the same class names. Switching paradigms is one import line + `await` keywords.

**Why:** preserves [progressive disclosure](design-principles.md#4-progressive-disclosure) — the sync ladder is unchanged and new users never see async. Mirrors stdlib (`urllib.request` / `urllib.parse`, `concurrent.futures`).

**Rejected alternatives:**

- *Method suffix* (`chat_async`, `run_async` on the existing classes). Python convention puts async-ness in the call site (`await`), not the method name — `asyncio`, `httpx`, `redis-py`, `aiohttp` all avoid `_async` suffix. The claimed "share one client across sync/async" benefit evaporates because `client.messages` isn't concurrency-safe.
- *Async-only library, sync via `asyncio.run`.* Halves the codebase but adds `asyncio.run(...)` ceremony to every sync call site. Hurts casual sync users (notebooks, Streamlit) more than it helps maintainers.
- *Smart context-detection* (same `client.chat()` routes by detecting a running loop). Return type becomes ambiguous (`str` vs coroutine); forgotten `await` silently returns a coroutine — violates [failures are apparent](design-principles.md#6-failures-are-apparent).
- *Async-first with hidden sync facade.* Sync users would carry an invisible event-loop thread and see asyncio frames in tracebacks — violates [plain Python](design-principles.md#1-plain-python).

## Decision 2: Full parity

Every sync class has an async twin in `aimu.aio`: `Agent`, `SkillAgent`, `Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`, `PlanExecuteEvaluator`, `OrchestratorAgent`, `MCPClient`.

**Why:** preserves [composability through uniform interfaces](design-principles.md#3-composability-through-uniform-interfaces) inside each world. Without full parity you can't drop an async `Agent` into an async `Chain`, and the substitutability principle breaks.

**Rejected alternatives:** "model client + `Parallel` only" (~800 LOC) and "model client only" (~400 LOC) both fail principle 3 — users would have to write their own structured concurrency over `agent.run()` calls, which is exactly what the library is supposed to factor out.

## Decision 3: Python 3.11+ with `asyncio.TaskGroup`

The Python floor is `>=3.11`. `asyncio.TaskGroup`, `asyncio.timeout()`, and native `ExceptionGroup` are used inside `aimu.aio`. No `anyio` in new code. (The existing sync `MCPClient` still uses `anyio.start_blocking_portal()` — that's unchanged.)

**Why:** structured concurrency in plain stdlib. `TaskGroup` cancels in-flight siblings cleanly when one task fails and surfaces all errors as an `ExceptionGroup`. Python 3.10 EOL is October 2026; 3.11 has been stable since October 2022.

**Behaviour change to note:** async `Parallel` and async `concurrent_tool_calls=True` cancel sibling work on first failure. The sync `ThreadPoolExecutor` versions let all workers run to completion. The async semantics are arguably better for debugging — surface all failures at once, don't waste cycles after an unrecoverable error — but they *are* different from the sync semantics.

**Rejected alternatives:**

- *Stay on 3.10, use `asyncio.gather`.* Loses structured concurrency; only the first exception surfaces. Worse semantics than even the sync `ThreadPoolExecutor` version.
- *Stay on 3.10, use `anyio.create_task_group()`.* Works on 3.10 and gets trio compatibility for free, but contributors must learn anyio idioms. With 3.11's stdlib `TaskGroup` available, anyio doesn't pull its weight in new code.

## Decision 4: MCP — sync `MCPClient` stays first-class

Both surfaces expose an `MCPClient` class with the same construction signature (`config=` / `server=` / `file=`) and the same method names. Sync `aimu.tools.MCPClient` keeps the `anyio.start_blocking_portal()` machinery and emits **no deprecation warning**. Async `aimu.aio.MCPClient` is a thin parallel (~30 LOC) using FastMCP's `Client` directly. A single `aimu.tools.mcp_format.mcp_tools_to_openai` helper is shared.

**Why:** the sync wrapper exists precisely so sync users can use MCP tools *without* learning asyncio. That's [progressive disclosure](design-principles.md#4-progressive-disclosure) working correctly — deprecating it would make the simple case harder. Symmetric class shape on both surfaces respects [uniform interfaces](design-principles.md#3-composability-through-uniform-interfaces).

**Rejected alternatives:**

- *Deprecate the sync wrapper.* The whole point is letting sync callers use MCP without `await`.
- *No async wrapper at all — expose `fastmcp.Client` directly.* Asymmetric surface (sync users learn AIMU's `MCPClient`; async users learn FastMCP's).

## Decision 5: Streaming type asymmetry — accept it

`aimu.chat(stream=True)` returns `Iterator[StreamChunk]`. `aio.chat(stream=True)` returns `AsyncIterator[StreamChunk]`. They are not unified.

**Why:** unifying them would need either `nest_asyncio` magic (violates [plain Python](design-principles.md#1-plain-python)) or context-detection magic (violates [failures are apparent](design-principles.md#6-failures-are-apparent)). The namespace boundary makes which one you get unambiguous: if you imported from `aimu.aio`, you `async for`; otherwise you `for`.

## Decision 6: Async tool detection at decoration time

`@tool` records `func.__tool_is_async__ = inspect.iscoroutinefunction(func)` when the function is decorated. Sync `_handle_tool_calls()` raises `ValueError` if it encounters an async tool, with a message pointing the user at `aimu.aio`. Async `_handle_tool_calls()` accepts both — it `await`s async tools and routes sync ones through `asyncio.to_thread` to keep the event loop free.

**Why:** same `@tool` function is usable on either surface as long as the surface and the tool's color match. Failures are visible at the dispatch site, not silent paradigm-crossing. Aligns with [failures are apparent](design-principles.md#6-failures-are-apparent).

**Rejected alternative:** sync dispatch auto-running async tools via `asyncio.run`. Silent paradigm crossing makes errors harder to localize — if a user mixes sync and async tools, that's a structural choice the library should not paper over.

## Decision 7: In-process providers wrap an existing sync client

`AsyncHuggingFaceClient` and `AsyncLlamaCppClient` do not load model weights independently. They take an existing sync client in their constructor and route through `asyncio.to_thread`. Calling `aio.client(HuggingFaceModel.X)` directly raises a `ValueError` pointing the user at the correct pattern:

```python
sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)   # loads weights once
async_client = aio.client(sync_client)                   # wraps; shares weights
```

State (`messages`, `system_message`, `tools`, `mcp_client`) is shared with the wrapped sync client. There is conceptually one client; the async wrapper just adds an awaitable interface.

**Why:** HF transformer models routinely weigh 10–100 GB. LlamaCpp GGUF files are similar. Twin construction would double the memory footprint and likely OOM the host. Even if memory were free, the GIL and the underlying CUDA stream serialize execution — there's no coroutine concurrency to gain from two "client" instances. Making the resource share explicit follows [failures are apparent](design-principles.md#6-failures-are-apparent): if you construct two clients, the second one tells you why that won't work.

**Caveat to know:** for in-process providers, "async" only buys event-loop integration (your async handler doesn't block on inference) — *not* coroutine-level concurrency. If you need batch throughput from one HF model, use `model.generate(input_ids=batched)` directly, not concurrent `await` calls.

**Rejected alternatives:**

- *Module-level weight cache keyed by `(model_id, kwargs)`.* Dedups transparently but hides global state; lifecycle becomes unclear (when does a cached tensor get freed?). Explicit wrapping makes the resource share obvious.
- *Twin classes that each load weights.* The OOM footgun is real and easy to hit accidentally.
- *Skip async support for in-process providers.* Forces an asymmetric mental model. Wrapping is cheap (~20 LOC per provider) and keeps the surface uniform.

## What follows from these decisions

The async surface is a strict mirror of the sync one — same names, same shapes, same `Runner` decision tree, same `StreamChunk` type. The two surfaces differ in three concrete places:

- **Construction site.** `from aimu import aio` (or `aio.client(...)`) instead of top-level imports.
- **Call site.** `await ...` and `async for ...`.
- **Concurrency primitive.** `asyncio.TaskGroup` instead of `ThreadPoolExecutor` for `Parallel` and `concurrent_tool_calls=True`. This is the *only* place async delivers behaviourally different semantics — and it's strictly better for debugging.

For in-process providers (HuggingFace, LlamaCpp), one further difference: you must build a sync client first and wrap it.

## See also

- [Architecture](architecture.md) — the shared shape that both surfaces follow.
- [Design principles](design-principles.md) — the six principles each decision above was weighed against.
- [StreamChunk model](streamchunk-model.md) — the chunk type yielded by both `Iterator[StreamChunk]` (sync) and `AsyncIterator[StreamChunk]` (async).
