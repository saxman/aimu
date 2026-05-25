# `aimu.aio`

Async surface. Mirrors the sync API one-for-one — same class names, different namespace. See [how-to: use async](../../how-to/use-async.md) for usage patterns and [explanation: async design](../../explanation/async-design.md) for why the surface is shaped this way.

Differences from the sync surface:

- Every `run()`, `chat()`, `generate()` is `async def`.
- Streaming returns `AsyncIterator[StreamChunk]` (consume with `async for`).
- `Parallel` and `concurrent_tool_calls=True` use `asyncio.TaskGroup` instead of `ThreadPoolExecutor`.
- In-process providers (`AsyncHuggingFaceClient`, `AsyncLlamaCppClient`) wrap an existing sync client; calling `aio.client(HuggingFaceModel.X)` directly raises.

## Top-level

::: aimu.aio.chat

::: aimu.aio.client

::: aimu.aio.AsyncModelClient

## Hierarchy

::: aimu.aio.AsyncRunner

## Agents

::: aimu.aio.Agent

::: aimu.aio.SkillAgent

::: aimu.aio.OrchestratorAgent

## Workflows

::: aimu.aio.Chain

::: aimu.aio.Router

::: aimu.aio.Parallel

::: aimu.aio.EvaluatorOptimizer

::: aimu.aio.PlanExecuteEvaluator

## Tools

::: aimu.aio.MCPClient
