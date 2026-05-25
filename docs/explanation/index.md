# Explanation

Understanding-oriented documentation. The *why* — what AIMU is, how its parts fit together, and what tradeoffs the design makes (and refuses to make).

Read these when you want to know *why* the API looks the way it does, not just how to use it.

- **[Architecture](architecture.md)** — the load-bearing shape: `BaseModelClient`, `Agent`, `Runner`, how providers dispatch.
- **[Design principles](design-principles.md)** — what AIMU deliberately *doesn't* do, and why (the line that separates AIMU from LangChain / LangGraph / Strands / PydanticAI).
- **[Agents vs workflows](agents-vs-workflows.md)** — Anthropic's taxonomy and when to pick which.
- **[StreamChunk model](streamchunk-model.md)** — why one chunk type instead of three.
- **[System message lifecycle](system-message-lifecycle.md)** — mutable, then locked at first chat, then unlocked by `reset()`.
- **[Tool integration](tool-integration.md)** — `@tool` vs MCP, dispatch order, when to pick which route.
- **[Async design](async-design.md)** — why `aimu.aio` is a submodule (not method suffixes or smart routing), why `asyncio.TaskGroup`, and why in-process providers wrap an existing sync client.

For task-oriented usage, see [how-to guides](../how-to/index.md). For the API surface, see [reference](../reference/index.md).
