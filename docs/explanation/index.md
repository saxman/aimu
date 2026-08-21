# Explanation

Understanding-oriented documentation. The *why*: what AIMU is for, how its parts fit together, and what tradeoffs the design makes (and refuses to make).

The short version: AIMU exists so you can find out what generative AI models and systems are actually capable of, and every design decision below is an attempt to keep the library from getting in the way of that.

Read these when you want to know *why* the API looks the way it does, not just how to use it.

- **[Architecture](architecture.md)**: the load-bearing shape: `BaseModelClient`, `Agent`, `Runner`, how providers dispatch.
- **[Design principles](design-principles.md)**: why AIMU exists, the six principles that follow from it, and what each one therefore rules out (the line that separates AIMU from LangChain / LangGraph / Strands / PydanticAI). Start here.
- **[Agents vs workflows](agents-vs-workflows.md)**: Anthropic's taxonomy and when to pick which.
- **[StreamChunk model](streamchunk-model.md)**: why one chunk type instead of three.
- **[System message lifecycle](system-message-lifecycle.md)**: seeded before the first chat, then swapped in place mid-conversation; `reset()` clears history.
- **[Tool integration](tool-integration.md)**: `@tool` vs MCP, dispatch order, when to pick which route.
- **[A2A vs MCP](a2a-vs-mcp.md)**: sharing whole *agents* over the wire vs sharing *tools*; why both are optional, adapter-shaped, and composable through `Runner`.
- **[Async design](async-design.md)**: why `aimu.aio` is a submodule (not method suffixes or smart routing), why `asyncio.TaskGroup`, and why in-process providers wrap an existing sync client.
- **[Thinking and the model context](thinking-and-context.md)**: where reasoning is stored, why the `"thinking"` message key is inert metadata, and why prior-turn reasoning is not re-fed to the model.

For task-oriented usage, see [how-to guides](../how-to/index.md). For the API surface, see [reference](../reference/index.md).
