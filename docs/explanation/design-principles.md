# Design principles

AIMU has a clear identity, deliberately positioned against the heavier alternatives. This page is the line AIMU won't cross — the patterns rejected on purpose, and the reasons.

## Six principles

These are guardrails. A change that violates one is rejected even if it would save keystrokes.

1. **Plain Python over framework primitives.** No `Runnable` protocol, no `BaseTool` hierarchy, no `|` LCEL operator, no `Agent[Deps, Out]` generics, no dependency injection container. The library is built from ordinary classes and functions you can read top-to-bottom.
2. **OpenAI message dicts as the only data model.** No `Message` / `ChatMessage` / `AIMessage` classes. No Pydantic for tool args — `@tool` inspects type hints and docstrings from the standard library.
3. **Explicit over magic.** No prompt-template objects, no implicit chaining, no callback or observer systems, no declarative YAML workflow DSL. Composition happens by instantiation.
4. **One way to do each thing.** Where dual paths existed (factory vs direct constructor, `AgentChunk` vs `StreamChunk`, public `AgenticModelClient` vs `as_model_client()`), one was kept and the other removed.
5. **Synchronous first.** Async is deferred until a concrete need surfaces. Adding `async def chat()` everywhere doubles the surface area; users who need async wrap `client.chat()` in `asyncio.to_thread()` in one line.
6. **Loud failures.** Silent fallbacks on malformed input become exceptions with actionable messages. Bad `@tool` signatures, malformed `SKILL.md` files, dead MCP connections, mutating `system_message` after the conversation starts — all raise with a clear message.

## What we explicitly don't do

These are tempting patterns lifted from neighbouring libraries. Each is rejected with a reason.

| Pattern | Found in | Rejected because |
|---|---|---|
| `Runnable` protocol / LCEL `\|` chaining | LangChain | Composition by class instantiation is clearer; the `\|` operator hides the argument flow. |
| Graph / state-machine workflow DSL | LangGraph | Anthropic's four patterns (chain, route, parallel, evaluator-optimizer) cover the documented cases. Users can write Python for the rest. |
| Pydantic models for tool args | PydanticAI | `@tool` on plain functions with type hints is half the cognitive load. JSON parsing for structured output stays manual where needed. |
| `Agent[Deps, Output]` typed generics | PydanticAI | Real benefit is small; cognitive overhead in error messages and IDE hover is large. |
| Prompt template objects (`PromptTemplate("Hi {name}")`) | LangChain | f-strings exist. |
| Callback / event handler system | LangChain | The `include=[...]` stream filter covers 90%; the rest is YAGNI until proven otherwise. |
| Dependency injection container | PydanticAI | Passing the model client into an `Agent` constructor is one line. |
| Plugin registry for tools | LangChain Hub, Strands tools | `from aimu.tools import builtin; tools=builtin.web` is enough. |
| Built-in retrievers / vector stores | LangChain, Strands | Out of scope. `SemanticMemoryStore` exists for episodic memory, not RAG infrastructure. |
| Built-in tracing / observability | LangChain, OpenLLMetry | Out of scope; users wire their own logging. |

The result: AIMU is the smallest of the four by import surface and by line count.

## How these principles cash out in the API

- `@tool` on plain Python functions — not `BaseTool` subclasses or Pydantic models.
- Message history is a `list[dict]` you can `print()` and edit by hand.
- `Agent(client, "system message", tools=[...])` — positional system message, two-line construction.
- Workflows are concrete classes (`Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`), not nodes in a graph.
- One streaming chunk type (`StreamChunk`) flows through everything; no separate event classes.
- The `include=` parameter is a list filter, not a subscription/observer.
- Tool dispatch order is documented and deterministic (Python `@tool` first, then MCP).
- `system_message` immutability is enforced with `RuntimeError` rather than silently dropped writes.

## What this means for contributing

If you propose a feature, the first question is: does this fit AIMU, or does it belong in a wrapper above AIMU? Patterns that match the principles above are welcome; patterns that violate them are best built externally — the small public surface makes that easy.

## See also

- [Architecture](architecture.md) — the shape that falls out of these principles.
- The library's CLAUDE.md (in the repo root) — engineering notes for Claude Code, which also encode these principles.
