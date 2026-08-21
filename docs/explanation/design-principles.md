# Design principles

## Why AIMU exists

AIMU exists so you can find out what generative AI models and systems are actually capable of, and understand how they work while you do it.

Every principle below is in service of that. Each one removes something that would otherwise stand between you and the model: a wrapper type, a hidden control flow, a silent fallback, a guessed capability. What's left is thin enough that when a model surprises you, the surprise is the model's and not the library's.

Building real applications is how you find out, so AIMU is a real library rather than a teaching toy. But when convenience and legibility conflict, legibility wins.

The six principles are ordered from foundation to consequence: read top to bottom and the API falls out.

## 1. Plain Python

The library is built from ordinary classes, functions, decorators, dataclasses, and type hints. Every file is readable top-to-bottom. There is no framework metalanguage to learn before you can build something. If a behaviour is hard to find by reading the source, that's a bug.

This is the first thing that makes model behaviour legible: when a run does something unexpected, you can read the whole path from your call to the provider request and see whether the library shaped it. Nothing is dispatched through machinery you can't follow.

*How this cashes out:* `@tool` is a decorator that calls `inspect.signature()`. The agent loop is a `for` loop in [agent.py](../reference/api/agents.md#aimu.agents.Agent) you can read in one screen. Workflows are concrete classes (`Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`), not graph nodes. Composition happens by passing objects to constructors, not by operator overloading.

## 2. Plain data

Conversation state is a `list[dict]` you can `print()`, edit by hand, JSON-serialise, or hand to any other library. The OpenAI message format is the lingua franca and the only persistent representation. Provider-specific formats (Anthropic's `tool_use` blocks, Ollama's message-level `images=` field, HuggingFace's PIL images) adapt at request time and never leak into `self.messages`.

The payoff for understanding is that the conversation is the thing you inspect. You can print exactly what the model was sent and what it sent back, diff two runs, or replay a transcript, without decoding a wrapper type first.

*How this cashes out:* `client.messages` is always introspectable; you can save it to JSON and reload it. `ConversationManager` is a TinyDB wrapper over the same dicts. `MessageHistory` is a type alias for `dict[str, list[dict]]`, not a class. There is no `Message` / `AIMessage` / `ChatMessage` wrapper to translate to and from.

## 3. Composability through uniform interfaces

Substitutability is the principle that makes the library scale up without scaling the surface area. Every provider implements `BaseModelClient`. Every agent and workflow implements `Runner`. Every memory backend implements `MemoryStore`. Every streaming source yields `StreamChunk`. You can swap any concrete implementation for any other at the same level without rewriting call sites.

Substitutability is also what makes comparison cheap. Running the same task across Ollama, Claude, and a local vLLM server is a string change, so the difference you observe is the models' and not your harness's. `Benchmark` and `FallbackClient` exist because that comparison is a first-class activity.

*How this cashes out:* `agent.as_model_client()` returns a `BaseModelClient` view, so an agent slots into anywhere a client is accepted, including inside a workflow that itself accepts agents as steps. `Benchmark` runs the same `chat()` across plain clients and agentic ones uniformly. A `Router`'s handlers can be agents, chains, or other routers. The `messages` property merges recursively across nested workflows.

## 4. Progressive disclosure

The first useful call is one line. Each layer beyond that exposes more capability without forcing you to engage it. Start with `aimu.chat()` for a one-shot. Reach for `aimu.client()` when you need a conversation. Reach for `Agent` when you need a tool-using loop. Reach for a workflow (`Chain` / `Router` / `Parallel` / `EvaluatorOptimizer`) when you need a fixed pipeline. Subclass `BaseModelClient` only when you're adding a provider.

The ladder is also a curriculum. Each rung introduces exactly one new idea, so you can learn what an agent adds over a chat call, or what an orchestrator adds over an agent, by climbing one step and reading the difference.

*How this cashes out:* You can be productive at any depth. A demo script never needs to know what `ModelSpec` is. A power user can still construct `LlamaCppClient(model, model_path=..., n_gpu_layers=-1, chat_format="chatml-function-calling")` directly. The top-level `aimu.chat()` is a thin wrapper over `ModelClient` is a thin wrapper over the concrete client; peeling each layer takes the reader one file deeper, no more.

## 5. Direct paths for common tasks

Common operations have one obvious, ergonomic entry point that takes minimal ceremony. One-shot chat is `aimu.chat("...", model="...")`. Building a tool-using agent is `Agent(client, "system msg", tools=[...])`. A three-step pipeline is `Chain.from_client(client, [...])`. Filtering a stream is `include=["generating"]`. Grouped helpers (`builtin.web`, `builtin.fs`, `builtin.compute`) let you grab an obvious set of tools without naming each one. The library does not offer parallel, equally-recommended ways to do the same job. If a second path exists, it's a power-user escape hatch, not a fork in the documentation.

One obvious path means the code you read in an example is the code you would write, and a run's behaviour can't be explained by having picked the wrong one of two equivalent entry points.

*How this cashes out:* The top-level `aimu.chat()` and `aimu.client()` short-circuit the full `ModelClient` constructor when you don't need to customise it. `Agent`'s constructor takes `system_message` positionally because every agent needs one. `Chain.from_client()`, `Router.from_client()`, `Parallel.from_client()` exist so the common case is one line; the full constructors are still available for the unusual case. `agent.as_model_client()` is a method instead of a class import because almost no one needs to subclass it.

## 6. Failures are apparent

Errors raise at the layer where the cause is actionable, with messages that name the problem. `@tool` signature problems raise at decoration time, not at the first model call. Malformed `SKILL.md` files raise on discovery, not silently skipped. Dead MCP connections raise on construction, not on the first tool call. The library does not silently fall back, and it does not wrap an error in a way that hides the original cause.

This principle is what keeps a model's real limits visible. A silent fallback would let you believe a model can do something it can't. So a capability AIMU can't deliver raises (an unknown model id, a vision call to a text-only model), and a portable request a backend can't honour warns and says what it dropped, rather than quietly changing what you asked for.

*How this cashes out:* `ToolSignatureError`, `SkillLoadError`, `SkillNotFoundError`, `MCPConnectionError`, and the `system_message` `RuntimeError` each name a specific failure mode. Each message tells the caller what to do next. Each chained exception preserves the original cause via `raise ... from exc`, so the traceback always points at the root.

## What follows from these principles

Each principle excludes certain patterns. **Plain Python** rules out operator-overloaded chaining, observer/callback systems, prompt-template classes, graph DSLs, and dependency-injection containers, which would all introduce framework metalanguage between the user and the code. **Plain data** rules out Pydantic message wrappers and typed `Agent[Deps, Output]` generics. **Composability through uniform interfaces** rules out parallel inheritance trees for the same concept. **Direct paths for common tasks** rules out tool plugin registries and configuration DSLs. If a path is the obvious one, there isn't a second equally-obvious one. None of these are missing because they were hard to add; they're absent because they would violate the principles above.

## What this means for contributing

Before proposing a change, ask which principle it serves. A new feature that doesn't extend at least one of the six is probably wrapper material, not library material. The small public surface makes external wrappers easy to write.

Then ask the question the principles exist to serve: **does this make a model's real behaviour more legible, or does it hide it?** A change that makes the library more convenient while obscuring what the model actually did is working against the reason AIMU exists, however well it scores on the other six.

The harder question is which principle a proposed change *violates*. If you can't tell, the principles aren't doing their job; open a discussion.

## See also

- [Architecture](architecture.md): the shape that falls out of these principles.
- [Agents vs workflows](agents-vs-workflows.md): applying principle #3 (uniform interfaces) and #4 (progressive disclosure) to the agent/workflow split.
