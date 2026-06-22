# Architecture

The whole library fits in your head. There are three load-bearing abstractions for text plus a parallel image-generation surface, nothing more.

```
┌──────────────────────────────────────────────────────────────┐
│  Top-level: aimu.chat() / aimu.client() / resolve_model_string│
│             aimu.image_client() / aimu.generate_image()      │
├──────────────────────────────────────────────────────────────┤
│  Text:  ModelClient (factory)                                │
│   ├─ OllamaClient / AnthropicClient / HuggingFaceClient      │
│   ├─ OpenAICompatClient (+ OpenAI, Gemini, LM Studio, ...)   │
│   └─ LlamaCppClient                                          │
│  Each implements BaseModelClient (chat / generate / _chat /  │
│   _generate / _update_generate_kwargs)                       │
├──────────────────────────────────────────────────────────────┤
│  Image: ImageClient (factory)                                │
│   ├─ HuggingFaceImageClient  (HF diffusers, local)           │
│   └─ GeminiImageClient       (Nano Banana, cloud)            │
│  Each implements BaseImageClient (generate / _generate)      │
├──────────────────────────────────────────────────────────────┤
│  Runner (ABC): single interface for everything runnable      │
│   ├─ Agent / SkillAgent / OrchestratorAgent                  │
│   │     (autonomous: LLM directs flow)                       │
│   └─ Chain / Router / Parallel /                             │
│      EvaluatorOptimizer / PlanExecuteEvaluator               │
│         (code-controlled: code directs flow)                 │
│  All concretes exported from aimu.agents.                    │
└──────────────────────────────────────────────────────────────┘
```

## The base contract: `BaseModelClient`

Every provider speaks the same shape. The contract:

```python
class BaseModelClient(ABC):
    model: Model
    messages: list[dict]
    system_message: str | None
    tools: list[Callable]   # @tool functions and/or MCPClient.as_tools() callables
    last_thinking: str

    def chat(user_message, generate_kwargs=None, use_tools=True,
             stream=False, images=None, include=None, tools=None) -> str | Iterator[StreamChunk]
    def generate(prompt, ...) -> str | Iterator[StreamChunk]
    def reset(system_message="__keep__") -> None
```

`chat()` and `generate()` are **concrete** on the base: they apply the `include=` stream filter and delegate to abstract `_chat()` / `_generate()` which each provider implements. This split means a new feature like `include=` lands in one place and works everywhere.

Message history is a plain `list[dict]` in OpenAI format. There is no `Message` class. Providers like Anthropic that need a different wire format adapt at request time, never mutating `self.messages`.

## The factory: `ModelClient`

```python
ModelClient(model)            # provider determined by enum type
ModelClient("ollama:qwen3.5:9b")   # or by string prefix
```

`ModelClient` is the single public entry point. It dispatches by checking the `Model` enum type against a registry, instantiates the matching concrete client, and delegates every method to it.

Provider client classes (`OllamaClient`, `AnthropicClient`, ...) still exist and can be imported, but `ModelClient` is the recommended path: it keeps your code provider-agnostic.

## Model definitions: `ModelSpec` + `Model` enum

Each provider has a `Model` subclass listing the models it supports:

```python
class AnthropicModel(Model):
    CLAUDE_SONNET_4_6 = ModelSpec("claude-sonnet-4-6", tools=True, thinking=True, vision=True)
    CLAUDE_OPUS_4_6   = ModelSpec("claude-opus-4-6",   tools=True, thinking=True, vision=True)
    CLAUDE_HAIKU_4_5  = ModelSpec("claude-haiku-4-5",  tools=True, vision=True)
```

`ModelSpec` is a frozen dataclass with the model id and capability flags (`tools`, `thinking`, `vision`, optional `generation_kwargs`). Equality and hash use `id` only, so a spec can hold a generation-kwargs dict and still serve as an enum value.

Each enum member exposes:

- `.value`: the provider id string (preserved for `provider.sdk(model="...")` calls)
- `.spec`: the underlying `ModelSpec`
- `.supports_tools` / `.supports_thinking` / `.supports_vision`: mirrored flags
- Classproperties `TOOL_MODELS` / `THINKING_MODELS` / `VISION_MODELS` derive automatically

## Agents and workflows: `Runner`

`Runner(ABC)` is the common interface for anything you can call `run()` on:

```python
class Runner(ABC):
    @abstractmethod
    def run(task, generate_kwargs=None, stream=False, images=None) -> str | Iterator[StreamChunk]: ...

    @property
    @abstractmethod
    def messages(self) -> dict[str, list[dict]]: ...
```

There's a single `Runner` ABC. The autonomous-vs-code-controlled split survives as a *categorisation of concrete classes*, not a type hierarchy:

- **Autonomous**: the LLM directs flow. `Agent` keeps looping until the model stops calling tools; `SkillAgent` and `OrchestratorAgent` extend that.
- **Code-controlled**: code directs flow. `Chain` runs steps in a fixed order; `Router` classifies and dispatches; `Parallel` fans out; `EvaluatorOptimizer` iterates generate→critique; `PlanExecuteEvaluator` plans→executes→scores→replans.

All concrete classes live in `aimu.agents`. This is the *Building Effective Agents* taxonomy made concrete. See [Agents vs workflows](agents-vs-workflows.md) for the underlying argument.

## Image generation: a parallel surface, not a forced fit

Image generation lives next to the text client, not inside it. The shape mirrors text one-for-one:

| Text | Image |
|---|---|
| `BaseModelClient` (ABC) | `BaseImageClient` (ABC) |
| `Model` (Enum base) | `ImageModel` (Enum base) |
| `ModelSpec` (dataclass) | `ImageSpec` (dataclass) + `HuggingFaceImageSpec` / `GeminiImageSpec` subclasses |
| `ModelClient` (factory) | `ImageClient` (factory) |
| `aimu.client()` / `aimu.chat()` | `aimu.image_client()` / `aimu.generate_image()` |

`BaseImageClient` does *not* subclass `BaseModelClient`; text and image have disjoint contracts. `chat()`, `messages`, `system_message`, `tools`, and `StreamChunk` phases are meaningless for stateless text-to-image. Forcing the modalities into one class would expand the public surface and bleed irrelevant fields into image users' code.

What the two surfaces share is *shape*, not implementation: the same factory-plus-concretes pattern, the same `provider:id` string format, the same per-provider extras (`[hf]` covers both HF text and HF image; `[google]` is image-only since text-side Gemini uses the OpenAI-compat endpoint), and the same dispatch story (enum / spec / string → concrete client via the factory).

Tool integration is what unites them in practice: the built-in `generate_image` tool wraps an `ImageClient` and lets any text chat agent call image generation when the user asks. The agent loop doesn't know or care that the tool ultimately hits a cloud API or local GPU.

See [How-to: generate images](../how-to/generate-images.md) for the runnable patterns.

## Two surfaces, one shape

Everything above describes the **sync** surface: `aimu.client()`, `aimu.agents.Agent`, etc. AIMU also ships an opt-in **async** surface under `aimu.aio` that mirrors this shape one-for-one. Same class names, same `Runner` decision tree, same `StreamChunk` type, same `_ChatStateMixin` for state mechanics. The only differences are at the call site (`await`), the streaming type (`AsyncIterator[StreamChunk]`), and the concurrency primitive used by `Parallel` and `concurrent_tool_calls` (`asyncio.TaskGroup` instead of `ThreadPoolExecutor`).

```
aimu.{chat, client, Agent, Chain, …}        # sync (default)
aimu.aio.{chat, client, Agent, Chain, …}    # async (opt-in)
```

The sync ladder is unchanged for users who don't need async. See [async design](async-design.md) for the design decisions.

## Streaming: one chunk type everywhere

`StreamChunk(phase, content, agent=None, iteration=0)` is the single chunk type yielded by every streaming path: `client.chat()`, `Agent.run()`, every workflow `run()`. Earlier versions had separate `AgentChunk` and `ChainChunk` named tuples; both were collapsed into `StreamChunk`.

The `agent` and `iteration` fields are populated by agents/workflows and default to `None` / `0` for plain chats. See [StreamChunk model](streamchunk-model.md) for the design argument.

## Tool integration: `@tool` and `MCPClient`

Two routes, both end up as callables in one registry (`client.tools`) from which the base client builds the OpenAI-format spec list it sends to the model:

- **`@tool` decorator** runs in-process. The decorator inspects the signature at decoration time and attaches a spec to `func.__tool_spec__`. The model client looks up tools by `__name__` and dispatches via direct function call.
- **`MCPClient.as_tools()`** wraps a FastMCP server's tools as callables (each closes over the client and calls `call_tool(name, args)` cross-process), carrying `__tool_spec__` just like a `@tool` function. Add them to `client.tools`.

Both kinds coexist in the same list and dispatch through one by-name lookup; on a name collision the last entry wins. See [Tool integration](tool-integration.md) for when to pick which.

## What lives where

| Package | Role |
|---|---|
| `aimu` | Top-level `chat()`, `client()`, `image_client()`, `generate_image()`, re-exports |
| `aimu.aio` | Async mirror of the public sync surface: same class names, `async def` everywhere |
| `aimu.models` | Text: `ModelClient`, `BaseModelClient`, `ModelSpec`, `StreamChunk`, provider clients. Image: `ImageClient`, `BaseImageClient`, `ImageSpec`, `HuggingFaceImageClient`, `GeminiImageClient` |
| `aimu.agents` | `Runner` ABC; `Agent`, `SkillAgent`, `OrchestratorAgent`; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` / `PlanExecuteEvaluator` |
| `aimu.tools` | `@tool` decorator, `MCPClient`, `builtin.*` tool groups |
| `aimu.skills` | `AgentSkill`, `SkillManager`, MCP server builder |
| `aimu.memory` | `SemanticMemoryStore`, `DocumentStore`, shared `MemoryStore` ABC |
| `aimu.history` | `ConversationManager` (TinyDB-backed persistence) |
| `aimu.prompts` | Versioned `PromptCatalog`, `PromptTuner` and concrete tuners, `Scorer` |
| `aimu.evals` | `Benchmark`, `BenchmarkResults`, DeepEval adapters |

Each package has its own optional dependency. `aimu[all]` installs everything; piecewise install is supported and missing modules degrade gracefully via `HAS_*` flags.

## See also

- [Design principles](design-principles.md): what AIMU deliberately doesn't do.
- [Agents vs workflows](agents-vs-workflows.md): the taxonomy in detail.
