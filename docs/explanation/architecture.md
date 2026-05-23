# Architecture

The whole library fits in your head. There are three load-bearing abstractions and a handful of supporting types — nothing more.

```
┌──────────────────────────────────────────────────────────────┐
│  Top-level: aimu.chat() / aimu.client() / resolve_model_string│
├──────────────────────────────────────────────────────────────┤
│  ModelClient (factory)                                       │
│   ├─ OllamaClient / AnthropicClient / HuggingFaceClient      │
│   ├─ OpenAICompatClient (+ OpenAI, Gemini, LM Studio, ...)   │
│   └─ LlamaCppClient                                          │
│  Each implements BaseModelClient (chat / generate / _chat /  │
│   _generate / _update_generate_kwargs)                       │
├──────────────────────────────────────────────────────────────┤
│  Runner (ABC)                                                │
│   ├─ BaseAgent     (autonomous: LLM directs flow)            │
│   │    └─ Agent / SkillAgent / OrchestratorAgent             │
│   └─ Workflow      (code-controlled: code directs flow)      │
│        └─ Chain / Router / Parallel / EvaluatorOptimizer     │
└──────────────────────────────────────────────────────────────┘
```

## The base contract: `BaseModelClient`

Every provider speaks the same shape. The contract:

```python
class BaseModelClient(ABC):
    model: Model
    messages: list[dict]
    system_message: str | None
    tools: list[Callable]
    mcp_client: MCPClient | None
    last_thinking: str

    def chat(user_message, generate_kwargs=None, use_tools=True,
             stream=False, images=None, include=None) -> str | Iterator[StreamChunk]
    def generate(prompt, ...) -> str | Iterator[StreamChunk]
    def reset(system_message="__keep__") -> None
```

`chat()` and `generate()` are **concrete** on the base — they apply the `include=` stream filter and delegate to abstract `_chat()` / `_generate()` which each provider implements. This split means a new feature like `include=` lands in one place and works everywhere.

Message history is a plain `list[dict]` in OpenAI format. There is no `Message` class — providers like Anthropic that need a different wire format adapt at request time, never mutating `self.messages`.

## The factory: `ModelClient`

```python
ModelClient(model)            # provider determined by enum type
ModelClient("ollama:qwen3.5:9b")   # or by string prefix
```

`ModelClient` is the single public entry point. It dispatches by checking the `Model` enum type against a registry, instantiates the matching concrete client, and delegates every method to it.

Provider client classes (`OllamaClient`, `AnthropicClient`, ...) still exist and can be imported, but `ModelClient` is the recommended path — it keeps your code provider-agnostic.

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

- `.value` — the provider id string (preserved for `provider.sdk(model="...")` calls)
- `.spec` — the underlying `ModelSpec`
- `.supports_tools` / `.supports_thinking` / `.supports_vision` — mirrored flags
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

Two marker ABCs split runners into autonomous and code-controlled:

- **`BaseAgent`** — the LLM directs flow (e.g. `Agent` keeps looping until the model stops calling tools).
- **`Workflow`** — code directs flow (e.g. `Chain` runs steps in a fixed order).

This is Anthropic's *Building Effective Agents* taxonomy made concrete. See [Agents vs workflows](agents-vs-workflows.md) for the underlying argument.

## Streaming: one chunk type everywhere

`StreamChunk(phase, content, agent=None, iteration=0)` is the single chunk type yielded by every streaming path — `client.chat()`, `Agent.run()`, every workflow `run()`. Earlier versions had separate `AgentChunk` and `ChainChunk` named tuples; those are now back-compat aliases.

The `agent` and `iteration` fields are populated by agents/workflows and default to `None` / `0` for plain chats. See [StreamChunk model](streamchunk-model.md) for the design argument.

## Tool integration: `@tool` and `MCPClient`

Two routes, both end up in the same place — a list of OpenAI-format tool specs that the base client sends to the model:

- **`@tool` decorator** runs in-process. The decorator inspects the signature at decoration time and attaches a spec to `func.__tool_spec__`. The model client looks up tools by `__name__` and dispatches via direct function call.
- **`MCPClient`** wraps a FastMCP server. The model client calls `mcp_client.get_tools()` to fetch the spec list and `mcp_client.call_tool(name, args)` to dispatch.

Both can be active on the same client. Python tools take precedence on name collision. See [Tool integration](tool-integration.md) for when to pick which.

## What lives where

| Package | Role |
|---|---|
| `aimu` | Top-level `chat()`, `client()`, `resolve_model_string()`, re-exports |
| `aimu.models` | `ModelClient`, `BaseModelClient`, `ModelSpec`, `StreamChunk`, provider clients |
| `aimu.agents` | `Runner` / `BaseAgent` / `Workflow` ABCs; `Agent`, `SkillAgent`, `OrchestratorAgent`; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` |
| `aimu.tools` | `@tool` decorator, `MCPClient`, `builtin.*` tool groups |
| `aimu.skills` | `AgentSkill`, `SkillManager`, MCP server builder |
| `aimu.memory` | `SemanticMemoryStore`, `DocumentStore`, shared `MemoryStore` ABC |
| `aimu.history` | `ConversationManager` (TinyDB-backed persistence) |
| `aimu.prompts` | Versioned `PromptCatalog`, `PromptTuner` and concrete tuners, `Scorer` |
| `aimu.evals` | `Benchmark`, `BenchmarkResults`, DeepEval adapters |

Each package has its own optional dependency. `aimu[all]` installs everything; piecewise install is supported and missing modules degrade gracefully via `HAS_*` flags.

## See also

- [Design principles](design-principles.md) — what AIMU deliberately doesn't do.
- [Agents vs workflows](agents-vs-workflows.md) — the taxonomy in detail.
