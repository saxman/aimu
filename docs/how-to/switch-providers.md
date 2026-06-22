# Switch providers

Every AIMU client implements the same `BaseModelClient` interface, so swapping backends is a one-line change. `ModelClient` is the factory; it accepts either a `Model` enum member or a `"provider:model_id"` string.

## Use a model string

```python
import aimu

client = aimu.client("anthropic:claude-sonnet-4-6")
client = aimu.client("ollama:qwen3.5:9b")
client = aimu.client("openai:gpt-4o-mini")
client = aimu.client("gemini:gemini-2.5-flash")
```

Model strings have the form `"provider:model_id"`. Colons inside the model id (e.g. Ollama's `qwen3.5:9b`) are preserved.

## Use an enum member

For IDE autocomplete and type checking:

```python
from aimu.models import ModelClient, OllamaModel, AnthropicModel

client = ModelClient(OllamaModel.QWEN_3_5_9B)
client = ModelClient(AnthropicModel.CLAUDE_SONNET_4_6)
```

## Resolve a name, string, or enum uniformly

When you accept a model from a CLI flag or config and don't know which form it'll arrive in, `resolve_model_enum` (text) and `resolve_image_model_enum` (image) normalise all three to an enum member:

```python
import aimu
from aimu.models import AnthropicModel

aimu.resolve_model_enum(AnthropicModel.CLAUDE_SONNET_4_6)   # enum member → returned unchanged
aimu.resolve_model_enum("anthropic:claude-sonnet-4-6")      # "provider:model_id" string
aimu.resolve_model_enum("CLAUDE_SONNET_4_6")                # bare enum-member name

aimu.resolve_image_model_enum("FLUX_2_KLEIN_4B")           # same, for image models
aimu.resolve_image_model_enum("gemini:nano-banana")
```

Pass the result straight to `aimu.client(...)` / `aimu.image_client(...)` (both also accept an enum member):

```python
client = aimu.client(aimu.resolve_model_enum(args.model))
```

A **bare text name is often ambiguous**: the same id ships under several providers (e.g. `QWEN_3_8B` is an Ollama, HuggingFace, and OpenAI-compat model). `resolve_model_enum` resolves the tie by preferring a provider where the model is *already available locally* (running Ollama → cached HuggingFace → reachable local server, tool-capable first), logging the choice at `WARNING`. If it isn't available anywhere, it raises `ValueError` listing the `"provider:model_id"` options, so pin the provider with the string form when you need a specific one. (`resolve_image_model_enum` has no local-availability notion and raises on the rare ambiguity.)

## Use whatever model is already available locally

Omit `model=` entirely and AIMU resolves a default: the `AIMU_LANGUAGE_MODEL` env var if set, otherwise the first *already-available* local model (running Ollama → cached HuggingFace → reachable local OpenAI-compat server, tool-capable preferred). A cloud provider is never auto-selected and weights are never downloaded.

```python
client = aimu.client()                          # auto-resolved default (logs the pick at WARNING)
```

```bash
export AIMU_LANGUAGE_MODEL="ollama:qwen3.5:9b"  # pin the default deterministically
```

To inspect or choose among what's available instead of taking the auto-pick:

```python
aimu.available_text_models()            # list[Model]: locally loadable models, provider-priority order
aimu.resolve_default_text_model_enum()  # the single auto-pick, as an enum member
```

## Provider keys

| Provider key | Extra | API key env var |
|---|---|---|
| `ollama` | `aimu[ollama]` | None |
| `hf` | `aimu[hf]` | None |
| `llamacpp` | `aimu[llamacpp]` | None (`model_path=` required) |
| `anthropic` | `aimu[anthropic]` | `ANTHROPIC_API_KEY` |
| `openai` | `aimu[openai_compat]` | `OPENAI_API_KEY` |
| `gemini` | `aimu[openai_compat]` | `GOOGLE_API_KEY` |
| `lmstudio` | `aimu[openai_compat]` | None (localhost:1234) |
| `ollama-openai` | `aimu[openai_compat]` | None (localhost:11434) |
| `hf-openai` | `aimu[openai_compat]` | None (localhost:8000) |
| `vllm` | `aimu[openai_compat]` | None (localhost:8000) |
| `llamaserver` | `aimu[openai_compat]` | None (localhost:8080) |
| `sglang` | `aimu[openai_compat]` | None (localhost:30000) |

See the [provider matrix](../reference/provider-matrix.md) for full details.

## Check what the provider supports

```python
client = aimu.client("ollama:qwen3.5:9b")
client.is_tool_using_model    # True
client.is_thinking_model      # True
client.is_vision_model        # False
```

## Pass provider-specific kwargs

Extra kwargs are forwarded to the underlying client constructor:

```python
# Ollama: keep the model warm for 5 minutes
aimu.client("ollama:qwen3.5:9b", model_keep_alive_seconds=300)

# llama.cpp: load a local GGUF file
aimu.client("llamacpp:qwen3-8b", model_path="/path/to/qwen3-8b.gguf")

# LM Studio: point at a non-default host
aimu.client("lmstudio:qwen3.5-9b", base_url="http://myserver:1234/v1")
```

Each provider client's constructor signature is in the [API reference](../reference/api/models.md).

## Timeouts and retries

Networked clients take `timeout` (seconds) and `max_retries`, forwarded straight to the provider SDK's own request timeout and bounded retry-on-transient-failure. AIMU adds no retry logic of its own.

```python
import aimu

# Cloud and local OpenAI-compat servers: both supported
client = aimu.client("anthropic:claude-sonnet-4-6", timeout=30, max_retries=5)
client = aimu.client("openai:gpt-4o", timeout=30, max_retries=5)
client = aimu.client("vllm:Qwen/Qwen3-8B", timeout=30, max_retries=5)
```

These are SDK-native (the `anthropic` and `openai` SDKs implement timeout + retry), so the names and behavior match those SDKs exactly. Unset values fall back to each SDK's defaults.

Two caveats:

- **Ollama** (native provider) supports `timeout` but has no request-retry; passing `max_retries` raises `ValueError`. Use the `ollama-openai` provider (which routes through the OpenAI SDK) if you need retries against Ollama.
- **In-process providers** (`hf:`, `llamacpp:`) run locally with no HTTP request, so they don't accept `timeout` / `max_retries` (passing them raises `TypeError`).

## Provider failover

Per-client `max_retries` retries the *same* provider. To fall back to a *different* provider when one is down, wrap an ordered list of clients in a `FallbackClient`:

```python
import aimu
from aimu.models import FallbackClient

client = FallbackClient([
    aimu.client("anthropic:claude-sonnet-4-6", timeout=30, max_retries=2),  # preferred
    aimu.client("openai:gpt-4o", timeout=30, max_retries=2),                # fallback
    aimu.client("ollama:qwen3.5:9b"),                                       # last resort, local
])

print(client.chat("Hello"))   # tries Anthropic; on error falls over to OpenAI, then Ollama
```

The first client that answers wins. A client that raises hands off to the next, carrying the **same conversation history**, so a multi-turn chat survives a mid-conversation failover. When every client fails, `FallbackExhaustedError` is raised with the last error chained as its cause.

`FallbackClient` is itself a `BaseModelClient`, so it composes everywhere a plain client does:

```python
from aimu.agents import Agent

agent = Agent(client, "You are a helpful assistant.", tools=[...])   # resilient agent
```

Notes:

- **What triggers failover.** By default any `Exception`. Narrow it with `retry_on=` (e.g. `FallbackClient([...], retry_on=(TimeoutError, ConnectionError))`) so permanent errors surface immediately instead of being masked.
- **Capabilities** (`is_thinking_model`, `model`, etc.) reflect the **first** client, so use capability-compatible clients in one set.
- **Streaming** fails over only before the first chunk is emitted; an error mid-stream propagates rather than replaying.
- **Async:** `aio.AsyncFallbackClient` is the one-for-one async twin.

Combine the two layers: `max_retries`/`timeout` give in-SDK retry against one provider, `FallbackClient` gives cross-provider failover on top.

## Failure modes

`aimu.client("foo:bar")` raises `ValueError` listing the available providers if the prefix is unknown, and raises with the available model ids if the prefix is valid but the id isn't:

```python
>>> aimu.client("foo:bar")
ValueError: Unknown provider 'foo'. Available providers (with installed deps): ['anthropic', 'ollama', ...]

>>> aimu.client("anthropic:claude-nonsense")
ValueError: Provider 'anthropic' has no model id 'claude-nonsense'. Available: ['claude-fable-5', 'claude-haiku-4-5', 'claude-opus-4-6', 'claude-opus-4-7', 'claude-opus-4-8', 'claude-sonnet-4-6']
```
