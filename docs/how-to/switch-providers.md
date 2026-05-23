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

## Provider keys

| Provider key | Extra | API key env var |
|---|---|---|
| `ollama` | `aimu[ollama]` | — |
| `hf` | `aimu[hf]` | — |
| `llamacpp` | `aimu[llamacpp]` | — (`model_path=` required) |
| `anthropic` | `aimu[anthropic]` | `ANTHROPIC_API_KEY` |
| `openai` | `aimu[openai_compat]` | `OPENAI_API_KEY` |
| `gemini` | `aimu[openai_compat]` | `GOOGLE_API_KEY` |
| `lmstudio` | `aimu[openai_compat]` | — (localhost:1234) |
| `ollama-openai` | `aimu[openai_compat]` | — (localhost:11434) |
| `hf-openai` | `aimu[openai_compat]` | — (localhost:8000) |
| `vllm` | `aimu[openai_compat]` | — (localhost:8000) |
| `llamaserver` | `aimu[openai_compat]` | — (localhost:8080) |
| `sglang` | `aimu[openai_compat]` | — (localhost:30000) |

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

## Failure modes

`aimu.client("foo:bar")` raises `ValueError` listing the available providers if the prefix is unknown, and raises with the available model ids if the prefix is valid but the id isn't:

```python
>>> aimu.client("foo:bar")
ValueError: Unknown provider 'foo'. Available providers (with installed deps): ['anthropic', 'ollama', ...]

>>> aimu.client("anthropic:claude-nonsense")
ValueError: Provider 'anthropic' has no model id 'claude-nonsense'. Available: ['claude-haiku-4-5', 'claude-opus-4-6', 'claude-sonnet-4-6']
```
