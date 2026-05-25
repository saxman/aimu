# Provider matrix

Every supported provider, with the extra needed to install it, the env var (if any) it reads, the default endpoint, and any provider-specific constructor kwargs.

| Provider key (in model string) | Client class | Extra | API key | Default endpoint | Provider-specific kwargs | Async kind |
|---|---|---|---|---|---|---|
| `ollama` | `OllamaClient` | `aimu[ollama]` | — | local | `model_keep_alive_seconds=60` | native (`ollama.AsyncClient`) |
| `hf` | `HuggingFaceClient` | `aimu[hf]` | — (HF login for gated models) | local | `model_kwargs={...}` (passed to `from_pretrained`) | wrapped sync via `asyncio.to_thread` |
| `anthropic` | `AnthropicClient` | `aimu[anthropic]` | `ANTHROPIC_API_KEY` | api.anthropic.com | `model_kwargs={...}` | native (`AsyncAnthropic`) |
| `openai` | `OpenAIClient` | `aimu[openai_compat]` | `OPENAI_API_KEY` | api.openai.com | `model_kwargs={...}` | native (`AsyncOpenAI`) |
| `gemini` | `GeminiClient` | `aimu[openai_compat]` | `GOOGLE_API_KEY` | generativelanguage.googleapis.com | `model_kwargs={...}` | native (`AsyncOpenAI`) |
| `lmstudio` | `LMStudioOpenAIClient` | `aimu[openai_compat]` | — | localhost:1234 | `base_url=` | native (`AsyncOpenAI`) |
| `ollama-openai` | `OllamaOpenAIClient` | `aimu[openai_compat]` | — | localhost:11434 | `base_url=` | native (`AsyncOpenAI`) |
| `hf-openai` | `HFOpenAIClient` | `aimu[openai_compat]` | — | localhost:8000 | `base_url=` | native (`AsyncOpenAI`) |
| `vllm` | `VLLMOpenAIClient` | `aimu[openai_compat]` | — | localhost:8000 | `base_url=` | native (`AsyncOpenAI`) |
| `llamaserver` | `LlamaServerOpenAIClient` | `aimu[openai_compat]` | — | localhost:8080 | `base_url=` | native (`AsyncOpenAI`) |
| `sglang` | `SGLangOpenAIClient` | `aimu[openai_compat]` | — | localhost:30000 | `base_url=` | native (`AsyncOpenAI`) |
| `llamacpp` | `LlamaCppClient` | `aimu[llamacpp]` | — | in-process | `model_path=` (required), `n_ctx`, `n_gpu_layers`, `chat_format`, `chat_handler`, `verbose` | wrapped sync via `asyncio.to_thread` |

The **async kind** column says how `aimu.aio` reaches each provider. *Native* providers use the SDK's own async client; the async surface delivers real coroutine concurrency. *Wrapped sync* providers (HF, LlamaCpp) load model weights in-process — the async surface buys you event-loop integration (your handler doesn't block) but **not** coroutine-level concurrency (the GIL and CUDA stream serialize execution). For these providers, async clients are built by wrapping an existing sync client; see [how-to: use async](../how-to/use-async.md#in-process-providers-huggingface-llamacpp).

## Passing provider kwargs

Provider-specific kwargs are forwarded through `ModelClient` and `aimu.client()`:

```python
import aimu

# Ollama: keep model warm
aimu.client("ollama:qwen3.5:9b", model_keep_alive_seconds=300)

# llama-cpp: required model_path
aimu.client("llamacpp:qwen3-8b", model_path="/path/to/qwen3-8b.gguf")

# LM Studio: non-default port
aimu.client("lmstudio:qwen3.5-9b", base_url="http://myserver:1234/v1")
```

## Notes per provider

- **`OpenAIClient`** overrides `_update_generate_kwargs` for o-series models (o1/o3/o4): renames `max_tokens → max_completion_tokens` and forces `temperature=1`.
- **`AnthropicClient`** stores `self.messages` in OpenAI format; conversion to Anthropic's format happens at request time. Thinking is native (`thinking={"type": "enabled", "budget_tokens": N}`), not `<think>` tag parsing.
- **`GeminiClient`** uses Google's OpenAI-compatible endpoint. Gemini 2.5 thinking models emit `<think>` tags on this endpoint, so the shared `_ThinkingParser` works as-is.
- **`OllamaClient`** (native API) supports vision via the message-level `images=` field; only inline base64 / data URLs work, http(s) URLs raise `ValueError`.
- **`LlamaCppClient`** loads GGUF files in-process. Vision needs an `mmproj` projector via the `chat_handler=` kwarg (e.g. `Llava15ChatHandler(clip_model_path=...)`).
- **`HuggingFaceClient`** uses `AutoProcessor` for vision when available (Gemma 3/4, Qwen 3.5/3.6 VL). The model's `tool_call_format` enum value (`XML` / `JSON_OBJECT` / `JSON_ARRAY` / `BRACKETED`) tells the client how to parse tool calls.
- **OpenAI-compatible local servers** (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve) all subclass `OpenAICompatClient` and differ only in default `base_url` and the format of their model id strings.

## See also

- [Model matrix](model-matrix.md) — every shipped model with capability flags.
- [Environment variables](env-vars.md) — every env var AIMU reads.
- [How-to: switch providers](../how-to/switch-providers.md) — practical patterns.
