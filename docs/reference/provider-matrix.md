# Provider matrix

Every supported provider, with the extra needed to install it, the env var (if any) it reads, the default endpoint, and any provider-specific constructor kwargs.

## Text providers (`aimu.client()` / `aimu.chat()`)

| Provider key (in model string) | Client class | Extra | API key | Default endpoint | Provider-specific kwargs | Async kind |
|---|---|---|---|---|---|---|
| `ollama` | `OllamaClient` | `aimu[ollama]` | none | local | `model_keep_alive_seconds=60` | native (`ollama.AsyncClient`) |
| `hf` | `HuggingFaceClient` | `aimu[hf]` | none (HF login for gated models) | local | `model_kwargs={...}` (passed to `from_pretrained`) | wrapped sync via `asyncio.to_thread` |
| `anthropic` | `AnthropicClient` | `aimu[anthropic]` | `ANTHROPIC_API_KEY` | api.anthropic.com | `model_kwargs={...}` | native (`AsyncAnthropic`) |
| `openai` | `OpenAIClient` | `aimu[openai_compat]` | `OPENAI_API_KEY` | api.openai.com | `model_kwargs={...}` | native (`AsyncOpenAI`) |
| `gemini` | `GeminiClient` | `aimu[openai_compat]` | `GOOGLE_API_KEY` | generativelanguage.googleapis.com | `model_kwargs={...}` | native (`AsyncOpenAI`) |
| `lmstudio` | `LMStudioOpenAIClient` | `aimu[openai_compat]` | none | localhost:1234 | `base_url=` | native (`AsyncOpenAI`) |
| `ollama-openai` | `OllamaOpenAIClient` | `aimu[openai_compat]` | none | localhost:11434 | `base_url=` | native (`AsyncOpenAI`) |
| `hf-openai` | `HFOpenAIClient` | `aimu[openai_compat]` | none | localhost:8000 | `base_url=` | native (`AsyncOpenAI`) |
| `vllm` | `VLLMOpenAIClient` | `aimu[openai_compat]` | none | localhost:8000 | `base_url=` | native (`AsyncOpenAI`) |
| `llamaserver` | `LlamaServerOpenAIClient` | `aimu[openai_compat]` | none | localhost:8080 | `base_url=` | native (`AsyncOpenAI`) |
| `sglang` | `SGLangOpenAIClient` | `aimu[openai_compat]` | none | localhost:30000 | `base_url=` | native (`AsyncOpenAI`) |
| `llamacpp` | `LlamaCppClient` | `aimu[llamacpp]` | none | in-process | `model_path=` (required), `n_ctx`, `n_gpu_layers`, `chat_format`, `chat_handler`, `verbose` | wrapped sync via `asyncio.to_thread` |

## Image providers (`aimu.image_client()` / `aimu.generate_image()`)

| Provider key (in model string) | Client class | Extra | API key | Default endpoint | Provider-specific kwargs | Async kind |
|---|---|---|---|---|---|---|
| `hf` | `HuggingFaceImageClient` | `aimu[hf]` | none (HF login for gated models) | local | `model_kwargs={...}` (passed to `pipeline.from_pretrained`) | wrapped sync via `asyncio.to_thread` |
| `gemini` | `GeminiImageClient` | `aimu[google]` | `GOOGLE_API_KEY` | generativelanguage.googleapis.com | `model_kwargs={"api_key": "..."}` | wrapped sync via `asyncio.to_thread` |

## Audio providers (`aimu.audio_client()` / `aimu.generate_audio()`)

| Provider key (in model string) | Client class | Extra | API key | Default endpoint | Provider-specific kwargs | Async kind |
|---|---|---|---|---|---|---|
| `hf` | `HuggingFaceAudioClient` | `aimu[hf]` | none (HF login for gated models) | local | none | wrapped sync via `asyncio.to_thread` |

Audio clients share the `provider:model_id` string format with text and image. The namespaces don't collide because they're consumed by separate factories (`AudioClient` vs `ModelClient` vs `ImageClient`). Per-call generation kwargs: `duration_s`, `num_inference_steps` (diffusers models only), `seed`, `num_audio`. All audio clients accept `format=` (`"numpy"` / `"path"` / `"bytes"` / `"data_url"`) and `output_dir=` from the shared `BaseAudioClient.generate()`.

The **async kind** column says how `aimu.aio` reaches each provider. HuggingFace audio models load weights in-process. The async surface buys you event-loop integration (your handler doesn't block) but **not** coroutine-level concurrency. Async clients are built by wrapping an existing sync client via `aio.audio_client(sync_client)`; see [how-to: use async](../how-to/use-async.md#in-process-providers-huggingface-llamacpp).

Image clients share the `provider:model_id` string format with text. The namespaces don't collide because they're consumed by separate factories (`ImageClient` vs `ModelClient`). Per-call generation kwargs differ by provider: HF takes `negative_prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`, `seed`, `reference_image`, `strength`; Gemini takes `aspect_ratio`, `image_size`, `reference_image`. All image clients accept `num_images=`, `format=` (`"pil"` / `"path"` / `"bytes"` / `"data_url"`), `output_dir=`, and `reference_image=` (image-to-image; accepts path, bytes, URL, data URL, or PIL Image) from the shared `BaseImageClient.generate()`. When `reference_image` is set, `width`/`height` are derived from the reference for HF models.

The **async kind** column says how `aimu.aio` reaches each provider. *Native* providers use the SDK's own async client; the async surface delivers real coroutine concurrency. *Wrapped sync* providers (HF, LlamaCpp) load model weights in-process. The async surface buys you event-loop integration (your handler doesn't block) but **not** coroutine-level concurrency (the GIL and CUDA stream serialize execution). For these providers, async clients are built by wrapping an existing sync client; see [how-to: use async](../how-to/use-async.md#in-process-providers-huggingface-llamacpp).

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
- **`AnthropicClient`** stores `self.messages` in OpenAI format; conversion to Anthropic's format happens at request time. Thinking is native (not `<think>` tag parsing), built per the model's `ThinkingStyle`: `enabled` (`{"type": "enabled", "budget_tokens": N}`) for Opus 4.6 / Sonnet 4.6 / Haiku 4.5, or `adaptive` (`{"type": "adaptive", "display": "summarized"}`, sampling params dropped) for Opus 4.7+ / Fable 5. See the [model matrix](model-matrix.md#anthropic-anthropicmodel).
- **`GeminiClient`** uses Google's OpenAI-compatible endpoint. Gemini 2.5 thinking models emit `<think>` tags on this endpoint, so the shared `_ThinkingParser` works as-is.
- **`OllamaClient`** (native API) supports vision via the message-level `images=` field; only inline base64 / data URLs work, http(s) URLs raise `ValueError`.
- **`LlamaCppClient`** loads GGUF files in-process. Vision needs an `mmproj` projector via the `chat_handler=` kwarg (e.g. `Llava15ChatHandler(clip_model_path=...)`).
- **`HuggingFaceClient`** uses `AutoProcessor` for vision when available (Gemma 3/4, Qwen 3.5/3.6 VL). The model's `tool_call_format` enum value (`XML` / `JSON_OBJECT` / `JSON_ARRAY` / `BRACKETED`) tells the client how to parse tool calls.
- **OpenAI-compatible local servers** (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve) all subclass `OpenAICompatClient` and differ only in default `base_url` and the format of their model id strings.
- **`HuggingFaceImageClient`** loads the `diffusers` pipeline lazily on the first `generate()` call. The spec's `pipeline_class` field names a class in the `diffusers` namespace (e.g. `"StableDiffusionXLPipeline"`, `"FluxPipeline"`). Placement is **memory-aware and automatic**: it measures the model size and each GPU's *free* VRAM (accounting for other processes), then pins to the freest GPU, or falls back to model / sequential CPU offload when the model is too big to fit. Override with `model_kwargs={"device": "cuda:1"}` (pin) or `model_kwargs={"device_map": ...}` (hand to diffusers/accelerate). The audio and speech HF clients take the same `{"device": ...}` hint but place on a single device (small models).
- **`GeminiImageClient`** uses Google's native `google-genai` SDK (not the OpenAI-compat endpoint). Calls `client.models.generate_content` with `response_modalities=[Modality.IMAGE]`; Nano Banana returns one image per call so `num_images > 1` issues N requests. Aliases like `"gemini:nano-banana"` resolve to `gemini-2.5-flash-image`.

## See also

- [Model matrix](model-matrix.md): every shipped model with capability flags.
- [Environment variables](env-vars.md): every env var AIMU reads.
- [How-to: switch providers](../how-to/switch-providers.md): practical patterns.
- [How-to: generate images](../how-to/generate-images.md): image surface details.
- [How-to: generate audio](../how-to/generate-audio.md): audio surface details.
