# Provider matrix

Every supported provider, with the extra needed to install it, the env var (if any) it reads, the default endpoint, and any provider-specific constructor kwargs.

## Text providers (`aimu.client()` / `aimu.chat()`)

| Provider key (in model string) | Client class | Extra | API key | Default endpoint | Provider-specific kwargs | Async kind |
|---|---|---|---|---|---|---|
| `ollama` | `OllamaClient` | `aimu[ollama]` | none | `OLLAMA_HOST`, else localhost:11434 | `host=`, `model_keep_alive_seconds=60` | native (`ollama.AsyncClient`) |
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
| `omlx` | `OMLXOpenAIClient` | `aimu[openai_compat]` | none | localhost:8000 | `base_url=` | native (`AsyncOpenAI`) |
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

# Ollama on another machine (or `"ollama:qwen3.5:9b@http://gpu-box:11434"`)
aimu.client("ollama:qwen3.5:9b", host="http://gpu-box:11434")

# llama-cpp: required model_path
aimu.client("llamacpp:qwen3-8b", model_path="/path/to/qwen3-8b.gguf")

# LM Studio: non-default port
aimu.client("lmstudio:qwen3.5-9b", base_url="http://myserver:1234/v1")
```

## Generation parameters

Sampling parameters come from four places. They are merged **per parameter** on every call, in
this order, so a lower tier only supplies what no higher tier set:

| Tier | Source | Typical use |
|------|--------|-------------|
| 1 | `Client.DEFAULT_GENERATE_KWARGS` | AIMU's own fallbacks for what nobody else sets: `max_tokens` 16000 on the cloud providers, 4096 on the local ones (HuggingFace as `max_new_tokens`); empty on Ollama |
| 2 | `ModelSpec.generation_kwargs` | the sampling the model card recommends (see the [model matrix](model-matrix.md)) |
| 3 | `client.default_generate_kwargs` | **your** standing choice for every call on this client |
| 4 | `chat(generate_kwargs={...})` | **your** choice for one call |

Tier 3 is the one to reach for when a setting should apply to a whole conversation:

```python
client = aimu.client("ollama:qwen3.5:9b")
client.default_generate_kwargs = {"temperature": 0.2, "context_length": 16384}

client.chat("summarise this")                                    # temperature 0.2
client.chat("now be creative", generate_kwargs={"temperature": 1.0})  # 1.0, just this call
```

It starts empty on every provider, and it is an *input*, not a report: reading it back shows what
you set, not the effective request. Assign a whole dict or mutate it in place; both work, and both
propagate through the `ModelClient` wrapper that `aimu.client()` returns, through
`agent.as_model_client()`, and down a `FallbackClient`'s chain.

Tier 1 sits *below* the model card on purpose: AIMU's generic `temperature=0.1` must not override
the tuned value a card specifies. Provider-specific rewrites run *after* the merge and can still
override you where an API demands it (see the notes below).

!!! note "Context length per provider"
    The portable [`context_length`](../how-to/set-context-length.md) key is renamed on the one
    backend that sizes the window per request and dropped with a warning on the rest, so a client
    default survives a provider swap instead of putting an unknown parameter on the wire.

    | Client | Per request? | Where it comes from instead |
    |---|---|---|
    | `OllamaClient` | **yes**, as `num_ctx` | -- |
    | `OllamaOpenAIClient` | no | `OLLAMA_CONTEXT_LENGTH` on the server |
    | other OpenAI-compat local servers | no | server launch (`--ctx-size`, `--max-model-len`, LM Studio's setting) |
    | `LlamaCppClient` | no | `LlamaCppClient(..., n_ctx=N)`, at load time |
    | `HuggingFaceClient` | no | the weights' own `max_position_embeddings` |
    | `AnthropicClient`, `OpenAIClient`, `GeminiClient` | no | fixed by the vendor |

    A client declares which case it is in as its `GENERATE_KWARG_SUPPORT["context_length"]` entry,
    alongside its verdict for the other seven portable keys (below); the warning fires once per
    client rather than once per call.

!!! note "Generation parameters per provider"
    Eight generation parameters have one portable name each, and each client declares what it does
    with every one of them in `GENERATE_KWARG_SUPPORT`: the backend's own spelling for the ones it
    accepts, a drop-with-a-warning for the ones it cannot honour. `temperature` and `top_p` are
    accepted everywhere under those names; the rest vary.

    | Client | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` | `max_tokens` |
    |---|---|---|---|---|---|
    | `OllamaClient` | `top_k` | dropped | `presence_penalty` | `repeat_penalty` | `num_predict` |
    | OpenAI-compat local servers (vLLM, SGLang, LM Studio, oMLX, HF Serve) | `extra_body` | `extra_body` | `presence_penalty` | `extra_body` | `max_tokens` |
    | `LlamaServerOpenAIClient` | `extra_body` | `extra_body` | `presence_penalty` | `extra_body`, as `repeat_penalty` | `max_tokens` |
    | `OllamaOpenAIClient` | dropped | dropped | `presence_penalty` | dropped | `max_tokens` |
    | `LlamaCppClient` | `top_k` | `min_p` | `presence_penalty` | `repeat_penalty` | `max_tokens` |
    | `HuggingFaceClient` | `top_k` | `min_p` | dropped | `repetition_penalty` | `max_new_tokens` |
    | `AnthropicClient` | `extra_body` | dropped | dropped | dropped | `max_tokens` |
    | `OpenAIClient`, `GeminiClient` | dropped | dropped | `presence_penalty` | dropped | `max_tokens` |

    "dropped" means the key is removed before the request and logged once per client, naming where
    to set it instead. `extra_body` means the key survives but moves: the OpenAI schema has no
    top-level place for it, and a local server reads it from the extra request field instead.
    `AnthropicClient` moves it for a different reason -- `anthropic` 1.x removed `temperature`,
    `top_p` and `top_k` from the `messages.create()` signature, so all three now travel in
    `extra_body`, and only on the models that still accept them (see the thinking section below).

    **The local OpenAI-compatible row is not uniformly confirmed.** It follows the sampling
    extensions vLLM and SGLang document, and the two servers that leave the row (`OllamaOpenAIClient`
    and `LlamaServerOpenAIClient`, below) were each checked against their own backend's reference.
    LM Studio, oMLX, and HF Transformers Serve inherit the row unconfirmed. The cell most in doubt is
    `repetition_penalty` on an llama.cpp-engine server: LM Studio runs llama.cpp, and llama-server
    wants `repeat_penalty`, so LM Studio plausibly does too. Treat that one as untested rather than
    settled, and confirm it against the server's reference before relying on it.

    Six entries are worth the detail:

    - **Ollama's `min_p`.** The `ollama` SDK types `options` as a pydantic `Options` model with no
      `min_p` field, so the value is discarded during request validation whatever the server
      supports. Set it in the model's Modelfile (`PARAMETER min_p`) instead.
    - **HuggingFace's `presence_penalty`.** Transformers' `generate()` has no such concept and
      raises on one; `repetition_penalty` is the nearest equivalent.
    - **`OllamaOpenAIClient`'s three knobs.** "OpenAI-compatible" describes the endpoint, not the
      sampling surface behind it: Ollama's shim maps a fixed OpenAI field set onto its native call and
      reads none of `top_k` / `min_p` / `repetition_penalty`, so putting them in `extra_body` would
      hand the server three fields it discards. They are declared unsupported instead, and the remedy
      names the native `ollama` provider (or, for `min_p`, which the native SDK cannot carry either,
      the model's Modelfile).
    - **`LlamaServerOpenAIClient`'s repetition knob.** llama-server accepts llama.cpp's own
      `/completion` sampling parameters on its OpenAI endpoint, where the knob is spelled
      `repeat_penalty`; vLLM and SGLang use `repetition_penalty`. One-key rename, and the `extra_body`
      routing follows it, because the OpenAI SDK's `create()` takes no arbitrary keywords.
    - **The cloud endpoints' `top_k`.** The OpenAI parameter set has none, and Google's
      OpenAI-compatibility reference documents no top-level `top_k` and no place for it under
      `extra_body` either, so it is declared unsupported on both: a parameter the endpoint rejects
      fails the whole request, where a dropped one only stops applying. **On Gemini that is an open
      question, not a settled impossibility.** The same reference does document
      `extra_body={"generation_config": ...}`, and the native Gemini API carries `topK` inside
      `generationConfig`, so that route may well work; it could be neither confirmed nor disproved
      without a live key. Revisit the Gemini cell with one in hand.
    - **o-series `max_tokens`.** `OpenAIClient` sends `max_completion_tokens` instead for o1/o3/o4.
      That rename depends on the model rather than the client, so it stays in the rewrite hook rather
      than the table; see [Notes per provider](#notes-per-provider) for what else the hook does there.

    Only a value *you* set is reported: one that came from the model card's sampling profile is
    dropped silently, since most cards carry `min_p` and `repetition_penalty` and a warning would
    otherwise fire once per client for a value you never chose. A `None` value means unset on every
    key, so a per-call `None` cancels a client default without reporting anything.

## Notes per provider

- **`OpenAIClient`** overrides `_rewrite_generate_kwargs` for o-series models (o1/o3/o4): renames `max_tokens → max_completion_tokens`, forces `temperature=1`, and drops `top_p` (the o-series exposes no sampling control).
- **`AnthropicClient`** stores `self.messages` in OpenAI format; conversion to Anthropic's format happens at request time. Thinking is native (not `<think>` tag parsing), built per the model's `ThinkingStyle`: `enabled` (`{"type": "enabled", "budget_tokens": N}`) for Haiku 4.5 alone, or `adaptive` (`{"type": "adaptive", "display": "summarized"}`) for every other member. Sampling parameters are dropped on Opus 4.7 and later, which reject them outright; the 4.6 line is adaptive and still accepts them. See the [model matrix](model-matrix.md#anthropic-anthropicmodel).

!!! note "Thinking control per provider"
    The portable [`thinking=`](../how-to/control-thinking.md) parameter reaches each backend
    differently, and two of them cannot express it at all today:

    | Client | Turn reasoning off | Set an effort level |
    |---|---|---|
    | `OllamaClient` | `think=False` | `think="low"/"medium"/"high"` |
    | OpenAI-compat local servers | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | `reasoning_effort`, with `high` sent as Qwen's `xhigh` |
    | `AnthropicClient` | `{"type": "disabled"}`, except Haiku 4.5 which omits the parameter; Fable 5 cannot be turned off | `output_config.effort` (`high` sent as `xhigh` where the model has it); `budget_tokens` 2048 / 8000 / 16000 on Haiku 4.5 |
    | `HuggingFaceClient` | `enable_thinking=False` template kwarg | `reasoning_effort` template kwarg |
    | `LlamaCppClient`, `OpenAIClient`, `GeminiClient` | nothing emitted | nothing emitted |

    `chat_template_kwargs` is a Qwen and vLLM *template* convention rather than part of the
    OpenAI API, so the two cloud subclasses opt out of it via
    `_SUPPORTS_CHAT_TEMPLATE_KWARGS = False` and warn instead of sending a field their endpoint
    would not honour. Gemini is a deferred case rather than an impossible one: Google's endpoint
    does accept `reasoning_effort`, but its vocabulary (`minimal/low/medium/high/none`) excludes
    the `xhigh` that AIMU's shared Qwen mapping sends for `high`, so a correct Gemini mapping
    needs its own effort vocabulary.

- **`GeminiClient`** uses Google's OpenAI-compatible endpoint. Gemini 2.5 thinking models emit `<think>` tags on this endpoint, so the shared `_ThinkingParser` works as-is.
- **`OllamaClient`** (native API) supports vision via the message-level `images=` field; only inline base64 / data URLs work, http(s) URLs raise `ValueError`.
- **`OllamaClient`, `AsyncOllamaClient`, and `OllamaEmbeddingClient`** take `host=` for a remote server, forwarded to the ollama SDK verbatim (bare host, `host:port`, or `scheme://host:port` all work). Pass the **server root**, not the `/v1` OpenAI-compatible path: the native API is served from the root, so a `/v1` suffix raises `ValueError` pointing at the `ollama-openai` provider. `OllamaClient` pulls the model eagerly in its constructor, so with `host=` set the pull runs on the remote machine. Give the embedding client the same `host=` as the text client -- embedding a corpus on one server and querying on another silently mixes vectors from two models.
- **`LlamaCppClient`** loads GGUF files in-process. Vision needs an `mmproj` projector via the `chat_handler=` kwarg (e.g. `Llava15ChatHandler(clip_model_path=...)`).
- **`HuggingFaceClient`** uses `AutoProcessor` for vision when available (Gemma 3/4, Qwen 3.5/3.6 VL). The model's `tool_call_format` enum value (`XML` / `JSON_OBJECT` / `JSON_ARRAY` / `BRACKETED`) tells the client how to parse tool calls.
- **MLX on Apple Silicon.** Three providers execute MLX-optimized weights: `omlx` (a dedicated MLX server), `lmstudio` (LM Studio's MLX engine, selected automatically for MLX weights), and `ollama` (0.19+ runs an MLX backend on Apple Silicon for its own model library, transparently — the tags are unchanged, and it wants >32 GB of unified memory). **`hf` and `llamacpp` are not MLX paths**: `HuggingFaceClient` is torch/`transformers` and `LlamaCppClient` is GGML/GGUF, and neither can load MLX's quantized safetensors layout. mlx-community models are *hosted* on the HuggingFace Hub but only mlx-lm/mlx-vlm can run them, so there is no in-process MLX client.
- **`OMLXOpenAIClient`** talks to [oMLX](https://github.com/jundot/omlx) (`omlx serve --model-dir ~/models`). oMLX discovers models from subdirectories, so a model id is **whatever the user named the folder** — the catalog ids follow the convention "directory name == the mlx-community repo's model segment", which is what a copy-pasted download produces. For any other layout use the ad-hoc form, and note that capability flags default to *false*, so spell them out: `omlx:my-dir;tools,thinking,vision`. oMLX also accepts a `<model>:<profile>` alias (`omlx:Qwen3.6-35B-A3B:fast`); the extra colon is fine, since only the first one splits off the provider. Its default port collides with vLLM and HF Transformers Serve, so pass `base_url=` / `@<base_url>` when running more than one.
- **OpenAI-compatible local servers** (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve, oMLX) all subclass `OpenAICompatClient` and differ only in default `base_url` and the format of their model id strings. The endpoint can be overridden inline in the model string with `@<base_url>`, and a model id not in the catalog can be run by declaring its capabilities with `;<flags>` (or via the generic `openai-compat` prefix); see [Point at a remote or custom OpenAI-compatible server](../how-to/switch-providers.md#point-at-a-remote-or-custom-openai-compatible-server).
- **`HuggingFaceImageClient`** loads the `diffusers` pipeline lazily on the first `generate()` call. The spec's `pipeline_class` field names a class in the `diffusers` namespace (e.g. `"StableDiffusionXLPipeline"`, `"FluxPipeline"`). Placement is **memory-aware and automatic**: it measures the model size and each GPU's *free* VRAM (accounting for other processes), then pins to the freest GPU, or falls back to model / sequential CPU offload when the model is too big to fit. Override with `model_kwargs={"device": "cuda:1"}` (pin) or `model_kwargs={"device_map": ...}` (hand to diffusers/accelerate). The audio and speech HF clients take the same `{"device": ...}` hint but place on a single device (small models).
- **`GeminiImageClient`** uses Google's native `google-genai` SDK (not the OpenAI-compat endpoint). Calls `client.models.generate_content` with `response_modalities=[Modality.IMAGE]`; Nano Banana returns one image per call so `num_images > 1` issues N requests. Aliases like `"gemini:nano-banana"` resolve to `gemini-2.5-flash-image`.

## See also

- [Model matrix](model-matrix.md): every shipped model with capability flags.
- [Environment variables](env-vars.md): every env var AIMU reads.
- [How-to: switch providers](../how-to/switch-providers.md): practical patterns.
- [How-to: generate images](../how-to/generate-images.md): image surface details.
- [How-to: generate audio](../how-to/generate-audio.md): audio surface details.
