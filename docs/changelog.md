# Changelog

## Unreleased — Audio input for text models

### Models

- **New** `audio: bool = False` field on `ModelSpec`. Audio-capable text models expose `supports_audio` on their enum members, `is_audio_model` on their client instances, and an `AUDIO_MODELS` classproperty (parallel to `TOOL_MODELS`, `THINKING_MODELS`, `VISION_MODELS`).
- **New** Audio-capable models added to the catalog: `OpenAIModel` GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano; `GeminiModel` 2.0 Flash, 2.0 Flash Lite, 2.5 Pro, 2.5 Flash; `HuggingFaceModel.GEMMA_4_E4B`, `GEMMA_4_12B`, `NEMOTRON_H_8B`. Ollama models remain `audio=False` with inline comments noting where the underlying weights support audio (upgrade path once the Ollama API adds audio input).

### `ModelClient.chat()` and `ModelClient.generate()`

- **New** `audio=` parameter on both `chat()` (stateful — turn persists in `self.messages`) and `generate()` (stateless one-shot — no history touched). Accepts any mix of: file path strings, `pathlib.Path`, raw bytes (WAV assumed), `https://` URLs (fetched eagerly), and `data:audio/...;base64,...` data URLs. Supported format strings: `wav`, `mp3`, `ogg`, `flac`, `m4a`, `webm` — inferred from file extension or MIME type.
- **New** Passing `audio=` to a model with `supports_audio=False` raises `ValueError` before any API call.
- **New** `images=` and `audio=` are mutually exclusive per turn; passing both raises `ValueError`.
- Internally normalised to OpenAI `input_audio` content blocks (`{"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}`). Provider adaptation happens at request time: OpenAI/Gemini/OpenAI-compat pass through; Anthropic converts to `{"type": "audio", "source": {"type": "base64", ...}}`; HuggingFace decodes to float32 numpy arrays via `soundfile` and passes them to the `AutoProcessor`; Ollama raises with a clear message (API does not yet support audio).
- Mirrored on the async surface (`aimu.aio`): same signature on `aio.chat()` and `aio.generate()`.
- **Fix** `ModelClient._generate` (and the async `AsyncModelClient._generate` / `_chat`) now accept and forward `audio=`. They were missing the parameter while the base `generate()`/`chat()` always pass it, so every `aimu.client().generate()` / `aimu.chat(...)` call through the factory raised `TypeError: _generate() got an unexpected keyword argument 'audio'`. (Concrete provider clients were unaffected, which is why the live test suite — which constructs them directly — didn't surface it.)

### Documentation

- **New** `docs/how-to/handle-audio-input.md` — accepted input forms, model selection, stateful vs. stateless, async surface, per-provider adaptation.
- **New** `notebooks/18 - Audio Input.ipynb` — capability flags, all input forms, multiple clips per turn, stateful/stateless split, multi-turn conversations, capability check, mutual-exclusion demo, Gemini and HuggingFace sections, async surface.

### Transcription (speech-to-text)

- **New** `aimu.transcription_client()` / `aimu.transcribe()` + `TranscriptionClient` factory + `BaseTranscriptionClient` ABC — a dedicated speech-to-text surface, parallel to TTS (`BaseSpeechClient`). Disjoint from the `audio=` parameter on text models, which handles audio analysis/QA by audio-capable chat models; this surface uses dedicated ASR models (Whisper family, gpt-4o-transcribe) optimised for transcription.
- **New** `OpenAITranscriptionClient` + `OpenAITranscriptionModel` — cloud ASR backed by `openai.audio.transcriptions.create()`. Models: `WHISPER_1`, `GPT_4O_TRANSCRIBE`, `GPT_4O_MINI_TRANSCRIBE`. Auth via `OPENAI_API_KEY`. Uses the same `openai` SDK already required by the `[openai_compat]` extra.
- **New** `HuggingFaceTranscriptionClient` + `HuggingFaceTranscriptionModel` — local ASR backed by `transformers.pipeline("automatic-speech-recognition")`. Models: `WHISPER_TINY`, `WHISPER_BASE`, `WHISPER_SMALL`, `WHISPER_MEDIUM`, `WHISPER_LARGE_V3`, `DISTIL_WHISPER_LARGE_V3`. Weight caching via module-level registry (same pattern as other HF clients).
- **New** `transcribe(audio, language=None, response_format="text", prompt=None, temperature=None) -> str | dict`. Accepted audio forms: file path, raw bytes, `https://` URL, `data:audio/...` URL — same set as `audio=` on `chat()`. `response_format="verbose_json"` returns a dict with `text`, `segments` (start/end/text), `language`, `duration`. `response_format` defaults to `"text"` (plain string).
- **New** `AIMU_TRANSCRIPTION_MODEL` env var — sets the default model for `aimu.transcription_client()` and `aimu.transcribe()` when `model=` is omitted.
- **New** Async mirror under `aimu.aio`: `AsyncTranscriptionClient`, `aio.transcription_client(sync_client)`, `await aio.transcribe(audio, *, model, ...)`. Wraps sync via `asyncio.to_thread` (Decision 7 — same as every other aio modality).
- **New** Built-in `transcribe_audio(audio_path: str) -> str` `@tool` in `aimu.tools.builtin`; `builtin.transcription` subgroup; included in `ALL_TOOLS`. Backed by a lazy `_transcription_client` singleton via `AIMU_TRANSCRIPTION_MODEL`. `make_transcription_tool(client)` binds a fresh tool to a caller-supplied client.
- **New** `docs/how-to/transcribe-audio.md` and `notebooks/19 - Transcription.ipynb`.

### Embeddings (text-to-vector)

- **New** `aimu.embedding_client()` / `aimu.embed()` + `EmbeddingClient` factory + `BaseEmbeddingClient` ABC — a dedicated text-embedding surface, parallel to the other modality clients. `embed()` takes one string (returns `list[float]`) or a list (returns `list[list[float]]`, order preserved); an empty list returns `[]` without a provider call. `client.dimensions` reports the spec's vector width.
- **New** `OpenAIEmbeddingClient` + `OpenAIEmbeddingModel` (`text-embedding-3-small/large`, `text-embedding-ada-002`) via `openai.embeddings.create()`; auth via `OPENAI_API_KEY`.
- **New** `OllamaEmbeddingClient` + `OllamaEmbeddingModel` (`nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, `all-minilm`) via `ollama.embed()`.
- **New** `HuggingFaceEmbeddingClient` + `HuggingFaceEmbeddingModel` (MiniLM-L6-v2, BGE small/base/large-en-v1.5, GTE-large, E5-large-v2, mxbai-embed-large-v1) backed by `sentence-transformers` so each model's own pooling/normalization config is honoured; lazy load + module-level weight cache (freed by `aimu.clear_hf_cache()`). Adds `sentence-transformers>=3` to the `[hf]` extra.
- **New** `SemanticMemoryStore(embedding_client=...)` — pluggable embedding model; default `None` keeps ChromaDB's built-in embedder (unchanged behaviour).
- **New** `AIMU_EMBEDDING_MODEL` env var sets the default model for `aimu.embedding_client()` / `aimu.embed()` when `model=` is omitted (raises if unset; no implicit download).
- **New** Async mirror: `aio.embedding_client(sync_client)` / `aio.embed()` wrap a sync client via `asyncio.to_thread`.
- **Docs** `docs/how-to/use-embeddings.md`, `notebooks/20 - Embeddings.ipynb`, API reference, and env-var reference.

### RAG primitives (retrieval-augmented generation)

- **New** `aimu.rag` — chunk/retrieve/rerank helpers as **plain functions over the `MemoryStore` interface** (no retriever/splitter/loader class hierarchy).
- **New** `split_text(text, *, chunk_size=1000, chunk_overlap=200, separators=None, length_function=len)` — recursive separator-based chunking (paragraphs → lines → sentences → words → characters) with overlap. `length_function` defaults to character count; pass a tokenizer's counter for token-aware chunking. Oversized unsplittable text hard-cuts at `chunk_size`.
- **New** `ingest(store, documents, *, chunk_size, chunk_overlap, separators, length_function) -> int` — splits one or many documents and stores each chunk via `store.store()`; returns the chunk count. `retrieve(store, query, *, n_results=5, **search_kwargs) -> list[str]` — a RAG-named pass-through to `store.search()` (forwards e.g. `max_distance=`). `format_context(chunks, *, separator="\n\n", numbered=False) -> str` — join chunks for prompt augmentation.
- **New** `rerank(query, documents, *, model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=None)` — cross-encoder reranking via `sentence-transformers` (the `[hf]` extra); lazy-loaded and cached. Empty input returns `[]` without loading the model.
- **New** `make_retrieval_tool(store, *, n_results=5)` in `aimu.tools.builtin` — wraps `retrieve` + `format_context` as a `retrieve_context(query)` agent tool (returns numbered context).
- **Docs** `docs/how-to/use-rag.md` and the `aimu.rag` API reference.
- Loaders and per-chunk metadata are intentionally out of scope: ingestion sources are covered by `read_file` / `get_webpage` (or any text-returning library), and chunks are stored as plain strings per the `MemoryStore` contract.

### Token usage surfacing

- **New** `client.last_usage` — token counts for the most recent non-streaming `chat()` / `generate()`, as `{"input_tokens", "output_tokens", "total_tokens"}` (or `None` when the provider/server omits usage). Captured for Anthropic, OpenAI-compat (incl. OpenAI/Gemini/local servers), and Ollama, on both sync and async surfaces, and delegated through the `ModelClient` / `AsyncModelClient` wrappers. Reset to `None` on streaming calls (streaming usage capture is a separate follow-up) and by `reset()`. Token counts only — dollar cost is derivable but intentionally not computed (no maintained price table).

### Anthropic models & adaptive thinking

- **New** `AnthropicModel` members: `CLAUDE_FABLE_5` (`claude-fable-5`), `CLAUDE_OPUS_4_8` (`claude-opus-4-8`), `CLAUDE_OPUS_4_7` (`claude-opus-4-7`) — all `tools=True, thinking=True, vision=True`.
- **New** `ThinkingStyle` enum (`ENABLED` / `ADAPTIVE`) carried as a per-member extra on `AnthropicModel` (analogous to HuggingFace's `ToolCallFormat`). `AnthropicClient._thinking_kwargs()` builds the request accordingly: `ENABLED` → `{"type": "enabled", "budget_tokens": N}`; `ADAPTIVE` → `{"type": "adaptive", "display": "summarized"}` with `temperature`/`top_p`/`top_k` dropped. Opus 4.7+ and Fable 5 are adaptive-only (the `enabled` form 400s on them); Opus 4.6, Sonnet 4.6, and Haiku 4.5 use the budget form.
- **Fix** `CLAUDE_HAIKU_4_5` now correctly has `thinking=True` — Haiku 4.5 supports extended thinking via the `enabled`/`budget_tokens` form (previously omitted, so thinking tests silently skipped).
- Adaptive models decide per request whether to think and may emit none on simple prompts; the thinking tests use a multi-step reasoning prompt and assert thinking emission rather than an exact answer.
- **Docs** Updated the [model matrix](reference/model-matrix.md#anthropic-anthropicmodel), [provider matrix](reference/provider-matrix.md), [add a new model](how-to/add-new-model.md#anthropic-provider-specific-extras), and CLAUDE.md (Thinking Models, AnthropicClient notes) to cover the two thinking styles and the new models.
- **Docs** Added an "Adaptive vs. budget thinking" section to `notebooks/01 - Model Client.ipynb` (section C) demonstrating `ThinkingStyle` and adaptive models skipping thinking on trivial prompts.

### Dependencies

- **Fix** Pinned the `[hf]` extra's `kernels` to `>=0.12,<0.13`. It was unconstrained and resolved to `kernels 0.15.2`, which is outside the range `transformers` supports (`<0.13`); `transformers` constructs `kernels.LayerRepository(...)` at import time and 0.13+ made `revision`/`version` mandatory, so `from transformers import AutoProcessor` raised `ValueError` — silently flipping `HAS_HF` to `False` (HuggingFace clients unavailable) and erroring every HF test on import.
- Pinned the `[hf]` extra's `transformers` to `>=5,<6` (the major the model catalog targets — Qwen 3.6, Gemma 4, GPT-OSS) so a future major can't reintroduce this class of import-time breakage on resolve.

## v0.7.0 (2026-06-08) — MCP tool unification, model resolvers, and agent improvements

### Breaking changes

- **Breaking** Removed the `model_client.mcp_client` attribute. MCP tools now integrate through the single `model_client.tools` registry: call `MCPClient(...).as_tools()` (sync) or `await aio.MCPClient.connect(...).as_tools()` (async) to turn a server's tools into `@tool`-style callables, then add them to `tools` (constructor `Agent(tools=...)`, `client.tools = ...`, or the per-call `chat(tools=...)` / `run(tools=...)` override). Migration: replace `client.mcp_client = mcp` with `client.tools = mcp.as_tools()` (concatenate with `@tool` functions as needed, e.g. `builtin.web + mcp.as_tools()`). Two consequences: dispatch is now one by-name lookup over `tools`, so on a name collision the **last** entry wins (previously Python `@tool` always beat a same-named MCP tool — to preserve that, append the Python tool *after* `mcp.as_tools()`); and `MCPClient.get_tools()` is no longer called on every `chat()` (the tool list is snapshotted by `as_tools()`), so call `as_tools()` again to pick up server-side tool changes. `SkillAgent` and the internal dispatch (`_handle_tool_calls(tool_calls)`, `_call_plain_tool(tc, tc_id)` — both lost their `tools` parameter) were updated accordingly. The `MCPClient` class, its `get_tools()` / `call_tool()` / `ping()`, and the `aio.MCPClient` parallel are unchanged.
- **Breaking** `system_message` is no longer immutable after the first `chat()`. The setter is now always live: assigning it mid-conversation rewrites the `{"role": "system"}` entry in `messages` in place (re-conditioning the model on the new prompt while preserving history), inserts one if absent, or removes it on `None`. Before the first chat it still just seeds the value. The previous behaviour raised `RuntimeError`; code that caught that error to gate a `reset()` can now assign directly. To change the prompt *and* drop history, use `reset(system_message="new")`. Two consequences are accepted by design: the transcript becomes counterfactual (prior assistant turns predate the new prompt), and there is no longer a guard against silently re-conditioning a `ModelClient` shared by another agent's in-flight conversation — don't share a live-conversation client across agents that each set `system_message`. The `_system_message_locked` flag has been removed. See [System message lifecycle](explanation/system-message-lifecycle.md).

### Models

- **New** `aimu.resolve_model_enum(model)` and `aimu.resolve_image_model_enum(model)` — resolve a model to its `Model` / `ImageModel` enum member from any of three input forms: an enum member (returned unchanged), a `"provider:model_id"` string (delegates to `resolve_model_string` / `resolve_image_model_string`), or a **bare enum-member name** (e.g. `"QWEN_3_8B"`, `"FLUX_2_KLEIN_4B"`, `"NANO_BANANA"`) looked up across every installed provider enum. Useful for CLIs/scripts that accept "enum, name, or string" uniformly. For text, an ambiguous bare name (the same id ships under many providers) is disambiguated the way the omitted-`model` default is — prefer a provider where the model is *actually available locally* (running Ollama → cached HuggingFace → reachable local OpenAI-compat server, tool-capable first), logged at WARNING; if it isn't available under any provider, `ValueError` lists the `"provider:model_id"` options. This availability probe runs only on the ambiguous path. `resolve_image_model_enum` has no local-availability notion (image catalogs don't collide) and raises on the rare ambiguity. Exported from `aimu.models` and top-level `aimu`.
- **New** `aimu.available_text_models(*, include_hf_cache=True)` — discovery: return locally *available* text models as `Model` enum members (running Ollama → cached HuggingFace → reachable local OpenAI-compat servers), in provider-priority order. Download-free and cloud-free. `aimu.resolve_default_text_model_enum(*, include_hf_cache=True)` returns the single auto-pick (env var → first available, tool-capable preferred) as an enum member — the enum-returning twin of the internal default resolver that backs `client()`/`chat()`/`agent()` when `model=` is omitted.
- **New** Gemma 4 12B added to every provider that can run it: `OllamaModel.GEMMA_4_12B` (`gemma4:12b`, tools/thinking/vision, with the shared Gemma sampling kwargs), `HuggingFaceModel.GEMMA_4_12B` (the instruction-tuned `google/gemma-4-12b-it`, tools/vision, processor `parse_response` path), and a `GEMMA_4_12B` member on every OpenAI-compat server enum (`OllamaOpenAIModel`, `LMStudioOpenAIModel`, `VLLMOpenAIModel`, `HFOpenAIModel`, `LlamaServerOpenAIModel`, `SGLangOpenAIModel`) plus `LlamaCppModel`. The server/llama.cpp entries are `tools=True`, matching the established `GEMMA_3_12B` convention for those catalogs. Resolvable via the usual `"provider:model_id"` strings (e.g. `"ollama:gemma4:12b"`, `"hf:google/gemma-4-12b-it"`, `"vllm:google/gemma-4-12b-it"`).

### Tools

- **New** The `tool` decorator is re-exported at the top level as `aimu.tool`. `@aimu.tool` is now the single recommended/documented form across the README, tutorials, how-tos, and notebook examples — it's namespaced, so it can't be silently shadowed by another library's same-named `tool` decorator (LangChain, smolagents, etc.). `from aimu.tools import tool` remains valid and unchanged (same object); it's the natural form for code already inside `aimu.tools`. The `ToolSignatureError` message prefix is now `@aimu.tool:` to match. No behaviour change to decoration or dispatch.
- **New** `MCPClient.as_tools()` (sync) and `aio.MCPClient.as_tools()` (async) return a server's tools as `@tool`-style callables, each closing over the client, invoking `call_tool()` cross-process, and carrying `__tool_spec__` / `__tool_is_async__` / `__tool_is_streaming__`. Drop them straight into `tools` (`client.tools = mcp.as_tools()`, `Agent(tools=builtin.web + mcp.as_tools())`). This unifies MCP and in-process tools onto the single `self.tools` registry and one dispatch path — see the breaking-change note above for the migration from `model_client.mcp_client`. New shared helper `aimu.tools.mcp_format.mcp_content_to_text(tool_response)` flattens a `call_tool` result to a string.
- **New** Per-call tool override: `chat(..., tools=None)` and `Agent.run(..., tools=None)` (both sync and `aimu.aio`) accept a `tools=` list that replaces the client's configured `self.tools` for a single call/run, restored afterward. `tools=None` (default) keeps the existing behaviour; `tools=[]` disables tools for the call (MCP tools, being callables in `self.tools` via `as_tools()`, are included in the swap). On an `Agent`, the override applies to every turn of the agentic loop. Implemented as a scoped `self.tools` swap (`_ChatStateMixin._tools_override`) covering both request-spec building and dispatch; the agent threads it through each loop `chat()` call so no new agent state is introduced. Not safe across concurrent `chat()` calls on a shared client — same contract as `self.messages`. Not added to the `Runner` ABC / workflow classes.

### Agents and workflows

- **New** `Agent.final_answer_prompt` (opt-in, default `None`; sync and `aimu.aio`) — guarantees a final answer when the agentic loop exhausts `max_iterations` while the model is still calling tools. Instead of returning whatever the last (possibly tool-only) turn produced — an empty or stub result — the agent sends this prompt once with tools disabled (`chat(..., tools=[])`), forcing the model to synthesize an answer from the context it has gathered. The trigger is the post-loop `_last_turn_called_tools()` check (no new counter); it fires only on the cap-with-pending-tools path (a natural finish — a turn with no tool calls — is unaffected) and the wrap-up turn is **not** counted against `max_iterations`. `OrchestratorAgent._init_orchestrator()` and `OrchestratorAgent.assemble(..., final_answer_prompt=...)` (sync + `aio`) forward it to the inner orchestrator agent, and it is accepted as a `from_config` key. Leaving it `None` preserves prior behaviour exactly.

### Fixes

- **Fix** `SkillAgent` skill injection no longer wipes conversation history when applied to an already-used client. It previously called `reset()` to unlock the setter (clearing `messages`); it now assigns `system_message` directly, which swaps the system entry in place.

## v0.6.0 (2026-06-04) — Output utilities, model weight caching, and experiment checkpointing

### Breaking changes

- **Breaking** Renamed `HuggingFaceImageModel.FLUX_DEV` → `FLUX_1_DEV` and `FLUX_SCHNELL` → `FLUX_1_SCHNELL` for naming consistency with the `FLUX_2_KLEIN_4B`/`FLUX_2_KLEIN_9B` members. The underlying model id strings (`black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell`) are unchanged. Update enum references; `"hf:black-forest-labs/FLUX.1-dev"` string-form usage is unaffected.
- **Behavior change** `builtin.compute` now includes `execute_python` alongside `calculate`. If you were passing `tools=builtin.compute` and want to exclude the sandboxed REPL, switch to `tools=[builtin.calculate]` explicitly. `ALL_TOOLS` and `make_tools()` are unchanged (opt-in only via `python_sandbox=True`).

### Output utilities

- **New** `aimu.parse_json_response(text, schema=None)` — extract JSON from any LLM response string using three extraction strategies (raw parse, fenced code block, `{…}` substring). Pass a dataclass class or Pydantic v2 `BaseModel` as `schema` to coerce the parsed dict into a typed object. Raises `ValueError` on all-strategy failure with the first 200 characters of the response included. Exported from `aimu.models._json`, `aimu.models`, and top-level `aimu`.
- **New** `aimu.generate_json(client, prompt, schema=None, *, retries=2, generate_kwargs=None)` — call `client.generate()` and parse the result as JSON, retrying up to `retries` times on parse failure. Convenience wrapper around `parse_json_response`.
- **New** `aimu.extract_tool_calls(messages)` — convert an OpenAI-format message list (e.g. `agent.model_client.messages`) into a flat `list[dict]` of `{iteration, tool, arguments, result}` records. Handles both `arguments` and `parameters` key names for cross-model compatibility. Replaces manual reconstruction boilerplate common in agentic scripts.

### Model weight caching

- **New** All four in-process HuggingFace clients (`HuggingFaceClient`, `HuggingFaceImageClient`, `HuggingFaceAudioClient`, `HuggingFaceSpeechClient`) now maintain a module-level weight registry keyed on `(spec.id, *sorted_model_kwargs)`. A second client instance with the same model and construction kwargs reuses already-loaded weights rather than calling `from_pretrained()` again. The text client checks on construction; the lazy-loading modality clients check on first load. `LlamaCppClient` has the same pattern with key `(model_path, n_ctx, n_gpu_layers, chat_format)`.
- **New** `aimu.clear_hf_cache(model=None)` — evict HuggingFace weight entries from all four modality registries and call `gc.collect()` + `cuda.empty_cache()`. Pass a model enum member to clear just that model; pass `None` to clear all.
- **New** `aimu.clear_llamacpp_cache(model=None)` — same for `LlamaCppClient`.

### Tools

- **New** `execute_python(code)` built-in tool in `builtin.compute`. Executes sandboxed Python in a fresh namespace per call, captures stdout, and returns the last expression value. Allowed imports: `math`, `statistics`, `json`, `re`, `itertools`, `functools`, `datetime`, and `numpy`/`pandas`/`scipy`/`matplotlib` when installed. Filesystem (`open`, `os`, `pathlib`) and subprocess access are blocked. **Not included in `ALL_TOOLS`** — opt in via `tools=builtin.compute` or `make_tools(python_sandbox=True)`.
- **New** `make_tools(..., python_sandbox=False)` — new `python_sandbox=` kwarg appends `execute_python` when `True`.
- **New** `make_memory_tools(store)` in `aimu.tools.builtin` — wraps any `MemoryStore` instance as three `@tool`-decorated functions (`store_memory`, `search_memories`, `list_memories`) for direct in-process agent use. Unlike the image/audio/speech built-in tools, there is no lazy singleton: the store is always explicit because persistence semantics (`persist_path`, backend, collection name) are meaningful caller choices. Works with `SemanticMemoryStore`, `DocumentStore`, or any `MemoryStore` subclass. For cross-process or multi-agent memory, the existing FastMCP servers (`aimu.memory.mcp` / `aimu.memory.document_mcp`) remain the recommended path.
- **New** `builtin.make_tools(..., memory_store=None)` — new `memory_store=` kwarg appends `make_memory_tools(store)` to the assembled tool list when provided.

### Agents and workflows

- **New** `Agent.restore(messages)` — restore an agent from a saved `list[dict]` (OpenAI message format) for resuming after failure. Calls `model_client.reset()`, strips the leading system message to prevent duplication on the next `chat()`, and sets `model_client.messages`. The live partial state after a failed run is on `agent.model_client.messages` (not the post-run snapshot from `agent.messages`).
- **New** `EvaluatorOptimizer.restore(messages)` — delegates to `generator.restore()`.
- **New** `Chain.restore(messages, step=0)` — restores the specified step's agent client.

### Documentation

- **New** `docs/how-to/using-llms-inside-tools.md` — covers the history pollution problem, `generate()` for stateless in-tool LLM calls, the HuggingFace weight caching model (including `clear_hf_cache()` / `clear_llamacpp_cache()`), and the save/restore checkpointing pattern with a full try/except example.

## v0.5.1 (2026-06-01) — Image-to-image, FLUX.2 Klein, and curated model catalog

### Image generation

- **New** Image-to-image (img2img) support: pass `reference_image=` to `BaseImageClient.generate()` (and all subclasses). Accepts a file path string, `pathlib.Path`, raw bytes, data URL, http(s) URL, or PIL Image. HuggingFace derives the img2img pipeline from the loaded txt2img pipeline via `from_pipe()` — shared weights, no extra VRAM. `strength=` (default `0.75`) controls deviation from the reference for FLUX.1-style pipelines. `width`/`height` are ignored; output size is derived from the reference image. Gemini passes the reference as inline PNG data in a multipart request, enabling image editing.
- **New** `HuggingFaceImageModel.FLUX_2_KLEIN_4B` and `FLUX_2_KLEIN_9B` — FLUX.2 Klein by Black Forest Labs. 4-step distilled model with improved text rendering, better hand/face quality, and higher resolution support. Uses `Flux2KleinPipeline` (diffusers 0.37+), a unified pipeline that handles both txt2img and img2img natively (`image=` parameter, no `strength`). `img2img_uses_strength=False` on the spec distinguishes it from FLUX.1-style img2img.
- **New** `HuggingFaceImageSpec.img2img_pipeline_class` — diffusers class name for the img2img variant (e.g. `"StableDiffusionImg2ImgPipeline"`); `None` for ad-hoc `"hf:<repo>"` strings.
- **New** `HuggingFaceImageSpec.img2img_uses_strength` — `True` (default) for strength-based pipelines; `False` for unified pipelines like FLUX.2 Klein that condition on the reference image directly.
- **New** `aimu.models._images._reference_image_to_pil()` — shared helper used by both HF and Gemini image clients to normalise any reference image input form to a PIL Image.
- **Changed** `scripts/hotdog_loop.py` absorbs `hotdog_climbing.py`: the two scripts shared identical structure and differed only in their acceptance policy. Pass `--strategy climbing` for hill-climbing behaviour (keep best, revert on non-improvement); `--strategy greedy` (default) preserves the original loop behaviour. `hotdog_climbing.py` is removed.
- **New** `scripts/hotdog_img2img.py` — iterative hotdog refinement via img2img + strength annealing. Hill-climbs in image space (always refines from the best image, not the most recent) while annealing `strength` from high (explore) to low (polish). Detects and warns when the active model does not support `strength` (e.g. FLUX.2 Klein).

### Negative prompts

- **New** `ImageSpec.supports_negative_prompt` capability flag. `True` by default; `False` for guidance-distilled / conversational models that have no negative-prompt parameter — `HuggingFaceImageModel.FLUX_2_KLEIN_4B`/`_9B` and the entire Gemini image family (`GeminiImageSpec` defaults it to `False`).
- **Behavior** `BaseImageClient.generate()` now raises `ValueError` if `negative_prompt=` is passed to a model whose spec sets `supports_negative_prompt=False`, instead of crashing deep in the pipeline (HuggingFace) or silently ignoring it (Gemini). Callers branch on `spec.supports_negative_prompt` and fold avoidance into the prose prompt for unsupporting models. The hotdog scripts do this via a new `negative_prompt_plan()` helper (native kwarg → summarizer-folded positive constraints → prompt suffix, by model).

### Curated model catalog (breaking for unknown ids)

- **Breaking** Model id strings must name a model AIMU ships a spec for. Passing an arbitrary `"hf:<unknown-repo>"` / `"gemini:<unknown-id>"` / `"openai:<unknown-id>"` to an image, audio, or speech client now raises `ValueError` (listing available ids) instead of fabricating a spec with guessed capabilities. Text was always strict (`resolve_model_string` raises); this brings the other modalities in line. For a one-off custom model, construct the provider spec and pass the object (e.g. `ImageClient(HuggingFaceImageSpec(...))`) — the explicit escape hatch.
- **Fixed** A `"provider:model_id"` string for a *known* model now resolves to the **same spec object** as the equivalent enum member, so capabilities are identical regardless of construction path. Previously the string form fabricated a default spec — e.g. `"hf:black-forest-labs/FLUX.2-klein-4B"` lost `supports_negative_prompt=False`/`img2img_uses_strength=False`, and `"hf:suno/bark"` lost BARK's `default_voice`.
- **Removed** The `_REPO_PIPELINE_HINTS` repo-prefix capability-guessing heuristics in the HuggingFace audio and speech clients (dead once unknown ids raise).

## v0.5.0 (2026-05-31) — Async, audio, speech, and default models

A feature release on top of the v0.4 redesign: a full async surface, two new output modalities (audio and speech), a cloud image provider, automatic default-model resolution, and streaming tools. No breaking changes to the v0.4 sync API.

### Async surface (`aimu.aio`)

- **New** `aimu.aio` mirrors the entire public sync API one-for-one — same class names, different namespace. Switch paradigms with one import line plus `await`. Exports `chat`, `client`, `Agent`, `SkillAgent`, `Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`, `PlanExecuteEvaluator`, `OrchestratorAgent`, `MCPClient`. Imported by default, so `from aimu import aio` needs no separate install.
- **New** `aio.Parallel` and `concurrent_tool_calls=True` use `asyncio.TaskGroup` for structured concurrency — sibling cancellation on first failure, `ExceptionGroup` aggregation.
- **New** Native async providers: Anthropic, OpenAI, Gemini, Ollama, and every OpenAI-compatible endpoint. In-process providers (HuggingFace, LlamaCpp) wrap an existing sync client so weights load only once (`aio.client(sync_client)`).
- **New** async `MCPClient` built on FastMCP's native async `Client` (no anyio portal); construct via `await MCPClient.connect(...)`. The sync `MCPClient` remains first-class.
- **New** `@tool` async detection (`__tool_is_async__`): `async def` tools are awaited directly; sync CPU-bound tools are routed through `asyncio.to_thread` so the event loop stays free.
- **Note** Streaming on the async surface returns `AsyncIterator[StreamChunk]` (consume with `async for`); the sync surface returns `Iterator[StreamChunk]`. The `StreamChunk` type itself is identical on both.
- **Requirement** Python 3.11+ is now required (the async surface uses `asyncio.TaskGroup`, `asyncio.timeout`, and native `ExceptionGroup`).

### Audio generation

- **New** `aimu.audio_client()` / `aimu.generate_audio()` + `AudioClient` factory + `BaseAudioClient` ABC, parallel to the text and image surfaces.
- **New** HuggingFace audio models: MusicGen small/medium/large (32 kHz, token-autoregressive), AudioLDM2 (16 kHz, diffusion), Stable Audio Open (44.1 kHz stereo, diffusion).
- **New** `AUDIO_GENERATING` `StreamChunk` phase + `StreamChunk.is_audio_progress()`. Streaming progress for diffusers-backed models.
- **New** `encode_audio()` output formats: `numpy` (default), `bytes`, `data_url`, `path` (WAV via `soundfile`).
- **New** Built-in `generate_audio` streaming tool + `make_audio_tool(client, duration_s=)`; `builtin.audio` subgroup.

### Speech (text-to-speech)

- **New** `aimu.speech_client()` / `aimu.generate_speech()` + `SpeechClient` factory + `BaseSpeechClient` ABC.
- **New** Providers: HuggingFace local (SpeechT5, MMS-TTS, BARK) and OpenAI cloud (`tts-1`, `tts-1-hd`).
- **New** `SPEECH_GENERATING` `StreamChunk` phase + `StreamChunk.is_speech_progress()`; OpenAI byte-chunk streaming.
- **New** Built-in `generate_speech` streaming tool + `make_speech_tool(client, voice=, speed=)`; `builtin.speech` subgroup.

### Image generation

- **New** Google Gemini "Nano Banana" cloud provider (`GeminiImageClient`, `gemini-2.5-flash-image`) under the `[google]` extra, dispatched via `aimu.image_client("gemini:...")`.
- **New** `aimu.image_client()` accepts ad-hoc `"hf:<repo_id>"` and `"gemini:<id>"` strings in addition to enum members.
- **New** Streaming image generation: `IMAGE_GENERATING` chunks during denoising, with optional per-step latent previews via `preview_every=N` (HuggingFace diffusers).
- **New** Built-in `generate_image` streaming tool, `make_image_tool(client, preview_every=)`, and `make_describe_image_tool(client)` (binds vision Q&A to a vision-capable chat client); `builtin.image` subgroup.

### Default-model resolution

- **New** `model=` is now optional on `aimu.chat()` / `aimu.client()` / `aimu.agent()`. When omitted, AIMU resolves a text default: `AIMU_LANGUAGE_MODEL` (`"provider:model_id"`) first, otherwise an *already-available* local model (running Ollama → cached HuggingFace model → running local OpenAI-compatible server), restricted to enum-known ids and preferring tool-capable ones. A cloud provider is never auto-selected and weights are never downloaded implicitly.
- **New** `AIMU_IMAGE_MODEL` / `AIMU_AUDIO_MODEL` / `AIMU_SPEECH_MODEL` provide defaults for the image/audio/speech entry points (env-var only; an unset var raises a clear `ValueError`).

### Tools and vision

- **New** Streaming tools: a generator-function `@tool` may `yield` `StreamChunk` objects mid-execution (flag `__tool_is_streaming__`); the agent forwards them through `agent.run(stream=True)`. The tool's recorded response resolves from its `return` value, the last chunk's `result`, or `str(last_chunk.content)`.
- **New** `images=` is now accepted on stateless `generate()` (one-shot vision Q&A that does not touch `self.messages`), in addition to stateful `chat()`.
- **New** `builtin.make_tools(base_client, image_client=None, audio_client=None, speech_client=None)` assembles the full built-in tool list with automatic image/vision/audio/speech wiring.

### Examples

- **Changed** The full-featured Streamlit chatbot (`web/streamlit_chatbot.py`) gains image, audio, and speech generation, plus optional TTS narration of completed responses.

## v0.4 (2026-05-26) — API redesign

Breaking changes across four areas, plus the new documentation site.

### Top-level API

- **New** `aimu.chat(user_message, *, model, ...)` — one-shot chat with a model string or enum.
- **New** `aimu.client(model, *, system=None, **kwargs)` — one-line `ModelClient` factory.
- **New** `aimu.resolve_model_string("provider:model_id")` — model-string parser.
- **New** `ModelClient` now accepts a `"provider:model_id"` string in addition to enum members.

### Model clients

- **New** `ModelSpec` frozen dataclass replaces positional enum tuples. All `Model` enums migrated.
- **New** `client.reset(system_message="__keep__")` clears history and unlocks the system-message setter.
- **Breaking** `system_message` is immutable after the first `chat()` call. The setter raises `RuntimeError`; call `reset()` to unlock.
- **New** `include=[...]` stream filter on `chat()` and `generate()` selects phases (`"thinking"`, `"tool_calling"`, `"generating"`, `"done"`).
- **Internal** Abstract methods renamed `chat → _chat`, `generate → _generate`. Concrete `chat`/`generate` on the base class apply the `include` filter and delegate.
- **New** Memory-aware GPU placement for `HuggingFaceImageClient`: on load it measures the pipeline size and each GPU's *free* VRAM (accounting for other processes), then pins to the freest GPU or falls back to model / sequential CPU offload so large models (SD3, FLUX) load without OOM. Override with `model_kwargs={"device": "cuda:1"}` or `{"device_map": ...}`. Audio/speech clients take the same `{"device": ...}` hint. Shared `aimu/models/_hf_device.py` helpers back all three.
- **New** `ImageSpec.max_prompt_tokens` records the model's text-encoder prompt budget (77 for CLIP, 256/512 for T5 models like SD3/FLUX, `None` for uncapped cloud models), exposed on `BaseImageClient`. Use it to size prompts to the model.
- **Changed** `HuggingFaceImageClient` now defaults `torch_dtype` per device (bf16 on CUDA, fp16 on MPS, fp32 on CPU) instead of `"auto"`, which could silently load in fp32 and double VRAM. Pass `model_kwargs={"torch_dtype": ...}` to override.

### Agents

- **Breaking** `Agent` constructor signature changed: `Agent(model_client, system_message=None, name=None, tools=None, ...)`. `system_message` is the second positional argument; `name` is optional (auto-derived).
- **Breaking** `AgenticModelClient` removed from the public API. Use `agent.as_model_client()` instead.
- **Breaking** `OrchestratorAgent._setup_orchestrator` renamed to `_init_orchestrator`.
- **New** `OrchestratorAgent.assemble(client, system_message, workers=[...])` factory builds an orchestrator without subclassing.
- **New** Workflow factories: `Chain.from_client(client, prompts)`, `Router.from_client(client, classifier_prompt, handlers)`, `Parallel.from_client(client, worker_prompts, aggregator_prompt=)`, `PlanExecuteEvaluator.from_client(client, ...)`.
- **Breaking** `BaseAgent` and `Workflow` ABCs removed. All concrete agents and workflows inherit directly from `Runner`. The agent-vs-workflow split survives as a conceptual category in the docs.
- **Breaking** `AgentChunk` and `ChainChunk` collapsed into `StreamChunk` — no back-compat aliases. `chunk.agent_name → chunk.agent`; `chunk.step → chunk.iteration`.

### Tools

- **New** `@tool` raises `ToolSignatureError` at decoration time on unsupported signatures (`*args`/`**kwargs`, params with no type hint and no default).
- **New** `Optional[T]` and `T | None` unwrap to the inner type in tool specs.
- **New** Built-in tool subgroups: `builtin.web`, `builtin.fs`, `builtin.compute`, `builtin.misc`.
- **New** `MCPClient` raises `MCPConnectionError` (rather than silently failing) on construction or call failure. Added `.ping()` method.

### Skills

- **Breaking** `SkillManager` raises `SkillLoadError` on malformed `SKILL.md` (instead of silently skipping).
- **Breaking** `SkillManager.get_skill_body()` raises `SkillNotFoundError` on unknown skill name (instead of returning a sentinel string).
- **New** Skill catalogue prompt includes script-derived tool names inline.
- **Breaking** `Skill` renamed to `AgentSkill` (no back-compat alias).
- **New** Skills logged at `INFO` on discovery.

### Documentation

- New documentation site built with MkDocs Material and hosted on GitHub Pages.
- Diátaxis structure: tutorials, how-to guides, reference, explanation.
- README slimmed to landing-page size.

## Earlier versions

This is the first formal changelog entry. Prior versions tracked changes via git history; consult `git log` on GitHub for v0.3.x and earlier.
