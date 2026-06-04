# Changelog

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
