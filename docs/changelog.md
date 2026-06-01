# Changelog

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
