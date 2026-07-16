# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMU (AI Modeling Utilities) is a Python library for building AI-powered applications, with language models as the primary building block. It provides a unified, provider-agnostic interface across modalities -- text generation, image generation, and audio generation -- spanning local backends (Ollama, HuggingFace, llama-cpp-python, any OpenAI-compatible server) and cloud providers (OpenAI, Anthropic, Google Gemini). MCP tool integration and typed streaming output (thinking, tool calling, generation phases) are built into the base model client, not layered on top.

AIMU implements the taxonomy from Anthropic's *[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)*. Concretely, that means **one** `Runner` ABC (no type-level split between agents and workflows) and the following concrete classes, all exported from `aimu.agents` and composable via the shared interface:

- **Autonomous** (the LLM directs flow):
  - `Agent`: the augmented-LLM tool-calling loop.
  - `SkillAgent`: `Agent` + filesystem-discovered skill injection.
  - `OrchestratorAgent`: Anthropic's "orchestrator-workers" pattern, expressed as an autonomous orchestrator whose tools are other agents. Prebuilt examples in `aimu.agents.prebuilt` (`ResearchReportAgent`, `CodeReviewAgent`, `ContentCreationAgent`).
- **Code-controlled workflows** (Python directs flow):
  - `Chain`: prompt chaining (step N's output → step N+1's input).
  - `Router`: routing (classifier dispatches to a specialist handler).
  - `Parallel`: parallelization (workers run concurrently, optional aggregator).
  - `EvaluatorOptimizer`: generate → critique → revise loop.
  - `PlanExecuteEvaluator`: AIMU's extension beyond Anthropic's original five: plan → execute with tools → score → replan on failure.

The agent-vs-workflow distinction is documentation-only; both inherit directly from `Runner`, so they compose freely (a `Router` can dispatch to an `Agent`; a `Chain` step can be an `OrchestratorAgent`). `agent.as_model_client()` adds one more composition path: any agent wraps as a `BaseModelClient`, so it slots anywhere a plain client is accepted (e.g. into a `Benchmark` or a `Chain` that consumes clients). `Runner.as_tool()` adds the inverse: any agent or workflow wraps as a `@tool`-style callable, so an autonomous `Agent` can call a *workflow* (or a remote agent) as a tool, and `OrchestratorAgent.assemble(workers=...)` accepts any `Runner`, not just `Agent`. A hill-climbing prompt tuner optimizes prompts against labelled data without ML machinery.

AIMU also ships an optional **async surface** under `aimu.aio` that mirrors the entire public sync API one-for-one (same class names, different namespace). Sync is the default; async is strictly opt-in. The async surface uses modern `asyncio.TaskGroup` for structured concurrency in `Parallel` and `concurrent_tool_calls`. See [docs/explanation/async-design.md](docs/explanation/async-design.md) for the seven design decisions behind this split.

For **cross-process / cross-vendor** composition, an optional **A2A (Agent2Agent) interop** layer under `aimu.agents.a2a` (extra: `aimu[a2a]`) exposes any `Runner` as an HTTP server (`serve_a2a`) and consumes a remote agent as a local `Runner` (`RemoteAgent`). It mirrors the MCP-for-tools pattern at the agent level and is adapter-shaped (A2A types never leak into `Runner`/`Agent` core). Because `RemoteAgent` is a `Runner`, a remote agent composes (via `as_tool()`, as a workflow step, or an `assemble()` worker) exactly like a local one. See [docs/explanation/a2a-vs-mcp.md](docs/explanation/a2a-vs-mcp.md).

For **always-on personal assistants** (OpenClaw / Hermes Agent style), three async-first primitives supply the pieces beyond an LLM call: a `Channel` transport ABC + `CLIChannel` (`aimu.aio.channels`), a `Scheduler` for proactive triggers (`aimu.aio.Scheduler`), and runtime skill authoring (`aimu.skills.write_skill` / `make_skill_authoring_tool` + `SkillManager.refresh()`). These are library primitives only; the assembled single-user daemon lives in `examples/personal-assistant/`. See the "Personal-Assistant Primitives" section below.

## Design Principles

These six principles drive every architectural decision in AIMU. When proposing changes, check the change against each one. If it violates a principle, it likely belongs in a wrapper above AIMU rather than in the library itself. Full rationale lives in [docs/explanation/design-principles.md](docs/explanation/design-principles.md).

1. **Plain Python.** Classes, functions, decorators, dataclasses, type hints. Every file is readable top-to-bottom. No framework metalanguage; no `Runnable` protocol, no LCEL, no graph DSL, no dependency-injection container.
2. **Plain data.** Conversation state is a `list[dict]` in OpenAI message format. No `Message` / `AIMessage` / `ChatMessage` classes. Provider-specific formats (Anthropic blocks, Ollama image fields, HF PIL images) adapt at request time and never leak into `self.messages`.
3. **Composability through uniform interfaces.** `BaseModelClient` for every provider, `Runner` for every agent and workflow, `MemoryStore` for every memory backend, `StreamChunk` for every streaming source. `agent.as_model_client()` exists specifically so agents are substitutable with plain clients.
4. **Progressive disclosure.** `aimu.chat()` → `aimu.client()` → `Agent` → workflows → custom `BaseModelClient` subclass. Each layer optional; the top wraps the next. New entry points should fit somewhere on this ladder, not parallel to it.
5. **Direct paths for common tasks.** Common operations have one obvious, ergonomic entry point: `aimu.chat()`, `Chain.from_client(...)`, `Agent(client, "system msg", tools=[...])`, `include=["generating"]`, `builtin.web`. The library does not offer parallel, equally-recommended ways to do the same job. If a second path exists, it's a power-user escape hatch (e.g. `Agent.from_config()`), not a documented alternative.
6. **Failures are apparent.** Errors raise at the layer where the cause is actionable, with messages that name the problem. `ToolSignatureError` at decoration time. `ToolArgumentError` at tool dispatch (surfaced to the model as a tool result). `SkillLoadError` on discovery. `MCPConnectionError` on construction. Silent fallbacks are bugs; chained exceptions preserve the original cause via `raise ... from exc`.

When reviewing a proposed change, ask which principle it serves and which (if any) it violates. The small public surface is itself a feature; keep it small.

## Development Commands

### Python version

AIMU requires **Python 3.11+**. The async surface (`aimu.aio`) uses `asyncio.TaskGroup`, `asyncio.timeout`, and native `ExceptionGroup`, all of which require 3.11. Ruff target is `py311`; tests use `pytest-asyncio` in `asyncio_mode = "auto"` with session-scoped event loops.

### Installation

Install all dependencies for development:
```bash
pip install -e '.[all,dev,notebooks]'
```

Or using uv (faster):
```bash
uv sync --all-extras
```

Extras fall into two kinds (plus development tooling). **Provider backends** (a model/inference provider):
```bash
pip install -e '.[ollama]'         # Ollama only
pip install -e '.[hf]'             # HuggingFace transformers (text) + diffusers (image) + audio/speech + embeddings
pip install -e '.[anthropic]'      # Anthropic Claude models
pip install -e '.[openai_compat]'  # OpenAI-compatible servers + cloud (OpenAI, Gemini, LM Studio, vLLM, etc.)
pip install -e '.[llamacpp]'       # Local GGUF models via llama-cpp-python (no external service)
pip install -e '.[google]'         # Google Gemini image generation (Nano Banana, google-genai SDK)
```
**Capabilities** (cross-provider features):
```bash
pip install -e '.[web]'            # Streamlit/Gradio chat apps + personal-assistant web UI
pip install -e '.[tuning]'         # Prompt tuners + evals Benchmark harness (pandas, tqdm)
pip install -e '.[evals]'          # DeepEval metric adapters (renamed from the old `deepeval` extra)
pip install -e '.[a2a]'            # Agent-to-Agent protocol interop
```
`[all]` is every provider backend + every capability (development extras `dev` / `notebooks` / `docs` excluded).

**`tuning` extra (pandas/tqdm).** The prompt-tuning subsystem (`aimu.prompts.tuner` + the `PromptTuner` subclasses) and the evals `Benchmark` harness import `pandas`/`tqdm`, which live in the optional `[tuning]` extra (included in `[all]`). `aimu.prompts` imports the tuner classes **lazily** (a module-level `__getattr__`), so `import aimu` and `from aimu.prompts import PromptCatalog` / `Scorer` / `LLMJudgeScorer` work without the extra; touching a tuner class (`from aimu.prompts import JudgedPromptTuner`) triggers the import and raises `ModuleNotFoundError` only if `[tuning]` isn't installed. `aimu.evals.Benchmark` is not lazy-wrapped, so `import aimu.evals` requires the extra.

### Testing

Run all tests:
```bash
pytest
```

Run tests for a specific client:
```bash
pytest tests/test_models.py --client=ollama
pytest tests/test_models.py --client=hf
pytest tests/test_models.py --client=openai
pytest tests/test_models.py --client=anthropic
pytest tests/test_models.py --client=gemini
pytest tests/test_models.py --client=lmstudio_openai
pytest tests/test_models.py --client=ollama_openai
pytest tests/test_models.py --client=hf_openai
pytest tests/test_models.py --client=vllm
pytest tests/test_models.py --client=llamaserver_openai
pytest tests/test_models.py --client=sglang_openai
pytest tests/test_models.py --client=llamacpp --model-path=/path/to/model.gguf
```

Run tests for a specific model:
```bash
pytest tests/test_models.py --client=ollama --model=GEMMA_4_E4B
pytest tests/test_models.py --client=hf --model=GPT_OSS_20B
```

Run specific test modules:
```bash
pytest tests/test_history.py
pytest tests/test_tools.py
pytest tests/test_prompt_catalog.py
pytest tests/test_memory.py
pytest tests/test_models_api.py  # mock-only unit tests of the model surface
```

Run live image-generation tests (parametrized like `test_models.py`):
```bash
pytest tests/test_images.py --image-client=hf --image-model=SD_1_5     # one HF model
pytest tests/test_images.py --image-client=gemini                       # all Gemini models
pytest tests/test_images.py --image-client=all                          # every installed provider
```

Run live audio-generation tests (opt-in; require HuggingFace model weights):
```bash
pytest tests/test_audio.py --audio-client=hf --audio-model=MUSICGEN_SMALL   # one model (fastest)
pytest tests/test_audio.py --audio-client=hf --audio-model=AUDIOLDM2        # diffusion model
pytest tests/test_audio.py --audio-client=hf                                # all HF audio models
```

Run async surface tests:
```bash
# Mock-only async tests (no backend required)
pytest tests/test_aio_models_api.py tests/test_aio_workflow_parallel.py tests/test_aio_agents.py tests/test_aio_tools.py

# Live async tests against Ollama (parallels tests/test_models.py)
pytest tests/test_aio_models.py --client=ollama --model=QWEN_3_8B
pytest tests/test_aio_models.py --client=anthropic --model=CLAUDE_SONNET_4_6
```

### Linting

The project uses Ruff for linting and formatting (configured in pyproject.toml with line length 120):
```bash
ruff check .
ruff format .
```

### Running the Chat UI

The `examples/web/` directory ships two Streamlit apps and a Gradio app (install the UI stack with `pip install aimu[web]`; `streamlit`/`gradio` are an optional extra, not core deps):

- **`streamlit_chatbot_basic.py`**: ~70-line showcase. Model/provider selector, streaming chat with `include=["generating"]`, built-in tools running silently. Start here to see how little code a working AIMU chatbot needs.
- **`streamlit_chatbot.py`**: Full-featured version. Adds image generation, agentic mode, thinking display, tool-call expanders, generation parameter sliders, and conversation persistence. Intended as an extensible foundation.

```bash
streamlit run examples/web/streamlit_chatbot_basic.py   # showcase
streamlit run examples/web/streamlit_chatbot.py         # full-featured
python examples/web/gradio_chatbot_basic.py             # Gradio variant
```

The **personal-assistant** example additionally ships a WebSocket front end -- `examples/personal-assistant/web_assistant.py` (a Starlette + `uvicorn` server) with an example-local `WebChannel` (a `Channel` adapter over a browser WebSocket) and a static page. It streams replies and pushes proactive scheduler messages to the browser, demonstrating that the `Channel` ABC accepts a network adapter with no change to the `Assistant` loop. Run: `python examples/personal-assistant/web_assistant.py --model ollama:qwen3:8b` then open `http://127.0.0.1:8000`. Also under `aimu[web]`.

## Architecture

### Top-Level API

- **[aimu/__init__.py](aimu/__init__.py)** exposes the small public surface most users start from:
  - `aimu.chat(user_message, *, model=None, system=None, generate_kwargs=None, stream=False, images=None, include=None)`: one-shot chat: builds a fresh client, sends one message, returns the response (or a `StreamChunk` iterator when `stream=True`).
  - `aimu.client(model=None, *, system=None, **provider_kwargs)`: one-line constructor that returns a fully configured `ModelClient`. Use this for multi-turn chats.
  - **Default model resolution (when `model` is omitted)**: `client()` / `chat()` / `agent()` resolve a text default via `aimu.models._internal.model_defaults.resolve_default_text_model()`: (1) the `AIMU_LANGUAGE_MODEL` env var (a `"provider:model_id"` string, read after `load_dotenv()`); else (2) an *already-available* local model (a running Ollama server (`ollama.list()`), a cached HuggingFace model (`huggingface_hub.scan_cache_dir()`, download-free), or a running local OpenAI-compat server probed at its default base_url (`models.list()`)), each restricted to ids matching a provider `Model` enum so capabilities are known, preferring tool-capable, logged at WARNING; else (3) a `ValueError` naming the remedies. A cloud provider is **never** auto-selected and weights are **never** downloaded implicitly. The async `aio.client()` reuses the same resolver with `include_hf_cache=False` (an `hf:` default would need an explicit sync-client wrap). The `ModelClient` factory itself still requires an explicit model; the default magic lives only at the ergonomic `aimu.*` layer.
  - `aimu.resolve_model_string(s)`: parses a `"provider:model_id"` string and returns the matching `Model` enum member. Raises `ValueError` with the list of valid ids on miss.
  - `aimu.resolve_model_enum(model)` / `aimu.resolve_image_model_enum(model)`: broaden the `resolve_*_model_string` lookups to also accept a `Model`/`ImageModel` enum member (returned unchanged) **and a bare enum-member name** (e.g. `"QWEN_3_8B"`, `"FLUX_2_KLEIN_4B"`, `"NANO_BANANA"`) searched across every installed provider enum. A `"provider:model_id"` string delegates to the string resolver. These are the resolvers scripts use to accept "enum member, bare name, or string" uniformly across text and image. **Ambiguous bare text names** (the same id ships under many providers) are resolved by `resolve_model_enum` the same way the omitted-`model` default is: prefer a provider where the model is *actually available locally* (running Ollama → cached HF → reachable local server, tool-capable first), logged at WARNING; if it's not available under any provider, `ValueError` lists the `"provider:model_id"` options. This availability tiebreaker runs **only on the ambiguous path** (enum / string / unambiguous-name inputs do no I/O). `resolve_image_model_enum` has no local-availability notion (and image catalogs don't collide), so it raises on the rare ambiguity.
  - `aimu.available_text_models(*, include_hf_cache=True)`: discovery: returns *locally available* text models as `Model` enum members (running Ollama → cached HuggingFace → reachable local OpenAI-compat servers), in provider-priority order. Download-free and cloud-free (same probes as the default resolver). `aimu.resolve_default_text_model_enum(*, include_hf_cache=True)` returns the single auto-pick (env var → first available, tool-capable preferred) as an enum member, the enum-returning twin of the internal `resolve_default_text_model()` (which returns the `"provider:model_id"` string used by `client()`).
  - `aimu.available_image_models()` / `available_audio_models()` / `available_speech_models()` / `available_transcription_models()` / `available_embedding_models()`: the modality analogs of `available_text_models()`. They **surface** locally available models (as provider enum members) but, unlike text, are **never auto-selected**; the modality entry points still require an explicit `model=` or env var (see the modality default note below). Discovery is download-free and cloud-free: image/audio/speech/transcription probe the HuggingFace cache only (no local server exists for these); embedding probes a running Ollama server then the HuggingFace cache (Ollama serves embedders, like text). Defined in `aimu.models._internal.model_defaults`, re-exported from `aimu.models` and top-level `aimu`.
  - `aimu.parse_json_response(text, schema=None)`: parse JSON from an LLM response string using three extraction strategies (raw parse, fenced block, `{…}` substring). Pass a dataclass class or Pydantic v2 `BaseModel` as `schema` to coerce into a typed object. Raises `ValueError` on failure.
  - `aimu.generate_json(client, prompt, schema=None, *, retries=2, generate_kwargs=None)`: call `client.generate()` and parse the result as JSON, retrying on parse failure. Convenience wrapper around `parse_json_response`.
  - `aimu.extract_tool_calls(messages)`: extract tool call/result pairs from an OpenAI-format message list (e.g. `agent.model_client.messages`). Returns `list[dict]` with keys `iteration`, `tool`, `arguments`, `result`. Replaces the manual reconstruction boilerplate common in agentic scripts.
  - `aimu.clear_hf_cache(model=None)`: release cached HuggingFace model weights across all modalities (text, image, audio, speech) and free GPU memory. Pass a model enum member to clear just that model; pass `None` to clear all. See HuggingFace client sections for caching details.
  - `aimu.clear_llamacpp_cache(model=None)`: same for LlamaCpp.
  - Re-exports: `BaseModelClient`, `Model`, `ModelClient`, `ModelSpec`, `StreamChunk`, `StreamingContentType`.
  - `aimu.aio`: async submodule. Imported by default (one-line `from . import aio`) so users can `from aimu import aio` without a separate install. See "Async Surface" below.

Model-string format: `"provider:model_id"`. Provider keys (when their optional dep is installed): `ollama`, `hf`, `anthropic`, `openai`, `gemini`, `lmstudio`, `ollama-openai`, `hf-openai`, `vllm`, `llamaserver`, `sglang`, `llamacpp`. The id is matched against the provider's `Model` enum; colons inside the id (e.g. `qwen3.5:9b`) are preserved.

### Async Surface (`aimu.aio`)

The sync API has an opt-in async twin under `aimu.aio` with the same class names. Users switch paradigms with one import line + `await` keywords. The sync ladder is unchanged.

```python
# Sync (default)
import aimu
reply = aimu.chat("Hi", model="anthropic:claude-sonnet-4-6")

# Async (opt-in)
from aimu import aio
async def main():
    client = aio.client("anthropic:claude-sonnet-4-6")
    agent = aio.Agent(client, tools=[my_async_tool])
    reply = await agent.run("Hello")
```

**Layout** (mirrors `aimu.{models,agents,tools}` one-for-one):

- **[aimu/aio/__init__.py](aimu/aio/__init__.py)**: exports `chat`, `client`, `AsyncModelClient`, `Agent`, `SkillAgent`, `Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`, `PlanExecuteEvaluator`, `OrchestratorAgent`, `MCPClient`, `AsyncRunner`, and (when the `a2a` extra is installed, guarded by `HAS_A2A`) `RemoteAgent`, `serve_a2a`, `build_a2a_app`, `A2AConnectionError`.
- **[aimu/aio/_base.py](aimu/aio/_base.py)**: `AsyncBaseModelClient` (ABC). Concrete `async def chat()`/`generate()` apply the `include` filter and route the `schema=` path; abstract `_chat()`/`_generate()` are coroutines. Tool *execution* lives in the async tool-loop engine (`aimu.aio._tool_loop`), not here; `chat()` is one model turn. Concurrent tool dispatch (`concurrent_tool_calls=True`) uses `asyncio.TaskGroup` in the engine.
- **[aimu/aio/_model_client.py](aimu/aio/_model_client.py)**: `AsyncModelClient` factory + top-level `client()`/`chat()`. Refuses direct construction with `HuggingFaceModel`/`LlamaCppModel` (see "In-process providers" below).
- **[aimu/aio/_mcp_client.py](aimu/aio/_mcp_client.py)**: async `MCPClient` (~30 LOC). Uses FastMCP's native async `Client` directly (no anyio portal). Construct via the `connect()` classmethod factory: `mcp = await MCPClient.connect(server=...)`.
- **[aimu/aio/providers/](aimu/aio/providers/)**: async provider clients:
  - `anthropic.py`: `AsyncAnthropicClient` using `anthropic.AsyncAnthropic`. Reuses sync class's pure format adapters via composition.
  - `openai_compat.py`: `AsyncOpenAICompatClient` using `openai.AsyncOpenAI`, plus `Async*Client` subclasses for OpenAI, Gemini, LM Studio, Ollama-OpenAI, HF-OpenAI, vLLM, llama-server, SGLang.
  - `ollama.py`: `AsyncOllamaClient` using `ollama.AsyncClient`.
  - `hf.py`: `AsyncHuggingFaceClient` wraps a sync `HuggingFaceClient`; see "In-process providers".
  - `llamacpp.py`: `AsyncLlamaCppClient` wraps a sync `LlamaCppClient`; see "In-process providers".
- **[aimu/aio/agent.py](aimu/aio/agent.py)**: `AsyncRunner` ABC + async `Agent`. Same loop semantics as sync; streaming returns `AsyncIterator[StreamChunk]`.
- **[aimu/aio/skill_agent.py](aimu/aio/skill_agent.py)**: async `SkillAgent`. `_setup_skills_async()` constructs an `aio.MCPClient` for skill discovery and caches `await mcp.as_tools()` on `_skills_tools`; the skill tools reach the model through the overridden `_effective_tools` (base tools + skill tools, deduped), which the tool-loop engine re-reads each round. `run()` prepares state and runs async skill setup once (before building the engine), then delegates to `Agent`'s post-prepare loop helpers `_run_loop(loop, ...)` / `_run_loop_streamed(loop, ...)` (which take a pre-built engine, shared with the base `Agent`) rather than re-implementing the loop. It therefore mirrors `aio.Agent.run()` in full: `deps=` (per-run `ToolContext` injection), `tool_approval=`, `schema=` (structured-output short-circuit; with `stream=True` it streams the structured turn via `_structured_stream_after_setup`, done after skill setup so the skills aren't wiped), and the `final_answer_prompt` forced-wrap-up on both the streamed and non-streamed paths.
- **[aimu/aio/agentic_client.py](aimu/aio/agentic_client.py)**: internal `_AsyncAgenticView` for `Agent.as_model_client()`.
- **[aimu/aio/orchestrator_agent.py](aimu/aio/orchestrator_agent.py)**: async `OrchestratorAgent` + `assemble()`. Worker dispatch wrappers are `async def`; tool dispatch awaits them under `asyncio.TaskGroup` when `concurrent_tool_calls=True`.
- **[aimu/aio/workflows/](aimu/aio/workflows/)**: async workflow patterns:
  - `chain.py`: async `Chain`, sequential `await` loop.
  - `router.py`: async `Router`.
  - `parallel.py`: async `Parallel`. **The headline change**: `asyncio.TaskGroup` replaces `ThreadPoolExecutor`. Structured concurrency: sibling cancellation on first failure, `ExceptionGroup` aggregation.
  - `evaluator.py`: async `EvaluatorOptimizer`.
  - `plan_execute_evaluator.py`: async `PlanExecuteEvaluator`.
- **[aimu/aio/scheduler.py](aimu/aio/scheduler.py)** + **[aimu/aio/channels/](aimu/aio/channels/)**: personal-assistant primitives (`Scheduler`, `Channel`/`CLIChannel`). See "Personal-Assistant Primitives" below.
- **[aimu/aio/run_handle.py](aimu/aio/run_handle.py)**: `RunHandle` (exported from `aimu.aio`), a thin wrapper over the `asyncio.Task` of an in-flight run (`RunHandle.start(coro)` = `asyncio.ensure_future`; `cancel()`, `async result()`, `done`/`cancelled`, `__await__`). Cooperative cancellation rides on asyncio task cancellation (no threaded token), so the run's `await chat(...)` points are the cancel boundaries; `CancelledError` is a `BaseException`, so the `except Exception` dispatch handlers never swallow it. The async `Agent` loop helpers (`_run_loop`, `_run_loop_streamed`, and the `schema=` paths in `Agent`/`SkillAgent.run`) snapshot `_last_messages` in a **`finally`**, so a cancelled run still records its partial turn (resume via `restore()` off the live `model_client.messages`). Async-only (sync has no cancellation primitive). The personal-assistant example wires `/stop` on top of this (runs each turn as a `RunHandle`). How-to: [docs/how-to/cancel-a-run.md](docs/how-to/cancel-a-run.md).

**Shared infrastructure** (used by both sync and async surfaces, no duplication):

- **[aimu/models/_internal/chat_state.py](aimu/models/_internal/chat_state.py)**: `_ChatStateMixin` provides `system_message` lifecycle (always-live setter that swaps the in-history system entry), `reset()`, `_append_user_turn()`, `_collect_python_tool_specs()`, and the capability properties (`is_thinking_model`, etc.). Both `BaseModelClient` and `AsyncBaseModelClient` inherit it. No state mechanics are duplicated.
- **[aimu/models/_internal/streaming.py](aimu/models/_internal/streaming.py)**: `resolve_include()`, `filter_chunks()` (sync), `afilter_chunks()` (async).
- **[aimu/models/_internal/json.py](aimu/models/_internal/json.py)**: `parse_json_response(text, schema=None)`, `generate_json(client, prompt, schema=None, *, retries=2, generate_kwargs=None)`, `extract_tool_calls(messages)`. Utilities for getting structured data out of model responses. Exported from `aimu.models` and top-level `aimu`.
- **[aimu/tools/mcp_format.py](aimu/tools/mcp_format.py)**: `mcp_tools_to_openai(tool_specs)` (spec conversion) and `mcp_content_to_text(tool_response)` (flatten a `call_tool` result to a string), shared by sync `MCPClient` and async `aio.MCPClient` (`get_tools()` / `as_tools()`).
- Provider format adapters (`_openai_messages_to_anthropic`, `_openai_tools_to_anthropic`, `_adapt_messages_for_ollama`, `_split_thinking`, `_ThinkingParser`, `_build_user_content_blocks`) are pure data transforms, reused by async providers via composition, no duplication.

**`@tool` decorator and async tools**:

- `@tool` records `func.__tool_is_async__ = inspect.iscoroutinefunction(func)` at decoration time.
- The sync tool-loop engine (`_ToolLoop._call_plain_tool`) raises `ValueError` if it encounters an async tool, with a message pointing the caller at `aimu.aio`.
- The async engine (`_AsyncToolLoop`) awaits async tools directly; sync (CPU-bound) tools are routed through `asyncio.to_thread` so the event loop stays free.
- `concurrent_tool_calls=True` on the async surface uses `asyncio.TaskGroup` instead of `ThreadPoolExecutor` (both in the engine).

**Streaming type asymmetry**:

- `client.chat(stream=True)` (sync) returns `Iterator[StreamChunk]`: consume with `for`.
- `await client.chat(stream=True)` (async) returns `AsyncIterator[StreamChunk]`: consume with `async for`.
- The `StreamChunk` named tuple itself is identical on both surfaces. They cannot be unified without a hidden event loop (rejected; see async-design.md).

**In-process providers: wrap an existing sync client (Decision 7)**:

`AsyncHuggingFaceClient` and `AsyncLlamaCppClient` do *not* load model weights independently. They take an existing sync client in their constructor and route through `asyncio.to_thread`. Constructing directly with a model enum raises a clear error:

```python
sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)   # loads weights once
async_client = aio.client(sync_client)                  # wraps; shares state
# aio.client(HuggingFaceModel.LLAMA_70B)  # ValueError pointing at the pattern above
```

State (`messages`, `system_message`, `tools`) is shared with the wrapped sync client; there is conceptually one client, and the async version just exposes an awaitable interface. For in-process providers, "async" buys event-loop integration (handler doesn't block on inference) but *not* coroutine-level concurrency (the GIL and CUDA stream serialize execution).

**MCP integration**:

- The sync `aimu.tools.MCPClient` (anyio portal wrapper) is **kept first-class**; no deprecation. It exists so sync users can use MCP tools without writing async code.
- The async `aimu.aio.MCPClient` is a thin parallel using FastMCP's `Client` directly. Same construction signature (`config=`/`server=`/`file=`/`url=`, plus `auth=`/`headers=` for `url=`); use `await MCPClient.connect(...)` to construct + connect.
- Both `MCPClient.get_tools()` implementations delegate to `aimu.tools.mcp_format.mcp_tools_to_openai()`; `as_tools()` (sync + async) builds callables over those specs and shares `mcp_content_to_text()`.

**Design decisions reference**: [docs/explanation/async-design.md](docs/explanation/async-design.md) records the seven decisions (API shape, scope, runtime, MCP, streaming asymmetry, `@tool` detection, in-process wrapping) with rationale and rejected alternatives. Mirror future async-related architectural decisions there.

### Model Client Architecture

The codebase uses an abstract base class pattern for model clients:

- **[aimu/models/base.py](aimu/models/base.py)**: Defines `BaseModelClient` abstract base class with:
  - `chat(user_message=None, generate_kwargs, use_tools=True, stream=False, images=None, include=None, tools=None, audio=None, schema=None)`: **one model turn** against the persistent message history; returns `str`, `Iterator[StreamChunk]`, or (when `schema=` is set) a validated dataclass/Pydantic instance. A single call issues **one** model request; if the model requests tools, they are **parsed and stored** on the assistant message (`content` + `tool_calls`) and the call returns — the client does **not** execute them (the tool-loop engine in `Agent` does; see "Tool Integration"). `user_message=None` (the continuation primitive the engine uses) runs a turn on the current messages **without appending a new user turn** — nothing is added to `self.messages` before the model call. Bare `chat("q", tools=[...])` on a tool-using turn therefore returns the tool turn (often empty prose) and leaves the tool *unexecuted*; use `Agent`/`agent.as_model_client()` for a full tool-using answer. Concrete clients implement `_chat()`; the public `chat()` applies the `include` stream filter and the `schema=` structured-output path (see "Structured Output").
  - `generate(prompt, generate_kwargs, stream=False, images=None, include=None, audio=None, schema=None)`: Single-turn stateless generation. Concrete clients implement `_generate()`; the public `generate()` is provided by the base. Accepts `images=` for one-shot vision or `audio=` for one-shot audio input (same forms as `chat`) **without** touching `self.messages`, and `schema=` for structured output. The base validates the respective capability flag (raises `ValueError` for non-vision / non-audio models). Passing both `images=` and `audio=` raises `ValueError`.
  - `images=` (vision-capable models only) accepts file paths, `pathlib.Path`, raw `bytes`, http(s) URLs, or `data:image/...` URLs. Available on both `chat()` (stateful) and `generate()` (stateless).
  - `audio=` (audio-capable models only) accepts file paths, `pathlib.Path`, raw `bytes`, `https://` URLs (fetched eagerly, since `input_audio` has no remote-URL field), or `data:audio/...;base64,...` URLs. Stored in `self.messages` as OpenAI `input_audio` content blocks. Available on both `chat()` and `generate()`. Mutually exclusive with `images=` per turn.
  - `include=` (streaming only): optional iterable of `StreamingContentType` values (or their string equivalents `"thinking"`, `"tool_calling"`, `"generating"`, `"done"`) selecting which phases to yield. Defaults to all phases.
  - `tools=` (per-call override): the tool set to **advertise** for this one call. `None` (default) uses the transient `self.tools` (usually `[]`); any other value (including `[]` to advertise none) replaces it for that single call and is restored afterward. Implemented as a scoped swap of `self.tools` via `_ChatStateMixin._tools_override` (covers request-spec building, which reads `self.tools`); the streaming path wraps stream *consumption* so the override stays live across lazy iteration. Not safe across concurrent `chat()` calls on a shared client; same contract as `self.messages`. `ModelClient`/`AsyncModelClient` need no special handling: `tools` is a delegating property, so the swap propagates to the inner client. The tool-loop engine passes the effective tools via `tools=` on **every** call in the loop, so this is how the `Agent` supplies its tools; there is no persistent registry on the client.
  - `reset(system_message="__keep__")`: clears the conversation history. Default keeps the existing system message; pass `None` to clear it or a new string to replace it.
  - `_chat_setup()`: Prepares messages and tools before a chat call; normalizes `images=` into OpenAI-format `image_url` content blocks, or `audio=` into OpenAI-format `input_audio` content blocks.
  - `_record_tool_calls(tool_calls, content="")`: parse-and-store helper. When the model requests tools, the concrete `_chat` calls this to append the assistant message carrying `tool_calls` (+ any prose/thinking) to `self.messages`. The client **does not execute** tools — that is the tool-loop engine's job (see "Tool Integration"). `chat()` is a single model turn.
  - `is_thinking_model` / `is_tool_using_model` / `is_vision_model` / `is_audio_model`: Read-only capability properties derived from `model.supports_thinking` / `model.supports_tools` / `model.supports_vision` / `model.supports_audio`.
  - `system_message` property: the setter is always live. Assigning it mid-conversation rewrites the system entry in `self.messages` in place (re-conditioning the model on the new prompt while preserving history); before the first `chat()` it just seeds the value. See "Message History Management" below for the re-condition semantics and the shared-client caveat.
  - Message history stored in `self.messages` (OpenAI-style format; user messages with images are content-block lists).
  - `self.tools` is an internal **per-call transient** (default `[]`) that the `tools=` override swaps for the duration of one `chat()` call; it is not a persistent public registry. Tool callables (both `@tool` functions and `MCPClient(...).as_tools()` wrappers) are supplied **per call** via `chat(tools=...)` — the `Agent` and its tool-loop engine own the durable tool list. There is no separate `mcp_client` attribute (one uniform tool interface).
  - Tool *execution*, iteration, approval, `ToolContext(deps)` injection, and `concurrent_tool_calls` live in the tool-loop engine composed by `Agent`, **not** on the model client (see "Tool Integration").

- **Supporting types in base.py**:
  - `StreamingContentType(str, Enum)`: `THINKING`, `TOOL_CALLING`, `GENERATING`, `DONE`; string values (`"thinking"`, etc.).
  - `StreamChunk(NamedTuple)`: `(phase, content, agent=None, iteration=0)`. The single chunk type used everywhere: `client.chat(stream=True)`, `Agent.run(stream=True)`, `image_client.generate(stream=True)`, streaming tools, all workflow runs. `content` shape depends on phase:
    - `str` for `THINKING` / `GENERATING` (token).
    - `dict {"name", "arguments", "response"}` for `TOOL_CALLING` (``arguments`` is the dict the model passed to the tool).
    - `dict {"step", "total_steps", "image", "final", "result"}` for `IMAGE_GENERATING`, emitted by image clients during denoising (HF diffusers per step; Gemini coarse start/done). ``image`` is an optional ``PIL.Image`` (None unless ``preview_every`` opted in); ``final=True`` marks the terminal chunk per image; ``result`` carries the encoded output (path / bytes / data-url per ``format=``) on the final chunk.

    Helpers `is_text()` / `is_tool_call()` / `is_image_progress()` dispatch on phase.
  - `ModelSpec`: frozen dataclass with `id: str`, `tools: bool`, `thinking: bool`, `vision: bool`, `audio: bool`, `structured_output: bool`, `generation_kwargs: dict | None`. Equality and hash use `id` only, so it can hold a dict and still be used as an enum value. Each `Model` enum member's value is a `ModelSpec`.
  - `Model(Enum)`: base enum. Each member's value is a `ModelSpec`. Members expose `.value` (the id string), `.spec` (the `ModelSpec`), `.supports_tools`, `.supports_thinking`, `.supports_vision`, `.supports_audio`, `.supports_structured_output`, `.generation_kwargs`.

- **[aimu/models/model_client.py](aimu/models/model_client.py)**: `ModelClient(BaseModelClient)` factory/wrapper class:
  - Single public construction path. Accepts a `Model` enum member or a `"provider:model_id"` string.
  - Detects provider from the model enum type and instantiates the corresponding concrete client.
  - Delegates `_chat`/`_generate` to the inner client; the public `chat`/`generate` from the base apply the `include` filter.
  - Provider-specific kwargs (e.g. `model_path`, `base_url`, `model_keep_alive_seconds`) are passed through to the concrete constructor.
  - Usage: `ModelClient(OllamaModel.QWEN_3_8B)`, `ModelClient("anthropic:claude-sonnet-4-6")`, `ModelClient(LlamaCppModel.QWEN_3_8B, model_path="/path/to/model.gguf")`.
  - Exports a helper `resolve_model_string(s)` that performs the string lookup independently, plus `resolve_model_enum(model)` (enum / string / bare-name) and the discovery functions `available_text_models()` / `resolve_default_text_model_enum()` (re-exported here from `_internal.model_defaults`, so `aimu.models` / `aimu` are their public home).

- **Model Implementations** (all concrete clients live under [aimu/models/providers/](aimu/models/providers/); see "Provider layout" below):
  - [aimu/models/providers/ollama.py](aimu/models/providers/ollama.py): `OllamaClient` for local Ollama models (native API)
  - [aimu/models/providers/hf/text.py](aimu/models/providers/hf/text.py): `HuggingFaceClient` for local Transformers models
  - [aimu/models/providers/anthropic.py](aimu/models/providers/anthropic.py): `AnthropicClient` for Anthropic Claude models (native `anthropic` SDK)
  - [aimu/models/providers/openai_compat.py](aimu/models/providers/openai_compat.py): the generic OpenAI-compatible base plus the **local-server** subclasses (require `openai_compat` extra):
    - `OpenAICompatClient`: generic base for any OpenAI-compatible server
    - `LMStudioOpenAIClient`: [LM Studio](https://lmstudio.ai/) (default: `localhost:1234`)
    - `OllamaOpenAIClient`: Ollama OpenAI-compat endpoint (default: `localhost:11434`)
    - `HFOpenAIClient`: HuggingFace Transformers Serve (default: `localhost:8000`)
    - `VLLMOpenAIClient`: vLLM (default: `localhost:8000`)
    - `LlamaServerOpenAIClient`: llama.cpp llama-server (default: `localhost:8080`)
    - `SGLangOpenAIClient`: SGLang (default: `localhost:30000`)
  - [aimu/models/providers/openai/text.py](aimu/models/providers/openai/text.py): `OpenAIClient`: [OpenAI](https://platform.openai.com/) cloud API (GPT-4o, GPT-4.1, o-series); reads `OPENAI_API_KEY`. Subclasses `OpenAICompatClient`.
  - [aimu/models/providers/gemini/text.py](aimu/models/providers/gemini/text.py): `GeminiClient`: [Google Gemini](https://ai.google.dev/) via Google's OpenAI-compat endpoint; reads `GOOGLE_API_KEY`. Subclasses `OpenAICompatClient`.
  - [aimu/models/providers/llamacpp.py](aimu/models/providers/llamacpp.py): `LlamaCppClient` for local GGUF models via llama-cpp-python (requires `llamacpp` extra):
    - Loads GGUF files in-process; no external service required
    - Constructor takes `model_path` (GGUF file path) plus `n_ctx`, `n_gpu_layers`, `chat_format`, `verbose`
    - GPU offloading via `n_gpu_layers=-1` (all layers); `n_gpu_layers=0` for CPU-only
    - Thinking model support via `<think>...</think>` tag parsing (same as OpenAI-compat clients)

- **Model Definitions**: Each client defines a `Model` enum (e.g., `OllamaModel`) where each member's value is a `ModelSpec(id, tools=..., thinking=..., vision=..., generation_kwargs=...)`. Provider-specific extras (e.g. HuggingFace `ToolCallFormat`, `think_opener_in_prompt`) become additional positional elements in the enum member tuple after the `ModelSpec`. Every concrete client exposes the derived classproperties:
  - `TOOL_MODELS`: members where `supports_tools=True`
  - `THINKING_MODELS`: members where `supports_thinking=True`
  - `VISION_MODELS`: members where `supports_vision=True`
  - `AUDIO_MODELS`: members where `supports_audio=True`

- **HuggingFace model weight caching**: All four in-process HuggingFace clients (`HuggingFaceClient`, `HuggingFaceImageClient`, `HuggingFaceAudioClient`, `HuggingFaceSpeechClient`) maintain a module-level `_model_registry` dict keyed on `(spec.id, *sorted_model_kwargs)`. A second client instance with the same model and kwargs reuses already-loaded weights rather than calling `from_pretrained()` again. For the eager-loading text client, the cache is checked in `__init__`; for the lazy-loading modality clients, it is checked in their respective `_load_pipeline()` / `_ensure_loaded()` methods. Use `aimu.clear_hf_cache(model=None)` to evict entries and free VRAM. `LlamaCppClient` has the same pattern with cache key `(model_path, n_ctx, n_gpu_layers, chat_format)` and `aimu.clear_llamacpp_cache()`. **Note**: creating clients with different `model_kwargs` (e.g. different `device_map`) produces separate cache entries and loads weights independently.

- **`aimu/models/_internal/json.py`**: utilities for processing model output: `parse_json_response(text, schema=None)`, `generate_json(client, prompt, schema=None, *, retries=2, generate_kwargs=None)`, `extract_tool_calls(messages)`. All three are exported from `aimu.models` and re-exported from top-level `aimu`. See Top-Level API section for details.

- **Optional Dependencies**: [aimu/models/__init__.py](aimu/models/__init__.py) gracefully handles missing dependencies; clients only import if their dependencies are installed (flags: `HAS_OLLAMA`, `HAS_HF`, `HAS_ANTHROPIC`, `HAS_OPENAI_COMPAT`, `HAS_LLAMACPP`). `OpenAIClient` and `GeminiClient` are part of `openai_compat` since they use the same `openai` SDK. `model_client.py` performs the same guarded imports independently to avoid a circular import with `__init__.py`. The `aimu/evals/` package uses the same guarded-import pattern with `HAS_DEEPEVAL`.

### Tool Integration

AIMU supports two tool registration routes that can be combined on the same client:

- **`@tool` decorator** ([aimu/tools/decorator.py](aimu/tools/decorator.py)): mark any plain Python function as a tool. The decorator is defined in `aimu.tools` and **re-exported at the top level as `aimu.tool`**: `@aimu.tool` is the single recommended/documented form (it's namespaced, so it can't be silently shadowed by another library's `tool` decorator); `from aimu.tools import tool` remains valid for code already inside `aimu.tools`. Both names are the same object. The decorator inspects the signature and docstring at decoration time and attaches an OpenAI-format spec at `func.__tool_spec__`. Hand decorated functions to an agent via `Agent(client, tools=[fn1, fn2])` (or set `client.tools = [fn1, fn2]` directly). Type-to-JSON mapping covers `str`/`int`/`float`/`bool`/`list`/`dict` plus subscripted generics (`list[str]`, `dict[str, int]`) and `Optional[T]` / `T | None`; a `Literal[...]` parameter becomes a JSON Schema `enum` advertising the exact allowed values. The docstring is parsed Google-style: the prose before the first section header (`Args:`, `Returns:`, ...) becomes the tool description, and an `Args:`/`Arguments:`/`Parameters:` section supplies per-parameter `description`s (`name: text` entries, continuation lines allowed); required vs. optional args are derived from default values. (This replaced an earlier first-paragraph-only description, which silently dropped guidance placed in later paragraphs and left `dict`/`Literal` params unstructured.) All tools (in-process `@tool` and MCP `as_tools()` callables) live in `self.tools`; dispatch is one by-name lookup and a name collision resolves to the last entry in the list.

  **Signature validation (raises `ToolSignatureError` at decoration time):**
  - `*args` / `**kwargs` (variadic parameters): declare each argument explicitly
  - parameter with no type hint *and* no default value
  - Missing return annotation is a warning, not an error.

  **Argument validation (model-supplied call args, raises `ToolArgumentError` at dispatch)**: the decorator also stores `func.__tool_param_adapters__` (a Pydantic `TypeAdapter` per annotated, model-facing parameter, built once at decoration time), `func.__tool_allowed_args__` (model-facing param names), and `func.__tool_required__` (the spec's `required` list). At dispatch the shared `_collect`-path helper `coerce_tool_arguments(fn, arguments)` (in `decorator.py`, exported from `aimu.tools`) validates and **lax-coerces** the model's arguments against the declared hints (`"5"` → `5`, `"true"` → `True`); uncoercible values, missing-required args, and unknown args raise `ToolArgumentError`. The full annotation is kept (not unwrapped) so `Optional[T]`/`T | None` still accepts `None`. Callables without `__tool_param_adapters__` (MCP `as_tools()` wrappers) pass through unchanged (the MCP server validates). `pydantic>=2` is a declared core dependency for this. See "Tool-argument validation" below.

  **Async detection**: the decorator also sets `func.__tool_is_async__ = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)`: true for plain `async def` tools and for `async def + yield` (async generator) tools.

  **Streaming tools**: the decorator sets `func.__tool_is_streaming__ = inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func)`. Generator tools may yield zero-or-more `StreamChunk` objects during execution; the agent forwards each yielded chunk through `agent.run(stream=True)` so callers see progress live. The tool's *response* (what goes into the tool message in the conversation history) is resolved in priority order:
  1. The generator's `return` value (captured via `StopIteration.value`); sync generators only.
  2. The last yielded chunk's `content["result"]` if it's a dict with that key, the convention used by `IMAGE_GENERATING` final chunks.
  3. `str(last_chunk.content)` as a last resort.

  Streaming tools are dispatched by the tool-loop engine's `_dispatch_streamed` (`_ToolLoop` / `_AsyncToolLoop`); the non-streaming `_dispatch` → `_call_plain_tool` path rejects them with a clear `ValueError` pointing at `stream=True`. Concurrent dispatch (`concurrent_tool_calls=True`) falls back to sequential when any tool in the batch is streaming, to avoid interleaving chunks from concurrent generators.

- **MCPClient** ([aimu/tools/client.py](aimu/tools/client.py)): wraps FastMCP 2.0 for cross-process tool servers.
  - Converts async FastMCP API to synchronous interface using an internal event loop
  - Constructor requires *exactly one* of `config=`, `server=`, `file=`, or `url=` (a remote HTTP/SSE server). `auth=` (a bearer-token string, the literal `"oauth"`, or a configured FastMCP `OAuth` / `httpx.Auth` **provider object**) and `headers=` apply only with `url=`. A string/absent `auth` is folded into a single-server `mcpServers` config so FastMCP infers SSE vs streamable-HTTP and applies them in one path; a **non-string** provider object is passed straight to `fastmcp.Client(url, auth=obj)` (the bare `url` is the transport) so callers can supply persistent OAuth token storage / a custom `redirect_handler`, and it cannot be combined with `headers=`. `_build_transport` (shared by sync + async in [aimu/tools/client.py](aimu/tools/client.py)) returns `(transport, client_auth)`; both `MCPClient`s call `Client(transport, auth=client_auth)`. Wrong count, `auth`/`headers` without `url`, an object `auth` with `headers`, or connection failure raises `MCPConnectionError` with the original exception chained.
  - `ping()`: verifies the connection is alive (calls `list_tools()`); raises `MCPConnectionError` if dead.
  - `get_tools()`: Returns tools in OpenAI function calling format
  - `as_tools()`: Returns the server's tools as `@tool`-style callables (each closes over the client, calls `call_tool()` cross-process, returns the result's text via `mcp_content_to_text()`, and carries `__tool_spec__` + `__tool_is_async__=False` + `__tool_is_streaming__=False`). This is the integration path: `Agent(client, tools=mcp.as_tools())` (or `tools=builtin.web + mcp.as_tools()`). Snapshot; call again to refresh, and keep the `MCPClient` (or the callables, which hold a ref) alive for the connection.
  - `call_tool()`: Synchronous tool execution; wraps backend errors in `MCPConnectionError`
  - `list_tools()`: Returns available tool names
  - **No `model_client.mcp_client` attribute**: MCP tools integrate by becoming `@tool`-style callables (via `as_tools()`) that you hand to an `Agent` alongside `@tool` functions; the engine dispatches both through the same by-name lookup. One uniform tool interface.

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)**: built-in general-purpose tools, each decorated with `@tool` for direct in-process use. Pass an entire subgroup with `tools=builtin.web + [my_tool]`:
  - **`builtin.web`**: `get_weather`, `get_webpage`, `get_webpage_html`, `search`, `wikipedia`
  - **`builtin.fs`**: `list_directory`, `read_file`
  - **`builtin.compute`**: `calculate`, `execute_python` (sandboxed Python REPL; see below)
  - **`builtin.misc`**: `echo`, `get_current_date_and_time`
  - **`builtin.ALL_TOOLS`**: flat list of every built-in (kept for back-compat)
  - **`make_memory_tools(store)`**: factory that returns `[store_memory, search_memories, list_memories]` as `@tool`-decorated functions closing over the provided `MemoryStore` instance. No lazy singleton; the store is always explicit because `persist_path` and backend are meaningful choices. Pass the result directly to `Agent(client, tools=make_memory_tools(store))` or via `make_tools(..., memory_store=store)`. For cross-process or multi-agent memory, use `aimu.memory.mcp` / `aimu.memory.document_mcp` instead.
  - **`make_document_tools(store)`**: factory parallel to `make_memory_tools` that exposes a `DocumentStore`'s richer path API as `[save_document, read_document, list_documents, search_documents]` `@tool`s (wrapping `write`/`read`/`list_paths`/`search_full_text`; `read_document` returns a "not found" message instead of raising on a missing path). The names are deliberately distinct from the `make_memory_tools` triad so one agent can carry **both** a `SemanticMemoryStore` (short facts) and a `DocumentStore` (longer documents) at once. `make_memory_tools`, `make_document_tools`, and `make_retrieval_tool` are re-exported from `aimu.aio.tools.builtin` (the async agent dispatches these sync tools via `asyncio.to_thread`).
  - **`make_subagent_tool(model, *, system_message=..., tools=None, agent_types=None, max_depth=1, max_iterations=10, concurrent_tool_calls=True, deps=None, tool_name="spawn_subagent")`**: factory returning a `spawn_subagent` `@tool` for **dynamic sub-agent spawning** — the LLM delegates an independent subtask to a *fresh* `Agent` with its own isolated context at runtime (the Claude-Code `Task` pattern; AIMU's *dynamic complement* to `OrchestratorAgent`'s static roster). Each invocation builds a fresh `ModelClient(model)` (isolated history, the `make_workers` idiom) and returns `agent.run(task)`. Two shapes chosen by `agent_types`: generic `spawn_subagent(task)`, or typed `spawn_subagent(agent_type, task)` over a registry (`name -> {"system_message", "tools"?, "model"?}`) whose names are folded into the tool description; an unknown `agent_type` is returned to the model as a tool result (self-correction), while programmer errors (`max_depth < 1`, empty/malformed `agent_types`) raise `ValueError` at factory-call time. **Parallelism is free**: with the *parent* agent's `concurrent_tool_calls=True`, multiple spawns in one turn run concurrently via the engine's existing `ThreadPoolExecutor` (sync) / `asyncio.TaskGroup` (async). `max_depth` (default 1, counting the caller as level 1) bounds recursion by only re-injecting a decremented spawn tool when `> 1`. Non-streaming by design (a streaming tool would disable the concurrent dispatch path and interleave concurrent sub-agents). It is a plain factory tool — **not** a `Runner` subclass. Async twin `make_async_subagent_tool` in [aimu/aio/tools/builtin.py](aimu/aio/tools/builtin.py) produces an `async def` tool building `aio.Agent`s; in-process providers (HuggingFace/LlamaCpp) are wrapped per spawn via a fresh sync client (aio can't build them from an enum; the weight cache prevents reload). Tests: `tests/test_subagent_tools.py`, `tests/test_aio_subagent_tools.py`. How-to: [docs/how-to/spawn-subagents.md](docs/how-to/spawn-subagents.md); demo: `examples/news-summarizer --method spawn`.

  - **`make_web_tools(*, session=None, timeout=15, max_content_chars=20000, user_agent=...)`**: factory returning `[find_forms, submit_form]` as `@tool`s closing over a shared `requests.Session`, so cookies persist across calls (a GET-then-POST form flow works: `find_forms` scrapes hidden fields incl. CSRF tokens, `submit_form` echoes them back with the session's cookies). `find_forms(url)` parses every `<form>` (stdlib `html.parser` via `_FormExtractor`; **no new dep**) into a text listing of index / resolved-absolute action / method / fields (name, type, value, including `type=hidden`). `submit_form(url, method="POST", data=None)` covers both verbs (`GET` → query params, `POST` → form body) and returns status + final URL + truncated body; `method="GET"` doubles as the session-aware raw fetch for pages behind a login. Pass alongside the stateless `get_webpage_html`: `Agent(client, tools=[get_webpage_html, *make_web_tools()])`. **Server-rendered HTML only** (no JavaScript execution): JS-rendered SPAs and anti-bot-protected pages are out of scope; a headless-browser backend is a deferred future addition. `submit_form` performs writes (POST) — gate it via the `tool_approval` hook when confirmation is wanted. Re-exported from `aimu.aio.tools.builtin` (dispatched via `asyncio.to_thread`). Tests: `tests/test_web_tools.py`.

  Tool reference:
  - `echo(echo_string)`: Returns input string
  - `get_current_date_and_time()`: Returns ISO format datetime
  - `get_weather(location)`: Current weather via Open-Meteo API (city name or coordinates)
  - `calculate(expression)`: Safe arithmetic expression evaluator
  - `execute_python(code)`: Sandboxed Python REPL. Executes `code` in a fresh `exec()` namespace per call; captures stdout and returns the last expression value. Allowed imports: `math`, `statistics`, `json`, `re`, `itertools`, `functools`, `datetime`, and `numpy`/`pandas`/`scipy`/`matplotlib` when installed. Filesystem (`open`, `os`, `pathlib`) and subprocess access are blocked. **Not in `ALL_TOOLS` by default**; opt in via `tools=builtin.compute` or `make_tools(..., python_sandbox=True)`.
  - `get_webpage(url)`: Fetches page and returns visible text with HTML stripped
  - `get_webpage_html(url)`: Fetches page and returns raw HTML markup (truncated); stateless, server-rendered HTML only. For form inspection/submission with cookies, use `make_web_tools()`
  - `search(query, num_results)`: Web search via SearXNG (`SEARXNG_BASE_URL` env var)
  - `wikipedia(query)`: Wikipedia article summary
  - `list_directory(path)`: Lists files and subdirectories
  - `read_file(path, max_lines)`: Reads local file contents

- **[aimu/tools/mcp.py](aimu/tools/mcp.py)**: thin FastMCP server that registers `builtin.ALL_TOOLS` for cross-process use. Run standalone: `python -m aimu.tools.mcp`. Single source of truth; the same callables back both routes.

- **The tool-loop engine** ([aimu/agents/_tool_loop.py](aimu/agents/_tool_loop.py) `_ToolLoop` + module-level `last_turn_called_tools`; async twin [aimu/aio/_tool_loop.py](aimu/aio/_tool_loop.py) `_AsyncToolLoop`): the internal middle layer that owns iterative tool calling. It is **not** public API — `Agent` composes one per run (`Agent._make_tool_loop(...)`), and the public ladder stays `chat()` → `Agent` → workflows. Constructor: `_ToolLoop(model_client, tools, *, deps=None, tool_approval=None, concurrent_tool_calls=False, max_rounds=10, final_answer_prompt=None)`; `tools` may be a list **or a zero-arg callable** returning one (re-read each round, so `SkillAgent` can surface a mid-run authored skill tool). Methods: `run(user_message=None, ...)` / `run_streamed(...)` drive the model↔tools loop (`max_rounds` caps total model turns); `_dispatch` / `_dispatch_streamed` execute the pending tool calls stored on the last assistant turn; `_call_plain_tool`, `_tool_call_kwargs` (arg coercion + `ToolContext(deps)` injection), and `_tool_call_approved` do the per-call work.

- **Tool Calling Flow**:
  1. The `Agent` builds a `_ToolLoop` for the run with the effective tools + policy and calls `loop.run(task)` (or `run_streamed`).
  2. The engine calls `model_client.chat(task, tools=...)` — one model turn. If the model requests tools, the provider parses them and stores them on the assistant message via `_record_tool_calls` (no execution).
  3. `last_turn_called_tools(messages)` sees the pending tool calls, so the engine's `_dispatch` runs them: one by-name lookup `{fn.__name__: fn for fn in tools}` per call; a match has its arguments validated/coerced (`_tool_call_kwargs` → `coerce_tool_arguments`) then is invoked in-process (a `@tool` directly, an `as_tools()` wrapper cross-process behind the same calling convention); a miss appends a "not found" tool message. Name collision resolves to the **last** entry in the tools list. Invalid arguments raise `ToolArgumentError`, caught at dispatch and appended as a tool message so the model can self-correct (distinct from a tool that runs and raises).
  4. Tool results are appended to `model_client.messages` with role "tool".
  5. The engine calls `model_client.chat()` again (no new user turn) to let the model react to the results, looping until a turn makes no tool calls (or `max_rounds`), then the optional `final_answer_prompt` wrap-up.

- **Tool-argument validation**: model-supplied tool-call arguments are validated and lax-coerced against each `@tool` function's type hints **before** invocation, on every dispatch path. The single integration point is the engine's `_tool_call_kwargs` (in `_ToolLoop` / `_AsyncToolLoop`), so one call to `coerce_tool_arguments(fn, arguments)` covers sync + async + streaming + concurrent dispatch. On failure it raises `ToolArgumentError` (in [aimu/tools/decorator.py](aimu/tools/decorator.py), exported from `aimu.tools`); each dispatch site (`_call_plain_tool` and the streaming branch of `_dispatch_streamed`, sync in `aimu/agents/_tool_loop.py` and async in `aimu/aio/_tool_loop.py`) has an `except ToolArgumentError` clause ahead of the generic handler that surfaces `str(exc)` as the tool message (no "raised an error" prefix). Coercion uses Pydantic `TypeAdapter.validate_python` (lax), so a stringified scalar is coerced but a stringified nested object (e.g. `'{"a":1}'` for a `dict` param) is rejected rather than parsed. Tests: `tests/test_tool_decorator.py` (sync), `tests/test_aio_tools.py` (async, incl. `concurrent_tool_calls=True`).

- **Tool-call approval (the `tool_approval` gate)**: an optional, additive hook run immediately before each tool invocation; default approves everything (no behavior change). `ToolApproval = Callable[[str, dict], bool | Awaitable[bool]]` + the default `approve_all` live in [aimu/tools/approval.py](aimu/tools/approval.py), exported from `aimu.tools` and re-exported from top-level `aimu`. The policy is configured on the `Agent` (`Agent(tool_approval=...)`) or per run (`Agent.run(tool_approval=...)`, mirroring `deps=`, on sync + async `Agent`/`SkillAgent`); the `Agent` passes it to the tool-loop engine, which enforces it (it is **not** stored on the model client). Gated at two dispatch points, `_call_plain_tool` (covers non-streaming + the concurrent fast path) and the streaming-generator branch of `_dispatch_streamed`; on deny the engine appends `"Tool '<name>' was not approved."` (same shape as the "not found" message) so the model can react. The async helper `_tool_call_approved` awaits a coroutine policy; the sync one raises `ValueError` if handed a coroutine (sync needs a sync policy). The policy sees the tool name + raw model arguments, so a host gates only risky tools. The personal-assistant example gates `add_skill_script` by default (its `CONFIRM_BEFORE` set + a terminal y/n policy). Tests: `tests/test_tool_approval.py`, `tests/test_aio_tool_approval.py`. How-to: [docs/how-to/gate-tool-calls.md](docs/how-to/gate-tool-calls.md).

### Conversation Management

- **[aimu/history.py](aimu/history.py)**: `ConversationManager` class
  - Uses TinyDB for JSON-based conversation storage
  - Tracks message history with timestamps
  - `create_new_conversation()`: Returns `(doc_id, messages list)`
  - `update_conversation(messages)`: Persists updated messages with timestamps
  - Integrates with ModelClient via `model_client.messages`

### Sessions (multi-user)

- **[aimu/sessions/](aimu/sessions/)**: per-conversation state keyed by `channel:sender`, so one process serves many users/chats. `ConversationManager` holds a single conversation; `SessionStore` generalizes it to keyed sessions. Sync (matches the `MemoryStore` / `ConversationManager` family); the async assistant loop calls it directly, as it already does `ConversationManager`.
  - `Session` dataclass: `key`, `messages: list[dict]` (OpenAI format, the same plain data the agent uses), `memory_namespace: str | None` (a scope a caller can pass to a `MemoryStore`), `metadata: dict`.
  - `SessionStore(ABC)`: `get(key) -> Session` (a detached snapshot; a fresh empty `Session` if none stored), `save(session)` (create/replace by `key`), `list_keys()`, `close()`.
  - `InMemorySessionStore` (non-durable dict; the trivial default + test target) and `TinyDBSessionStore` (durable; one row per session, reusing `ConversationManager`'s TinyDB mechanics; no new dep).
  - `session_key(channel, sender) -> "channel:sender"`; single-user collapses to `"default:default"` (no ceremony).
  - `SessionLocks`: lazy per-key `asyncio.Lock` so one session's turns serialize while different sessions run concurrently (`async with locks(key):`).
  - **Routing pattern** (no new agent primitive): per inbound message, under that session's lock, `agent.restore(session.messages)` -> `await agent.run(text)` -> snapshot `agent.model_client.messages` back -> `store.save(session)`. Agents never share a live `messages` list across sessions. **Gotcha**: put the system prompt on the *client* (`aio.client(system=...)`), not the agent; an agent with `system_message` set resets its client each run (`_prepare_run`) and would wipe the restored history (the personal-assistant example wires it this way). (Part of the personal-assistant substrate roadmap; durable scheduler, network channel adapters, and run-safety hooks are separate follow-ups.)

### Semantic Memory

- **[aimu/memory/base.py](aimu/memory/base.py)**: `MemoryStore` abstract base class
  - Defines the shared interface: `store(content)`, `search(query, n_results)`, `delete(identifier)`, `list_all()`, `__len__()`
  - All concrete stores implement this interface

- **[aimu/memory/semantic_store.py](aimu/memory/semantic_store.py)**: `SemanticMemoryStore(MemoryStore)` for semantic fact storage
  - Uses ChromaDB with cosine similarity for vector search
  - `store(fact)`: Store a subject-predicate-object string; auto-assigns UUID
  - `search(topic, n_results, max_distance)`: Semantic vector search by topic; `max_distance` is an optional cosine-distance cutoff (0 = identical, 2 = maximally dissimilar; useful range ~0.3 to 0.6; default `None` = no cutoff)
  - `delete(fact)`: Remove by exact-string match; no-op if not found
  - `list_all()`: Return all stored fact strings
  - Constructor: `SemanticMemoryStore(collection_name="memories", persist_path=None)`; `persist_path=None` → ephemeral

- **[aimu/memory/document_store.py](aimu/memory/document_store.py)**: `DocumentStore(MemoryStore)`: path-based document store
  - Mirrors Anthropic's Managed Agents Memory API for drop-in compatibility
  - `write(path, content)`: Create/overwrite document at path (recommended ≤ 100 KB)
  - `read(path)`: Retrieve document; raises `KeyError` if not found
  - `edit(path, old_str, new_str)`: Replace first occurrence; raises `ValueError` if not found
  - `list_paths(prefix=None)`: Sorted paths, optionally filtered by prefix
  - `search_full_text(query, n_results)`: Case-insensitive substring search; returns `[{"path": ..., "content": ...}]`
  - `MemoryStore` interface: `store()` auto-assigns `/note-{uuid}.md` path; `search()` wraps `search_full_text()`; `delete()` treats identifier as path; `list_all()` delegates to `list_paths()`
  - Constructor: `DocumentStore(persist_path=None)`; `persist_path=None` → ephemeral

- **[aimu/memory/mcp.py](aimu/memory/mcp.py)**: FastMCP server exposing `SemanticMemoryStore` as tools
  - `search_memories(search_request)`, `add_memories(memories)`, `delete_memory(memory)`, `list_memories()`
  - Storage path via `MEMORY_STORE_PATH` env var; run: `python -m aimu.memory.mcp`

- **[aimu/memory/document_mcp.py](aimu/memory/document_mcp.py)**: FastMCP server exposing `DocumentStore` as tools (matches Anthropic's Managed Agents Memory API naming)
  - `memory_list(path_prefix)`, `memory_search(query)`, `memory_read(path)`, `memory_write(path, content)`, `memory_edit(path, old_str, new_str)`, `memory_delete(path)`
  - Storage path via `DOCUMENT_STORE_PATH` env var; run: `python -m aimu.memory.document_mcp`

### Retrieval-Augmented Generation (RAG)

- **[aimu/rag/](aimu/rag/)**: chunk / retrieve / rerank primitives as **plain functions over the `MemoryStore` interface**: deliberately *no* `BaseRetriever` / `BaseSplitter` / `BaseLoader` class hierarchy. The retriever interface already exists as `MemoryStore.search()`; a parallel retriever tree would violate "composability through uniform interfaces" and "no parallel inheritance trees for the same concept" (see `docs/explanation/design-principles.md`). If polymorphism is ever needed, the idiomatic move is a `typing.Protocol`, not an ABC subtree.
  - **[aimu/rag/splitter.py](aimu/rag/splitter.py)**: `split_text(text, *, chunk_size=1000, chunk_overlap=200, separators=None, length_function=len)`: recursive separator-based chunking. Tries the largest separator that keeps pieces under `chunk_size` (default hierarchy: `["\n\n", "\n", ". ", " ", ""]`), recursing into oversized pieces; `""` is the base case (character split) so no chunk exceeds `chunk_size`. `_merge` greedily packs splits with `chunk_overlap` carried at piece granularity. `length_function` defaults to `len` (chars); pass a tokenizer's counter for token-aware chunking. Empty/whitespace input → `[]`; `chunk_overlap >= chunk_size` raises `ValueError`.
  - **[aimu/rag/pipeline.py](aimu/rag/pipeline.py)**: `ingest(store, documents, *, chunk_size, chunk_overlap, separators, length_function) -> int` (splits one-or-many docs, calls `store.store()` per chunk, returns count); `retrieve(store, query, *, n_results=5, **search_kwargs) -> list[str]` (RAG-named pass-through to `store.search()`, forwarding e.g. `max_distance=`); `format_context(chunks, *, separator="\n\n", numbered=False) -> str`.
  - **[aimu/rag/rerank.py](aimu/rag/rerank.py)**: `rerank(query, documents, *, model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=None)` via `sentence_transformers.CrossEncoder`; lazy-imported (raises `ImportError` pointing at the `[hf]` extra if absent) and weight-cached in a module-level registry. Empty input returns `[]` without loading the model.
  - RAG-as-augmentation is a **workflow you compose** (`ingest` → `retrieve` → `format_context` → `chat`), not a new abstraction. `make_retrieval_tool(store, *, n_results=5)` in [aimu/tools/builtin.py](aimu/tools/builtin.py) wraps `retrieve` + `format_context` as a `retrieve_context(query)` agent tool (numbered context; "No relevant context found." when empty), next to `make_memory_tools`.
  - **Out of scope (by design)**: document loaders (`builtin.fs.read_file` / `get_webpage` cover ingestion sources) and per-chunk metadata (chunks are plain strings per the `MemoryStore` `store(content: str)` contract. The higher-leverage future change for citations/filtering would be widening that value type, *not* adding loader/retriever classes).
  - **Tests**: `tests/test_rag.py` (mock-only): splitter (size, overlap, hard-cut, token-aware via `length_function`, custom separators, guards), `ingest`/`retrieve` over a deterministic substring store, `format_context`, `rerank` with a stubbed `CrossEncoder`, and `make_retrieval_tool` spec + output.

### Agent Skills

- **[aimu/skills/](aimu/skills/)**: Filesystem-discovered skills that inject instructions and tools into agents
  - **[aimu/skills/manager.py](aimu/skills/manager.py)**: `SkillManager` discovers `SKILL.md` files from project and user dirs
    - Default search paths (project-level wins on collision): `.agents/skills/`, `.claude/skills/`, `~/.agents/skills/`, `~/.claude/skills/`
    - Logs discovered count and paths at `INFO` level on first access
    - `catalog_prompt()`: Returns XML-formatted skill listing for system prompt injection. Each entry includes a `<tools>` block listing the names of script-derived tools (`{skill}__{script_stem}`) so the model can call them directly without first invoking `activate_skill`.
    - `get_skill_body(name)`: Returns full SKILL.md instructions body; raises `SkillNotFoundError` (subclass of `KeyError`) if the name is unknown.
    - Malformed `SKILL.md` files raise `SkillLoadError` (subclass of `ValueError`) on parse; no longer silently skipped.
  - **[aimu/skills/skill.py](aimu/skills/skill.py)**: `AgentSkill` dataclass with `name`, `description`, `path`, `load_body()`, `script_tool_names()`.
  - **[aimu/skills/mcp.py](aimu/skills/mcp.py)**: `build_skills_server(manager)` creates a FastMCP server
    - Registers `activate_skill(name)` tool
    - Dynamically registers each `*.py` file in a skill's `scripts/` directory as a tool (`{skill_name}__{script_stem}()`)
  - Skill SKILL.md requires YAML frontmatter with `name` and `description` fields

### Agentic Workflows

AIMU follows the taxonomy from Anthropic's *[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)*. All runnable units share **one** `Runner` ABC with `run(task, stream=False)` and a `messages` property; there is no type-level split between agents and workflows. The agent-vs-workflow distinction is a *categorisation of concrete classes*, not a type hierarchy; it lives in the module docstring at the top of [aimu/agents/base.py](aimu/agents/base.py), which carries a decision tree for picking the right class.

The mapping from Anthropic's patterns to AIMU's concrete classes:

| Anthropic pattern | AIMU class | Category | Module |
|---|---|---|---|
| Augmented LLM (base building block) | `BaseModelClient` / `ModelClient` | n/a | `aimu.models` |
| Autonomous agent (open-ended tool loop) | `Agent` | autonomous | `aimu.agents.agent` |
| Autonomous agent + skill discovery | `SkillAgent` | autonomous | `aimu.agents.skill_agent` |
| Orchestrator-workers | `OrchestratorAgent` (+ `assemble()` factory) | autonomous (workers as tools) | `aimu.agents.orchestrator_agent` |
| Prompt chaining | `Chain` (+ `Chain.from_client()`) | code-controlled | `aimu.agents.workflows.chain` |
| Routing | `Router` (+ `Router.from_client()`) | code-controlled | `aimu.agents.workflows.router` |
| Parallelization | `Parallel` (+ `Parallel.from_client()`) | code-controlled | `aimu.agents.workflows.parallel` |
| Evaluator-optimizer | `EvaluatorOptimizer` | code-controlled | `aimu.agents.workflows.evaluator` |
| *AIMU extension*: plan → execute → score → replan | `PlanExecuteEvaluator` (+ `from_client()`) | code-controlled | `aimu.agents.workflows.plan_execute_evaluator` |

**Notes on the mapping**:
- Anthropic places "orchestrator-workers" under *workflows*. AIMU implements it as an autonomous `OrchestratorAgent` because the dispatch decision (which worker to call, with what task) lives inside the orchestrator LLM's tool-calling loop (i.e., the LLM directs the flow within the orchestrator). The workers are wrapped as `@tool`-decorated callables (`OrchestratorAgent.assemble()` does this automatically) so the orchestrator's `concurrent_tool_calls=True` overlaps worker execution. Each worker still runs its own agentic loop.
- `PlanExecuteEvaluator` is not in Anthropic's original five patterns; it composes the planner (a `SkillAgent`), executor (an `Agent`), and a `Scorer` into a plan → execute → judge → replan loop, useful for tasks with measurable success criteria.
- All concrete classes are re-exported from `aimu.agents` (`from aimu.agents import Agent, Chain, Router, Parallel, EvaluatorOptimizer, PlanExecuteEvaluator, OrchestratorAgent, SkillAgent`).
- The async mirror under `aimu.aio` re-implements the same set of classes (same names, `async def run()`) with `asyncio.TaskGroup` replacing `ThreadPoolExecutor` in `Parallel` and `concurrent_tool_calls`.

See [docs/explanation/agents-vs-workflows.md](docs/explanation/agents-vs-workflows.md) for the *why*: when each one is the right tool, and how the split shapes API choices.

- **[aimu/agents/base.py](aimu/agents/base.py)**: Single `Runner` ABC for the agent/workflow hierarchy
  - `Runner(ABC)`: abstract `run(task, generate_kwargs, stream=False, images=None)` and `messages` property. Every concrete agent and workflow inherits directly from `Runner`; the agent-vs-workflow distinction is documentation only.
  - `Runner.as_tool(*, name=None, description=None) -> Callable`: concrete (non-abstract) method that wraps `self.run(task)` as a `@tool`-decorated callable (`tool(task: str) -> str`). Lets any agent or workflow be handed to another agent as a tool. `name` defaults to `self.name` (sanitised); `description` to the first line of `self.system_message` if present (covers `Agent`/`SkillAgent`), else a generic delegation string (workflows have no `system_message`). `AsyncRunner.as_tool()` is the async twin (the dispatch is `async def`, so the `@tool` decorator marks it `__tool_is_async__=True`). `OrchestratorAgent.assemble()` builds its worker tools via this method.
  - `MessageHistory`: type alias for `dict[str, list[dict]]`, the return type of `.messages` across the hierarchy; exported from `aimu.agents`.

- **[aimu/agents/agent.py](aimu/agents/agent.py)**: `Agent(Runner)` for autonomous, multi-round tool execution
  - Constructor: `Agent(model_client, system_message=None, name=None, tools=None, max_iterations=10, continuation_prompt=..., reset_messages_on_run=False, final_answer_prompt=None, deps=None, tool_approval=None, concurrent_tool_calls=False)`. `system_message` is the second positional argument. `name` is optional, auto-derived from `id(agent)` as `"agent-{hex}"` when unset. Tool configuration (`tools`, `deps`, `tool_approval`, `concurrent_tool_calls`) lives on the `Agent`, not the model client.
  - Wraps any `BaseModelClient` and **composes a tool-loop engine** (`_ToolLoop` / `_AsyncToolLoop`) per run via `_make_tool_loop(...)`. `chat()` is a **single model turn** (it parses and stores any requested tool calls; it does **not** execute them). The engine is the loop over that: `chat(task, tools=...)`, then dispatch the requested tools, then `chat()` (no user message — a continuation turn), until a turn makes no tool calls, or `max_iterations` turns are reached. **No synthetic continuation prompt is injected** into the transcript.
  - Stop condition (module-level `last_turn_called_tools` in `aimu/agents/_tool_loop.py`): reverse-scan `model_client.messages` for the most recent model turn — a trailing `"tool"` result (or an `assistant` message that still carries `tool_calls`) means "continue"; a plain `assistant` answer means "stop".
  - **Deprecated (kept as accepted no-ops for back-compat):** `continuation_prompt` / `DEFAULT_CONTINUATION_PROMPT` — the loop no longer injects a prompt (it continues via `chat()` with no user message), so `PROVENANCE_CONTINUATION` is no longer produced (the constant remains for legacy transcripts).
  - `final_answer_prompt` (opt-in, default `None`): guarantees a final answer when the loop exhausts `max_iterations` while the model is still calling tools. Instead of returning whatever the last (possibly tool-only) turn produced (the empty-output failure mode), the engine sends this prompt **once with tools disabled** (`chat(..., tools=[])`), forcing synthesis from the gathered context. It fires only on the cap-with-pending-tools path, a natural finish is unaffected, and the wrap-up turn is **not** counted against `max_iterations`. Mirrored in `aio.Agent`; forwarded by `OrchestratorAgent._init_orchestrator()` / `assemble(..., final_answer_prompt=...)` to the inner agent (sync + async).
  - `tools: list[Callable]` field accepts Python functions decorated with `@aimu.tool`; `_make_tool_loop()` hands them (via a re-read-each-round callable, `_effective_tools`) to the engine, which advertises them on each `chat(tools=...)` and dispatches them. Combine freely with `MCPClient(...).as_tools()` for cross-process tools (just concatenate into the `tools` list). The `Agent` never mutates `model_client.tools`.
  - `run(task, generate_kwargs, stream=False, images=None, tools=None, deps=None, tool_approval=None, schema=None)`: returns final response string or `Iterator[StreamChunk]`. Streamed chunks carry `agent` (the agent name) and `iteration` (loop index) fields. `tools=` / `deps=` / `tool_approval=` are per-run overrides of the agent's configured fields (`None` uses the field). `schema=` short-circuits to a single structured-output turn (no tool loop). The overrides are passed to the engine (`_make_tool_loop`), which advertises the effective tools on every `chat()` call; after the run, `model_client.tools` is the transient default (`[]`) since the `Agent` supplies tools per call. Not added to the `Runner` ABC or workflow classes (they compose sub-runners and have no single tool list).
  - `as_model_client() -> BaseModelClient`: returns a `BaseModelClient` view (internal `_AgenticView`) whose `chat()` runs the full agent loop. Use this to drop an agent into an API that expects a model client; for direct use call `run()` instead.
  - `messages` property: returns `{name: snapshot}` where `snapshot` is a copy of `model_client.messages` taken at the end of the most recent `run()` call; stable even when agents share a `ModelClient` and `_prepare_run()` clears the live messages.
  - `from_config(config, model_client)`: factory from plain dict (keys: `name`, `system_message`, `max_iterations`, `continuation_prompt`, `final_answer_prompt`).
  - `restore(messages)`: Restore agent state from a saved `list[dict]` (OpenAI message format) for resuming after failure. Calls `model_client.reset()` (clears messages, preserves `system_message` value), strips the leading system message from *messages* to prevent duplication on the next `chat()`, then sets `model_client.messages`. The live partial state after a failed run is in `agent.model_client.messages` (not the post-run snapshot from `agent.messages`). See `docs/how-to/using-llms-inside-tools.md` for the full save/restore pattern. **`restore()` is uniform across runners** (each delegates down to an `Agent.restore`): `Chain.restore(messages, step=0)`, `Router.restore(messages, route=None)` (`None` → routing classifier; route key → that handler, `KeyError` on miss), `Parallel.restore(messages, worker=0)`, `OrchestratorAgent.restore(messages)` (inner orchestrator), and `EvaluatorOptimizer.restore(messages)` (generator). It is **not** on the `Runner` ABC (signatures vary by selector). The full set is mirrored one-for-one on `aimu.aio` (the async surface previously had none). Tests: `tests/test_checkpointing.py`, `tests/test_aio_checkpointing.py`.

- **[aimu/agents/skill_agent.py](aimu/agents/skill_agent.py)**: `SkillAgent(Agent)`: adds skill discovery and injection
  - Extends `Agent` with a `skill_manager: SkillManager` field (defaults to `SkillManager()` for auto-discovery)
  - On first `run()` (or after a message reset), `_setup_skills()` appends the skill catalog to the system message and builds the skills server's tools (via `MCPClient(server=...).as_tools()`, built once and cached on `_skills_tools`); the model can call `activate_skill` for the full body, or call any script-derived `{skill}__{script}` tool directly (the catalog lists those tool names inline). The skill tools reach the model through the overridden `_effective_tools` (agent tools + skill tools, deduped by name), which the tool-loop engine re-reads each round — so a skill authored mid-run via `reload_skills()` is advertised and dispatchable without restarting the conversation. `_setup_skills()` does **not** mutate `model_client.tools`.
  - `_prepare_run()` resets the `_skills_setup_done` flag when messages are cleared, so skills are re-injected on each fresh run
  - `from_config(config, model_client)`: factory from plain dict (keys: `name`, `system_message`, `max_iterations`, `continuation_prompt`, `skill_dirs`); omit `skill_dirs` to auto-discover

- **[aimu/agents/agentic_client.py](aimu/agents/agentic_client.py)**: `_AgenticView(BaseModelClient)` is the internal class returned by `Agent.as_model_client()`. It is *not* part of the public API; use the method instead of importing the class. The legacy public name `AgenticModelClient` has been removed; call sites should switch to `agent.as_model_client()`.

- **[aimu/agents/workflows/chain.py](aimu/agents/workflows/chain.py)**: `Chain(Runner)`: **Prompt Chaining** pattern (re-exported from `aimu.agents`)
  - Output of step N (accumulated `GENERATING` chunks) becomes task input to step N+1
  - `run(task, generate_kwargs, stream=False)`: returns final string or `Iterator[StreamChunk]`; streamed chunks carry the chain step in `iteration`.
  - `Chain.from_client(client, prompts, *, name="chain")`: build a Chain from a single shared client and a list of step system_messages. Each step gets `reset_messages_on_run=True`.
  - `from_config(configs, client)`: builds from a list of dicts and a single `ModelClient`; the client is shared across steps.
  - `agents: list[Runner]`: steps may be `Agent` or nested workflows
  - `messages` property: merges `agent.messages` for all steps into one `MessageHistory` dict

- **[aimu/agents/workflows/router.py](aimu/agents/workflows/router.py)**: `Router(Runner)`: **Routing** pattern (re-exported from `aimu.agents`)
  - `routing_agent` classifies the task; output (stripped, lowercased) is matched against `handlers: dict[str, Runner]`
  - Falls back to `fallback: Runner` if set; raises `ValueError` otherwise
  - `run(stream=True)` yields routing chunks then the matched handler's chunks
  - `Router.from_client(client, classifier_prompt, handlers, *, fallback=None, name="router")`: build a Router from a single client and a classifier system_message.
  - `from_config(routing_config, handler_configs, client, fallback_config)`: factory from dicts
  - `messages` property: merges `routing_agent.messages` + all `handler.messages` + `fallback.messages` (if set)

- **[aimu/agents/workflows/parallel.py](aimu/agents/workflows/parallel.py)**: `Parallel(Runner)`: **Parallelization** pattern (re-exported from `aimu.agents`)
  - Runs `workers: list[Runner]` concurrently via `ThreadPoolExecutor`; workers complete before streaming to avoid interleaved output
  - `aggregator: Runner` (optional) receives all worker outputs joined by `separator`; without an aggregator, the joined string is returned directly
  - `run(stream=True)` streams only the aggregator phase (or yields a single GENERATING chunk)
  - `Parallel.from_client(client, worker_prompts, *, aggregator_prompt=None, separator=..., name="parallel")`: build a Parallel from a single client and a list of worker system_messages; pass `aggregator_prompt` to add an aggregator step.
  - `messages` property: merges all `worker.messages` + `aggregator.messages` (if set)

- **[aimu/agents/workflows/evaluator.py](aimu/agents/workflows/evaluator.py)**: `EvaluatorOptimizer(Runner)`: **Evaluator-Optimizer** pattern (re-exported from `aimu.agents`)
  - `generator` produces output; `evaluator` reviews it against the original task
  - Acceptance is decided by one of three mechanisms (priority order): `stop_when` (a predicate over the evaluator's output: raw text, or the typed verdict when `verdict_schema` is set); `verdict_schema` (a dataclass / Pydantic model the evaluator returns via `Agent.run(schema=...)`; acceptance reads its `passed` bool, revision uses its `feedback` str; `passed_attr`/`feedback_attr` configurable; a malformed verdict raises); or `pass_keyword` (default substring match in the evaluator's text). The loop also stops at `max_rounds`.
  - `run(stream=True)` yields a single GENERATING chunk with the final output (intermediate drafts not streamed)
  - `messages` property: merges `generator.messages` + `evaluator.messages`

- **[aimu/agents/workflows/plan_execute_evaluator.py](aimu/agents/workflows/plan_execute_evaluator.py)**: `PlanExecuteEvaluator(Runner)`: **Plan-Execute-Evaluate** pattern (re-exported from `aimu.agents`)
  - `planner` (a `SkillAgent`) produces a plan; `executor` (an `Agent`) runs it with tools; `scorer` (a `Scorer`) judges the output
  - On score below `pass_threshold`, planner is invoked again with prior round's feedback to produce a *new* plan
  - Bounds at `max_rounds`; returns the highest-scoring attempt
  - Two criteria modes: user-supplied (`criteria="..."`) or planner-invented (`criteria=None`; planner emits `## Evaluation criteria` and `## Plan` sections)
  - `PlanExecuteEvaluator.from_client(client, *, criteria=None, judge_client=None, executor_tools=None, ...)`: factory that builds a `SkillAgent` planner, an `Agent` executor with your tools, and an `LLMJudgeScorer` over `judge_client` (defaults to `client`)
  - `last_attempts: list[dict]`: per-round log with `plan`, `criteria`, `output`, `score`, `feedback`

- **[aimu/agents/orchestrator_agent.py](aimu/agents/orchestrator_agent.py)**: `OrchestratorAgent(Runner, ABC)` base class for orchestrator agents
  - Two construction paths:
    1. **Subclass**: define worker `Agent` instances and `@tool`-decorated dispatch functions in `__init__`, then call `self._init_orchestrator(model_client, *, name, system_message, tools=[...], concurrent_tool_calls=False, final_answer_prompt=None)` (renamed from the legacy `_setup_orchestrator`) to wire everything together.
    2. **Factory**: `OrchestratorAgent.assemble(client, system_message, *, workers: list[Runner], name="orchestrator", concurrent_tool_calls=True, final_answer_prompt=None)`. Each worker is auto-wrapped as a `@tool` via `Runner.as_tool()` (named after the worker's `name`, described from its `system_message`). Workers may be **any `Runner`** (an `Agent`, a `Chain`/`Router`/`Parallel` workflow, or a remote A2A `RemoteAgent`), not just `Agent` instances. No subclass needed.
  - `final_answer_prompt` (both construction paths) is forwarded to the inner orchestrator `Agent`, so an orchestrator that exhausts its iterations while still dispatching to workers still produces a final answer (see `Agent.final_answer_prompt`).
  - Provides `run(task, generate_kwargs, stream, images)` and `messages`; subclasses need not re-implement them
  - Exported from `aimu.agents`

- **[aimu/agents/prebuilt/](aimu/agents/prebuilt/)**: Ready-to-use `OrchestratorAgent` subclasses demonstrating the orchestrator + worker tools pattern
  - Each class accepts a `ModelClient`, defines `@tool`-decorated wrappers around worker `Agent.run()` calls, and passes them to `_init_orchestrator(tools=[...])`
  - `ResearchReportAgent`: orchestrator + `research_overview`, `find_examples`, `find_counterpoints` worker tools; accepts optional `worker_tools: list[Callable]` to give workers live tool access (e.g. `[builtin.web_search, builtin.get_webpage]`) and automatically enables `concurrent_tool_calls` on the orchestrator
  - `CodeReviewAgent`: orchestrator + `review_security`, `review_performance`, `review_readability` worker tools
  - `ContentCreationAgent`: orchestrator + `research_topic`, `create_outline`, `write_section` worker tools
  - All three inherit from `OrchestratorAgent` and expose `run(task, stream=False)` and `messages`

- All agents/workflows use the public `chat()` / `chat(..., stream=True)` API only; no changes to `ModelClient` subclasses
- `messages` is composable: nested workflows (e.g. a `Router` dispatching to a `Chain`) merge recursively, so all sub-agent names appear at the top level

### Agent-to-Agent (A2A) Interop

Optional cross-process / cross-vendor agent interop under **[aimu/agents/a2a/](aimu/agents/a2a/)** (extra: `aimu[a2a]`; guarded `HAS_A2A` flag exported from `aimu.agents`). The agent-level analog of the MCP-for-tools surface (`aimu.tools.MCPClient` / `python -m aimu.tools.mcp`): MCP shares *tools*, A2A shares whole *agents*. Backed by `a2a-sdk` pinned to the `0.3.x` line (pydantic-native; the `1.x` protobuf/gRPC rework is a tracked future migration). **A2A types never leak into `Runner`/`Agent` core**; they adapt only at the boundary (`run()` still takes/returns `str`).

- **[aimu/agents/a2a/_card.py](aimu/agents/a2a/_card.py)**: pure adapters shared by sync + async, server + client (the agent-level analog of `aimu/tools/mcp_format.py`): `build_agent_card()`, `make_send_request()`/`make_stream_request()`, `result_to_text()`/`parts_to_text()`, `runner_name()`/`runner_description()`.
- **[aimu/agents/a2a/client.py](aimu/agents/a2a/client.py)**: `RemoteAgent(Runner)` + `A2AConnectionError`. `RemoteAgent.connect(url, *, name=None, agent_card_path="/.well-known/agent-card.json", timeout=60.0)` resolves the remote agent card and returns a local `Runner`. The sync client drives the async `a2a-sdk` through an anyio blocking portal (same mechanism as `aimu.tools.MCPClient`); it sends to the URL you connected to (`A2AClient(..., url=url)`), not the card's advertised url, so a card behind a proxy still works. `run(stream=True)` yields the full response as a single `GENERATING` chunk on the sync surface. `RemoteAgent.system_message` is set to the card's description so the inherited `as_tool()` produces a meaningful tool description. `messages` is best-effort (the local `(user, assistant)` exchange, not the remote transcript). `close()`/`__del__` tear down the portal + httpx client; `__deepcopy__` returns self (holds a live connection).
- **[aimu/agents/a2a/server.py](aimu/agents/a2a/server.py)**: `serve_a2a(runner, *, host, port, url=None, name=None, description=None, skills=None, **uvicorn_kwargs)` (blocking) and `build_a2a_app(runner, *, url, ...)` (returns a Starlette ASGI app, used by tests/embedding). Internal `_RunnerExecutor(AgentExecutor)` offloads the blocking sync `runner.run` via `asyncio.to_thread`. The card advertises `streaming=True` (the `DefaultRequestHandler` serves the SSE endpoint, emitting the final message as one event).
- **[aimu/agents/a2a/__main__.py](aimu/agents/a2a/__main__.py)**: `python -m aimu.agents.a2a --model ... --system ... --name ... --host ... --port ...` builds an `Agent` and serves it (agent-level analog of `python -m aimu.tools.mcp`).
- **Async twin** under **[aimu/aio/a2a/](aimu/aio/a2a/)** (exported from `aimu.aio` when `HAS_A2A`): `RemoteAgent(AsyncRunner)` uses the `a2a-sdk` async client natively (no portal, mirrors `aimu.aio.MCPClient`) and supports real incremental `message/stream` streaming mapped to `StreamChunk`s; async `serve_a2a`/`build_a2a_app` await `runner.run` directly. Reuses the sync `_card.py` helpers and `A2AConnectionError`.
- **Composition payoff**: because `RemoteAgent` is a `Runner`, a remote agent drops into `Chain`/`Router`/`Parallel`/`assemble(workers=[...])` and into any local agent's tools via `remote.as_tool()` with no A2A-specific wiring; the direct analog of `MCPClient.as_tools()`, at the agent level.
- **Tests**: `tests/test_a2a.py` (sync) and `tests/test_aio_a2a.py` (async), both gated on `pytest.importorskip("a2a")`. Mock-only unit tests (card helpers, `result_to_text`, app routes via Starlette `TestClient`) plus a real HTTP round-trip (uvicorn in a background thread) proving `RemoteAgent` composes as tool / `Chain` step / orchestrator worker, and the streaming + dead-URL paths.

### Personal-Assistant Primitives (channels, scheduler, skill authoring)

Three **async-first** primitives (under `aimu.aio`, plus filesystem-shared skill authoring under `aimu.skills`) supply the pieces a long-running, always-on personal assistant (OpenClaw / Hermes Agent style) needs that the rest of AIMU did not: a chat transport, proactive triggers, and runtime skill creation. They are **library primitives, not an app** -- the assembled daemon lives in `examples/personal-assistant/` (see "Project Structure"). Multi-user sessions are deliberately out of scope: a *personal* assistant serves one user, so `ConversationManager` suffices. Each primitive follows the existing "one ABC / small utility, no new core deps" shape; network channel SDKs (Telegram/Slack) are a deferred follow-up behind an optional extra + `HAS_*` guard, mirroring `a2a`.

- **Channel transport** -- **[aimu/aio/channels/](aimu/aio/channels/)** (package):
  - **[aimu/aio/channels/base.py](aimu/aio/channels/base.py)**: `Channel(ABC)` -- a uniform transport interface alongside `AsyncRunner`/`AsyncBaseModelClient`/`MemoryStore`. Methods: `receive() -> AsyncIterator[ChannelMessage]` (an async generator the assistant loop iterates), `async send(content: str | AsyncIterator[StreamChunk], *, reply_to: ChannelMessage | None = None)`, and `async aclose()` (default no-op). `ChannelMessage` is a plain dataclass (`text`, `sender`, `channel`, `images`, `metadata`) -- transport-level data, distinct from LLM `list[dict]` state; `text`/`images` map directly onto `agent.run(task, images=...)`. The `reply_to` arg is the single seam that lets a future network adapter route a reply to the right chat while keeping `send()` stable; single-user adapters ignore it.
  - **[aimu/aio/channels/cli.py](aimu/aio/channels/cli.py)**: `CLIChannel(Channel)`. `receive()` reads stdin via `await asyncio.to_thread(sys.stdin.readline)` (idiomatic, dependency-free; avoids POSIX `connect_read_pipe`), yielding one `ChannelMessage` per non-blank line and ending on EOF (Ctrl-D). `send()` prints a finished string, or relays an `AsyncIterator[StreamChunk]` token-by-token (`GENERATING`; plus `THINKING` and `TOOL_CALLING` as labelled lines when the opt-in `show_thinking` / `show_tools` flags are set, both off by default to keep the bare channel minimal). A blocked `readline` thread can't be cancelled, so shutdown relies on EOF or process exit.
  - **[aimu/aio/channels/web.py](aimu/aio/channels/web.py)**: `WebChannel(Channel)`, the WebSocket twin of `CLIChannel`. Bridges one browser WebSocket onto the `Channel` ABC: the app's server pump calls `feed(text)` per inbound frame (`feed(None)` on disconnect, the sentinel that ends `receive()`), and `send()` relays a finished string or a streamed reply as JSON frames a browser page renders. Frame protocol: `{"type": "message", "text", "proactive"}` (a finished message; `proactive` is true when there is no `reply_to`, e.g. a scheduler push), `{"type": "token", "text"}` (`GENERATING`), `{"type": "thinking", "text"}` (`THINKING`, only when `show_thinking`), `{"type": "tool", "name", "arguments"}` (`TOOL_CALLING`, only when `show_tools`), and `{"type": "done"}` (terminates a stream). The public `async send_frame(frame)` is the single point through which every frame reaches the socket (closed-guard: a late frame is dropped, not raised) and the blessed subclass seam for apps adding their own frame types (e.g. conversation lists, approval prompts). The websocket is duck-typed (`send_json`/`close`), so the module has **no** hard `starlette` import and is unit-testable with a fake; exported unconditionally (no `HAS_*` guard). The Starlette server, route, and HTML page are **app-side, not library** (an app wires its own; see [examples/personal-assistant/web_assistant.py](examples/personal-assistant/web_assistant.py)). Tests: [tests/test_aio_web_channel.py](tests/test_aio_web_channel.py).
  - Network adapters (e.g. `telegram.py`) would live here behind a `telegram` extra + `HAS_TELEGRAM` guard, constructed via a `TelegramChannel.connect(token=...)` classmethod factory (mirrors `aio.MCPClient.connect`). Not built yet.
- **Scheduler** -- **[aimu/aio/scheduler.py](aimu/aio/scheduler.py)**: `Scheduler` runs interval and one-shot async jobs concurrently under a single `asyncio.TaskGroup` (the `aimu/aio/workflows/parallel.py` idiom). API: `every(seconds, callback, *, name=None, first_delay=None) -> str`, `at(delay_seconds, callback, *, name=None) -> str`, `cancel(name) -> bool`, `stop()`, and `async run()` (blocks until `stop()`). A `Job` is a zero-arg coroutine factory; bind context (an agent, a channel) with a closure, keeping the scheduler decoupled from what it fires. **Job-exception isolation**: a job that raises is logged and (for interval jobs) continues on the next tick -- one misbehaving reminder must not tear down the daemon; only `stop()`-driven `CancelledError` unwinds the run loop. **Single-use** by design: a `stop()` signalled before `run()` (e.g. by a sibling task that finished first) is honored and `run()` returns immediately -- there is no `self._stop.clear()` (it would create a lost-stop race); use a fresh `Scheduler` to run again. Persistence is out of scope (jobs are non-serializable callables; durable cron belongs in a wrapper).
- **Skill authoring** -- **[aimu/skills/authoring.py](aimu/skills/authoring.py)** (sync filesystem work, surface-shared) + `SkillManager.refresh()`:
  - `write_skill(name, description, body, *, skills_dir, overwrite=False, metadata=None) -> Path`: writes `skills_dir/<name>/SKILL.md` (YAML frontmatter + markdown body, the format `SkillManager` discovers). Validates that `name` is a slug (lowercase-with-hyphens, no path separators -> traversal guard) and `description` is non-empty; refuses to clobber unless `overwrite=True`; round-trips through the manager parser so a malformed authored file fails loudly at the write site.
  - `make_skill_authoring_tool(manager, skills_dir) -> Callable`: returns an async `@tool author_skill(name, description, body)` (Hermes-style self-improvement) that calls `write_skill(...)` then `manager.refresh()`. Both bound by closure (no module globals).
  - `SkillManager.refresh()` (the only edit to an existing core module): clears the cached `_skills` and re-discovers, so a skill authored mid-run is visible. **Known limitation**: after `refresh()`, `activate_skill` finds the new skill (its closure reads `manager.skills` live), but a skill catalog already injected into an in-flight system prompt is not retroactively updated -- it surfaces from the next fresh conversation. Don't attempt live MCP tool re-registration.
  - Exported from `aimu.skills` (`write_skill`, `make_skill_authoring_tool`); intentionally *not* re-exported from `aimu.aio` (surface-agnostic filesystem code that returns an async tool).
- **Exports**: `Channel`, `ChannelMessage`, `CLIChannel`, `Scheduler` are added to `aimu.aio.__all__` (no optional guard -- stdlib + existing core types only).
- **Tests** (mock-only): `tests/test_aio_channels.py` (Channel ABC abstract, `CLIChannel` receive-to-EOF + string/stream send + ignored `reply_to`), `tests/test_aio_scheduler.py` (one-shot/interval firing, `cancel`, job-exception isolation, clean `stop()`, add-job-after-start, stop-before-run), `tests/test_aio_skill_authoring.py` (`write_skill` discoverability + slug/clobber/empty guards, `refresh()` cache invalidation, `author_skill` tool spec + write+refresh), `tests/test_aio_assistant_exports.py` (export smoke), and `examples/personal-assistant/tests/test_assistant.py` (the daemon wiring).

### Evaluations & Benchmarking

- **[aimu/evals/](aimu/evals/)**: DeepEval adapters (`aimu[evals]` extra) plus a multi-model benchmark harness (`aimu[tuning]` extra: it uses `pandas`)
  - **[aimu/evals/deepeval.py](aimu/evals/deepeval.py)**: `DeepEvalModel(DeepEvalBaseLLM)` adapter
    - Wraps any `BaseModelClient` to satisfy DeepEval's judge interface
    - `get_model_name()`: returns the model enum member name
    - `generate(prompt, schema=None)`: calls `model_client.generate(prompt)`; when `schema` is a Pydantic `BaseModel` subclass, parses JSON from the response using a 3-stage strategy (raw → fenced block → `{...}` substring) and validates into the schema
    - `a_generate(prompt, schema=None)`: async wrapper using `run_in_executor` (AIMU clients are synchronous)
    - Pass to any DeepEval metric via `model=DeepEvalModel(client)`: `GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, etc.
  - **[aimu/evals/deepeval_scorer.py](aimu/evals/deepeval_scorer.py)**: `DeepEvalScorer(Scorer)`: drop-in `Scorer` for `JudgedPromptTuner` and `Benchmark` backed by DeepEval metrics
    - Constructor: `DeepEvalScorer(metrics: list[BaseMetric], input_field="content", output_field="output")`
    - `score(row)` builds an `LLMTestCase(input=row[input_field], actual_output=row[output_field], expected_output=row.get("reference"))`, calls `metric.measure()` on every metric, returns `(mean(scores), "\n".join(reasons))`
  - **[aimu/evals/benchmark.py](aimu/evals/benchmark.py)**: `Benchmark` and `BenchmarkResults` for multi-model comparison
    - `Benchmark(prompt: str, data: pd.DataFrame, scorer: Scorer, pass_threshold: float = 0.7, generate_kwargs: dict | None = None)`: validates that `data` has a unique index and a `content` column up front; optional `reference` column is forwarded to scorers
    - `run(clients: dict[str, BaseModelClient]) -> BenchmarkResults`: for each client, snapshots `client.messages`, then for every row calls `client.reset()` and `client.chat(prompt.format(content=row.content), generate_kwargs)`, scoring each output via `scorer.score(row_with_output)`. Original messages are restored in a `try/finally` so a failure during one client doesn't leak state to the next.
    - **Critical**: drives rows through `chat()` (not `generate()`) because the `_AgenticView` returned by `Agent.as_model_client()` only engages the agent loop via `chat()`. `client.reset()` preserves `system_message` (it's stored separately on the client) and clears history for the next chat.
    - Aggregates per-client to `{score: mean(scores), pass_rate: mean(scores >= pass_threshold)}`
    - `BenchmarkResults`: dataclass holding `metrics: pd.DataFrame` (rows = client names, cols = metrics) and the `prompt` used; exposes `to_csv(path)`, `to_json(path)`, and `to_catalog(catalog: PromptCatalog, prompt_name: str)` which stores one `Prompt(name=prompt_name, model_id=client_name, prompt=self.prompt, metrics=row)` per client; catalog auto-versions on each store so re-runs append new versions
    - Out of scope (deferred): multi-prompt grids, classification/extraction harnesses, per-row retention, cross-client concurrency
  - **[aimu/evals/__init__.py](aimu/evals/__init__.py)**: unconditional exports of `Benchmark` and `BenchmarkResults`; guarded imports for `DeepEvalModel`, `DeepEvalScorer`, and `HAS_DEEPEVAL` flag

### Prompt Management

- **[aimu/prompts/catalog.py](aimu/prompts/catalog.py)**: `PromptCatalog` for versioned prompt storage
  - Uses SQLAlchemy ORM; `Prompt` model tracks `id`, `parent_id`, `name`, `prompt`, `model_id`, `version` (auto-assigned), `mutation_prompt`, `reasoning_prompt`, `metrics` (JSON), `created_at` (auto-set)
  - All queries keyed on `(name, model_id)` pair to support multiple prompt lineages per model
  - `store_prompt(prompt)`: auto-assigns version and timestamp; `retrieve_last(name, model_id)`, `retrieve_all(name, model_id)`, `delete_all(name, model_id)`, `retrieve_model_ids()`, `retrieve_names()`
- **[aimu/prompts/mcp.py](aimu/prompts/mcp.py)**: FastMCP server exposing catalog as MCP tools
  - `get_prompt(name, model_id)`, `list_prompts()`, `store_prompt_version(name, model_id, prompt, metrics)`
  - DB path via `PROMPT_CATALOG_PATH` env var; run standalone: `python -m aimu.prompts.mcp`
- **[aimu/prompts/tuner.py](aimu/prompts/tuner.py)**: `PromptTuner` abstract base class for hill-climbing prompt optimization
  - Abstract methods: `apply_prompt(prompt, data)`, `evaluate(data) -> dict`, `mutation_prompt(current_prompt, items) -> str`
  - `tune(training_data, initial_prompt, max_iterations=20, max_examples=5, catalog=None, prompt_name=None) -> str`
  - Separate `generate_kwargs` (classification, 3 tokens) and `mutation_kwargs` (prompt generation, 512 tokens)
  - Thinking models automatically get higher token limits and temperature
  - Caches best state; regressing mutations revert without re-evaluation
  - Saves each improvement to catalog when `catalog` and `prompt_name` are provided
  - `score(metrics) -> float`: overrideable optimization target; default `metrics["accuracy"]`; override to maximise a different metric (e.g. `metrics["score"]` for `JudgedPromptTuner`)
  - `extract_mutated_prompt(result) -> str`: overrideable parser for mutation LLM responses; default extracts `<prompt>...</prompt>` tag content
- **[aimu/prompts/tuners/classification.py](aimu/prompts/tuners/classification.py)**: `ClassificationPromptTuner(PromptTuner)`
  - Binary YES/NO classification; prompt template must contain `{content}` placeholder
  - Data must have `content` (text) and `actual_class` (bool) columns
  - `classify_data(prompt, data)`: runs prompt on all rows, adds `predicted_class` column
  - `evaluate_results(data)`: returns `{accuracy, precision, recall}`
- **[aimu/prompts/tuners/multiclass.py](aimu/prompts/tuners/multiclass.py)**: `MultiClassPromptTuner(PromptTuner)`
  - N-way classification; constructor takes `classes: list[str]`
  - Prompt template must contain `{content}`; model output must include `[ClassName]` for one of the registered classes
  - Data must have `content` (str) and `actual_class` (str) columns
  - `classify_data(prompt, data)`: scans output for first `[ClassName]` match (case-insensitive); raises `ValueError` if none found
  - `evaluate_results(data)`: returns `{accuracy, macro_f1, per_class_{name}_precision/recall/f1}` for each class
  - Mutation prompt groups failures by `(predicted, actual)` confusion pair
- **[aimu/prompts/tuners/extraction.py](aimu/prompts/tuners/extraction.py)**: `ExtractionPromptTuner(PromptTuner)`
  - Structured field extraction; constructor takes `fields: list[str]`
  - Prompt template must contain `{content}`; model output must be parseable as JSON (raw, fenced block, or `{...}` substring)
  - Data must have `content` (str) and `expected` (dict mapping field→value) columns
  - `extract_data(prompt, data)`: calls model, parses JSON, stores result in `extracted` column
  - `_correct` per row: all fields in `self.fields` match between `extracted` and `expected`
  - `evaluate_results(data)`: returns `{accuracy, field_{name}_accuracy}` for each field
  - Mutation prompt shows field-by-field expected vs. actual mismatches per failed row
- **[aimu/prompts/tuners/judged.py](aimu/prompts/tuners/judged.py)**: `JudgedPromptTuner(PromptTuner)`
  - Open-ended generation evaluated by a pluggable [`Scorer`](aimu/prompts/tuners/scorers.py); constructor takes `scorer: Scorer`, `pass_threshold: float = 0.7`
  - Data must have `content` (str) column; optional `reference` (str) column is forwarded to scorers that consume it
  - `generate_responses(prompt, data)`: applies prompt to each row, stores output in `output` column using `response_kwargs` (longer token budget than `generate_kwargs`)
  - `judge_responses(data)`: calls `scorer.score(row)` per row; stores `judge_score` (0-1) and `judge_feedback` (string)
  - `_correct`: `judge_score >= pass_threshold`
  - `evaluate_results(data)`: returns `{score: mean judge_score, pass_rate}`
  - **Overrides `score()` to return `metrics["score"]`**; optimises mean quality instead of binary pass rate
  - Mutation prompt includes input, output, score, and judge feedback for each failing example
- **[aimu/prompts/tuners/scorers.py](aimu/prompts/tuners/scorers.py)**: `Scorer` ABC and built-in `LLMJudgeScorer`
  - `Scorer.score(row) -> tuple[float, str]`: returns normalised score (0-1) and feedback string for one row
  - `LLMJudgeScorer(judge_client, criteria, prompt_template=None, generate_kwargs=None)`: asks `judge_client` to rate 1-10 and parses the first standalone integer in that range; defaults to 0.5 on parse failure
- **[aimu/evals/deepeval_scorer.py](aimu/evals/deepeval_scorer.py)**: `DeepEvalScorer(Scorer)` (requires `aimu[evals]`)
  - Constructor: `DeepEvalScorer(metrics: list[BaseMetric], input_field="content", output_field="output")`
  - Each row is converted to `LLMTestCase(input=row[input_field], actual_output=row[output_field], expected_output=row.get("reference"))`
  - Calls `metric.measure(test_case)` for every metric; returns `(mean(metric.score for metric in metrics), "\n".join(f"{type(metric).__name__}: {metric.reason}" for metric in metrics))`
  - Use any DeepEval metric (`GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, etc.) wired with a `DeepEvalModel(judge_client)` as the metric's `model`

### Key Design Patterns

These are *implementation* patterns: the *how*. The *why* lives in the "Design Principles" section above.

1. **Single public construction path**: `ModelClient` is the factory. Provider clients are still importable but no longer the recommended public surface. The top-level `aimu.client()` and `aimu.chat()` wrap `ModelClient` for one-line use.
2. **Optional Dependencies**: Graceful degradation when model backends aren't installed. `HAS_*` flags expose installation state; missing optional deps don't break the import of the package.
3. **Tool Calling Abstraction**: The base class handles tool calls uniformly across providers. Concrete clients implement `_chat` / `_generate`; the public `chat` / `generate` wrap them with the `include` filter. `@tool` functions and `MCPClient.as_tools()` callables share one `self.tools` registry and dispatch through one by-name lookup (last entry wins on name collision).
4. **Streaming chunk type**: All clients support streaming via `chat(..., stream=True)`, `generate(..., stream=True)`, and image clients via `generate(..., stream=True)`. They yield `StreamChunk(phase, content, agent, iteration)` named tuples. `phase` is a `StreamingContentType` (THINKING, TOOL_CALLING, GENERATING, IMAGE_GENERATING, DONE); see the `StreamChunk` entry above for content shapes per phase. Streaming tools (generator functions decorated with `@tool`) yield their own chunks that flow through the agent's stream; see "Streaming tools" below. Use `chunk.is_text()` / `chunk.is_tool_call()` / `chunk.is_image_progress()` to dispatch on phase.
5. **Model Capability Flags**: `supports_tools`, `supports_thinking`, `supports_vision`, `supports_audio`, and `supports_structured_output` are encoded on the `ModelSpec` value of each `Model` enum member and mirrored as attributes for direct read access. `TOOL_MODELS` / `THINKING_MODELS` / `VISION_MODELS` / `AUDIO_MODELS` / `STRUCTURED_MODELS` classproperties are derived automatically.

## Important Implementation Notes

### Adding New Models

**Curated-catalog policy (all modalities).** AIMU ships specs for best-of-class models only. A model must be a member of its provider enum (or a hand-built spec object passed explicitly). Passing an arbitrary `"provider:some/unknown-repo"` string **raises `ValueError`**; the clients never fabricate a spec with guessed capabilities, because a wrong guess (`pipeline_class`, `supports_negative_prompt`, tool/thinking/vision flags, voice/step defaults) causes hard-to-debug runtime failures. The string form resolves to the *same* spec object as the enum member; unknown ids raise. This is enforced in each provider's `_parse_model_string` (text's `resolve_model_string` was always strict). See [docs/how-to/add-new-model.md](docs/how-to/add-new-model.md).

When adding new models to a client:
1. Add an enum member to the client's `Model` class with a `ModelSpec(id, tools=..., thinking=..., vision=..., audio=..., generation_kwargs=...)` value. Example: `QWEN_3_8B = ModelSpec("qwen3:8b", tools=True, thinking=True)`.
2. `TOOL_MODELS`, `THINKING_MODELS`, `VISION_MODELS`, and `AUDIO_MODELS` lists are derived automatically from the capability flags; no manual update needed.
3. For HuggingFace models, the enum value is a tuple `(ModelSpec(...), ToolCallFormat.XXX[, think_opener_in_prompt])` because HF carries provider-specific extras alongside the spec.
4. Test with `pytest tests/test_models.py --client=<client> --model=<MODEL_NAME>`

**Other modalities** use the same pattern with their own spec type and enum: image (`HuggingFaceImageSpec`/`GeminiImageSpec` in `HuggingFaceImageModel`/`GeminiImageModel`), audio (`HuggingFaceAudioSpec` in `HuggingFaceAudioModel`), speech (`HuggingFaceSpeechSpec`/`OpenAISpeechSpec` in `HuggingFaceSpeechModel`/`OpenAISpeechModel`). For a one-off custom model, construct the spec object and pass it to the factory (`ImageClient(spec)`), the explicit escape hatch, rather than a string. Test with the modality's live suite (e.g. `pytest tests/test_images.py --image-client=hf --image-model=<NAME>`).

### Tool Calling Compatibility

- Tool calls use OpenAI format: `{"type": "function", "function": {"name": "...", "arguments": {...}}}`
- Some models (e.g., Llama 3.1) use `parameters` instead of `arguments`; this is normalized in `_prepare_tool_calls()` (the parse-and-store path, [aimu/models/_base/text.py](aimu/models/_base/text.py))
- Tool responses must be added to messages with role "tool" and include `tool_call_id`
- HuggingFace client uses `ToolCallFormat` enum to detect and parse different tool call response formats per model

### Message History Management

- System message is automatically prepended to `messages` on the first `chat()` call if `system_message` is set.
- `system_message` is **mutable at any time**; the setter is always live. Before the first `chat()` it seeds the value (injected on the first call). Mid-conversation, assigning it **swaps the `{"role": "system"}` entry in `messages` in place** (rewriting it, inserting one if absent, or removing it on `None`), so the change takes effect on the next request. The model is re-conditioned on the new prompt for every subsequent turn (useful for changing a chat's persona mid-conversation) while the conversation history is preserved. This means the active system prompt is always the system entry in `messages`; `self._system_message` is only a seed consulted once when `messages` is empty (see `aimu/models/_internal/chat_state.py`). Note the deliberate tradeoff: prior assistant turns remain in the transcript even though they were generated under the old prompt (a counterfactual but seamless record), and there is no longer a guard against silently re-conditioning a `ModelClient` shared by another agent's in-flight conversation; don't share a live-conversation client across agents that each set `system_message`. To change the prompt *and* drop history, use `reset(system_message="new")` instead.
- Message history persists across `chat()` calls on the same client instance.
- Use `ConversationManager` to persist conversations across sessions.

### Thinking Models

Models with `supports_thinking=True` support extended reasoning:
- Reasoning process stored in `self.last_thinking`
- Streamed as `StreamChunk(phase=StreamingContentType.THINKING, content=str)` chunks
- Filter via `include=["generating", "tool_calling"]` on `chat(...)` / `generate(...)`.
- Thinking content is also attached to the assistant message in `self.messages` under a non-standard `"thinking"` key, uniformly across all thinking providers (HuggingFace, Ollama, llama-cpp, OpenAI-compat local servers; sync + async) and on **every** assistant message that had reasoning, including the tool-call message in an agentic loop (the reasoning that preceded a tool call is attached to the message carrying `tool_calls` via the `msgs_before` index idiom, matching HuggingFace/Ollama). The key is omitted when a turn produced no reasoning. It is **inert metadata** for UIs (`examples/web/streamlit_chatbot.py` renders it) and persistence (`ConversationManager`): chat templates and the Anthropic/HuggingFace adapters read only `role`/`content`/`tool_calls`, and the OpenAI-compat and Ollama request paths (which otherwise forward unknown message-dict keys verbatim) call `strip_inert_keys()` first, so the key is never sent back to the model, and prior-turn reasoning is **not** re-fed on subsequent turns (matches Qwen3/Gemma/DeepSeek-R1 guidance). `last_thinking` holds only the latest turn; the message key holds per-turn reasoning. See [docs/explanation/thinking-and-context.md](docs/explanation/thinking-and-context.md).

### Message Provenance (agent-loop vs user input)

Framework-injected turns are marked with a non-standard `"provenance"` key so a replayed or persisted transcript can distinguish them from genuine user input (both are `{"role": "user", ...}` and were otherwise byte-for-byte identical). The `Agent` loop tags its `final_answer_prompt` turn `PROVENANCE_FINAL_ANSWER`; the personal-assistant example tags scheduler pushes `PROVENANCE_PROACTIVE`. (`PROVENANCE_CONTINUATION` is **legacy / no longer produced** — the agent used to inject a "continue" user turn between tool rounds but now continues via `chat()` with no user message, so nothing is injected; the constant is kept only so older persisted transcripts still parse.) Genuine user input and ordinary assistant turns are **left untagged** (absence = ordinary), so display logic is just `message.get(PROVENANCE_KEY)`. The constants and the key live in [aimu/models/_internal/message_meta.py](aimu/models/_internal/message_meta.py), re-exported from `aimu.models` and top-level `aimu` (`PROVENANCE_KEY`, `PROVENANCE_CONTINUATION`, `PROVENANCE_FINAL_ANSWER`, `PROVENANCE_PROACTIVE`). Tagging is done by index after the injecting `chat()` turn completes (the shared `_AgentLoopMixin._tag_injected_turn`, sync + async), mirroring how providers attach the `"thinking"` key by index, so the public `chat()` signature is untouched. `"provenance"` is in `INERT_MESSAGE_KEYS`, so `strip_inert_keys()` removes it (alongside `"thinking"` and the `ConversationManager` `"timestamp"`) from every OpenAI-compat and Ollama request; Anthropic/HuggingFace drop it by rebuilding the payload. The Streamlit examples hide/mute tagged turns when re-rendering history. Streaming needs no marker: the injected prompts are *input*, never emitted as chunks, so the problem is purely at the replayed-history level. Tests: `tests/test_provenance.py` (+ async mirror in `tests/test_aio_agents.py`).

`supports_thinking` is the universal flag; *how* reasoning is requested is provider-specific and handled inside each client (Anthropic `thinking` param, HuggingFace `enable_thinking` template kwarg, `<think>`-tag parsing for OpenAI-compat/Ollama/llama-cpp, `reasoning_effort` for OpenAI o-series). The Anthropic client distinguishes two request shapes via a `ThinkingStyle` enum carried on each `AnthropicModel` member (analogous to HuggingFace's `ToolCallFormat`):
- **`ThinkingStyle.ENABLED`**: `thinking={"type": "enabled", "budget_tokens": N}`; the model always thinks up to the budget. Opus 4.6, Sonnet 4.6, Haiku 4.5.
- **`ThinkingStyle.ADAPTIVE`**: `thinking={"type": "adaptive", "display": "summarized"}`; the model decides per request whether and how much to think (it may not think on simple prompts), and `temperature`/`top_p`/`top_k` are dropped. Required by Opus 4.7+ and Fable 5, which reject the `enabled` form with a 400. `display="summarized"` is set so reasoning still surfaces as `THINKING` chunks (the API default `"omitted"` yields empty thinking text).

Because adaptive models may legitimately emit no thinking on trivial prompts, tests that assert thinking-presence use a multi-step reasoning prompt and treat adaptive models like the other "doesn't consistently emit thinking" models in the tool-call assertions.

### Vision Input

Models with `supports_vision=True` accept images alongside the user prompt:
- Pass `images=[...]` to `chat()` (stateful, persisted in `self.messages`) **or** to `generate()` (stateless single-turn, not persisted). Each item may be a file path string, `pathlib.Path`, raw `bytes`, an `http(s)://` URL, or a `data:image/...;base64,...` URL
- Internally normalized to OpenAI-format content blocks (`{"type": "image_url", "image_url": {"url": ...}}`); for `chat()` these live in `self.messages`, for `generate()` they're built into a one-off request message and discarded. Helpers live in [aimu/models/_internal/image_input.py](aimu/models/_internal/image_input.py) (`_normalize_image`, `_build_user_content_blocks`, plus per-provider adapters)
- Vision support is validated once at the public layer: `_ChatStateMixin._require_vision()` raises `ValueError` for a non-vision model; called from `_append_user_turn()` (the `chat` path) and from `BaseModelClient.generate()` / `AsyncBaseModelClient.generate()` (the `generate` path)
- Per-provider adaptation is shared between `chat` and `generate`: `generate(images=...)` reuses the same pure adapters (`_build_user_content_blocks`, `_openai_blocks_to_anthropic`, `_ollama_split_message` via `OllamaClient._extract_ollama_images`, HF's `_extract_pil_images` template path) rather than duplicating per-provider request building
- Per-provider adaptation happens at request time without mutating `self.messages`:
  - `OpenAICompatClient` (and subclasses): pass-through (the `openai` SDK speaks `image_url` blocks natively)
  - `AnthropicClient`: `_openai_messages_to_anthropic` calls `_openai_blocks_to_anthropic` to rewrite `image_url` → `image` blocks (`base64` source for data URLs, `url` source for http(s))
  - `OllamaClient` (native): `_adapt_messages_for_ollama` is called before each `ollama.chat()` invocation to extract `image_url` content blocks into Ollama's message-level `images=[<bare base64>]` field. The `generate()` path uses `OllamaClient._extract_ollama_images()` to pass the same bare-base64 list to `ollama.generate(images=...)`. Only inline base64 (`data:image/...`) is supported; http(s) URLs raise `ValueError`
  - `HuggingFaceClient`: when the model has a processor (Gemma 3 / Gemma 4), `_apply_chat_template` decodes images via `_extract_pil_images` and passes them as `images=` to the processor; `image_url` blocks are rewritten to `{"type": "image"}` placeholders for the chat template
  - `LlamaCppClient`: pass-through; vision requires loading an `mmproj` projector via the new `chat_handler=` constructor kwarg (e.g. `Llava15ChatHandler(clip_model_path=...)`)
- Capability flag: each `ModelSpec` accepts `vision: bool` (default `False`). `BaseModelClient.is_vision_model` and `VISION_MODELS` classproperty mirror the thinking/tools equivalents
- `images=` is also threaded through `Agent.run()` (only the initial task turn carries images; continuation prompts are text-only), through `Agent.as_model_client().chat()`, and through workflow `Runner.run()` overrides (`Chain` forwards to step 0, `Router` forwards to the dispatched handler, `Parallel` forwards to every worker, `EvaluatorOptimizer` forwards only to the initial generator turn)

### Audio Input

`audio=` on `chat()` and `generate()` is the parallel surface for audio-capable text models (same stateful/stateless split as vision). Stored internally as OpenAI `input_audio` content blocks; the canonical format for audio in OpenAI message format.

- Pass `audio=[...]` to `chat()` (stateful, persisted in `self.messages`) or `generate()` (stateless single-turn, not persisted). Each item may be a file path string, `pathlib.Path`, raw `bytes`, an `https://` URL (fetched eagerly, since `input_audio` has no remote-URL field), or a `data:audio/...;base64,...` URL. Accepted format strings: `wav`, `mp3`, `ogg`, `flac`, `m4a`, `webm`.
- `images=` and `audio=` are mutually exclusive per turn. Passing both raises `ValueError` at the public layer.
- Audio support is validated once at the public layer: `_ChatStateMixin._require_audio()` raises `ValueError` for non-audio models; called from `_append_user_turn()` (the `chat` path) and from `BaseModelClient.generate()` (the `generate` path).
- Normalization lives in `aimu/models/_internal/audio_input.py` (mirrors `image_input.py`): `_normalize_audio()`, `_build_audio_content_blocks()`, `_extract_audio_arrays()` (HuggingFace processor path), `_replace_audio_with_placeholder()`.
- Per-provider adaptation at request time:
  - `OpenAICompatClient` (and subclasses, including OpenAI and Gemini): pass-through (`input_audio` is already OpenAI format)
  - `AnthropicClient`: `_openai_blocks_to_anthropic()` in `image_input.py` converts `input_audio` blocks to Anthropic `{"type": "audio", "source": {"type": "base64", "media_type": "audio/<fmt>", "data": "..."}}` format
  - `HuggingFaceClient`: `_extract_audio_arrays()` decodes blocks to float32 numpy arrays; `_replace_audio_with_placeholder()` rewrites them to `{"type": "audio"}` placeholders for the chat template; arrays passed as `audio=`/`sampling_rate=16000` to the processor
  - `OllamaClient`: all models have `audio=False`; `_require_audio()` raises before any Ollama-specific code runs
- Capability flag: `ModelSpec` accepts `audio: bool = False`. `BaseModelClient.is_audio_model` and `AUDIO_MODELS` classproperty mirror the vision equivalents.
- Audio-capable text models: OpenAI GPT-4o/4.1 series, Gemini 2.0/2.5, HuggingFace Gemma 4 E4B/12B, Nemotron-H-8B.
- `soundfile` (already in the `[hf]` extra) is used only for the HuggingFace extraction path; all other providers decode client-side or pass through base64 directly.

### Structured Output

`schema=` on `chat()` / `generate()` (sync and async) returns a validated instance of a **dataclass or Pydantic v2 model** instead of a string. One additive parameter; no new public class, no breakage.

- **Auto-escalate, branching on the static capability flag** (not on catching a runtime error):
  - `supports_structured_output=True` → native provider enforcement. The base converts the schema to a JSON Schema once ([aimu/models/_internal/structured.py](aimu/models/_internal/structured.py): `schema_to_json_schema`, reusing the `@tool` decorator's type map) and threads it as `response_format` (a dict) to the provider's `_chat`/`_generate`. If the native call itself errors, it **raises**; it does not cascade to parse (that would mask a provider error).
  - `supports_structured_output=False` → the base appends a JSON-Schema instruction to the prompt and parses the result. Providers never receive `response_format` on this path.
  - Either way the base coerces the returned text via `parse_json_response(text, schema)`; parse failure raises `ValueError`.
- **Per-provider native envelope** (only these three accept `response_format`; async mirrors + the `ModelClient`/`AsyncModelClient` factories forward it; HuggingFace/llama-cpp are parse-path and untouched):
  - `OpenAICompatClient` (OpenAI, Gemini): `response_format={"type":"json_schema","json_schema":{"name", "schema", "strict": False}}` via `_with_response_format()`. `strict: False` so arbitrary user schemas (optional fields, defaults) don't trip OpenAI strict-mode's subset rules.
  - `OllamaClient`: native `format=<json schema>` on `ollama.chat`/`generate`, grammar-enforced for *any* Ollama model.
  - `AnthropicClient`: forced single tool whose `input_schema` is the JSON Schema (`tool_choice={"type":"tool","name":...}`), returning the tool input as a JSON string for uniform coercion. Thinking is incompatible with a forced `tool_choice`, so the structured path routes around `_thinking_kwargs`.
- **Streaming (`schema=` + `stream=True`)**: supported. `chat()`/`generate()` return a `StreamChunk` iterator (thinking/generation live), ending in a terminal `StreamChunk(DONE, {"result": <validated object>})`; the object is also stored on `client.last_structured` once the stream is fully consumed (mirrors `last_usage`; proxied through `ModelClient`/`_AgenticView`/`FallbackClient` + async twins; cleared by `reset()`). The base `_chat_structured_streamed`/`_generate_structured_streamed` (sync in [aimu/models/_base/text.py](aimu/models/_base/text.py), async in [aimu/aio/_base.py](aimu/aio/_base.py)) call the provider with `stream=True`, accumulate the GENERATING text, `parse_json_response` it, set `last_structured`, and emit the terminal chunk (always, regardless of `include=`). Ollama threads `format=` into its streamed call and OpenAI-compat already folds `response_format` into `generate_kwargs` before the stream branch, so both stream thinking alongside the constrained answer; **Anthropic** uses a streaming forced-tool path (`_structured_call_streamed`) that yields the tool-input JSON as GENERATING (`input_json_delta`) with **no** THINKING (forced `tool_choice` ⊥ extended thinking; no regression vs non-streaming). `StreamChunk.is_done()` dispatches the terminal chunk.
- **Invariants/guards**: `self.messages` stays plain strings (the typed object is return-only). On Anthropic, `schema=` + active `tools=` raises (its native mode *is* a forced tool, which conflicts with action tools); OpenAI-compat and parse-path providers compose `schema=` with tools fine.
- **Capability flag**: `ModelSpec.structured_output: bool = False` → `supports_structured_output` property (on `_ChatStateMixin`) + `STRUCTURED_MODELS` classproperty. Set on the OpenAI, Gemini, Ollama (all models), and Anthropic catalogs.
- **`Agent.run(schema=...)`** (sync + async): pass a dataclass / Pydantic v2 model to make the run a single structured-output turn that returns a validated instance instead of running the tool-calling loop, for an agent whose job is to return a typed object (e.g. a critic's verdict). Mutually exclusive with the tool loop. With `stream=True` the run streams the structured turn (chunks + terminal `DONE` result; `_run_structured_streamed`, tagging chunks with the agent name). `EvaluatorOptimizer(verdict_schema=...)` uses the non-streamed form to get a typed `Verdict(passed, feedback)` from its evaluator.
- **Deferred**: a `strict=True` (native-or-raise) knob; native HuggingFace/llama-cpp enforcement (would need an `xgrammar`/`outlines` backend). `generate_json` remains the older prompt-and-parse-with-retries convenience.
- **Tests**: `tests/test_structured_api.py` (schema helper, base native/parse logic via fakes, OpenAI envelope, Ollama `format=`, Anthropic forced-tool + tools-conflict, guards, plain-messages invariant) and `tests/test_aio_structured_api.py` (async mirror).

### OpenAICompatClient Notes

- `OpenAICompatClient` uses the `openai` Python SDK with a configurable `base_url`; works with any server that speaks the OpenAI REST API
- Service-specific subclasses each define their own `Model` enum with service-appropriate model IDs and a default `base_url`
- Thinking model support uses `<think>...</think>` tag parsing (via `_split_thinking` and `_ThinkingParser` in [aimu/models/providers/_thinking.py](aimu/models/providers/_thinking.py)) since OpenAI-compat endpoints embed thinking in content rather than a dedicated field; `LlamaCppClient` uses the same shared utilities
- Pass `tools=openai.NOT_GIVEN` (not `tools=[]`) when no tools are available; some local servers reject an empty tools array
- Model ID format varies by service: LM Studio uses loaded model keys, Ollama uses `name:tag` format, HF Transformers Serve, vLLM, and SGLang use HuggingFace repo paths (e.g. `Qwen/Qwen3-8B`); llama-server uses GGUF filenames (the model field is ignored server-side, so these are capability-lookup only)
- Test with `pytest tests/test_models.py --client=lmstudio_openai` (or `ollama_openai`, `hf_openai`, `vllm_openai`, `llamaserver_openai`, `sglang_openai`)

### LlamaCppClient Notes

- `LlamaCppClient` loads a GGUF file in-process via `llama_cpp.Llama`; no external service or server required
- Constructor requires `model_path` (path to GGUF file); `model` enum specifies capability flags (`supports_tools`, `supports_thinking`, `supports_vision`)
- `n_gpu_layers=-1` offloads all layers to GPU (default); `n_gpu_layers=0` forces CPU-only inference
- `chat_format=None` (default) auto-detects from GGUF metadata; specify explicitly (e.g. `"chatml-function-calling"`) for older GGUFs that lack embedded templates
- Optional `chat_handler=` constructor kwarg for vision: pass a `Llava15ChatHandler(clip_model_path=...)` (or similar) to enable image input via `mmproj` projector files. Without it, vision-capable model enums still pass image_url blocks through but the model will error
- llama-cpp-python returns plain Python dicts, not OpenAI SDK objects; all response access uses dict indexing (`response["choices"][0]["message"]["content"]`)
- Tool calling requires both `supports_tools=True` on the model enum AND a compatible `chat_format`; modern GGUFs auto-detect this
- Thinking support works the same as `OpenAICompatClient`; `<think>...</think>` tags in content, parsed by shared utilities in `_thinking.py`
- Test with `pytest tests/test_models.py --client=llamacpp --model-path=/path/to/model.gguf`

### Image Generation

AIMU supports text-to-image generation via HuggingFace `diffusers` (`HuggingFaceImageClient`) and Google's Gemini Nano Banana (`GeminiImageClient`) as a **parallel surface** to the text chat client. The image hierarchy mirrors the text one (one base class, one factory class, per-provider concrete clients), but the modality interfaces are disjoint: `chat()` / `messages` / `StreamChunk` are meaningless for stateless text-to-image, so `BaseImageClient` is its own ABC rather than a subclass of `BaseModelClient`. The two surfaces live side-by-side: `aimu.client()` / `ModelClient` for text, `aimu.image_client()` / `ImageClient` for image.

- **[aimu/models/base.py](aimu/models/base.py)**: base classes (text ↔ image parallel):
  - `ImageSpec` (sibling to `ModelSpec`): minimal common spec with `id` and `max_prompt_tokens` (the text-encoder prompt budget, the image analog of `ModelSpec`'s capability flags; `77` = CLIP default, `256`/`512` for T5 models, `None` = no cap for cloud); hashed/equal by id only. Subclassed with `@dataclass(eq=False)` so the parent's id-only equality isn't shadowed by the dataclass-default per-field `__eq__`. `BaseImageClient` exposes `max_prompt_tokens`.
  - `HuggingFaceImageSpec(ImageSpec)`: adds `pipeline_class` (resolved lazily via `getattr(diffusers, name)`), `img2img_pipeline_class` (diffusers class for image-to-image; `None` for ad-hoc strings), `img2img_uses_strength` (bool, default `True`; `False` for unified pipelines like `Flux2KleinPipeline` that accept `image=` directly without a `strength` param), `default_steps`, `default_guidance`, `default_width`, `default_height`, `default_negative_prompt`, `pipeline_kwargs`.
  - `GeminiImageSpec(ImageSpec)`: adds `default_aspect_ratio`, `default_image_size`, `image_config_kwargs` (consumed by the SDK's `ImageConfig`); overrides `max_prompt_tokens` default to `None` (cloud models take natural-language prompts).
  - `ImageModel(Enum)` (sibling to `Model`): each member's value is an `ImageSpec` (or subclass); `__init__` sets `_value_=spec.id` and stores `self.spec=spec`. Provider enums (`HuggingFaceImageModel`, `GeminiImageModel`) inherit this.
  - `BaseImageClient(ABC)` (sibling to `BaseModelClient`): concrete `generate(prompt, *, num_images, format, output_dir, stream, preview_every, reference_image=None, **kwargs)` does `num_images` validation + format conversion via `encode_image()`. `reference_image` accepts the same input forms as vision input (path string, `pathlib.Path`, raw bytes, data URL, http(s) URL, or PIL Image); when set, the call becomes image-to-image. Subclasses implement `_generate(prompt, *, num_images, reference_image=None, **kwargs) -> list[Image]` returning PIL images.
- **[aimu/models/image_client.py](aimu/models/image_client.py)**: `ImageClient` factory (sibling to `ModelClient`) + `resolve_image_model_string()` + `resolve_image_model_enum()` (enum / `"provider:id"` string / bare enum-member name, parallel to the text `resolve_model_enum`). Accepts a provider enum / spec / `"provider:id"` string; dispatches to the right concrete client. String-form (`"hf:<repo>"`, `"gemini:<id_or_alias>"`) supports ad-hoc model ids not in the enums, by routing on the prefix before falling back to enum lookup.

- **[aimu/models/_internal/image_output.py](aimu/models/_internal/image_output.py)**: `encode_image(image, format, *, prompt="", output_dir=None)` is the shared format-conversion helper. Sibling to `image_input.py` (which decodes images for vision *input*); this one encodes images for diffusion *output*.
  - `format="pil"` → return PIL Image unchanged.
  - `format="bytes"` → PNG-encoded bytes.
  - `format="data_url"` → `data:image/png;base64,...`.
  - `format="path"` → save to `output_dir or paths.output/"images"`, filename `{YYYYMMDD-HHMMSS}-{sha1(prompt+ts)[:8]}.png`; mkdir parents on demand.
  - Unknown format raises `ValueError`.

- **[aimu/models/providers/hf/image.py](aimu/models/providers/hf/image.py)**: `HuggingFaceImageClient` + `HuggingFaceImageModel` enum.
  - `HuggingFaceImageModel` catalog: `SD_1_5`, `SDXL_BASE`, `SD_3_5_MEDIUM`, `FLUX_1_DEV`, `FLUX_1_SCHNELL`, `FLUX_2_KLEIN_4B`, `FLUX_2_KLEIN_9B`. Each member's value is a `HuggingFaceImageSpec`. The two Klein models use `Flux2KleinPipeline` (diffusers 0.37+), which is a unified pipeline: the same class handles txt2img (`image=None`) and img2img (`image=<PIL>`). Klein sets `img2img_uses_strength=False` because it conditions on the reference image directly rather than blending via a noise `strength` scalar.
  - `HuggingFaceImageClient(model, model_kwargs=None)` accepts a `HuggingFaceImageModel` member, a `HuggingFaceImageSpec`, or a `"hf:<repo_id>"` string (for ad-hoc HF models not in the enum; defaults to `DiffusionPipeline` auto-detect loader).
  - Lazy pipeline load on first `generate()` call: `pipeline_cls = getattr(diffusers, spec.pipeline_class); self._pipe = pipeline_cls.from_pretrained(spec.id, **kwargs)`. Placement is **memory-aware and automatic** via `_hf_device.auto_place_pipeline()`: it measures the loaded pipeline's size and each GPU's *free* memory (`torch.cuda.mem_get_info`, so other processes like a local LLM server are accounted for), then picks the cheapest fit: pin to the freest GPU → `enable_model_cpu_offload` (largest component fits) → `enable_sequential_cpu_offload` (always fits). Falls back to CUDA/MPS/CPU with no CUDA or missing `accelerate` (offload needs it). Override with `model_kwargs={"device": "cuda:1"}` (pin) or `model_kwargs={"device_map": ...}` (hand to diffusers/accelerate). `torch_dtype` also defaults per-device via `_hf_device.default_torch_dtype()` (bf16 on CUDA, fp16 on MPS, fp32 on CPU) unless the caller sets it; avoids `"auto"` silently resolving to fp32 (which doubles VRAM).
  - `generate(prompt, *, negative_prompt=None, width=None, height=None, num_inference_steps=None, guidance_scale=None, seed=None, num_images=1, format="pil", output_dir=None, stream=False, preview_every=None, reference_image=None, strength=None)`. Unset params fall back to `spec.default_*`. Seed plumbed through `torch.Generator`. Returns a single image (or list when `num_images > 1`) in the chosen format. When `reference_image` is provided, the client calls `_get_img2img_pipeline()` which lazily derives the img2img pipeline from the loaded txt2img pipeline via `from_pipe()` (shared weights, no extra VRAM). `strength` defaults to `0.75` for FLUX.1-style pipelines (not used for Klein, where `img2img_uses_strength=False`). `width` and `height` are dropped from the pipeline call in img2img mode; output size is derived from the reference image.
  - `stream=True` returns an `Iterator[StreamChunk]` with phase `IMAGE_GENERATING`. Implemented via `callback_on_step_end` on the diffusers pipeline: a background thread runs the pipeline, the callback pushes per-step chunks onto a `queue.Queue`, and the public generator drains the queue. `preview_every=N` opts into per-step latent-decoded previews: when set, the callback decodes via `pipe.vae.decode` every N steps and includes the PIL in the chunk (~50 to 200 ms per decode on GPU). Final chunks (one per image when `num_images > 1`) carry `final=True` and the encoded `result` per `format=`.
  - `import diffusers` is a hard module-load import so `HAS_HF_IMAGE` accurately reflects "the optional dep is installed" (matches the HF convention).

- **[aimu/__init__.py](aimu/__init__.py)**: top-level entry points parallel to `client()` / `chat()`:
  - `aimu.image_client(model=None, **kwargs) -> HuggingFaceImageClient`: one-line constructor.
  - `aimu.generate_image(prompt, *, model=None, format="pil", **kwargs)`: one-shot.
  - When `model` is omitted, image/audio/speech entry points read their env var (`AIMU_IMAGE_MODEL` / `AIMU_AUDIO_MODEL` / `AIMU_SPEECH_MODEL`) via `resolve_default_modality_model()`; **unset raises `ValueError`** (these modalities are never auto-selected, since a wrong silent pick is costly, and weights are never downloaded implicitly). The error message lists any locally available models (via the modality `available_*_models()` discovery probe) so the explicit choice is easy; discovery failing never masks the real error.
  - Both raise `ImportError` with a helpful install hint when `HAS_HF_IMAGE` is False.

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)**: chat integration via a built-in **streaming** `@tool`:
  - `generate_image(prompt: str)` is a **generator tool** (decorated with `@tool`): it yields `IMAGE_GENERATING` chunks during denoising and returns the saved file path as its final value. The tool-loop engine's `_dispatch_streamed` forwards the yielded chunks through the agent's stream so callers see live progress.
  - Backed by a lazy module-level singleton `_image_client` constructed on first call via `aimu.image_client()`. The model comes from the `AIMU_IMAGE_MODEL` env var (**required**; the tool raises `ValueError` if unset; no default is downloaded).
  - `make_image_tool(client, *, preview_every=None)` factory binds a fresh streaming tool to a caller-supplied image client, with optional intermediate-image previews every N denoising steps (HF only; Gemini ignores it).
  - `make_describe_image_tool(client, *, default_instruction=...)` builds a `describe_image(image_path, instruction=...)` tool bound to a **vision-capable chat client** (raises `ValueError` if `client.model.supports_vision` is False). The tool snapshots `client.messages` before the vision call and restores it afterwards, so the agent's conversation log isn't polluted with one-off image-Q&A turns; `use_tools=False` is passed to prevent recursive tool calls during the vision call. This is a factory rather than a lazy singleton because the most useful binding is to the *same* model the agent is already running: same knowledge, same provider, no extra API key. The Streamlit chatbot appends this tool to the agent's tool list automatically when the active chat model is vision-capable.
  - New subgroup `image = [generate_image]` next to `web`, `fs`, `compute`, `misc`. Appended to `ALL_TOOLS`.

- **Skill integration (deeper, optional)**: for `SkillAgent` users, drop a `SKILL.md` under `.agents/skills/image-generation/` listing `generate_image` and adding prompt-engineering guidance (composition, style descriptors, negative-prompt hints). Skills are filesystem-discovered, not shipped with the library; see `notebooks/18-image-generation.qmd` for a copyable example.

- **Async twin**: full mirror under `aimu.aio`:
  - **[aimu/aio/providers/hf_image.py](aimu/aio/providers/hf_image.py)**: `AsyncHuggingFaceImageClient` wraps a sync `HuggingFaceImageClient` (no second weight load). Properties (`model`, `spec`, `pipeline`, `model_kwargs`) delegate to `self._sync`. `generate()` is `async` and routes through `asyncio.to_thread`. Does *not* inherit `AsyncBaseModelClient`; diffusion has no chat lifecycle to inherit.
  - **[aimu/aio/image.py](aimu/aio/image.py)**: `aio.image_client(sync_client)` factory + `aio.generate_image(prompt, *, model, ...)` one-shot. The factory refuses direct enum / string / `HuggingFaceImageSpec` construction with a helpful error pointing at the wrap pattern; same shape as the HF/LlamaCpp refusal in `aimu/aio/_model_client.py:140-143`.
  - **[aimu/aio/tools/builtin.py](aimu/aio/tools/builtin.py)**: async `@tool async def generate_image(prompt: str) -> str` using its own lazy `AsyncHuggingFaceImageClient` singleton; `make_async_image_tool(client)` accepts either a sync `HuggingFaceImageClient` or an existing `AsyncHuggingFaceImageClient`. Re-exports the rest of `aimu.tools.builtin` so async agents get a complete namespace.

- **Cloud image provider: Google Nano Banana (`gemini-2.5-flash-image`)**:
  - **[aimu/models/providers/gemini/image.py](aimu/models/providers/gemini/image.py)**: `GeminiImageClient` + `GeminiImageModel` enum (`NANO_BANANA`, `NANO_BANANA_PREVIEW`). Sits next to the Gemini text client in `aimu/models/providers/gemini/`; both share the `aimu.image_client()` / `aimu.generate_image()` (image) and `aimu.client()` (text) surfaces.
  - **[aimu/models/base.py](aimu/models/base.py)**: `GeminiImageSpec` sits next to `HuggingFaceImageSpec`. Disjoint fields: cloud API has no `pipeline_class`, `default_steps`, or `default_guidance`. Carries `id`, `default_aspect_ratio`, `default_image_size`, `image_config_kwargs`. Hashed and equal by `id` only.
  - Uses `google.genai.Client().models.generate_content(model=..., contents=..., config=GenerateContentConfig(response_modalities=[Modality.IMAGE], image_config=ImageConfig(...)))`. Without a reference image, `contents=[prompt]` (bare string). With `reference_image`, `contents` becomes a `genai_types.Content` with two parts (text + `inlineData` PNG blob), enabling image editing. The response's first inline-data part is decoded to a PIL image; `encode_image()` handles the format conversion (`pil`/`bytes`/`path`/`data_url`), the same helper diffusers uses, no duplication.
  - Auth: reads `GOOGLE_API_KEY` from env (or `.env` via `python-dotenv`), or accepts `api_key=` in `model_kwargs`. Raises `RuntimeError` with a clear hint if neither is set.
  - `num_images > 1` issues N separate API calls (Nano Banana returns one image per call). Aspect ratio defaults to the spec's `default_aspect_ratio` if set; otherwise the model picks based on the prompt.
  - **Dispatch via `aimu.image_client()`**: `"gemini:nano-banana"` / `"gemini:gemini-2.5-flash-image"` / `GeminiImageModel.NANO_BANANA` / `GeminiImageSpec(...)` all route to `GeminiImageClient`; `"hf:..."` / `HuggingFaceImageModel.*` / `HuggingFaceImageSpec(...)` route to `HuggingFaceImageClient`. The factory raises `ImportError` with the install hint when the relevant `HAS_*` flag is False.
  - **Async**: `aio.image_client(sync_gemini_client)` returns an `AsyncGeminiImageClient` that wraps via `asyncio.to_thread`. Cloud calls aren't GIL/CUDA-bound so a native async path (`google.genai.aio`) would be slightly faster, but is kept as a follow-up for surface consistency with the in-process wrap pattern.
  - Lives under the `[google]` extra (`google-genai`, `Pillow`); `HAS_GEMINI_IMAGE` is set independently of `HAS_HF_IMAGE` so users who only want one of the two providers install accordingly. `HAS_HF_IMAGE` requires the `[hf]` extra (`diffusers`, `safetensors`, `Pillow`, plus torch/transformers shared with HF text).

- **Tests**: mirror the text-side `test_models.py` / `test_models_api.py` split:
  - `tests/test_images_api.py` (mock-only, single file for both providers): stubs `diffusers` in `sys.modules` (with a `_FakePipeline` that has `from_pipe()`, returning PIL images) and patches `genai.Client` per-test via the `fake_genai` fixture. Covers spec equality-by-id, the shared `encode_image` format matrix, `HuggingFaceImageClient` construction + string parsing + lazy load + per-call kwarg threading + seed plumbing + unknown-pipeline error, img2img via `reference_image=` (PIL/path/bytes input, `from_pipe()` caching, `strength` default, `width`/`height` omitted, streaming path, no-img2img-class error, `FLUX_2_KLEIN_4B` unified-pipeline path without `strength`), `GeminiImageClient` construction + alias resolution + API-key resolution + aspect-ratio threading + `reference_image` multipart-content path + backward-compat plain-string path, and the top-level `aimu.image_client()` / `aimu.generate_image()` dispatch for both providers. The `_install_diffusers_stub` helper is re-used by `test_image_tools.py` and `test_aio_image_api.py`.
  - `tests/test_image_tools.py` (mock-only): patches the singleton; asserts `generate_image.__tool_spec__` shape, `make_image_tool` returns a fresh tool bound to its client, env var honoured.
  - `tests/test_aio_image_api.py` (mock-only): refusal of direct enum/string construction for both `HuggingFaceImageClient` and `GeminiImageClient`, `AsyncHuggingFaceImageClient` / `AsyncGeminiImageClient` share state with wrapped sync clients, async tool detected via `__tool_is_async__`.
  - `tests/test_images.py` (live, opt-in): parametrized across image providers, the analog of `tests/test_models.py` for the image modality. Reads `--image-client` (`hf` / `gemini` / `all`) and `--image-model` (enum member name or `all`) from `conftest.py`; `helpers.resolve_image_model_params()` builds the parametrize list, `helpers.create_real_image_client()` constructs the concrete client. Cross-provider tests (PIL output, format matrix, num_images) run on every parametrized client; provider-specific tests (`seed=` reproducibility for HF, `aspect_ratio=` for Gemini) skip with a clear message when the active client doesn't support them. Skipped entirely when `--image-client` is omitted. Example: `pytest tests/test_images.py --image-client=hf --image-model=SD_1_5`.

### Audio Generation

AIMU supports text-to-audio generation via HuggingFace as a **parallel surface** to the text and image clients. The hierarchy mirrors the image one (one base class, one factory class, per-provider concrete clients), but the interface is disjoint from both text and image; `BaseAudioClient` is its own ABC. The two modality surfaces live side-by-side: `aimu.client()` / `ModelClient` for text, `aimu.image_client()` / `ImageClient` for image, `aimu.audio_client()` / `AudioClient` for audio.

Scope: music and sound generation (not TTS). Three pipeline families ship under `HuggingFaceAudioModel`: `MUSICGEN_SMALL/MEDIUM/LARGE` (token-autoregressive via `transformers`), `AUDIOLDM2` (latent diffusion, 16 kHz), `STABLE_AUDIO_OPEN` (latent diffusion, 44.1 kHz stereo).

Key files and their roles:

- **[aimu/models/base.py](aimu/models/base.py)**: extended with audio base types (parallel to image types):
  - `AUDIO_GENERATING` added to `StreamingContentType`. Content shape: `{"step": int, "total_steps": int, "final": bool, "result": Any, "duration_s": float}`.
  - `StreamChunk.is_audio_progress()` helper (parallel to `is_image_progress()`).
  - `AudioSpec` (id-only base, parallel to `ImageSpec`), `HuggingFaceAudioSpec(AudioSpec)` (adds `pipeline_type`, `default_duration_s`, `default_steps`), `AudioModel(Enum)` base, `BaseAudioClient(ABC)`.
  - `BaseAudioClient.generate(prompt, *, duration_s, num_inference_steps, num_audio, format, output_dir, seed, stream)`: validates args, dispatches to `_generate()` or `_generate_streamed()`, encodes output via `encode_audio()`.

- **[aimu/models/_internal/audio_output.py](aimu/models/_internal/audio_output.py)**: `encode_audio(audio, sample_rate, format, *, prompt, output_dir)` (parallel to `image_output.py`):
  - `"numpy"` → `(sample_rate, audio)` unchanged.
  - `"bytes"` → WAV bytes via `soundfile.write` into `BytesIO`.
  - `"data_url"` → `data:audio/wav;base64,...`.
  - `"path"` → saves WAV to `paths.output/"audio"/{timestamp}-{hash}.wav`, returns path string.

- **[aimu/models/providers/hf/audio.py](aimu/models/providers/hf/audio.py)**: `HuggingFaceAudioClient` + `HuggingFaceAudioModel` enum:
  - Lazy pipeline load (same pattern as `HuggingFaceImageClient`). Auto-detects CUDA/MPS via the shared `_hf_device.py` helpers; `model_kwargs={"device": "cuda:1"}` targets a specific GPU. No sharding default; audio models are small enough for one card.
  - Constructor accepts `HuggingFaceAudioModel` member, `HuggingFaceAudioSpec`, or `"hf:<repo_id>"` string (pipeline type inferred from known repo prefixes, defaulting to `"musicgen"`).
  - Three loader methods dispatched by `spec.pipeline_type`: `_load_musicgen()` (transformers `AutoProcessor` + `MusicgenForConditionalGeneration`), `_load_audioldm2()` (diffusers `AudioLDM2Pipeline`), `_load_stable_audio()` (diffusers `StableAudioPipeline`).
  - Three generator methods: `_run_musicgen()` (duration → `max_new_tokens = int(duration_s * 50)`), `_run_audioldm2()`, `_run_stable_audio()`. All return `list[(sample_rate, ndarray)]`.
  - Streaming for diffusers models: background thread + `queue.Queue` + `callback_on_step_end`, draining into `AUDIO_GENERATING` chunks. MusicGen yields one final chunk (no per-step API). No intermediate audio decode (unlike image latent previews); intermediate chunks carry step/total_steps/`final=False`/`result=None`.
  - `import soundfile` is a hard module-load import so `HAS_HF_AUDIO` accurately reflects installed state.

- **[aimu/models/audio_client.py](aimu/models/audio_client.py)**: `AudioClient` factory + `resolve_audio_model_string()` (parallel to `image_client.py`). Accepts model enum / spec / `"hf:<repo_id>"` string. Only `"hf:"` prefix registered; additional providers added as the surface grows.

- **[aimu/__init__.py](aimu/__init__.py)**: top-level entry points parallel to `client()` / `image_client()`:
  - `aimu.audio_client(model, **kwargs)`: one-line constructor.
  - `aimu.generate_audio(prompt, *, model, format="numpy", **kwargs)`: one-shot.
  - Both raise `ImportError` with install hint when `HAS_HF_AUDIO` is False.

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)**: `generate_audio` generator tool + `make_audio_tool()` factory (parallel to image equivalents):
  - `generate_audio(prompt: str)` is a generator `@tool` yielding `AUDIO_GENERATING` chunks and returning the saved file path. Backed by a lazy `_audio_client` singleton; model via `AIMU_AUDIO_MODEL` env var (**required**; the tool raises if unset; no default is downloaded).
  - `make_audio_tool(client, *, duration_s=None)`: binds a fresh tool to a caller-supplied client.
  - `make_tools(base_client, image_client=None, preview_every=None, audio_client=None, speech_client=None, memory_store=None, python_sandbox=False)`: `audio_client` and `speech_client` kwargs replace their respective singletons when provided; `memory_store` appends `make_memory_tools(store)` when set; `python_sandbox=True` appends `execute_python`.
  - New subgroup `audio = [generate_audio]` (parallel to `image`); included in `ALL_TOOLS`.

- **Async twin**: full mirror under `aimu.aio`:
  - **[aimu/aio/providers/hf_audio.py](aimu/aio/providers/hf_audio.py)**: `AsyncHuggingFaceAudioClient` wraps a sync `HuggingFaceAudioClient` via `asyncio.to_thread`. Properties delegate to the wrapped sync client; no second weight load.
  - **[aimu/aio/audio.py](aimu/aio/audio.py)**: `AsyncAudioClient` factory + `aio.audio_client(sync_client)` + `aio.generate_audio(prompt, *, model, ...)`. The factory refuses direct enum/string construction with a pointer to the wrap pattern.
  - **[aimu/aio/tools/builtin.py](aimu/aio/tools/builtin.py)**: re-exports sync tools from `aimu.tools.builtin`. Async-native `generate_image` lives here; the `generate_audio` tool in `aimu.tools.builtin` is a sync generator and is re-exported as-is (dispatched via `asyncio.to_thread` by the async agent).

- **Tests**:
  - `tests/test_audio_api.py` (mock-only): stubs `soundfile`, `transformers`, and diffusers audio pipelines in `sys.modules`. Covers spec equality, `encode_audio` format matrix, `HuggingFaceAudioClient` construction + string parsing + kwarg threading + seed plumbing + streaming (musicgen=1 final chunk; diffusers=N progress + 1 final), `AudioClient` factory dispatch, top-level `aimu.audio_client()` / `aimu.generate_audio()`. Exports `_install_audio_stubs()` for downstream test files.
  - `tests/test_audio_tools.py` (mock-only): `generate_audio.__tool_spec__` shape, `__tool_is_streaming__ is True`, singleton lifecycle, `AIMU_AUDIO_MODEL` env var, `make_audio_tool` factory.
  - `tests/test_aio_audio_api.py` (mock-only): wrap refusal, state sharing, async generate.
  - `tests/test_audio.py` (live, opt-in): `--audio-client` / `--audio-model` CLI flags; cross-model tests (numpy output, format matrix, `num_audio > 1`); model-specific tests (`seed=` for MusicGen, `num_inference_steps=` for diffusers). Skipped when `--audio-client` is omitted.

### Speech Generation

AIMU supports text-to-speech (TTS) generation as a **parallel surface** to the text, image, and audio clients. The hierarchy mirrors audio: one base ABC (`BaseSpeechClient`), one factory class (`SpeechClient`), per-provider concrete clients. Speech is distinct from audio: audio generates music and sounds from a descriptive prompt; speech converts literal text to spoken audio. `BaseSpeechClient` is scoped to TTS only; STT uses `BaseTranscriptionClient` (see "Transcription (Speech-to-Text)" below).

Two providers ship: **HuggingFace** for local generation (MMS-TTS, BARK), **OpenAI** for cloud TTS (tts-1, tts-1-hd). Both reuse `encode_audio()` for output; no new encoder needed.

Key files and their roles:

- **[aimu/models/base.py](aimu/models/base.py)**: extended with speech base types (parallel to audio types):
  - `SPEECH_GENERATING` added to `StreamingContentType`. Content shape: `{"chunk_index": int, "total_chunks": int | None, "final": bool, "result": Any}`. `total_chunks=None` for OpenAI streaming (unknown upfront); `total_chunks=N` for HuggingFace (single-pass).
  - `StreamChunk.is_speech_progress()` helper (parallel to `is_audio_progress()`).
  - `SpeechSpec` dataclass with `id`, `default_voice`, `default_speed`; id-only `__hash__` and `__eq__`.
  - `HuggingFaceSpeechSpec(SpeechSpec)` with `@dataclass(eq=False)`, adds `pipeline_type` (`"tts_pipeline"` | `"bark"`).
  - `OpenAISpeechSpec(SpeechSpec)` with `@dataclass(eq=False)`.
  - `SpeechModel(Enum)` base: `__init__` sets `_value_ = spec.id` and stores `self.spec`.
  - `BaseSpeechClient(ABC)`: concrete `generate(text, *, voice, speed, num_audio, format, output_dir, stream)` validates args, resolves voice/speed from spec defaults, delegates to abstract `_generate()`, encodes output via `encode_audio()`.

- **[aimu/models/providers/hf/speech.py](aimu/models/providers/hf/speech.py)**: `HuggingFaceSpeechClient` + `HuggingFaceSpeechModel` enum:
  - `HuggingFaceSpeechModel` members: `MMS_TTS_ENG` (`facebook/mms-tts-eng`, `tts_pipeline`), `SPEECHT5` (`microsoft/speecht5_tts`, `speecht5`), `BARK` (`suno/bark`, `bark`).
  - Lazy pipeline load on first `generate()` call; auto-moves to CUDA/MPS via the shared `_hf_device.py` helpers. `model_kwargs={"device": "cuda:1"}` targets a specific GPU (no sharding default; TTS models are small). The SpeechT5/BARK loaders move to the accelerator by default; the MMS-TTS `pipeline` stays on CPU unless a `device` hint is given.
  - Constructor accepts `HuggingFaceSpeechModel` member, `HuggingFaceSpeechSpec`, or `"hf:<repo_id>"` string (pipeline type inferred from known repo prefixes).
  - Three loader methods: `_load_tts_pipeline()` (HuggingFace `pipeline("text-to-speech", ...)`), `_load_speecht5()` (`SpeechT5ForTextToSpeech` + `SpeechT5HifiGan` vocoder + default CMU Arctic speaker embedding from `datasets`), `_load_bark()` (`BarkProcessor` + `BarkModel`). All return `(sample_rate, np.ndarray)`.
  - SpeechT5 voice selection: `voice=None` uses the pre-loaded default embedding (CMU Arctic index 7306); `voice="N"` loads index N from the xvectors dataset (cached on the client after first lookup). Sample rate: 16 kHz.
  - `import soundfile` is a hard module-load import so `HAS_HF_SPEECH` accurately reflects installed state.

- **[aimu/models/providers/openai/speech.py](aimu/models/providers/openai/speech.py)**: `OpenAISpeechClient` + `OpenAISpeechModel` enum:
  - `OpenAISpeechModel` members: `TTS_1` (`tts-1`), `TTS_1_HD` (`tts-1-hd`).
  - Auth via `OPENAI_API_KEY` env var. Uses `openai.audio.speech.create()` with `response_format="pcm"` (raw 24 kHz 16-bit PCM), decoded to `float32` via `np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0`.
  - Streaming via `client.audio.with_streaming_response.speech.create(...)`: yields HTTP byte chunks as `SPEECH_GENERATING` progress chunks with `total_chunks=None`.

- **[aimu/models/speech_client.py](aimu/models/speech_client.py)**: `SpeechClient` factory + `resolve_speech_model_string()` (parallel to `audio_client.py`). Accepts model enum / spec / `"provider:model_id"` string; dispatches on `"hf:"` or `"openai:"` prefix.

- **[aimu/__init__.py](aimu/__init__.py)**: top-level entry points:
  - `aimu.speech_client(model, **kwargs)`: one-line constructor.
  - `aimu.generate_speech(text, *, model, format="path", **kwargs)`: one-shot.
  - Both raise `ImportError` with install hint when neither `HAS_HF_SPEECH` nor `HAS_OPENAI_SPEECH` is True.

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)**: `generate_speech` generator tool + `make_speech_tool()` factory:
  - `generate_speech(text: str)` is a generator `@tool` yielding `SPEECH_GENERATING` chunks and returning the saved WAV path. Backed by a lazy `_speech_client` singleton; model via `AIMU_SPEECH_MODEL` env var (**required**; the tool raises if unset; no default is downloaded).
  - `make_speech_tool(client, *, voice: Optional[str] = None, speed: Optional[float] = None)`: binds a fresh tool to a caller-supplied client.
  - `make_tools(base_client, image_client=None, preview_every=None, audio_client=None, speech_client=None, memory_store=None)`: `speech_client` kwarg replaces the default singleton when provided; `memory_store` appends `make_memory_tools(store)` when set.
  - Subgroup `speech = [generate_speech]` (parallel to `audio`, `image`); included in `ALL_TOOLS`.

- **Async twin**: full mirror under `aimu.aio`:
  - **[aimu/aio/providers/hf_speech.py](aimu/aio/providers/hf_speech.py)**: `AsyncHuggingFaceSpeechClient` wraps a sync `HuggingFaceSpeechClient` via `asyncio.to_thread`. Properties delegate to the wrapped sync client; no second weight load.
  - **[aimu/aio/providers/openai_speech.py](aimu/aio/providers/openai_speech.py)**: `AsyncOpenAISpeechClient` wraps a sync `OpenAISpeechClient` via `asyncio.to_thread`.
  - **[aimu/aio/speech.py](aimu/aio/speech.py)**: `AsyncSpeechClient` factory + `aio.speech_client(sync_client)` + `aio.generate_speech(text, *, model, ...)`. The factory refuses direct enum/string construction with a pointer to the wrap pattern. `generate_speech` accepts both existing sync clients and enum/string model identifiers (mirrors sync surface).

- **Optional dep flags**: `HAS_HF_SPEECH` (hard `import soundfile` at top of `hf_speech_client.py`), `HAS_OPENAI_SPEECH` (hard `import openai` at top of `openai_speech_client.py`). Both set independently so a partial install works.

- **Tests**:
  - `tests/test_speech_api.py` (mock-only): stubs `soundfile`, `transformers`, and `openai` in `sys.modules`. Covers spec equality, `encode_audio` reuse, `HuggingFaceSpeechClient` construction + pipeline dispatch + lazy load, `OpenAISpeechClient` construction + PCM decode + streaming, `SpeechClient` factory dispatch, top-level `aimu.speech_client()` / `aimu.generate_speech()`. Exports `_install_speech_stubs()` for downstream test files.
  - `tests/test_speech_tools.py` (mock-only): `generate_speech.__tool_spec__` shape, `__tool_is_streaming__ is True`, singleton lifecycle, `AIMU_SPEECH_MODEL` env var, voice/speed threading in `make_speech_tool`.
  - `tests/test_aio_speech_api.py` (mock-only): wrap refusal, state sharing, async generate, streaming path.
  - `tests/test_speech.py` (live, opt-in): `--speech-client` / `--speech-model` CLI flags; cross-model tests (path output, format matrix); model-specific tests (voice param, speed param). Skipped when `--speech-client` is omitted.

### Transcription (Speech-to-Text)

AIMU supports speech-to-text (STT) as a **parallel surface** to TTS (`BaseSpeechClient`). The hierarchy follows the same pattern: one base ABC (`BaseTranscriptionClient`), one factory class (`TranscriptionClient`), per-provider concrete clients. This surface is disjoint from the `audio=` parameter on text models, which handles audio analysis/QA by audio-capable chat models (GPT-4o, Gemini, etc.); the transcription surface uses dedicated ASR models (the Whisper family, gpt-4o-transcribe) optimised for accurate transcription.

Two providers ship: **OpenAI** for cloud ASR (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe), **HuggingFace** for local ASR (Whisper tiny/base/small/medium/large-v3, distil-whisper/distil-large-v3).

Key files and their roles:

- **[aimu/models/base.py](aimu/models/base.py)**: extended with transcription base types:
  - `TranscriptionSpec` dataclass with `id`, `default_language`, `supports_timestamps`, `supports_translation`; id-only `__hash__` and `__eq__`.
  - `HuggingFaceTranscriptionSpec(TranscriptionSpec)` with `@dataclass(eq=False)` (no additional fields in the first catalog).
  - `OpenAITranscriptionSpec(TranscriptionSpec)` with `@dataclass(eq=False)`.
  - `TranscriptionModel(Enum)` base: `__init__` sets `_value_ = spec.id` and stores `self.spec`.
  - `BaseTranscriptionClient(ABC)`: concrete `transcribe(audio, *, language, response_format, prompt, temperature) -> str | dict` validates args, normalises `audio` input (same forms as `audio=` on `chat()`: file path, raw bytes, `https://` URL, `data:audio/...` URL), delegates to abstract `_transcribe()`.
  - `response_format` options: `"text"` (default, plain string), `"json"` (dict with `text` key), `"verbose_json"` (dict with `text`, `segments` start/end/text, `language`, `duration`). `"verbose_json"` requires `supports_timestamps=True` on the spec.

- **[aimu/models/providers/openai/transcription.py](aimu/models/providers/openai/transcription.py)**: `OpenAITranscriptionClient` + `OpenAITranscriptionModel` enum:
  - `OpenAITranscriptionModel` members: `WHISPER_1` (`whisper-1`, timestamps+translation), `GPT_4O_TRANSCRIBE` (`gpt-4o-transcribe`, timestamps only), `GPT_4O_MINI_TRANSCRIBE` (`gpt-4o-mini-transcribe`, timestamps only).
  - Auth via `OPENAI_API_KEY` env var. Uses `openai.audio.transcriptions.create()` from the same `openai` SDK as `OpenAICompatClient`.

- **[aimu/models/providers/hf/transcription.py](aimu/models/providers/hf/transcription.py)**: `HuggingFaceTranscriptionClient` + `HuggingFaceTranscriptionModel` enum:
  - `HuggingFaceTranscriptionModel` members: `WHISPER_TINY` (`openai/whisper-tiny`), `WHISPER_BASE` (`openai/whisper-base`), `WHISPER_SMALL` (`openai/whisper-small`), `WHISPER_MEDIUM` (`openai/whisper-medium`), `WHISPER_LARGE_V3` (`openai/whisper-large-v3`), `DISTIL_WHISPER_LARGE_V3` (`distil-whisper/distil-large-v3`). All have `supports_timestamps=True, supports_translation=True`.
  - Lazy pipeline load on first `transcribe()` call via `transformers.pipeline("automatic-speech-recognition", ...)`; auto-moves to CUDA/MPS via the shared `_hf_device.py` helpers.
  - Module-level weight caching via `_model_registry` keyed on `(spec.id, *sorted_model_kwargs)`, the same pattern as all other HF clients.

- **[aimu/models/transcription_client.py](aimu/models/transcription_client.py)**: `TranscriptionClient` factory + `resolve_transcription_model_string()` (parallel to `speech_client.py`). Accepts model enum / spec / `"provider:model_id"` string; dispatches on `"hf:"` or `"openai:"` prefix.

- **[aimu/__init__.py](aimu/__init__.py)**: top-level entry points:
  - `aimu.transcription_client(model=None, **kwargs)`: one-line constructor; reads `AIMU_TRANSCRIPTION_MODEL` env var when `model=` is omitted.
  - `aimu.transcribe(audio, *, model=None, **kwargs)`: one-shot.
  - Both raise `ImportError` with install hint when neither `HAS_HF_TRANSCRIPTION` nor `HAS_OPENAI_TRANSCRIPTION` is True.

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)**: `transcribe_audio` tool + `make_transcription_tool()` factory:
  - `transcribe_audio(audio_path: str) -> str` is a `@tool`-decorated function returning the transcribed text. Backed by a lazy `_transcription_client` singleton; model via `AIMU_TRANSCRIPTION_MODEL` env var (**required**; the tool raises if unset; no default is downloaded).
  - `make_transcription_tool(client)`: binds a fresh tool to a caller-supplied client.
  - Subgroup `transcription = [transcribe_audio]` (parallel to `audio`, `speech`, `image`); included in `ALL_TOOLS`.

- **Async twin**: full mirror under `aimu.aio`:
  - `AsyncTranscriptionClient` wraps a sync `BaseTranscriptionClient` via `asyncio.to_thread`. Properties delegate to the wrapped sync client; no second weight load.
  - `aio.transcription_client(sync_client)` factory refuses direct enum/string construction with a pointer to the wrap pattern (same as every other aio modality).
  - `await aio.transcribe(audio, *, model, ...)` one-shot (same signature as sync `aimu.transcribe()` plus `await`).

- **Optional dep flags**: `HAS_HF_TRANSCRIPTION` (guarded `transformers` import in `hf/transcription.py`), `HAS_OPENAI_TRANSCRIPTION` (guarded `openai` import in `openai/transcription.py`). Both set independently so a partial install works.

- **Tests**:
  - `tests/test_transcription_api.py` (mock-only): stubs `transformers` and `openai` in `sys.modules`. Covers spec equality, accepted audio forms, `HuggingFaceTranscriptionClient` construction + lazy load + weight caching + `verbose_json` path, `OpenAITranscriptionClient` construction + language threading + `response_format` options, `TranscriptionClient` factory dispatch, top-level `aimu.transcription_client()` / `aimu.transcribe()`.
  - `tests/test_transcription_tools.py` (mock-only): `transcribe_audio.__tool_spec__` shape, singleton lifecycle, `AIMU_TRANSCRIPTION_MODEL` env var, `make_transcription_tool` factory.
  - `tests/test_aio_transcription_api.py` (mock-only): wrap refusal, state sharing, async transcribe.
  - `tests/test_transcription.py` (live, opt-in): `--transcription-client` / `--transcription-model` CLI flags; cross-model tests (text output, `verbose_json`, language hint). Skipped when `--transcription-client` is omitted.

### Embeddings (Text-to-Vector)

AIMU provides a dedicated text-embedding surface via `aimu.embedding_client()` and `aimu.embed()`, a **parallel surface** to the other modality clients. Embeddings map text to fixed-length vectors (semantic similarity, clustering, retrieval, semantic memory). The interface is disjoint from chat (no message history, no streaming), so `BaseEmbeddingClient` is its own ABC.

Three providers ship: **OpenAI** (cloud), **Ollama** (local server), **HuggingFace** (local, via `sentence-transformers`).

- **[aimu/models/_base/embedding.py](aimu/models/_base/embedding.py)**: base types (parallel to `_base/audio.py`):
  - `EmbeddingSpec` (id-only equality/hash): `id`, `dimensions`, `max_input_tokens`. Subclasses `OpenAIEmbeddingSpec`, `OllamaEmbeddingSpec`, `HuggingFaceEmbeddingSpec` (adds `normalize: bool = True`), all `@dataclass(eq=False)`.
  - `EmbeddingModel(Enum)` base: `__init__` sets `_value_ = spec.id` and stores `self.spec`.
  - `BaseEmbeddingClient(ABC)`: concrete `embed(texts, **kwargs)` normalizes a single `str` to one vector (`list[float]`) and a list to a list of vectors (`list[list[float]]`, order preserved); empty list → `[]`; non-string items raise `ValueError`. Delegates to abstract `_embed(texts: list[str], **kwargs) -> list[list[float]]`. Exposes `dimensions` (from the spec).
- **[aimu/models/providers/openai/embedding.py](aimu/models/providers/openai/embedding.py)**: `OpenAIEmbeddingClient` + `OpenAIEmbeddingModel` (`TEXT_EMBEDDING_3_SMALL/LARGE`, `TEXT_EMBEDDING_ADA_002`). Uses `openai.embeddings.create()`; reads `OPENAI_API_KEY`. `dimensions=` (OpenAI's vector-truncation param) is forwarded via `embed(..., dimensions=N)`.
- **[aimu/models/providers/ollama.py](aimu/models/providers/ollama.py)**: `OllamaEmbeddingClient` + `OllamaEmbeddingModel` (`nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, `all-minilm`). Uses `ollama.embed()`; pulls the model on construction (same as `OllamaClient`).
- **[aimu/models/providers/hf/embedding.py](aimu/models/providers/hf/embedding.py)**: `HuggingFaceEmbeddingClient` + `HuggingFaceEmbeddingModel` (MiniLM-L6-v2, BGE small/base/large-en-v1.5, GTE-large, E5-large-v2, mxbai-embed-large-v1). Backed by `sentence_transformers.SentenceTransformer` so each model's own pooling/normalization config is honoured (avoids hand-rolled pooling that is silently wrong per model). Lazy load + module-level `_model_registry` weight cache (same pattern as the other HF clients; cleared by `aimu.clear_hf_cache()`, which includes `models.providers.hf.embedding`). `_embed` defaults `normalize_embeddings` to `spec.normalize` (overridable per call). Hard `import sentence_transformers` so `HAS_HF_EMBEDDING` reflects installed state. Retrieval-tuned models (BGE / E5) expect `"query: "` / `"passage: "` prefixes for asymmetric retrieval; the client does not auto-prepend, so pass prefixed strings when needed.
- **[aimu/models/embedding_client.py](aimu/models/embedding_client.py)**: `EmbeddingClient` factory + `resolve_embedding_model_string()` (mirrors `transcription_client.py`). Dispatches on `"openai:"` / `"ollama:"` / `"hf:"`.
- **[aimu/__init__.py](aimu/__init__.py)**: `aimu.embedding_client(model=None, **kwargs)` and `aimu.embed(texts, *, model=None, **kwargs)`. When `model` is omitted, reads `AIMU_EMBEDDING_MODEL` via `resolve_default_modality_model()`; unset raises `ValueError` listing any locally available embedders (running Ollama → HF cache, via `available_embedding_models()`); never auto-selected, since a mismatched embedder silently corrupts a persisted vector store; no implicit download.
- **Memory integration**: `SemanticMemoryStore(embedding_client=...)` adapts the client to a ChromaDB `EmbeddingFunction` (`_EmbeddingClientFunction`, which returns `NotImplemented` from `get_config()` to skip config persistence). Default `None` keeps ChromaDB's built-in embedder, unchanged behaviour. A custom embedding model isn't persisted in the collection config, so reopen a persistent store with the same `embedding_client=`.
- **Async** ([aimu/aio/embedding.py](aimu/aio/embedding.py)): `aio.embedding_client(sync_client)` wraps any sync `BaseEmbeddingClient` via `asyncio.to_thread` (one provider-agnostic wrapper, since all embedding clients share `embed()`); refuses enum/string construction with a pointer to the wrap pattern. `aio.embed(texts, *, model=...)` accepts a sync client or builds one via `aimu.embedding_client(model)`.
- **Optional dep flags**: `HAS_OPENAI_EMBEDDING`, `HAS_OLLAMA_EMBEDDING`, `HAS_HF_EMBEDDING` (the last requires `sentence-transformers`, added to the `[hf]` extra). `available_embedding_clients()` returns installed provider clients.
- **Tests**: `tests/test_embeddings_api.py` (mock-only; HF tests stub `sentence_transformers` and import the provider fresh), `tests/test_aio_embeddings_api.py` (wrap refusal, state sharing, async embed), `tests/test_embeddings.py` (live, opt-in; `--embedding-client` / `--embedding-model` flags).

### Token Usage Surfacing

`client.last_usage` holds token counts for the most recent `chat()` / `generate()` as `{"input_tokens", "output_tokens", "total_tokens"}`, or `None` when the provider/server omitted usage. Normalizers live in [aimu/models/_internal/usage.py](aimu/models/_internal/usage.py) (`usage_from_openai` / `usage_from_anthropic` / `usage_from_ollama`); each provider sets `self.last_usage` after its final response. Captured for Anthropic, OpenAI-compat (incl. OpenAI/Gemini/local servers), and Ollama, on both sync and async surfaces; delegated through the `ModelClient` / `AsyncModelClient` wrappers (mirroring `last_thinking`). For Anthropic, `cache_creation_input_tokens`/`cache_read_input_tokens` are added when the response reports them (prompt caching; see the `cache_prompt` flag in Cloud Provider Client Notes).

**Streaming (P1-B):** `last_usage` also populates after a streamed call, but **only once the stream is fully consumed** (the generator captures usage when it reaches the terminal chunk/event; reading mid-stream yields `None`). Per provider: OpenAI-compat passes `stream_options={"include_usage": True}` and reads the terminal usage chunk in `_iter_stream` / the `_chat_streamed` first-pass loop (both guard the empty-`choices` chunk against `IndexError`); Ollama reads the final streamed part's eval counts; Anthropic reads `stream.get_final_message().usage` after iteration (`get_final_message()` is a coroutine on the **async** client → `await`ed there, plain call on sync). Like non-streaming, the **final** turn's counts win when a streamed chat makes a tool-follow-up call. A server that doesn't report usage (or doesn't support `stream_options`) leaves it `None` (no regression). `reset()` clears it. In-process providers (HuggingFace, LlamaCpp) expose no streaming counts and leave it `None`. Token counts only; dollar cost is intentionally not computed (it would need a maintained per-model price table). `client.last_structured` (the validated object from a `schema=` call) follows the same "populated only after the stream is fully consumed" contract on the streamed structured-output path (see Structured Output).

### Resilience (timeout / retries)

Networked model clients accept `timeout: Optional[float]` and `max_retries: Optional[int]` constructor kwargs, forwarded **verbatim** to the underlying SDK (the names match the `anthropic` / `openai` SDKs exactly; no mapping). They flow through the `ModelClient` / `AsyncModelClient` factories unchanged (the factories already thread `**kwargs`). A shared helper [aimu/models/_internal/sdk_config.py](aimu/models/_internal/sdk_config.py) `sdk_client_kwargs(timeout, max_retries)` builds the forward dict, **omitting unset values** so each SDK's defaults stay intact when the caller doesn't opt in. AIMU implements **no** retry/backoff itself; this is pure passthrough (the P0-C roadmap item).

- **Anthropic / OpenAI / Gemini / local OpenAI-compat servers**: both kwargs forwarded to `anthropic.Anthropic` / `openai.OpenAI` (sync) and their `Async*` twins. The local-server subclasses inherit via `**kwargs` (sync) / `super()` forwarding (async). `OpenAIClient` / `GeminiClient` add the params explicitly (no `**kwargs` in their signatures).
- **Ollama**: `timeout` only, forwarded to `ollama.Client(timeout=...)` / `ollama.AsyncClient(timeout=...)` (httpx). The **sync** `OllamaClient` now holds an `ollama.Client` instance (`self._client`) and calls `self._client.chat/generate/pull` instead of the module-level `ollama.*` functions (the async client already used `ollama.AsyncClient`). Ollama has no native request-retry, so passing `max_retries` raises `ValueError` naming the `ollama-openai` provider (which routes through the OpenAI SDK and does support it). `OllamaEmbeddingClient` is out of scope and still uses module-level `ollama.*`.
- **In-process providers (HuggingFace, LlamaCpp)**: not networked; their constructors don't accept these kwargs, so passing `timeout`/`max_retries` raises `TypeError` (documented networked-only). The aio HF/llamacpp wrappers wrap an existing sync client, so any config lives on that sync client.
- Tests: `tests/test_resilience_api.py` (sync), `tests/test_aio_resilience_api.py` (async); both monkeypatch the SDK constructor to assert forwarding, the omit-when-unset path, the Ollama `max_retries` guard, and the in-process `TypeError`.

### Fallback / failover (`FallbackClient`)

`FallbackClient` ([aimu/models/fallback.py](aimu/models/fallback.py)) wraps an **ordered list** of `BaseModelClient`s and fails over to the next on error: the first that answers wins; a client raising an exception in `retry_on` (default `(Exception,)`) hands off to the next; when all fail, `FallbackExhaustedError` is raised with the last error as `__cause__` and all `(client, exception)` pairs on `.errors`. It is the P0-B roadmap item, the principle-aligned answer to provider failover (one composable class, not a gateway). The async twin is `aio.AsyncFallbackClient` ([aimu/aio/fallback.py](aimu/aio/fallback.py)). Exported from `aimu`, `aimu.models`, and `aimu.aio`.

- **It is a `BaseModelClient`**, so it composes everywhere a plain client does (`Agent`, workflows, `Benchmark`, `agent.as_model_client()`) with no failover-specific wiring. This is the payoff of "composability through uniform interfaces."
- **State model:** `FallbackClient` holds the canonical conversation (`messages` / `system_message` / `tools`); `_load_state` loads it into whichever inner client runs (via the inner client's public `reset`/`messages`/`tools` setters, so it works for concrete clients **and** `ModelClient` wrappers), and `_adopt_state` adopts the winning client's resulting state. This preserves multi-turn history across a failover. Each attempt gets a fresh `deepcopy` of the canonical messages, so a failed client's partial appends don't leak into the next attempt.
- **Implementation:** overrides the **public** `chat`/`generate` (so each inner client's own `chat()` applies its include-filter / schema / tool loop); the abstract `_chat`/`_generate` are unreachable stubs. Construction + state plumbing live in a shared `_FallbackStateMixin` reused by the sync and async classes; `FallbackClient.__init__` mirrors `BaseModelClient.__init__` (the base `__init__` is abstract/provider-specific, so it isn't called).
- **Capability flags** (`is_thinking_model`, `model`, etc.) reflect the **first** client; document that one fallback set should use capability-compatible clients.
- **Streaming** fails over only *before the first chunk is emitted* (buffer first chunk via `next`/`__anext__`); a mid-stream error propagates rather than replaying. Pure policy layer (no backoff/sleep); pair with per-client `timeout`/`max_retries` for in-SDK retry plus cross-provider failover.
- Tests: `tests/test_fallback_api.py` (sync, incl. an `Agent`-composition test), `tests/test_aio_fallback_api.py` (async).

### Cloud Provider Client Notes

- **`OpenAIClient`** (`aimu[openai_compat]`): thin subclass of `OpenAICompatClient` pointing at `https://api.openai.com/v1`; reads `OPENAI_API_KEY`. Overrides `_update_generate_kwargs` to rename `max_tokens → max_completion_tokens` and force `temperature=1` for o-series models (o1/o3/o4 prefix). Lives in [aimu/models/providers/openai/text.py](aimu/models/providers/openai/text.py).
- **`AnthropicClient`** (`aimu[anthropic]`): full implementation using the native `anthropic` SDK; reads `ANTHROPIC_API_KEY`. `self.messages` is always stored in OpenAI format; conversion to Anthropic format (`_openai_messages_to_anthropic`) and tool format conversion (`_openai_tools_to_anthropic`) happen at call time. When the model calls tools, `_chat` stores the assistant message with Anthropic's **real** `tool_use` block IDs directly (via `_record_tool_calls`), so the engine's tool-result messages match them without a later id-patching pass. Thinking is native (not `<think>` tag parsing) and `_thinking_kwargs()` builds the request per the model's `ThinkingStyle`: `ENABLED` → `{"type": "enabled", "budget_tokens": N}` (Opus 4.6, Sonnet 4.6, Haiku 4.5); `ADAPTIVE` → `{"type": "adaptive", "display": "summarized"}` with sampling params stripped (Opus 4.7, Opus 4.8, Fable 5, which 400 on the `enabled` form). Catalog: `CLAUDE_FABLE_5`, `CLAUDE_OPUS_4_8`, `CLAUDE_OPUS_4_7`, `CLAUDE_OPUS_4_6`, `CLAUDE_SONNET_4_6`, `CLAUDE_HAIKU_4_5`. See "Thinking Models" above for the `ThinkingStyle` rationale. **Opt-in prompt caching** (P1-A): the `cache_prompt: bool = False` constructor kwarg (Anthropic-only; threads via `aimu.client("anthropic:...", cache_prompt=True)`) makes the two format adapters mark the **system prompt** (returned as a `[{type:text, ..., cache_control:ephemeral}]` block list instead of a bare string) and the **last tool** (one breakpoint caches the whole tools array) with `cache_control: {"type":"ephemeral"}`. Because the markers live in the adapters, every request path is covered (chat, tool-follow-up, streaming, structured) with no per-call-site edits; the existing `system=system_str if system_str else NOT_GIVEN` guards still hold (a non-empty list is truthy). Below the provider minimum (1024 tokens; 2048 for Haiku) the API silently skips caching, so the flag is safe on. `usage_from_anthropic` adds `cache_creation_input_tokens`/`cache_read_input_tokens` to `last_usage` when present. Caching conversation *messages* is out of scope (they grow each turn). Tests: `tests/test_anthropic_caching.py`.
- **`GeminiClient`** (`aimu[openai_compat]`): thin subclass of `OpenAICompatClient` pointing at `https://generativelanguage.googleapis.com/v1beta/openai/`; reads `GOOGLE_API_KEY`. Gemini 2.5 thinking models emit `<think>` tags on this endpoint, so `_ThinkingParser` works as-is. Lives in [aimu/models/providers/gemini/text.py](aimu/models/providers/gemini/text.py).
- Test with `pytest tests/test_models.py --client=openai` (or `anthropic`, `gemini`)

## Project Structure

```
aimu/
├── __init__.py          # Top-level API (aimu.chat, aimu.client, resolve_model_string) + `from . import aio`
├── aio/                 # Async surface: mirrors public sync API one-for-one (opt-in)
│   ├── __init__.py      # Exports chat, client, AsyncModelClient, Agent, SkillAgent, Chain, Router, Parallel, EvaluatorOptimizer, PlanExecuteEvaluator, OrchestratorAgent, MCPClient
│   ├── _base.py         # AsyncBaseModelClient (pure provider adapter; chat() = one model turn)
│   ├── _tool_loop.py    # _AsyncToolLoop: async iterative tool-calling engine (asyncio.TaskGroup)
│   ├── _model_client.py # AsyncModelClient factory + top-level client()/chat()
│   ├── _mcp_client.py   # async MCPClient (FastMCP Client direct; no anyio portal)
│   ├── agent.py         # AsyncRunner ABC + async Agent
│   ├── skill_agent.py   # async SkillAgent
│   ├── agentic_client.py # internal _AsyncAgenticView for Agent.as_model_client()
│   ├── orchestrator_agent.py # async OrchestratorAgent + assemble()
│   ├── a2a/             # async A2A interop (a2a extra): RemoteAgent + serve_a2a (native async client)
│   │   ├── client.py        # async RemoteAgent(AsyncRunner): native a2a-sdk client + message/stream
│   │   └── server.py        # async serve_a2a/build_a2a_app (awaits runner.run directly)
│   ├── providers/       # Async provider clients (mirrors aimu/models/providers/)
│   │   ├── anthropic.py      # AsyncAnthropicClient (anthropic.AsyncAnthropic)
│   │   ├── openai_compat.py  # AsyncOpenAICompatClient base + local-server subclasses
│   │   ├── ollama.py         # AsyncOllamaClient (ollama.AsyncClient)
│   │   ├── llamacpp.py       # AsyncLlamaCppClient (wraps sync LlamaCppClient, Decision 7)
│   │   ├── hf/               # HuggingFace async clients (wrap sync, Decision 7)
│   │   │   ├── text.py           # AsyncHuggingFaceClient
│   │   │   ├── image.py          # AsyncHuggingFaceImageClient
│   │   │   ├── audio.py          # AsyncHuggingFaceAudioClient
│   │   │   ├── speech.py         # AsyncHuggingFaceSpeechClient
│   │   │   └── transcription.py  # AsyncHuggingFaceTranscriptionClient (wraps sync)
│   │   ├── openai/           # OpenAI cloud async clients
│   │   │   ├── text.py           # AsyncOpenAIClient (subclasses AsyncOpenAICompatClient)
│   │   │   ├── speech.py         # AsyncOpenAISpeechClient (wraps sync OpenAISpeechClient)
│   │   │   └── transcription.py  # AsyncOpenAITranscriptionClient (wraps sync OpenAITranscriptionClient)
│   │   └── gemini/          # Google Gemini async clients
│   │       ├── text.py           # AsyncGeminiClient (subclasses AsyncOpenAICompatClient)
│   │       └── image.py          # AsyncGeminiImageClient (wraps sync GeminiImageClient)
│   ├── image.py         # AsyncImageClient factory + aio.image_client() / aio.generate_image()
│   ├── audio.py         # AsyncAudioClient factory + aio.audio_client() / aio.generate_audio()
│   ├── speech.py        # AsyncSpeechClient factory + aio.speech_client() / aio.generate_speech()
│   ├── transcription.py # AsyncTranscriptionClient factory + aio.transcription_client() / aio.transcribe()
│   ├── embedding.py     # AsyncEmbeddingClient (wraps any sync BaseEmbeddingClient) + aio.embedding_client() / aio.embed()
│   ├── scheduler.py     # Scheduler: interval/one-shot async jobs (proactive triggers) under asyncio.TaskGroup
│   ├── channels/        # Channel transport for personal assistants
│   │   ├── base.py          # Channel(ABC) + ChannelMessage (receive/send/aclose)
│   │   ├── cli.py           # CLIChannel (stdin via asyncio.to_thread; streaming stdout)
│   │   └── web.py           # WebChannel (browser WebSocket <-> Channel; server/HTML stay app-side)
│   ├── tools/           # Async tools (mirrors aimu.tools)
│   │   └── builtin.py        # Async generate_image + make_async_subagent_tool + re-exports of sync built-ins (incl. generate_audio, generate_speech, transcribe_audio)
│   └── workflows/       # Async workflow patterns (asyncio.TaskGroup for Parallel)
│       ├── chain.py
│       ├── router.py
│       ├── parallel.py        # asyncio.TaskGroup; structured concurrency
│       ├── evaluator.py
│       └── plan_execute_evaluator.py
├── models/              # Model client implementations
│   ├── base.py          # Re-export hub for the per-modality base types (back-compat import location)
│   ├── _base/           # Per-modality base types, split out of base.py
│   │   ├── shared.py        # StreamChunk, StreamingContentType, classproperty (cross-modality)
│   │   ├── text.py          # ModelSpec, Model, BaseModelClient
│   │   ├── image.py         # ImageSpec (+ HuggingFaceImageSpec, GeminiImageSpec), ImageModel, BaseImageClient
│   │   ├── audio.py         # AudioSpec (+ HuggingFaceAudioSpec), AudioModel, BaseAudioClient
│   │   ├── speech.py        # SpeechSpec (+ HuggingFaceSpeechSpec, OpenAISpeechSpec), SpeechModel, BaseSpeechClient
│   │   ├── transcription.py # TranscriptionSpec (+ HF/OpenAI subclasses), TranscriptionModel, BaseTranscriptionClient
│   │   └── embedding.py     # EmbeddingSpec (+ OpenAI/Ollama/HuggingFace subclasses), EmbeddingModel, BaseEmbeddingClient
│   ├── model_client.py  # ModelClient factory/wrapper + resolve_model_string + resolve_model_enum + available_text_models/resolve_default_text_model_enum (re-export)
│   ├── image_client.py  # ImageClient factory + resolve_image_model_string + resolve_image_model_enum
│   ├── audio_client.py  # AudioClient factory + resolve_audio_model_string
│   ├── speech_client.py # SpeechClient factory + resolve_speech_model_string
│   ├── transcription_client.py # TranscriptionClient factory + resolve_transcription_model_string
│   ├── embedding_client.py # EmbeddingClient factory + resolve_embedding_model_string
│   ├── _internal/       # Private cross-cutting helpers (never re-exported)
│   │   ├── chat_state.py    # _ChatStateMixin: system_message/reset/append (shared sync+async)
│   │   ├── streaming.py     # resolve_include, filter_chunks, afilter_chunks: stream helpers
│   │   ├── json.py          # parse_json_response, generate_json, extract_tool_calls
│   │   ├── usage.py         # usage_from_openai/anthropic/ollama: normalize token usage → client.last_usage
│   │   ├── model_defaults.py # default-model resolution + local-availability discovery (env + local probes); backs available_text_models / resolve_default_text_model_enum
│   │   ├── factory.py       # ProviderEntry + build_client/resolve_model_string/FactoryDelegate: shared dispatch for the 5 modality factories (image/audio/speech/transcription/embedding); factories take provider kwargs directly (**kwargs)
│   │   ├── image_input.py   # vision-input normalization (_normalize_image, per-provider adapters)
│   │   ├── image_output.py  # encode_image(): diffusion-output format conversion
│   │   └── audio_output.py  # encode_audio(): audio/speech-output conversion (WAV); reused by speech
│   └── providers/       # Concrete provider clients, one module per provider
│       ├── anthropic.py     # AnthropicClient + AnthropicModel
│       ├── ollama.py        # OllamaClient + OllamaModel (native API) + OllamaEmbeddingClient + OllamaEmbeddingModel
│       ├── llamacpp.py      # LlamaCppClient + LlamaCppModel (requires llamacpp extra)
│       ├── openai_compat.py # OpenAICompatClient base + local-server subclasses
│       │                    #   (LMStudio/OllamaOpenAI/HFOpenAI/VLLM/LlamaServer/SGLang) (requires openai package)
│       ├── _thinking.py     # _split_thinking / _ThinkingParser: used by llamacpp + openai_compat
│       ├── hf/              # HuggingFace in-process clients, by modality
│       │   ├── _device.py       # GPU-placement helpers (HF-only): device hint, cuda/mps fallback, memory-aware auto_place_pipeline()
│       │   ├── text.py          # HuggingFaceClient, HuggingFaceModel, ToolCallFormat
│       │   ├── image.py         # HuggingFaceImageClient + HuggingFaceImageModel (image extra)
│       │   ├── audio.py         # HuggingFaceAudioClient + HuggingFaceAudioModel (MusicGen/AudioLDM2/StableAudio; hf extra)
│       │   ├── speech.py        # HuggingFaceSpeechClient + HuggingFaceSpeechModel (MMS-TTS, BARK; hf extra)
│       │   ├── transcription.py # HuggingFaceTranscriptionClient + HuggingFaceTranscriptionModel (Whisper family; hf extra)
│       │   └── embedding.py     # HuggingFaceEmbeddingClient + HuggingFaceEmbeddingModel (sentence-transformers; hf extra)
│       ├── openai/          # OpenAI cloud clients
│       │   ├── text.py          # OpenAIClient + OpenAIModel (GPT/o-series; subclasses OpenAICompatClient)
│       │   ├── speech.py        # OpenAISpeechClient + OpenAISpeechModel (tts-1, tts-1-hd; openai_compat extra)
│       │   ├── transcription.py # OpenAITranscriptionClient + OpenAITranscriptionModel (whisper-1, gpt-4o-transcribe; openai_compat extra)
│       │   └── embedding.py     # OpenAIEmbeddingClient + OpenAIEmbeddingModel (text-embedding-3-*; openai_compat extra)
│       └── gemini/          # Google Gemini clients
│           ├── text.py          # GeminiClient + GeminiModel (subclasses OpenAICompatClient)
│           └── image.py         # GeminiImageClient + GeminiImageModel (Nano Banana; image extra)
├── tools/               # Tool integration (in-process @tool + cross-process MCP)
│   ├── decorator.py     # @tool decorator + ToolSignatureError + ToolArgumentError + coerce_tool_arguments; sets __tool_is_async__ flag
│   ├── builtin.py       # Built-in @tool functions + web/fs/compute/misc/image/audio/speech/transcription subgroups (incl. lazy generate_image, generate_audio, generate_speech, transcribe_audio) + make_memory_tools() / make_document_tools() / make_subagent_tool() factories
│   ├── client.py        # MCPClient wrapper + MCPConnectionError + .ping() + .as_tools()
│   ├── mcp_format.py    # mcp_tools_to_openai() + mcp_content_to_text(): shared by sync & async MCPClient (get_tools/as_tools)
│   └── mcp.py           # FastMCP server registering builtin.ALL_TOOLS
├── agents/              # Agents and workflow patterns (single Runner ABC)
│   ├── base.py          # Runner ABC (+ Runner.as_tool()) + MessageHistory + decision-tree docstring
│   ├── _loop.py         # _AgentLoopMixin: _prepare_run/restore shared by sync + aio Agent
│   ├── _tool_loop.py    # _ToolLoop: iterative tool-calling engine (dispatch/approval/deps/loop) + last_turn_called_tools
│   ├── _as_tool.py      # build_as_tool(): shared name/description resolution for Runner.as_tool() (sync + aio)
│   ├── _agentic_view.py # _AgenticViewMixin: state delegation shared by sync + aio _AgenticView
│   ├── agent.py         # Agent (agentic loop) + as_model_client()
│   ├── skill_agent.py   # SkillAgent (Agent + skill discovery/injection)
│   ├── agentic_client.py # Internal _AgenticView (not public; use Agent.as_model_client())
│   ├── orchestrator_agent.py # OrchestratorAgent + _init_orchestrator() + assemble(workers: list[Runner])
│   ├── a2a/             # A2A interop (a2a extra; guarded HAS_A2A): RemoteAgent + serve_a2a
│   │   ├── _card.py         # pure A2A<->AIMU adapters (card build, message build, text extract): shared sync+async
│   │   ├── client.py        # RemoteAgent(Runner) + A2AConnectionError (anyio portal over a2a-sdk)
│   │   ├── server.py        # serve_a2a() + build_a2a_app() wrapping any Runner as an A2A server
│   │   └── __main__.py      # python -m aimu.agents.a2a (serve an Agent from the CLI)
│   ├── workflows/       # Code-controlled workflow patterns (re-exported from aimu.agents)
│   │   ├── chain.py     # Chain + Chain.from_client() (prompt chaining)
│   │   ├── router.py    # Router + Router.from_client() (routing)
│   │   ├── parallel.py  # Parallel + Parallel.from_client() (parallelization)
│   │   ├── evaluator.py # EvaluatorOptimizer (generate-evaluate-revise)
│   │   └── plan_execute_evaluator.py # PlanExecuteEvaluator + .from_client()
│   └── prebuilt/        # Prebuilt orchestrator agents (ready to use or copy)
│       ├── _base.py            # make_workers(): shared worker-Agent construction for the prebuilt orchestrators
│       ├── research_report.py  # ResearchReportAgent
│       ├── code_review.py      # CodeReviewAgent
│       └── content_creation.py # ContentCreationAgent
├── history.py           # Conversation management (TinyDB; single conversation)
├── sessions/            # Multi-user session store (per-conversation state keyed by channel:sender)
│   ├── base.py          # Session dataclass + SessionStore ABC + session_key() + SessionLocks
│   ├── memory.py        # InMemorySessionStore (non-durable)
│   └── tinydb.py        # TinyDBSessionStore (durable; reuses ConversationManager TinyDB mechanics)
├── memory/              # Memory stores
│   ├── base.py          # MemoryStore abstract base class
│   ├── semantic_store.py # SemanticMemoryStore (ChromaDB vector search)
│   ├── document_store.py # DocumentStore (path-based, Anthropic API-compatible)
│   ├── mcp.py           # FastMCP server for SemanticMemoryStore
│   └── document_mcp.py  # FastMCP server for DocumentStore
├── rag/                 # Retrieval-augmented generation primitives (plain functions over MemoryStore)
│   ├── splitter.py      # split_text (recursive separators + overlap + length_function)
│   ├── pipeline.py      # ingest, retrieve, format_context
│   └── rerank.py        # rerank (cross-encoder via sentence-transformers; lazy + cached)
├── skills/              # Agent skill discovery, authoring, and MCP server
│   ├── skill.py         # AgentSkill dataclass; load_body, script_tool_names
│   ├── manager.py       # SkillManager (+ refresh()) + SkillLoadError + SkillNotFoundError
│   ├── authoring.py     # write_skill() + make_skill_authoring_tool(): runtime skill creation
│   └── mcp.py           # build_skills_server(): FastMCP server from SkillManager
├── evals/               # Evaluation adapters and benchmark harness
│   ├── __init__.py        # Exports Benchmark + BenchmarkResults; guarded DeepEval imports
│   ├── benchmark.py       # Benchmark + BenchmarkResults (uses client.reset() per row)
│   ├── deepeval.py        # DeepEvalModel(DeepEvalBaseLLM) adapter
│   └── deepeval_scorer.py # DeepEvalScorer(Scorer)
├── prompts/             # Prompt storage and management
│   ├── catalog.py       # PromptCatalog (SQLAlchemy, versioned, (name, model_id) keyed)
│   ├── tuner.py         # PromptTuner ABC
│   ├── mcp.py           # FastMCP server for prompt catalog
│   └── tuners/
│       ├── classification.py  # ClassificationPromptTuner
│       ├── multiclass.py      # MultiClassPromptTuner
│       ├── extraction.py      # ExtractionPromptTuner
│       ├── judged.py          # JudgedPromptTuner
│       └── scorers.py         # Scorer ABC + LLMJudgeScorer
└── paths.py             # Path configuration (root, tests, package, output, skills)

tests/                   # Pytest test suite
├── conftest.py          # CLI options: --client/--model (text), --image-client/--image-model (image), --speech-client/--speech-model (speech), --model-path (llamacpp)
├── helpers.py           # Shared text + image parametrization helpers (resolve_model_params, create_real_model_client, resolve_image_model_params, create_real_image_client, MockModelClient)
├── test_models.py       # Live text-model tests, parametrized via --client/--model
├── test_vision.py       # Vision tests (_normalize_image, per-provider adapters, agentic forwarding)
├── test_history.py      # Conversation management tests
├── test_memory.py       # Semantic memory tests
├── test_tools.py        # MCP tools tests
├── test_agents.py       # Agent, as_model_client, OrchestratorAgent, assemble()
├── test_workflow_chain.py                   # Chain
├── test_workflow_router.py                  # Router
├── test_workflow_parallel.py                # Parallel
├── test_workflow_evaluator_optimizer.py     # EvaluatorOptimizer
├── test_workflow_plan_execute_evaluator.py  # PlanExecuteEvaluator
├── test_skills.py       # Skill discovery + raises on malformed SKILL.md
├── test_prompt_*.py     # Prompt catalog and tuning tests
├── test_benchmark.py    # Benchmark harness tests (uses client.reset())
├── test_models_api.py   # Mock-only unit tests of the model surface
│                        # (resolve_model_string, ModelSpec, StreamChunk helpers,
│                        #  system_message lifecycle, include= stream filter)
├── test_images.py             # Live image-generation tests, parametrized via --image-client/--image-model (mirrors test_models.py)
├── test_images_api.py         # Mock-only sync image surface for both providers (mirrors test_models_api.py)
├── test_image_tools.py        # Mock-only built-in generate_image tool + make_image_tool
├── test_aio_image_api.py      # Mock-only async image surface + wrap refusal
├── test_audio.py              # Live audio-generation tests, parametrized via --audio-client/--audio-model (mirrors test_images.py)
├── test_audio_api.py          # Mock-only sync audio surface; exports _install_audio_stubs() for downstream files
├── test_audio_tools.py        # Mock-only built-in generate_audio tool + make_audio_tool
├── test_aio_audio_api.py      # Mock-only async audio surface + wrap refusal
├── test_speech_api.py         # Mock-only sync speech surface; exports _install_speech_stubs() for downstream files
├── test_speech_tools.py       # Mock-only built-in generate_speech tool + make_speech_tool
├── test_aio_speech_api.py     # Mock-only async speech surface + wrap refusal
├── test_transcription_api.py  # Mock-only sync transcription surface
├── test_transcription_tools.py # Mock-only built-in transcribe_audio tool + make_transcription_tool
├── test_aio_transcription_api.py # Mock-only async transcription surface + wrap refusal
├── test_memory_tools.py       # Mock-only make_memory_tools + make_document_tools factories + make_tools memory_store= integration
├── helpers_aio.py       # MockAsyncModelClient + live-backend dispatch for async tests
├── test_aio_models_api.py        # Mock-only async surface tests
├── test_aio_workflow_parallel.py # Verifies asyncio.TaskGroup overlap + sibling cancellation
├── test_aio_agents.py            # Async Agent, async @tool, concurrent_tool_calls under async
├── test_aio_tools.py             # aio.MCPClient lifecycle + mcp_tools_to_openai
├── test_aio_channels.py          # Channel ABC + CLIChannel (mock stdin/stdout)
├── test_aio_scheduler.py         # Scheduler: firing, cancel, exception isolation, clean stop
├── test_aio_skill_authoring.py   # write_skill + SkillManager.refresh() + author_skill tool
├── test_aio_assistant_exports.py # Export smoke for the personal-assistant primitives
└── test_aio_models.py            # Live-backend async tests (mirrors test_models.py)

notebooks/               # Quarto notebook demos (.qmd, kebab-case NN-title; _quarto.yml sets eval:false; see docs/contributing.md)

examples/                       # Runnable, real-world programs (kept out of the default pytest run via testpaths)
├── README.md                   # Index of all examples
├── text-refinement/            # "epic" family: generate→judge→refine over text (loop / Agent / EvaluatorOptimizer / annealing)
│   └── tests/                  # On pythonpath; run via `pytest examples/`
├── image-refinement/           # "hotdog" family: same loop over images (adds diffusers + vision eval + img2img); needs [hf]
│   └── tests/
├── news-summarizer/            # One task via Agent / Chain / Parallel / OrchestratorAgent (--method)
├── personal-assistant/         # Always-on single-user assistant: CLIChannel + Scheduler + skill-authoring SkillAgent
│   ├── assistant.py            # CLI entry point (CLIChannel)
│   ├── web_assistant.py        # WebSocket front end: Starlette + uvicorn server (needs [web])
│   ├── web_channel.py          # WebChannel: example-local Channel adapter over a browser WebSocket
│   ├── static/index.html       # Dependency-free chat page (streaming + proactive messages)
│   └── tests/                  # On pythonpath; run via `pytest examples/`
├── web/                        # Streamlit/Gradio chat UIs (needs [web] extra); moved here from /web
│   ├── streamlit_chatbot_basic.py # ~70-line showcase; model selector, streaming chat, silent tools
│   ├── streamlit_chatbot.py       # Full-featured; image/audio/speech gen, agentic mode, thinking, sliders
│   └── gradio_chatbot_basic.py    # Gradio chat interface with streaming
└── skills/                     # Demo SKILL.md skills (haiku-poet, unit-converter); exposed as aimu.paths.skills
```

## Testing Strategy

- Tests use pytest fixtures and parameterization
- `conftest.py` adds `--client` and `--model` CLI options for targeted testing
- Tests can run against all clients or specific client/model combinations
- Model clients are auto-pulled/downloaded on first use
- MCP tools tested with mock FastMCP servers
- `test_models_api.py` covers the public model surface (top-level `resolve_model_string`, `ModelSpec`, `StreamChunk` helpers, `system_message` lifecycle, `include=` stream filter) with mocks only; no backend required. Workflow factory and orchestrator coverage lives in each workflow's dedicated test file.
- **Async tests** (`test_aio_*.py`) use `pytest-asyncio` in `asyncio_mode = "auto"`. Session-scoped event loop (`asyncio_default_fixture_loop_scope = "session"`, `asyncio_default_test_loop_scope = "session"`) so a single live `httpx.AsyncClient` survives across multiple tests parametrized on the same fixture. Async mocks live in `tests/helpers_aio.py` (`MockAsyncModelClient`, `resolve_async_model_params`, `create_real_async_model_client`).
- The `test_aio_workflow_parallel.py` correctness pair (`test_parallel_overlaps_workers` + `test_parallel_cancels_siblings_on_failure`) validates the `asyncio.TaskGroup` win: overlap < 0.9s for two 0.5s workers, and sibling cancellation + `ExceptionGroup` on first failure.
