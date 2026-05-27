# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMU (AI Modeling Utilities) is a Python library for building AI-powered applications, with language models as the primary building block. It provides a unified, provider-agnostic interface across modalities -- text generation, image generation, and audio (coming soon) -- spanning local backends (Ollama, HuggingFace, llama-cpp-python, any OpenAI-compatible server) and cloud providers (OpenAI, Anthropic, Google Gemini). MCP tool integration and typed streaming output (thinking, tool calling, generation phases) are built into the base model client, not layered on top.

AIMU implements the taxonomy from Anthropic's *[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)*. Concretely, that means **one** `Runner` ABC (no type-level split between agents and workflows) and the following concrete classes, all exported from `aimu.agents` and composable via the shared interface:

- **Autonomous** (the LLM directs flow):
  - `Agent` — the augmented-LLM tool-calling loop.
  - `SkillAgent` — `Agent` + filesystem-discovered skill injection.
  - `OrchestratorAgent` — Anthropic's "orchestrator-workers" pattern, expressed as an autonomous orchestrator whose tools are other agents. Prebuilt examples in `aimu.agents.prebuilt` (`ResearchReportAgent`, `CodeReviewAgent`, `ContentCreationAgent`).
- **Code-controlled workflows** (Python directs flow):
  - `Chain` — prompt chaining (step N's output → step N+1's input).
  - `Router` — routing (classifier dispatches to a specialist handler).
  - `Parallel` — parallelization (workers run concurrently, optional aggregator).
  - `EvaluatorOptimizer` — generate → critique → revise loop.
  - `PlanExecuteEvaluator` — AIMU's extension beyond Anthropic's original five: plan → execute with tools → score → replan on failure.

The agent-vs-workflow distinction is documentation-only — both inherit directly from `Runner`, so they compose freely (a `Router` can dispatch to an `Agent`; a `Chain` step can be an `OrchestratorAgent`). `agent.as_model_client()` adds one more composition path: any agent wraps as a `BaseModelClient`, so it slots anywhere a plain client is accepted (e.g. into a `Benchmark` or a `Chain` that consumes clients). A hill-climbing prompt tuner optimizes prompts against labelled data without ML machinery.

AIMU also ships an optional **async surface** under `aimu.aio` that mirrors the entire public sync API one-for-one (same class names, different namespace). Sync is the default; async is strictly opt-in. The async surface uses modern `asyncio.TaskGroup` for structured concurrency in `Parallel` and `concurrent_tool_calls`. See [docs/explanation/async-design.md](docs/explanation/async-design.md) for the seven design decisions behind this split.

## Design Principles

These six principles drive every architectural decision in AIMU. When proposing changes, check the change against each one — if it violates a principle, it likely belongs in a wrapper above AIMU rather than in the library itself. Full rationale lives in [docs/explanation/design-principles.md](docs/explanation/design-principles.md).

1. **Plain Python.** Classes, functions, decorators, dataclasses, type hints. Every file is readable top-to-bottom. No framework metalanguage; no `Runnable` protocol, no LCEL, no graph DSL, no dependency-injection container.
2. **Plain data.** Conversation state is a `list[dict]` in OpenAI message format. No `Message` / `AIMessage` / `ChatMessage` classes. Provider-specific formats (Anthropic blocks, Ollama image fields, HF PIL images) adapt at request time and never leak into `self.messages`.
3. **Composability through uniform interfaces.** `BaseModelClient` for every provider, `Runner` for every agent and workflow, `MemoryStore` for every memory backend, `StreamChunk` for every streaming source. `agent.as_model_client()` exists specifically so agents are substitutable with plain clients.
4. **Progressive disclosure.** `aimu.chat()` → `aimu.client()` → `Agent` → workflows → custom `BaseModelClient` subclass. Each layer optional; the top wraps the next. New entry points should fit somewhere on this ladder, not parallel to it.
5. **Direct paths for common tasks.** Common operations have one obvious, ergonomic entry point: `aimu.chat()`, `Chain.from_client(...)`, `Agent(client, "system msg", tools=[...])`, `include=["generating"]`, `builtin.web`. The library does not offer parallel, equally-recommended ways to do the same job — if a second path exists, it's a power-user escape hatch (e.g. `Agent.from_config()`), not a documented alternative.
6. **Failures are apparent.** Errors raise at the layer where the cause is actionable, with messages that name the problem. `ToolSignatureError` at decoration time. `SkillLoadError` on discovery. `MCPConnectionError` on construction. `RuntimeError` for locked-`system_message` mutations. Silent fallbacks are bugs; chained exceptions preserve the original cause via `raise ... from exc`.

When reviewing a proposed change, ask which principle it serves and which (if any) it violates. The small public surface is itself a feature — keep it small.

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

For specific model backends only:
```bash
pip install -e '.[ollama]'         # Ollama only
pip install -e '.[hf]'             # HuggingFace transformers (text) + diffusers (image)
pip install -e '.[anthropic]'      # Anthropic Claude models
pip install -e '.[openai_compat]'  # OpenAI-compatible servers + cloud (OpenAI, Gemini, LM Studio, vLLM, etc.)
pip install -e '.[llamacpp]'       # Local GGUF models via llama-cpp-python (no external service)
pip install -e '.[google]'         # Google Gemini image generation (Nano Banana — google-genai SDK)
pip install -e '.[deepeval]'      # DeepEval evaluation metrics
```

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

The `web/` directory ships two Streamlit apps and a Gradio app:

- **`streamlit_chatbot_basic.py`** — ~70-line showcase. Model/provider selector, streaming chat with `include=["generating"]`, built-in tools running silently. Start here to see how little code a working AIMU chatbot needs.
- **`streamlit_chatbot.py`** — Full-featured version. Adds image generation, agentic mode, thinking display, tool-call expanders, generation parameter sliders, and conversation persistence. Intended as an extensible foundation.

```bash
streamlit run web/streamlit_chatbot_basic.py   # showcase
streamlit run web/streamlit_chatbot.py         # full-featured
python web/gradio_chatbot.py                   # Gradio variant
```

## Architecture

### Top-Level API

- **[aimu/__init__.py](aimu/__init__.py)** exposes the small public surface most users start from:
  - `aimu.chat(user_message, *, model, system=None, generate_kwargs=None, stream=False, images=None, include=None)` — one-shot chat: builds a fresh client, sends one message, returns the response (or a `StreamChunk` iterator when `stream=True`).
  - `aimu.client(model, *, system=None, **provider_kwargs)` — one-line constructor that returns a fully configured `ModelClient`. Use this for multi-turn chats.
  - `aimu.resolve_model_string(s)` — parses a `"provider:model_id"` string and returns the matching `Model` enum member. Raises `ValueError` with the list of valid ids on miss.
  - Re-exports: `BaseModelClient`, `Model`, `ModelClient`, `ModelSpec`, `StreamChunk`, `StreamingContentType`.
  - `aimu.aio` — async submodule. Imported by default (one-line `from . import aio`) so users can `from aimu import aio` without a separate install. See "Async Surface" below.

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

- **[aimu/aio/__init__.py](aimu/aio/__init__.py)** — exports `chat`, `client`, `AsyncModelClient`, `Agent`, `SkillAgent`, `Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`, `PlanExecuteEvaluator`, `OrchestratorAgent`, `MCPClient`, `AsyncRunner`.
- **[aimu/aio/_base.py](aimu/aio/_base.py)** — `AsyncBaseModelClient` (ABC). Concrete `async def chat()`/`generate()` apply the `include` filter; abstract `_chat()`/`_generate()` are coroutines. `_handle_tool_calls()` uses `asyncio.TaskGroup` when `concurrent_tool_calls=True`.
- **[aimu/aio/_model_client.py](aimu/aio/_model_client.py)** — `AsyncModelClient` factory + top-level `client()`/`chat()`. Refuses direct construction with `HuggingFaceModel`/`LlamaCppModel` (see "In-process providers" below).
- **[aimu/aio/_mcp_client.py](aimu/aio/_mcp_client.py)** — async `MCPClient` (~30 LOC). Uses FastMCP's native async `Client` directly (no anyio portal). Construct via the `connect()` classmethod factory: `mcp = await MCPClient.connect(server=...)`.
- **[aimu/aio/providers/](aimu/aio/providers/)** — async provider clients:
  - `anthropic.py` — `AsyncAnthropicClient` using `anthropic.AsyncAnthropic`. Reuses sync class's pure format adapters via composition.
  - `openai_compat.py` — `AsyncOpenAICompatClient` using `openai.AsyncOpenAI`, plus `Async*Client` subclasses for OpenAI, Gemini, LM Studio, Ollama-OpenAI, HF-OpenAI, vLLM, llama-server, SGLang.
  - `ollama.py` — `AsyncOllamaClient` using `ollama.AsyncClient`.
  - `hf.py` — `AsyncHuggingFaceClient` wraps a sync `HuggingFaceClient`; see "In-process providers".
  - `llamacpp.py` — `AsyncLlamaCppClient` wraps a sync `LlamaCppClient`; see "In-process providers".
- **[aimu/aio/agent.py](aimu/aio/agent.py)** — `AsyncRunner` ABC + async `Agent`. Same loop semantics as sync; streaming returns `AsyncIterator[StreamChunk]`.
- **[aimu/aio/skill_agent.py](aimu/aio/skill_agent.py)** — async `SkillAgent`. `_setup_skills_async()` constructs an `aio.MCPClient` for skill discovery.
- **[aimu/aio/agentic_client.py](aimu/aio/agentic_client.py)** — internal `_AsyncAgenticView` for `Agent.as_model_client()`.
- **[aimu/aio/orchestrator_agent.py](aimu/aio/orchestrator_agent.py)** — async `OrchestratorAgent` + `assemble()`. Worker dispatch wrappers are `async def`; tool dispatch awaits them under `asyncio.TaskGroup` when `concurrent_tool_calls=True`.
- **[aimu/aio/workflows/](aimu/aio/workflows/)** — async workflow patterns:
  - `chain.py` — async `Chain`, sequential `await` loop.
  - `router.py` — async `Router`.
  - `parallel.py` — async `Parallel`. **The headline change**: `asyncio.TaskGroup` replaces `ThreadPoolExecutor`. Structured concurrency: sibling cancellation on first failure, `ExceptionGroup` aggregation.
  - `evaluator.py` — async `EvaluatorOptimizer`.
  - `plan_execute_evaluator.py` — async `PlanExecuteEvaluator`.

**Shared infrastructure** (used by both sync and async surfaces, no duplication):

- **[aimu/models/_chat_state.py](aimu/models/_chat_state.py)** — `_ChatStateMixin` provides `system_message` lifecycle (setter + lock), `reset()`, `_append_user_turn()`, `_collect_python_tool_specs()`, and the capability properties (`is_thinking_model`, etc.). Both `BaseModelClient` and `AsyncBaseModelClient` inherit it. No state mechanics are duplicated.
- **[aimu/models/_streaming.py](aimu/models/_streaming.py)** — `resolve_include()`, `filter_chunks()` (sync), `afilter_chunks()` (async).
- **[aimu/tools/mcp_format.py](aimu/tools/mcp_format.py)** — `mcp_tools_to_openai(tool_specs)` shared by sync `MCPClient.get_tools()` and async `aio.MCPClient.get_tools()`.
- Provider format adapters (`_openai_messages_to_anthropic`, `_openai_tools_to_anthropic`, `_adapt_messages_for_ollama`, `_split_thinking`, `_ThinkingParser`, `_build_user_content_blocks`) are pure data transforms — reused by async providers via composition, no duplication.

**`@tool` decorator and async tools**:

- `@tool` records `func.__tool_is_async__ = inspect.iscoroutinefunction(func)` at decoration time.
- Sync `BaseModelClient._handle_tool_calls()` raises `ValueError` if it encounters an async tool, with a message pointing the caller at `aimu.aio`.
- Async `AsyncBaseModelClient._handle_tool_calls()` awaits async tools directly; sync (CPU-bound) tools are routed through `asyncio.to_thread` so the event loop stays free.
- `concurrent_tool_calls=True` on the async surface uses `asyncio.TaskGroup` instead of `ThreadPoolExecutor`.

**Streaming type asymmetry**:

- `client.chat(stream=True)` (sync) returns `Iterator[StreamChunk]` — consume with `for`.
- `await client.chat(stream=True)` (async) returns `AsyncIterator[StreamChunk]` — consume with `async for`.
- The `StreamChunk` named tuple itself is identical on both surfaces. They cannot be unified without a hidden event loop (rejected — see async-design.md).

**In-process providers — wrap an existing sync client (Decision 7)**:

`AsyncHuggingFaceClient` and `AsyncLlamaCppClient` do *not* load model weights independently. They take an existing sync client in their constructor and route through `asyncio.to_thread`. Constructing directly with a model enum raises a clear error:

```python
sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)   # loads weights once
async_client = aio.client(sync_client)                  # wraps; shares state
# aio.client(HuggingFaceModel.LLAMA_70B)  # ValueError pointing at the pattern above
```

State (`messages`, `system_message`, `tools`, `mcp_client`) is shared with the wrapped sync client — there is conceptually one client; the async version just exposes an awaitable interface. For in-process providers, "async" buys event-loop integration (handler doesn't block on inference) but *not* coroutine-level concurrency (the GIL and CUDA stream serialize execution).

**MCP integration**:

- The sync `aimu.tools.MCPClient` (anyio portal wrapper) is **kept first-class** — no deprecation. It exists so sync users can use MCP tools without writing async code.
- The async `aimu.aio.MCPClient` is a thin parallel using FastMCP's `Client` directly. Same construction signature (`config=`/`server=`/`file=`); use `await MCPClient.connect(...)` to construct + connect.
- Both `MCPClient.get_tools()` implementations delegate to `aimu.tools.mcp_format.mcp_tools_to_openai()`.

**Design decisions reference**: [docs/explanation/async-design.md](docs/explanation/async-design.md) records the seven decisions (API shape, scope, runtime, MCP, streaming asymmetry, `@tool` detection, in-process wrapping) with rationale and rejected alternatives. Mirror future async-related architectural decisions there.

### Model Client Architecture

The codebase uses an abstract base class pattern for model clients:

- **[aimu/models/base.py](aimu/models/base.py)**: Defines `BaseModelClient` abstract base class with:
  - `chat(user_message, generate_kwargs, use_tools=True, stream=False, images=None, include=None)`: Multi-turn chat with message history; returns `str` or `Iterator[StreamChunk]`. Concrete clients implement `_chat()`; the public `chat()` is provided by the base and applies the `include` stream filter.
  - `generate(prompt, generate_kwargs, stream=False, include=None)`: Single-turn stateless generation. Concrete clients implement `_generate()`; the public `generate()` is provided by the base.
  - `images=` (vision-capable models only) accepts file paths, `pathlib.Path`, raw `bytes`, http(s) URLs, or `data:image/...` URLs.
  - `include=` (streaming only): optional iterable of `StreamingContentType` values (or their string equivalents `"thinking"`, `"tool_calling"`, `"generating"`, `"done"`) selecting which phases to yield. Defaults to all phases.
  - `reset(system_message="__keep__")`: clears the conversation history and unlocks `system_message`. Default keeps the existing system message; pass `None` to clear it or a new string to replace it.
  - `_chat_setup()`: Prepares messages and tools before a chat call; normalizes `images=` into OpenAI-format `image_url` content blocks; locks `system_message` after the first user turn.
  - `_handle_tool_calls()`: Universal tool calling logic (MCP or Python functions).
  - `is_thinking_model` / `is_tool_using_model` / `is_vision_model`: Read-only capability properties derived from `model.supports_thinking` / `model.supports_tools` / `model.supports_vision`.
  - `system_message` property: setter raises `RuntimeError` once the conversation has started (after the first `chat()`); call `client.reset()` to unlock.
  - Message history stored in `self.messages` (OpenAI-style format; user messages with images are content-block lists).
  - Optional `mcp_client` attribute for MCP tool integration.
  - `concurrent_tool_calls: bool = False` — when `True`, multiple tool calls returned in a single model response execute concurrently via `ThreadPoolExecutor`; results are always appended in original order.

- **Supporting types in base.py**:
  - `StreamingContentType(str, Enum)`: `THINKING`, `TOOL_CALLING`, `GENERATING`, `DONE`; string values (`"thinking"`, etc.).
  - `StreamChunk(NamedTuple)`: `(phase, content, agent=None, iteration=0)`. The single chunk type used everywhere — `client.chat(stream=True)`, `Agent.run(stream=True)`, `image_client.generate(stream=True)`, streaming tools, all workflow runs. `content` shape depends on phase:
    - `str` for `THINKING` / `GENERATING` (token).
    - `dict {"name", "arguments", "response"}` for `TOOL_CALLING` (``arguments`` is the dict the model passed to the tool).
    - `dict {"step", "total_steps", "image", "final", "result"}` for `IMAGE_GENERATING` — emitted by image clients during denoising (HF diffusers per step; Gemini coarse start/done). ``image`` is an optional ``PIL.Image`` (None unless ``preview_every`` opted in); ``final=True`` marks the terminal chunk per image; ``result`` carries the encoded output (path / bytes / data-url per ``format=``) on the final chunk.

    Helpers `is_text()` / `is_tool_call()` / `is_image_progress()` dispatch on phase.
  - `ModelSpec`: frozen dataclass with `id: str`, `tools: bool`, `thinking: bool`, `vision: bool`, `generation_kwargs: dict | None`. Equality and hash use `id` only, so it can hold a dict and still be used as an enum value. Each `Model` enum member's value is a `ModelSpec`.
  - `Model(Enum)`: base enum. Each member's value is a `ModelSpec`. Members expose `.value` (the id string), `.spec` (the `ModelSpec`), `.supports_tools`, `.supports_thinking`, `.supports_vision`, `.generation_kwargs`.

- **[aimu/models/model_client.py](aimu/models/model_client.py)**: `ModelClient(BaseModelClient)` factory/wrapper class:
  - Single public construction path. Accepts a `Model` enum member or a `"provider:model_id"` string.
  - Detects provider from the model enum type and instantiates the corresponding concrete client.
  - Delegates `_chat`/`_generate` to the inner client; the public `chat`/`generate` from the base apply the `include` filter.
  - Provider-specific kwargs (e.g. `model_path`, `base_url`, `model_keep_alive_seconds`) are passed through to the concrete constructor.
  - Usage: `ModelClient(OllamaModel.QWEN_3_8B)`, `ModelClient("anthropic:claude-sonnet-4-6")`, `ModelClient(LlamaCppModel.QWEN_3_8B, model_path="/path/to/model.gguf")`.
  - Exports a helper `resolve_model_string(s)` that performs the string lookup independently.

- **Model Implementations**:
  - [aimu/models/ollama/ollama_client.py](aimu/models/ollama/ollama_client.py): `OllamaClient` for local Ollama models (native API)
  - [aimu/models/hf/hf_client.py](aimu/models/hf/hf_client.py): `HuggingFaceClient` for local Transformers models
  - [aimu/models/anthropic/anthropic_client.py](aimu/models/anthropic/anthropic_client.py): `AnthropicClient` for Anthropic Claude models (native `anthropic` SDK)
  - [aimu/models/openai_compat/](aimu/models/openai_compat/): All `openai`-SDK clients (require `openai_compat` extra):
    - `OpenAICompatClient`: generic base for any OpenAI-compatible server
    - `OpenAIClient`: [OpenAI](https://platform.openai.com/) cloud API (GPT-4o, GPT-4.1, o-series); reads `OPENAI_API_KEY`
    - `GeminiClient`: [Google Gemini](https://ai.google.dev/) via Google's OpenAI-compat endpoint; reads `GOOGLE_API_KEY`
    - `LMStudioOpenAIClient`: [LM Studio](https://lmstudio.ai/) (default: `localhost:1234`)
    - `OllamaOpenAIClient`: Ollama OpenAI-compat endpoint (default: `localhost:11434`)
    - `HFOpenAIClient`: HuggingFace Transformers Serve (default: `localhost:8000`)
    - `VLLMOpenAIClient`: vLLM (default: `localhost:8000`)
    - `LlamaServerOpenAIClient`: llama.cpp llama-server (default: `localhost:8080`)
    - `SGLangOpenAIClient`: SGLang (default: `localhost:30000`)
  - [aimu/models/llamacpp/llamacpp_client.py](aimu/models/llamacpp/llamacpp_client.py): `LlamaCppClient` for local GGUF models via llama-cpp-python (requires `llamacpp` extra):
    - Loads GGUF files in-process; no external service required
    - Constructor takes `model_path` (GGUF file path) plus `n_ctx`, `n_gpu_layers`, `chat_format`, `verbose`
    - GPU offloading via `n_gpu_layers=-1` (all layers); `n_gpu_layers=0` for CPU-only
    - Thinking model support via `<think>...</think>` tag parsing (same as OpenAI-compat clients)

- **Model Definitions**: Each client defines a `Model` enum (e.g., `OllamaModel`) where each member's value is a `ModelSpec(id, tools=..., thinking=..., vision=..., generation_kwargs=...)`. Provider-specific extras (e.g. HuggingFace `ToolCallFormat`, `think_opener_in_prompt`) become additional positional elements in the enum member tuple after the `ModelSpec`. Every concrete client exposes the derived classproperties:
  - `TOOL_MODELS`: members where `supports_tools=True`
  - `THINKING_MODELS`: members where `supports_thinking=True`
  - `VISION_MODELS`: members where `supports_vision=True`

- **Optional Dependencies**: [aimu/models/__init__.py](aimu/models/__init__.py) gracefully handles missing dependencies; clients only import if their dependencies are installed (flags: `HAS_OLLAMA`, `HAS_HF`, `HAS_ANTHROPIC`, `HAS_OPENAI_COMPAT`, `HAS_LLAMACPP`). `OpenAIClient` and `GeminiClient` are part of `openai_compat` since they use the same `openai` SDK. `model_client.py` performs the same guarded imports independently to avoid a circular import with `__init__.py`. The `aimu/evals/` package uses the same guarded-import pattern with `HAS_DEEPEVAL`.

### Tool Integration

AIMU supports two tool registration routes that can be combined on the same client:

- **`@tool` decorator** ([aimu/tools/decorator.py](aimu/tools/decorator.py)): mark any plain Python function as a tool. The decorator inspects the signature and docstring at decoration time and attaches an OpenAI-format spec at `func.__tool_spec__`. Hand decorated functions to an agent via `Agent(client, tools=[fn1, fn2])` (or set `client.tools = [fn1, fn2]` directly). Type-to-JSON mapping covers `str`/`int`/`float`/`bool`/`list`/`dict` plus subscripted generics (`list[str]`, `dict[str, int]`) and `Optional[T]` / `T | None`. The first paragraph of the docstring becomes the tool description; required vs. optional args are derived from default values. Python tools dispatch in-process and take precedence over MCP tools of the same name.

  **Validation (raises `ToolSignatureError`):**
  - `*args` / `**kwargs` (variadic parameters) — declare each argument explicitly
  - parameter with no type hint *and* no default value
  - Missing return annotation is a warning, not an error.

  **Async detection**: the decorator also sets `func.__tool_is_async__ = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)` — true for plain `async def` tools and for `async def + yield` (async generator) tools.

  **Streaming tools**: the decorator sets `func.__tool_is_streaming__ = inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func)`. Generator tools may yield zero-or-more `StreamChunk` objects during execution; the agent forwards each yielded chunk through `agent.run(stream=True)` so callers see progress live. The tool's *response* (what goes into the tool message in the conversation history) is resolved in priority order:
  1. The generator's `return` value (captured via `StopIteration.value`) — sync generators only.
  2. The last yielded chunk's `content["result"]` if it's a dict with that key — the convention used by `IMAGE_GENERATING` final chunks.
  3. `str(last_chunk.content)` as a last resort.

  Streaming tools are dispatched via `BaseModelClient._handle_tool_calls_streamed` (and the async sibling on `AsyncBaseModelClient`); the non-streaming `_handle_tool_calls` path rejects them with a clear `ValueError` pointing at `stream=True`. Concurrent dispatch (`concurrent_tool_calls=True`) falls back to sequential when any tool in the batch is streaming, to avoid interleaving chunks from concurrent generators.

- **MCPClient** ([aimu/tools/client.py](aimu/tools/client.py)): wraps FastMCP 2.0 for cross-process tool servers.
  - Converts async FastMCP API to synchronous interface using an internal event loop
  - Constructor requires *exactly one* of `config=`, `server=`, or `file=`. Wrong count or connection failure raises `MCPConnectionError` with the original exception chained.
  - `ping()`: verifies the connection is alive (calls `list_tools()`); raises `MCPConnectionError` if dead.
  - `get_tools()`: Returns tools in OpenAI function calling format
  - `call_tool()`: Synchronous tool execution; wraps backend errors in `MCPConnectionError`
  - `list_tools()`: Returns available tool names
  - Integrates with ModelClient via `model_client.mcp_client` attribute

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)**: built-in general-purpose tools, each decorated with `@tool` for direct in-process use. Pass an entire subgroup with `tools=builtin.web + [my_tool]`:
  - **`builtin.web`** — `get_weather`, `get_webpage`, `search`, `wikipedia`
  - **`builtin.fs`** — `list_directory`, `read_file`
  - **`builtin.compute`** — `calculate`
  - **`builtin.misc`** — `echo`, `get_current_date_and_time`
  - **`builtin.ALL_TOOLS`** — flat list of every built-in (kept for back-compat)

  Tool reference:
  - `echo(echo_string)`: Returns input string
  - `get_current_date_and_time()`: Returns ISO format datetime
  - `get_weather(location)`: Current weather via Open-Meteo API (city name or coordinates)
  - `calculate(expression)`: Safe arithmetic expression evaluator
  - `get_webpage(url)`: Fetches page and returns visible text with HTML stripped
  - `search(query, num_results)`: Web search via SearXNG (`SEARXNG_BASE_URL` env var)
  - `wikipedia(query)`: Wikipedia article summary
  - `list_directory(path)`: Lists files and subdirectories
  - `read_file(path, max_lines)`: Reads local file contents

- **[aimu/tools/mcp.py](aimu/tools/mcp.py)**: thin FastMCP server that registers `builtin.ALL_TOOLS` for cross-process use. Run standalone: `python -m aimu.tools.mcp`. Single source of truth — the same callables back both routes.

- **Tool Calling Flow**:
  1. If model supports tools, `_chat_setup` builds the request `tools` list by combining `mcp_client.get_tools()` (if set) with `__tool_spec__` from every callable in `self.tools`
  2. Model returns tool calls in response
  3. `_handle_tool_calls()` dispatches each call: Python `@tool` functions first (looked up by `__name__`), then MCP tools, then a "not found" tool message
  4. Tool results added to message history with role "tool"
  5. Model called again with tool results to generate final response

### Conversation Management

- **[aimu/history.py](aimu/history.py)**: `ConversationManager` class
  - Uses TinyDB for JSON-based conversation storage
  - Tracks message history with timestamps
  - `create_new_conversation()`: Returns `(doc_id, messages list)`
  - `update_conversation(messages)`: Persists updated messages with timestamps
  - Integrates with ModelClient via `model_client.messages`

### Semantic Memory

- **[aimu/memory/base.py](aimu/memory/base.py)**: `MemoryStore` abstract base class
  - Defines the shared interface: `store(content)`, `search(query, n_results)`, `delete(identifier)`, `list_all()`, `__len__()`
  - All concrete stores implement this interface

- **[aimu/memory/semantic_store.py](aimu/memory/semantic_store.py)**: `SemanticMemoryStore(MemoryStore)` for semantic fact storage
  - Uses ChromaDB with cosine similarity for vector search
  - `store(fact)`: Store a subject-predicate-object string; auto-assigns UUID
  - `search(topic, n_results, max_distance)`: Semantic vector search by topic; `max_distance` is an optional cosine-distance cutoff (0 = identical, 2 = maximally dissimilar; useful range ~0.3–0.6; default `None` = no cutoff)
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

### Agent Skills

- **[aimu/skills/](aimu/skills/)**: Filesystem-discovered skills that inject instructions and tools into agents
  - **[aimu/skills/manager.py](aimu/skills/manager.py)**: `SkillManager` discovers `SKILL.md` files from project and user dirs
    - Default search paths (project-level wins on collision): `.agents/skills/`, `.claude/skills/`, `~/.agents/skills/`, `~/.claude/skills/`
    - Logs discovered count and paths at `INFO` level on first access
    - `catalog_prompt()`: Returns XML-formatted skill listing for system prompt injection. Each entry includes a `<tools>` block listing the names of script-derived tools (`{skill}__{script_stem}`) so the model can call them directly without first invoking `activate_skill`.
    - `get_skill_body(name)`: Returns full SKILL.md instructions body; raises `SkillNotFoundError` (subclass of `KeyError`) if the name is unknown.
    - Malformed `SKILL.md` files raise `SkillLoadError` (subclass of `ValueError`) on parse — no longer silently skipped.
  - **[aimu/skills/skill.py](aimu/skills/skill.py)**: `AgentSkill` dataclass with `name`, `description`, `path`, `load_body()`, `script_tool_names()`.
  - **[aimu/skills/mcp.py](aimu/skills/mcp.py)**: `build_skills_server(manager)` creates a FastMCP server
    - Registers `activate_skill(name)` tool
    - Dynamically registers each `*.py` file in a skill's `scripts/` directory as a tool (`{skill_name}__{script_stem}()`)
  - Skill SKILL.md requires YAML frontmatter with `name` and `description` fields

### Agentic Workflows

AIMU follows the taxonomy from Anthropic's *[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)*. All runnable units share **one** `Runner` ABC with `run(task, stream=False)` and a `messages` property — there is no type-level split between agents and workflows. The agent-vs-workflow distinction is a *categorisation of concrete classes*, not a type hierarchy; it lives in the module docstring at the top of [aimu/agents/base.py](aimu/agents/base.py), which carries a decision tree for picking the right class.

The mapping from Anthropic's patterns to AIMU's concrete classes:

| Anthropic pattern | AIMU class | Category | Module |
|---|---|---|---|
| Augmented LLM (base building block) | `BaseModelClient` / `ModelClient` | — | `aimu.models` |
| Autonomous agent (open-ended tool loop) | `Agent` | autonomous | `aimu.agents.agent` |
| Autonomous agent + skill discovery | `SkillAgent` | autonomous | `aimu.agents.skill_agent` |
| Orchestrator-workers | `OrchestratorAgent` (+ `assemble()` factory) | autonomous (workers as tools) | `aimu.agents.orchestrator_agent` |
| Prompt chaining | `Chain` (+ `Chain.from_client()`) | code-controlled | `aimu.agents.workflows.chain` |
| Routing | `Router` (+ `Router.from_client()`) | code-controlled | `aimu.agents.workflows.router` |
| Parallelization | `Parallel` (+ `Parallel.from_client()`) | code-controlled | `aimu.agents.workflows.parallel` |
| Evaluator-optimizer | `EvaluatorOptimizer` | code-controlled | `aimu.agents.workflows.evaluator` |
| *AIMU extension*: plan → execute → score → replan | `PlanExecuteEvaluator` (+ `from_client()`) | code-controlled | `aimu.agents.workflows.plan_execute_evaluator` |

**Notes on the mapping**:
- Anthropic places "orchestrator-workers" under *workflows*. AIMU implements it as an autonomous `OrchestratorAgent` because the dispatch decision (which worker to call, with what task) lives inside the orchestrator LLM's tool-calling loop — i.e., the LLM directs the flow within the orchestrator. The workers are wrapped as `@tool`-decorated callables (`OrchestratorAgent.assemble()` does this automatically) so the orchestrator's `concurrent_tool_calls=True` overlaps worker execution. Each worker still runs its own agentic loop.
- `PlanExecuteEvaluator` is not in Anthropic's original five patterns — it composes the planner (a `SkillAgent`), executor (an `Agent`), and a `Scorer` into a plan → execute → judge → replan loop, useful for tasks with measurable success criteria.
- All concrete classes are re-exported from `aimu.agents` (`from aimu.agents import Agent, Chain, Router, Parallel, EvaluatorOptimizer, PlanExecuteEvaluator, OrchestratorAgent, SkillAgent`).
- The async mirror under `aimu.aio` re-implements the same set of classes (same names, `async def run()`) with `asyncio.TaskGroup` replacing `ThreadPoolExecutor` in `Parallel` and `concurrent_tool_calls`.

See [docs/explanation/agents-vs-workflows.md](docs/explanation/agents-vs-workflows.md) for the *why* — when each one is the right tool, and how the split shapes API choices.

- **[aimu/agents/base.py](aimu/agents/base.py)**: Single `Runner` ABC for the agent/workflow hierarchy
  - `Runner(ABC)`: abstract `run(task, generate_kwargs, stream=False, images=None)` and `messages` property. Every concrete agent and workflow inherits directly from `Runner` — the agent-vs-workflow distinction is documentation only.
  - `MessageHistory`: type alias for `dict[str, list[dict]]`, the return type of `.messages` across the hierarchy; exported from `aimu.agents`.

- **[aimu/agents/agent.py](aimu/agents/agent.py)**: `Agent(Runner)` for autonomous, multi-round tool execution
  - Constructor: `Agent(model_client, system_message=None, name=None, tools=None, max_iterations=10, continuation_prompt=..., reset_messages_on_run=False)`. `system_message` is the second positional argument. `name` is optional — auto-derived from `id(agent)` as `"agent-{hex}"` when unset.
  - Wraps any `BaseModelClient`; runs `chat()` in a loop until no tools are called.
  - Stop condition: scans `model_client.messages` in reverse; if a `"tool"` role message is found before the last `"user"` message, the agent sends `continuation_prompt` and loops again.
  - `tools: list[Callable]` field accepts Python functions decorated with `@aimu.tools.tool`; `_prepare_run()` copies them to `model_client.tools` before each run. Combine freely with `model_client.mcp_client` for cross-process tools.
  - `run(task, generate_kwargs, stream=False, images=None)`: returns final response string or `Iterator[StreamChunk]`. Streamed chunks carry `agent` (the agent name) and `iteration` (loop index) fields.
  - `as_model_client() -> BaseModelClient`: returns a `BaseModelClient` view (internal `_AgenticView`) whose `chat()` runs the full agent loop. Use this to drop an agent into an API that expects a model client; for direct use call `run()` instead.
  - `messages` property: returns `{name: snapshot}` where `snapshot` is a copy of `model_client.messages` taken at the end of the most recent `run()` call; stable even when agents share a `ModelClient` and `_prepare_run()` clears the live messages.
  - `from_config(config, model_client)`: factory from plain dict (keys: `name`, `system_message`, `max_iterations`, `continuation_prompt`).

- **[aimu/agents/skill_agent.py](aimu/agents/skill_agent.py)**: `SkillAgent(Agent)`: adds skill discovery and injection
  - Extends `Agent` with a `skill_manager: SkillManager` field (defaults to `SkillManager()` for auto-discovery)
  - On first `run()` (or after a message reset), `_setup_skills()` appends the skill catalog to the system message and attaches a skills MCPClient; the model can call `activate_skill` for the full body, or call any script-derived `{skill}__{script}` tool directly (the catalog now lists those tool names inline).
  - `_prepare_run()` resets the `_skills_setup_done` flag when messages are cleared, so skills are re-injected on each fresh run
  - `from_config(config, model_client)`: factory from plain dict (keys: `name`, `system_message`, `max_iterations`, `continuation_prompt`, `skill_dirs`); omit `skill_dirs` to auto-discover

- **[aimu/agents/agentic_client.py](aimu/agents/agentic_client.py)**: `_AgenticView(BaseModelClient)` is the internal class returned by `Agent.as_model_client()`. It is *not* part of the public API; use the method instead of importing the class. The legacy public name `AgenticModelClient` has been removed — call sites should switch to `agent.as_model_client()`.

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
  - Loop stops when `pass_keyword` appears in evaluator feedback or `max_rounds` is reached
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
    1. **Subclass**: define worker `Agent` instances and `@tool`-decorated dispatch functions in `__init__`, then call `self._init_orchestrator(model_client, *, name, system_message, tools=[...], concurrent_tool_calls=False)` (renamed from the legacy `_setup_orchestrator`) to wire everything together.
    2. **Factory**: `OrchestratorAgent.assemble(client, system_message, *, workers=[agent1, agent2, ...], name="orchestrator", concurrent_tool_calls=True)`. Each worker is auto-wrapped as a `@tool` named after the worker's `name`, with description taken from the worker's `system_message`. No subclass needed.
  - Provides `run(task, generate_kwargs, stream, images)` and `messages` — subclasses need not re-implement them
  - Exported from `aimu.agents`

- **[aimu/agents/prebuilt/](aimu/agents/prebuilt/)**: Ready-to-use `OrchestratorAgent` subclasses demonstrating the orchestrator + worker tools pattern
  - Each class accepts a `ModelClient`, defines `@tool`-decorated wrappers around worker `Agent.run()` calls, and passes them to `_init_orchestrator(tools=[...])`
  - `ResearchReportAgent`: orchestrator + `research_overview`, `find_examples`, `find_counterpoints` worker tools; accepts optional `worker_tools: list[Callable]` to give workers live tool access (e.g. `[builtin.search, builtin.get_webpage]`) and automatically enables `concurrent_tool_calls` on the orchestrator
  - `CodeReviewAgent`: orchestrator + `review_security`, `review_performance`, `review_readability` worker tools
  - `ContentCreationAgent`: orchestrator + `research_topic`, `create_outline`, `write_section` worker tools
  - All three inherit from `OrchestratorAgent` and expose `run(task, stream=False)` and `messages`

- All agents/workflows use the public `chat()` / `chat(..., stream=True)` API only; no changes to `ModelClient` subclasses
- `messages` is composable: nested workflows (e.g. a `Router` dispatching to a `Chain`) merge recursively, so all sub-agent names appear at the top level

### Evaluations & Benchmarking

- **[aimu/evals/](aimu/evals/)**: DeepEval adapters (`aimu[deepeval]` extra) plus a multi-model benchmark harness (no extras required)
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
    - **Critical**: drives rows through `chat()` (not `generate()`) because the `_AgenticView` returned by `Agent.as_model_client()` only engages the agent loop via `chat()`. `client.reset()` preserves `system_message` (it's stored separately on the client) and unlocks it for the next chat.
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
- **[aimu/evals/deepeval_scorer.py](aimu/evals/deepeval_scorer.py)**: `DeepEvalScorer(Scorer)` (requires `aimu[deepeval]`)
  - Constructor: `DeepEvalScorer(metrics: list[BaseMetric], input_field="content", output_field="output")`
  - Each row is converted to `LLMTestCase(input=row[input_field], actual_output=row[output_field], expected_output=row.get("reference"))`
  - Calls `metric.measure(test_case)` for every metric; returns `(mean(metric.score for metric in metrics), "\n".join(f"{type(metric).__name__}: {metric.reason}" for metric in metrics))`
  - Use any DeepEval metric (`GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, etc.) wired with a `DeepEvalModel(judge_client)` as the metric's `model`

### Key Design Patterns

These are *implementation* patterns — the *how*. The *why* lives in the "Design Principles" section above.

1. **Single public construction path**: `ModelClient` is the factory. Provider clients are still importable but no longer the recommended public surface. The top-level `aimu.client()` and `aimu.chat()` wrap `ModelClient` for one-line use.
2. **Optional Dependencies**: Graceful degradation when model backends aren't installed. `HAS_*` flags expose installation state; missing optional deps don't break the import of the package.
3. **Tool Calling Abstraction**: The base class handles tool calls uniformly across providers. Concrete clients implement `_chat` / `_generate`; the public `chat` / `generate` wrap them with the `include` filter. Python `@tool` functions take precedence over MCP tools of the same name; both routes can be active on the same client.
4. **Streaming chunk type**: All clients support streaming via `chat(..., stream=True)`, `generate(..., stream=True)`, and image clients via `generate(..., stream=True)`. They yield `StreamChunk(phase, content, agent, iteration)` named tuples. `phase` is a `StreamingContentType` (THINKING, TOOL_CALLING, GENERATING, IMAGE_GENERATING, DONE); see the `StreamChunk` entry above for content shapes per phase. Streaming tools (generator functions decorated with `@tool`) yield their own chunks that flow through the agent's stream — see "Streaming tools" below. Use `chunk.is_text()` / `chunk.is_tool_call()` / `chunk.is_image_progress()` to dispatch on phase.
5. **Model Capability Flags**: `supports_tools`, `supports_thinking`, and `supports_vision` are encoded on the `ModelSpec` value of each `Model` enum member and mirrored as attributes for direct read access. `TOOL_MODELS` / `THINKING_MODELS` / `VISION_MODELS` classproperties are derived automatically.

## Important Implementation Notes

### Adding New Models

When adding new models to a client:
1. Add an enum member to the client's `Model` class with a `ModelSpec(id, tools=..., thinking=..., vision=..., generation_kwargs=...)` value. Example: `QWEN_3_8B = ModelSpec("qwen3:8b", tools=True, thinking=True)`.
2. `TOOL_MODELS`, `THINKING_MODELS`, and `VISION_MODELS` lists are derived automatically from the capability flags; no manual update needed.
3. For HuggingFace models, the enum value is a tuple `(ModelSpec(...), ToolCallFormat.XXX[, think_opener_in_prompt])` because HF carries provider-specific extras alongside the spec.
4. Test with `pytest tests/test_models.py --client=<client> --model=<MODEL_NAME>`

### Tool Calling Compatibility

- Tool calls use OpenAI format: `{"type": "function", "function": {"name": "...", "arguments": {...}}}`
- Some models (e.g., Llama 3.1) use `parameters` instead of `arguments`; this is normalized in `_handle_tool_calls()`
- Tool responses must be added to messages with role "tool" and include `tool_call_id`
- HuggingFace client uses `ToolCallFormat` enum to detect and parse different tool call response formats per model

### Message History Management

- System message is automatically prepended to `messages` on the first `chat()` call if `system_message` is set.
- `system_message` is **immutable after the first `chat()`** — the setter raises `RuntimeError`. Call `client.reset()` to clear messages and unlock the setter; `reset()` preserves the existing system_message by default. Pass `reset(system_message=None)` to clear it, or `reset(system_message="new")` to replace it.
- Message history persists across `chat()` calls on the same client instance.
- Use `ConversationManager` to persist conversations across sessions.

### Thinking Models

Models with `supports_thinking=True` support extended reasoning:
- Reasoning process stored in `self.last_thinking`
- Streamed as `StreamChunk(phase=StreamingContentType.THINKING, content=str)` chunks
- Filter via `include=["generating", "tool_calling"]` on `chat(...)` / `generate(...)`.
- Thinking content may also be stored in the messages dict under a "thinking" key

### Vision Input

Models with `supports_vision=True` accept images alongside the user prompt:
- Pass `images=[...]` to `chat()`. Each item may be a file path string, `pathlib.Path`, raw `bytes`, an `http(s)://` URL, or a `data:image/...;base64,...` URL
- Internally normalized to OpenAI-format content blocks (`{"type": "image_url", "image_url": {"url": ...}}`) inside `self.messages`; helpers live in [aimu/models/_images.py](aimu/models/_images.py) (`_normalize_image`, `_build_user_content_blocks`, plus per-provider adapters)
- `BaseModelClient._chat_setup()` raises `ValueError` if `images=` is passed for a non-vision model
- Per-provider adaptation happens at request time without mutating `self.messages`:
  - `OpenAICompatClient` (and subclasses): pass-through (the `openai` SDK speaks `image_url` blocks natively)
  - `AnthropicClient`: `_openai_messages_to_anthropic` calls `_openai_blocks_to_anthropic` to rewrite `image_url` → `image` blocks (`base64` source for data URLs, `url` source for http(s))
  - `OllamaClient` (native): `_adapt_messages_for_ollama` is called before each `ollama.chat()` invocation to extract `image_url` content blocks into Ollama's message-level `images=[<bare base64>]` field. Only inline base64 (`data:image/...`) is supported; http(s) URLs raise `ValueError`
  - `HuggingFaceClient`: when the model has a processor (Gemma 3 / Gemma 4), `_apply_chat_template` decodes images via `_extract_pil_images` and passes them as `images=` to the processor; `image_url` blocks are rewritten to `{"type": "image"}` placeholders for the chat template
  - `LlamaCppClient`: pass-through; vision requires loading an `mmproj` projector via the new `chat_handler=` constructor kwarg (e.g. `Llava15ChatHandler(clip_model_path=...)`)
- Capability flag: each `ModelSpec` accepts `vision: bool` (default `False`). `BaseModelClient.is_vision_model` and `VISION_MODELS` classproperty mirror the thinking/tools equivalents
- `images=` is also threaded through `Agent.run()` (only the initial task turn carries images; continuation prompts are text-only), through `Agent.as_model_client().chat()`, and through workflow `Runner.run()` overrides (`Chain` forwards to step 0, `Router` forwards to the dispatched handler, `Parallel` forwards to every worker, `EvaluatorOptimizer` forwards only to the initial generator turn)

### OpenAICompatClient Notes

- `OpenAICompatClient` uses the `openai` Python SDK with a configurable `base_url`; works with any server that speaks the OpenAI REST API
- Service-specific subclasses each define their own `Model` enum with service-appropriate model IDs and a default `base_url`
- Thinking model support uses `<think>...</think>` tag parsing (via `_split_thinking` and `_ThinkingParser` in [aimu/models/_thinking.py](aimu/models/_thinking.py)) since OpenAI-compat endpoints embed thinking in content rather than a dedicated field; `LlamaCppClient` uses the same shared utilities
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

AIMU supports text-to-image generation via HuggingFace `diffusers` (`HuggingFaceImageClient`) and Google's Gemini Nano Banana (`GeminiImageClient`) as a **parallel surface** to the text chat client. The image hierarchy mirrors the text one (one base class, one factory class, per-provider concrete clients), but the modality interfaces are disjoint — `chat()` / `messages` / `StreamChunk` are meaningless for stateless text-to-image, so `BaseImageClient` is its own ABC rather than a subclass of `BaseModelClient`. The two surfaces live side-by-side: `aimu.client()` / `ModelClient` for text, `aimu.image_client()` / `ImageClient` for image.

- **[aimu/models/base.py](aimu/models/base.py)** — base classes (text ↔ image parallel):
  - `ImageSpec` (sibling to `ModelSpec`): minimal common spec with `id`, hashed/equal by id only. Subclassed with `@dataclass(eq=False)` so the parent's id-only equality isn't shadowed by the dataclass-default per-field `__eq__`.
  - `HuggingFaceImageSpec(ImageSpec)`: adds `pipeline_class` (resolved lazily via `getattr(diffusers, name)`), `default_steps`, `default_guidance`, `default_width`, `default_height`, `default_negative_prompt`, `pipeline_kwargs`.
  - `GeminiImageSpec(ImageSpec)`: adds `default_aspect_ratio`, `default_image_size`, `image_config_kwargs` (consumed by the SDK's `ImageConfig`).
  - `ImageModel(Enum)` (sibling to `Model`): each member's value is an `ImageSpec` (or subclass); `__init__` sets `_value_=spec.id` and stores `self.spec=spec`. Provider enums (`HuggingFaceImageModel`, `GeminiImageModel`) inherit this.
  - `BaseImageClient(ABC)` (sibling to `BaseModelClient`): concrete `generate(prompt, *, num_images, format, output_dir, **kwargs)` does `num_images` validation + format conversion via `encode_image()`. Subclasses implement `_generate(prompt, *, num_images, **kwargs) -> list[Image]` returning PIL images.
- **[aimu/models/image_client.py](aimu/models/image_client.py)** — `ImageClient` factory (sibling to `ModelClient`) + `resolve_image_model_string()`. Accepts a provider enum / spec / `"provider:id"` string; dispatches to the right concrete client. String-form (`"hf:<repo>"`, `"gemini:<id_or_alias>"`) supports ad-hoc model ids not in the enums, by routing on the prefix before falling back to enum lookup.

- **[aimu/models/_image_output.py](aimu/models/_image_output.py)**: `encode_image(image, format, *, prompt="", output_dir=None)` is the shared format-conversion helper. Sibling to `_images.py` (which decodes images for vision *input*); this one encodes images for diffusion *output*.
  - `format="pil"` → return PIL Image unchanged.
  - `format="bytes"` → PNG-encoded bytes.
  - `format="data_url"` → `data:image/png;base64,...`.
  - `format="path"` → save to `output_dir or paths.output/"images"`, filename `{YYYYMMDD-HHMMSS}-{sha1(prompt+ts)[:8]}.png`; mkdir parents on demand.
  - Unknown format raises `ValueError`.

- **[aimu/models/hf_image/hf_image_client.py](aimu/models/hf_image/hf_image_client.py)**: `HuggingFaceImageClient` + `HuggingFaceImageModel` enum.
  - `HuggingFaceImageModel` catalog: `SD_1_5`, `SDXL_BASE`, `SD_3_5_MEDIUM`, `FLUX_DEV`, `FLUX_SCHNELL`. Each member's value is a `HuggingFaceImageSpec`.
  - `HuggingFaceImageClient(model, model_kwargs=None)` accepts a `HuggingFaceImageModel` member, a `HuggingFaceImageSpec`, or a `"hf:<repo_id>"` string (for ad-hoc HF models not in the enum; defaults to `DiffusionPipeline` auto-detect loader).
  - Lazy pipeline load on first `generate()` call: `pipeline_cls = getattr(diffusers, spec.pipeline_class); self._pipe = pipeline_cls.from_pretrained(spec.id, **kwargs)`. Auto-moves to CUDA or MPS if available.
  - `generate(prompt, *, negative_prompt=None, width=None, height=None, num_inference_steps=None, guidance_scale=None, seed=None, num_images=1, format="pil", output_dir=None, stream=False, preview_every=None)`. Unset params fall back to `spec.default_*`. Seed plumbed through `torch.Generator`. Returns a single image (or list when `num_images > 1`) in the chosen format.
  - `stream=True` returns an `Iterator[StreamChunk]` with phase `IMAGE_GENERATING`. Implemented via `callback_on_step_end` on the diffusers pipeline: a background thread runs the pipeline, the callback pushes per-step chunks onto a `queue.Queue`, and the public generator drains the queue. `preview_every=N` opts into per-step latent-decoded previews — when set, the callback decodes via `pipe.vae.decode` every N steps and includes the PIL in the chunk (~50–200 ms per decode on GPU). Final chunks (one per image when `num_images > 1`) carry `final=True` and the encoded `result` per `format=`.
  - `import diffusers` is a hard module-load import so `HAS_HF_IMAGE` accurately reflects "the optional dep is installed" (matches the HF convention).

- **[aimu/__init__.py](aimu/__init__.py)** — top-level entry points parallel to `client()` / `chat()`:
  - `aimu.image_client(model, **kwargs) -> HuggingFaceImageClient` — one-line constructor.
  - `aimu.generate_image(prompt, *, model, format="pil", **kwargs)` — one-shot.
  - Both raise `ImportError` with a helpful install hint when `HAS_HF_IMAGE` is False.

- **[aimu/tools/builtin.py](aimu/tools/builtin.py)** — chat integration via a built-in **streaming** `@tool`:
  - `generate_image(prompt: str)` is a **generator tool** (decorated with `@tool`) — it yields `IMAGE_GENERATING` chunks during denoising and returns the saved file path as its final value. The agent's `_handle_tool_calls_streamed` forwards the yielded chunks through the agent's stream so callers see live progress.
  - Backed by a lazy module-level singleton `_image_client` constructed on first call. Default model is `hf:stabilityai/stable-diffusion-xl-base-1.0`; override via the `AIMU_IMAGE_MODEL` env var.
  - `make_image_tool(client, *, preview_every=None)` factory binds a fresh streaming tool to a caller-supplied image client, with optional intermediate-image previews every N denoising steps (HF only; Gemini ignores it).
  - `make_describe_image_tool(client, *, default_instruction=...)` builds a `describe_image(image_path, instruction=...)` tool bound to a **vision-capable chat client** (raises `ValueError` if `client.model.supports_vision` is False). The tool snapshots `client.messages` before the vision call and restores it afterwards, so the agent's conversation log isn't polluted with one-off image-Q&A turns; `use_tools=False` is passed to prevent recursive tool calls during the vision call. This is a factory rather than a lazy singleton because the most useful binding is to the *same* model the agent is already running — same knowledge, same provider, no extra API key. The Streamlit chatbot appends this tool to the agent's tool list automatically when the active chat model is vision-capable.
  - New subgroup `image = [generate_image]` next to `web`, `fs`, `compute`, `misc`. Appended to `ALL_TOOLS`.

- **Skill integration (deeper, optional)** — for `SkillAgent` users, drop a `SKILL.md` under `.agents/skills/image-generation/` listing `generate_image` and adding prompt-engineering guidance (composition, style descriptors, negative-prompt hints). Skills are filesystem-discovered, not shipped with the library; see `notebooks/15 - Image Generation.ipynb` for a copyable example.

- **Async twin** — full mirror under `aimu.aio`:
  - **[aimu/aio/providers/hf_image.py](aimu/aio/providers/hf_image.py)**: `AsyncHuggingFaceImageClient` wraps a sync `HuggingFaceImageClient` (no second weight load). Properties (`model`, `spec`, `pipeline`, `model_kwargs`) delegate to `self._sync`. `generate()` is `async` and routes through `asyncio.to_thread`. Does *not* inherit `AsyncBaseModelClient` — diffusion has no chat lifecycle to inherit.
  - **[aimu/aio/image.py](aimu/aio/image.py)**: `aio.image_client(sync_client)` factory + `aio.generate_image(prompt, *, model, ...)` one-shot. The factory refuses direct enum / string / `HuggingFaceImageSpec` construction with a helpful error pointing at the wrap pattern — same shape as the HF/LlamaCpp refusal in `aimu/aio/_model_client.py:140-143`.
  - **[aimu/aio/tools/builtin.py](aimu/aio/tools/builtin.py)**: async `@tool async def generate_image(prompt: str) -> str` using its own lazy `AsyncHuggingFaceImageClient` singleton; `make_async_image_tool(client)` accepts either a sync `HuggingFaceImageClient` or an existing `AsyncHuggingFaceImageClient`. Re-exports the rest of `aimu.tools.builtin` so async agents get a complete namespace.

- **Cloud image provider — Google Nano Banana (`gemini-2.5-flash-image`)**:
  - **[aimu/models/gemini_image/gemini_image_client.py](aimu/models/gemini_image/gemini_image_client.py)**: `GeminiImageClient` + `GeminiImageModel` enum (`NANO_BANANA`, `NANO_BANANA_PREVIEW`). Lives in the same package as `HuggingFaceImageClient` because the surface (`aimu.image_client()` / `aimu.generate_image()`) is shared.
  - **[aimu/models/base.py](aimu/models/base.py)**: `GeminiImageSpec` sits next to `HuggingFaceImageSpec`. Disjoint fields — cloud API has no `pipeline_class`, `default_steps`, or `default_guidance`. Carries `id`, `default_aspect_ratio`, `default_image_size`, `image_config_kwargs`. Hashed and equal by `id` only.
  - Uses `google.genai.Client().models.generate_content(model=..., contents=[prompt], config=GenerateContentConfig(response_modalities=[Modality.IMAGE], image_config=ImageConfig(...)))`. The response's first inline-data part is decoded to a PIL image; `encode_image()` handles the format conversion (`pil`/`bytes`/`path`/`data_url`) — same helper diffusers uses, no duplication.
  - Auth: reads `GOOGLE_API_KEY` from env (or `.env` via `python-dotenv`), or accepts `api_key=` in `model_kwargs`. Raises `RuntimeError` with a clear hint if neither is set.
  - `num_images > 1` issues N separate API calls (Nano Banana returns one image per call). Aspect ratio defaults to the spec's `default_aspect_ratio` if set; otherwise the model picks based on the prompt.
  - **Dispatch via `aimu.image_client()`**: `"gemini:nano-banana"` / `"gemini:gemini-2.5-flash-image"` / `GeminiImageModel.NANO_BANANA` / `GeminiImageSpec(...)` all route to `GeminiImageClient`; `"hf:..."` / `HuggingFaceImageModel.*` / `HuggingFaceImageSpec(...)` route to `HuggingFaceImageClient`. The factory raises `ImportError` with the install hint when the relevant `HAS_*` flag is False.
  - **Async**: `aio.image_client(sync_gemini_client)` returns an `AsyncGeminiImageClient` that wraps via `asyncio.to_thread`. Cloud calls aren't GIL/CUDA-bound so a native async path (`google.genai.aio`) would be slightly faster — kept as a follow-up for surface consistency with the in-process wrap pattern.
  - Lives under the `[google]` extra (`google-genai`, `Pillow`); `HAS_GEMINI_IMAGE` is set independently of `HAS_HF_IMAGE` so users who only want one of the two providers install accordingly. `HAS_HF_IMAGE` requires the `[hf]` extra (`diffusers`, `safetensors`, `Pillow`, plus torch/transformers shared with HF text).

- **Tests** — mirror the text-side `test_models.py` / `test_models_api.py` split:
  - `tests/test_images_api.py` (mock-only, single file for both providers): stubs `diffusers` in `sys.modules` (with a `_FakePipeline` returning PIL images) and patches `genai.Client` per-test via the `fake_genai` fixture. Covers spec equality-by-id, the shared `encode_image` format matrix, `HuggingFaceImageClient` construction + string parsing + lazy load + per-call kwarg threading + seed plumbing + unknown-pipeline error, `GeminiImageClient` construction + alias resolution + API-key resolution + aspect-ratio threading into `ImageConfig` + no-image-data failure path, and the top-level `aimu.image_client()` / `aimu.generate_image()` dispatch for both providers. The `_install_diffusers_stub` helper is re-used by `test_image_tools.py` and `test_aio_image_api.py`.
  - `tests/test_image_tools.py` (mock-only): patches the singleton; asserts `generate_image.__tool_spec__` shape, `make_image_tool` returns a fresh tool bound to its client, env var honoured.
  - `tests/test_aio_image_api.py` (mock-only): refusal of direct enum/string construction for both `HuggingFaceImageClient` and `GeminiImageClient`, `AsyncHuggingFaceImageClient` / `AsyncGeminiImageClient` share state with wrapped sync clients, async tool detected via `__tool_is_async__`.
  - `tests/test_images.py` (live, opt-in): parametrized across image providers — the analog of `tests/test_models.py` for the image modality. Reads `--image-client` (`hf` / `gemini` / `all`) and `--image-model` (enum member name or `all`) from `conftest.py`; `helpers.resolve_image_model_params()` builds the parametrize list, `helpers.create_real_image_client()` constructs the concrete client. Cross-provider tests (PIL output, format matrix, num_images) run on every parametrized client; provider-specific tests (`seed=` reproducibility for HF, `aspect_ratio=` for Gemini) skip with a clear message when the active client doesn't support them. Skipped entirely when `--image-client` is omitted. Example: `pytest tests/test_images.py --image-client=hf --image-model=SD_1_5`.

### Cloud Provider Client Notes

- **`OpenAIClient`** (`aimu[openai_compat]`): thin subclass of `OpenAICompatClient` pointing at `https://api.openai.com/v1`; reads `OPENAI_API_KEY`. Overrides `_update_generate_kwargs` to rename `max_tokens → max_completion_tokens` and force `temperature=1` for o-series models (o1/o3/o4 prefix). Lives in [aimu/models/openai_compat/openai_client.py](aimu/models/openai_compat/openai_client.py).
- **`AnthropicClient`** (`aimu[anthropic]`): full implementation using the native `anthropic` SDK; reads `ANTHROPIC_API_KEY`. `self.messages` is always stored in OpenAI format; conversion to Anthropic format (`_openai_messages_to_anthropic`) and tool format conversion (`_openai_tools_to_anthropic`) happen at call time. After `_handle_tool_calls()` assigns random IDs, `_patch_tool_ids()` replaces them with Anthropic's original `tool_use` block IDs so that tool results are correctly matched. Thinking is native (`thinking={"type": "enabled", "budget_tokens": N}`), not `<think>` tag parsing.
- **`GeminiClient`** (`aimu[openai_compat]`): thin subclass of `OpenAICompatClient` pointing at `https://generativelanguage.googleapis.com/v1beta/openai/`; reads `GOOGLE_API_KEY`. Gemini 2.5 thinking models emit `<think>` tags on this endpoint, so `_ThinkingParser` works as-is. Lives in [aimu/models/openai_compat/gemini_client.py](aimu/models/openai_compat/gemini_client.py).
- Test with `pytest tests/test_models.py --client=openai` (or `anthropic`, `gemini`)

## Project Structure

```
aimu/
├── __init__.py          # Top-level API (aimu.chat, aimu.client, resolve_model_string) + `from . import aio`
├── aio/                 # Async surface — mirrors public sync API one-for-one (opt-in)
│   ├── __init__.py      # Exports chat, client, AsyncModelClient, Agent, SkillAgent, Chain, Router, Parallel, EvaluatorOptimizer, PlanExecuteEvaluator, OrchestratorAgent, MCPClient
│   ├── _base.py         # AsyncBaseModelClient (uses asyncio.TaskGroup for concurrent_tool_calls)
│   ├── _model_client.py # AsyncModelClient factory + top-level client()/chat()
│   ├── _mcp_client.py   # async MCPClient (FastMCP Client direct; no anyio portal)
│   ├── agent.py         # AsyncRunner ABC + async Agent
│   ├── skill_agent.py   # async SkillAgent
│   ├── agentic_client.py # internal _AsyncAgenticView for Agent.as_model_client()
│   ├── orchestrator_agent.py # async OrchestratorAgent + assemble()
│   ├── providers/       # Async provider clients
│   │   ├── anthropic.py      # AsyncAnthropicClient (anthropic.AsyncAnthropic)
│   │   ├── openai_compat.py  # AsyncOpenAICompatClient + service-specific subclasses
│   │   ├── ollama.py         # AsyncOllamaClient (ollama.AsyncClient)
│   │   ├── hf.py             # AsyncHuggingFaceClient (wraps sync HuggingFaceClient — Decision 7)
│   │   ├── llamacpp.py       # AsyncLlamaCppClient (wraps sync LlamaCppClient — Decision 7)
│   │   ├── hf_image.py       # AsyncHuggingFaceImageClient (wraps sync HuggingFaceImageClient)
│   │   └── gemini_image.py   # AsyncGeminiImageClient (wraps sync GeminiImageClient)
│   ├── image.py         # AsyncImageClient factory + aio.image_client() / aio.generate_image()
│   ├── tools/           # Async tools (mirrors aimu.tools)
│   │   └── builtin.py        # Async generate_image + re-exports of sync built-ins
│   └── workflows/       # Async workflow patterns (asyncio.TaskGroup for Parallel)
│       ├── chain.py
│       ├── router.py
│       ├── parallel.py        # asyncio.TaskGroup; structured concurrency
│       ├── evaluator.py
│       └── plan_execute_evaluator.py
├── models/              # Model client implementations
│   ├── base.py          # Abstract bases — text: BaseModelClient, Model, ModelSpec, StreamChunk, StreamingContentType; image: BaseImageClient, ImageModel, ImageSpec (+ HuggingFaceImageSpec, GeminiImageSpec)
│   ├── model_client.py  # ModelClient factory/wrapper + resolve_model_string
│   ├── _chat_state.py   # _ChatStateMixin — system_message/reset/append shared by sync and async base classes
│   ├── _streaming.py    # resolve_include, filter_chunks, afilter_chunks — shared stream helpers
│   ├── _thinking.py     # Shared thinking utilities (_split_thinking, _ThinkingParser)
│   ├── _images.py       # Shared vision utilities (_normalize_image, per-provider adapters)
│   ├── ollama/          # Ollama client (OllamaClient, OllamaModel)
│   ├── hf/              # Hugging Face client (HuggingFaceClient, HuggingFaceModel, ToolCallFormat)
│   ├── anthropic/       # Anthropic Claude client (AnthropicClient, AnthropicModel)
│   ├── openai_compat/   # OpenAI-compatible clients (requires openai package)
│   │   ├── openai_compat_client.py      # OpenAICompatClient base
│   │   ├── lmstudio_openai_client.py    # LMStudioOpenAIClient + LMStudioOpenAIModel
│   │   ├── ollama_openai_client.py      # OllamaOpenAIClient + OllamaOpenAIModel
│   │   ├── hf_openai_client.py          # HFOpenAIClient + HFOpenAIModel
│   │   ├── vllm_openai_client.py        # VLLMOpenAIClient + VLLMOpenAIModel
│   │   ├── openai_client.py             # OpenAIClient + OpenAIModel (OpenAI GPT/o-series)
│   │   ├── gemini_client.py             # GeminiClient + GeminiModel (Google Gemini)
│   │   ├── llamaserver_openai_client.py # LlamaServerOpenAIClient + LlamaServerOpenAIModel
│   │   └── sglang_openai_client.py      # SGLangOpenAIClient + SGLangOpenAIModel
│   ├── llamacpp/        # llama-cpp-python client (requires llamacpp package)
│   │   └── llamacpp_client.py        # LlamaCppClient + LlamaCppModel
│   ├── image_client.py  # ImageClient factory + resolve_image_model_string
│   ├── _image_output.py # encode_image() — shared image-output format conversion
│   ├── hf_image/        # HuggingFace diffusers text-to-image (requires image extra)
│   │   └── hf_image_client.py        # HuggingFaceImageClient + HuggingFaceImageModel
│   └── gemini_image/    # Google Gemini image (Nano Banana; requires image extra)
│       └── gemini_image_client.py    # GeminiImageClient + GeminiImageModel
├── tools/               # Tool integration (in-process @tool + cross-process MCP)
│   ├── decorator.py     # @tool decorator + ToolSignatureError; sets __tool_is_async__ flag
│   ├── builtin.py       # Built-in @tool functions + web/fs/compute/misc/image subgroups (incl. lazy generate_image)
│   ├── client.py        # MCPClient wrapper + MCPConnectionError + .ping()
│   ├── mcp_format.py    # mcp_tools_to_openai() — shared by sync and async MCPClient.get_tools()
│   └── mcp.py           # FastMCP server registering builtin.ALL_TOOLS
├── agents/              # Agents and workflow patterns (single Runner ABC)
│   ├── base.py          # Runner ABC + MessageHistory + decision-tree docstring
│   ├── agent.py         # Agent (agentic loop) + as_model_client()
│   ├── skill_agent.py   # SkillAgent (Agent + skill discovery/injection)
│   ├── agentic_client.py # Internal _AgenticView (not public; use Agent.as_model_client())
│   ├── orchestrator_agent.py # OrchestratorAgent + _init_orchestrator() + assemble()
│   ├── workflows/       # Code-controlled workflow patterns (re-exported from aimu.agents)
│   │   ├── chain.py     # Chain + Chain.from_client() (prompt chaining)
│   │   ├── router.py    # Router + Router.from_client() (routing)
│   │   ├── parallel.py  # Parallel + Parallel.from_client() (parallelization)
│   │   ├── evaluator.py # EvaluatorOptimizer (generate-evaluate-revise)
│   │   └── plan_execute_evaluator.py # PlanExecuteEvaluator + .from_client()
│   └── prebuilt/        # Prebuilt orchestrator agents (ready to use or copy)
│       ├── research_report.py  # ResearchReportAgent
│       ├── code_review.py      # CodeReviewAgent
│       └── content_creation.py # ContentCreationAgent
├── history.py           # Conversation management (TinyDB)
├── memory/              # Memory stores
│   ├── base.py          # MemoryStore abstract base class
│   ├── semantic_store.py # SemanticMemoryStore (ChromaDB vector search)
│   ├── document_store.py # DocumentStore (path-based, Anthropic API-compatible)
│   ├── mcp.py           # FastMCP server for SemanticMemoryStore
│   └── document_mcp.py  # FastMCP server for DocumentStore
├── skills/              # Agent skill discovery and MCP server
│   ├── skill.py         # AgentSkill dataclass; load_body, script_tool_names
│   ├── manager.py       # SkillManager + SkillLoadError + SkillNotFoundError
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
├── conftest.py          # CLI options: --client/--model (text), --image-client/--image-model (image), --model-path (llamacpp)
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
├── helpers_aio.py       # MockAsyncModelClient + live-backend dispatch for async tests
├── test_aio_models_api.py        # Mock-only async surface tests
├── test_aio_workflow_parallel.py # Verifies asyncio.TaskGroup overlap + sibling cancellation
├── test_aio_agents.py            # Async Agent, async @tool, concurrent_tool_calls under async
├── test_aio_tools.py             # aio.MCPClient lifecycle + mcp_tools_to_openai
└── test_aio_models.py            # Live-backend async tests (mirrors test_models.py)

web/                           # Example chat UIs
├── streamlit_chatbot_basic.py # ~70-line showcase; model selector, streaming chat, silent tools
├── streamlit_chatbot.py       # Full-featured; image gen, agentic mode, thinking, sliders
└── gradio_chatbot.py          # Gradio chat interface with streaming

notebooks/               # Jupyter notebook demos
```

## Testing Strategy

- Tests use pytest fixtures and parameterization
- `conftest.py` adds `--client` and `--model` CLI options for targeted testing
- Tests can run against all clients or specific client/model combinations
- Model clients are auto-pulled/downloaded on first use
- MCP tools tested with mock FastMCP servers
- `test_models_api.py` covers the public model surface (top-level `resolve_model_string`, `ModelSpec`, `StreamChunk` helpers, `system_message` lifecycle, `include=` stream filter) with mocks only; no backend required. Workflow factory and orchestrator coverage lives in each workflow's dedicated test file.
- **Async tests** (`test_aio_*.py`) use `pytest-asyncio` in `asyncio_mode = "auto"`. Session-scoped event loop (`asyncio_default_fixture_loop_scope = "session"`, `asyncio_default_test_loop_scope = "session"`) so a single live `httpx.AsyncClient` survives across multiple tests parametrized on the same fixture. Async mocks live in `tests/helpers_aio.py` (`MockAsyncModelClient`, `resolve_async_model_params`, `create_real_async_model_client`).
- The `test_aio_workflow_parallel.py` correctness pair (`test_parallel_overlaps_workers` + `test_parallel_cancels_siblings_on_failure`) validates the `asyncio.TaskGroup` win — overlap < 0.9s for two 0.5s workers, and sibling cancellation + `ExceptionGroup` on first failure.
