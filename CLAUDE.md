# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMU (AI Model Utilities) is a Python package for working with language models, designed for running models locally via Ollama, Hugging Face Transformers, or any OpenAI-compatible local serving framework, with direct cloud provider support for OpenAI, Anthropic, and Google Gemini. The package provides unified interfaces for model interaction, MCP tool integration, conversation management, semantic memory storage, and prompt storage/tuning.

## Development Commands

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
pip install -e '.[hf]'             # Hugging Face only
pip install -e '.[anthropic]'      # Anthropic Claude models
pip install -e '.[openai_compat]'  # OpenAI-compatible servers + cloud (OpenAI, Gemini, LM Studio, vLLM, etc.)
pip install -e '.[llamacpp]'       # Local GGUF models via llama-cpp-python (no external service)
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
pytest tests/test_models.py --client=ollama --model=LLAMA_3_1_8B
pytest tests/test_models.py --client=hf --model=GPT_OSS_20B
```

Run specific test modules:
```bash
pytest tests/test_history.py
pytest tests/test_tools.py
pytest tests/test_prompt_catalog.py
pytest tests/test_memory.py
```

### Linting

The project uses Ruff for linting and formatting (configured in pyproject.toml with line length 120):
```bash
ruff check .
ruff format .
```

### Running the Chat UI

Launch the Streamlit chat application:
```bash
streamlit run web/streamlit_chatbot.py
```

Launch the Gradio chat application:
```bash
python web/gradio_chatbot.py
```

## Architecture

### Model Client Architecture

The codebase uses an abstract base class pattern for model clients:

- **[aimu/models/base_client.py](aimu/models/base_client.py)**: Defines `ModelClient` abstract base class with:
  - `generate()` / `generate_streamed()`: Single-turn text generation
  - `chat()` / `chat_streamed()`: Multi-turn chat with message history
  - `_chat_setup()`: Prepares messages and tools before a chat call
  - `_handle_tool_calls()`: Universal tool calling logic (MCP or Python functions)
  - `is_thinking_model` / `is_tool_using_model`: Read-only capability properties derived from `model.supports_thinking` / `model.supports_tools`
  - `chat_streamed()` / `generate_streamed()` yield `StreamChunk(phase, content)` — type is self-contained in each chunk
  - `system_message` property with setter that manages the system message in the messages list
  - Message history stored in `self.messages` (OpenAI-style format)
  - Optional `mcp_client` attribute for MCP tool integration

- **Supporting types in base_client.py**:
  - `ContentType(str, Enum)`: THINKING, TOOL_CALLING, GENERATING, DONE — string values (`"thinking"`, etc.)
  - `Model(Enum)`: Base enum class; each value carries `supports_tools` and `supports_thinking` boolean attributes

- **Model Implementations**:
  - [aimu/models/ollama/ollama_client.py](aimu/models/ollama/ollama_client.py): `OllamaClient` for local Ollama models (native API)
  - [aimu/models/hf/hf_client.py](aimu/models/hf/hf_client.py): `HuggingFaceClient` for local Transformers models
  - [aimu/models/anthropic/anthropic_client.py](aimu/models/anthropic/anthropic_client.py): `AnthropicClient` for Anthropic Claude models (native `anthropic` SDK)
  - [aimu/models/openai_compat/](aimu/models/openai_compat/): All `openai`-SDK clients (require `openai_compat` extra):
    - `OpenAICompatClient` — generic base for any OpenAI-compatible server
    - `OpenAIClient` — [OpenAI](https://platform.openai.com/) cloud API (GPT-4o, GPT-4.1, o-series); reads `OPENAI_API_KEY`
    - `GeminiClient` — [Google Gemini](https://ai.google.dev/) via Google's OpenAI-compat endpoint; reads `GOOGLE_API_KEY`
    - `LMStudioOpenAIClient` — [LM Studio](https://lmstudio.ai/) (default: `localhost:1234`)
    - `OllamaOpenAIClient` — Ollama OpenAI-compat endpoint (default: `localhost:11434`)
    - `HFOpenAIClient` — HuggingFace Transformers Serve (default: `localhost:8000`)
    - `VLLMOpenAIClient` — vLLM (default: `localhost:8000`)
    - `LlamaServerOpenAIClient` — llama.cpp llama-server (default: `localhost:8080`)
    - `SGLangOpenAIClient` — SGLang (default: `localhost:30000`)
  - [aimu/models/llamacpp/llamacpp_client.py](aimu/models/llamacpp/llamacpp_client.py): `LlamaCppClient` for local GGUF models via llama-cpp-python (requires `llamacpp` extra):
    - Loads GGUF files in-process — no external service required
    - Constructor takes `model_path` (GGUF file path) plus `n_ctx`, `n_gpu_layers`, `chat_format`, `verbose`
    - GPU offloading via `n_gpu_layers=-1` (all layers); `n_gpu_layers=0` for CPU-only
    - Thinking model support via `<think>...</think>` tag parsing (same as OpenAI-compat clients)

- **Model Definitions**: Each client defines a `Model` enum (e.g., `OllamaModel`) where each member encodes `(model_id, supports_tools, supports_thinking)`. Ollama and HuggingFace clients also expose:
  - `TOOL_MODELS`: Derived list of models where `supports_tools=True`
  - `THINKING_MODELS`: Derived list of models where `supports_thinking=True`

- **Optional Dependencies**: [aimu/models/__init__.py](aimu/models/__init__.py) gracefully handles missing dependencies — clients only import if their dependencies are installed (flags: `HAS_OLLAMA`, `HAS_HF`, `HAS_ANTHROPIC`, `HAS_OPENAI_COMPAT`, `HAS_LLAMACPP`). `OpenAIClient` and `GeminiClient` are part of `openai_compat` since they use the same `openai` SDK.

### MCP Tools Integration

- **[aimu/tools/client.py](aimu/tools/client.py)**: `MCPClient` wraps FastMCP 2.0 client
  - Converts async FastMCP API to synchronous interface using an internal event loop
  - `get_tools()`: Returns tools in OpenAI function calling format
  - `call_tool()`: Synchronous tool execution
  - `list_tools()`: Returns available tool names
  - Integrates with ModelClient via `model_client.mcp_client` attribute

- **[aimu/tools/mcp.py](aimu/tools/mcp.py)**: FastMCP server with built-in general-purpose tools:
  - `echo(echo_string)`: Returns input string
  - `get_current_date_and_time()`: Returns ISO format datetime
  - `get_weather(location)`: Current weather via Open-Meteo API (city name or coordinates)
  - `calculate(expression)`: Safe arithmetic expression evaluator
  - `get_webpage(url)`: Fetches page and returns visible text with HTML stripped
  - `search(query, num_results)`: Web search via SearXNG (`SEARXNG_BASE_URL` env var)
  - `list_directory(path)`: Lists files and subdirectories
  - `read_file(path, max_lines)`: Reads local file contents
  - Run standalone: `python -m aimu.tools.mcp`

- **Tool Calling Flow**:
  1. If model supports tools and `mcp_client` is set, tools are passed to model
  2. Model returns tool calls in response
  3. `_handle_tool_calls()` executes tools via MCP client
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

- **[aimu/memory/store.py](aimu/memory/store.py)**: `MemoryStore` class for semantic fact storage
  - Uses ChromaDB with cosine similarity for vector search
  - `store_fact(fact)`: Store a subject-predicate-object string
  - `retrieve_facts(topic, n_results)`: Vector semantic search
  - `delete_fact(fact)`: Remove exact fact
  - `__len__()`: Count stored facts
- **[aimu/memory/mcp.py](aimu/memory/mcp.py)**: FastMCP server exposing `search_memories` / `add_memories` tools
  - Run as MCP server: `python -m aimu.memory` or `python -m aimu.memory.mcp`
  - Storage path via `MEMORY_STORE_PATH` env var

### Agent Skills

- **[aimu/skills/](aimu/skills/)**: Filesystem-discovered skills that inject instructions and tools into agents
  - **[aimu/skills/manager.py](aimu/skills/manager.py)**: `SkillManager` discovers `SKILL.md` files from project and user dirs
    - Default search paths (project-level wins on collision): `.agents/skills/`, `.claude/skills/`, `~/.agents/skills/`, `~/.claude/skills/`
    - `catalog_prompt()`: Returns XML-formatted skill listing for system prompt injection
    - `get_skill_body(name)`: Returns full SKILL.md instructions body
  - **[aimu/skills/skill.py](aimu/skills/skill.py)**: `Skill` dataclass with `name`, `description`, `path`; `load_body()` strips YAML frontmatter
  - **[aimu/skills/mcp.py](aimu/skills/mcp.py)**: `build_skills_server(manager)` creates a FastMCP server
    - Registers `activate_skill(name)` tool
    - Dynamically registers each `*.py` file in a skill's `scripts/` directory as a tool (`{skill_name}__{script_stem}()`)
  - Skill SKILL.md requires YAML frontmatter with `name` and `description` fields

### Agentic Workflows

- **[aimu/agents/base_agent.py](aimu/agents/base_agent.py)**: `Agent` ABC — shared interface for all agent runners
  - Abstract `run(task, generate_kwargs)` and `run_streamed(task, generate_kwargs)` methods
  - `AgentChunk(NamedTuple)`: `(agent_name, iteration, phase: ContentType, content: Any)`

- **[aimu/agents/simple_agent.py](aimu/agents/simple_agent.py)**: `SimpleAgent(Agent)` for autonomous, multi-round tool execution
  - Wraps any `ModelClient`; runs `chat()` in a loop until no tools are called
  - Stop condition: scans `model_client.messages` in reverse — if a `"tool"` role message is found before the last `"user"` message, the agent sends `continuation_prompt` and loops again
  - `run(task, generate_kwargs)`: synchronous agentic loop, returns final response string
  - `run_streamed(task, generate_kwargs)`: yields `AgentChunk(agent_name, iteration, phase, content)`
  - `from_config(config, model_client)`: factory from plain dict (keys: `name`, `system_message`, `max_iterations`, `continuation_prompt`, `skill_dirs`, `use_skills`)
  - Optional `skill_manager`: if set, `_setup_skills()` injects catalog into system message and attaches skills MCP client on first run

- **[aimu/agents/agentic_client.py](aimu/agents/agentic_client.py)**: `AgenticModelClient(ModelClient)` — drop-in agentic wrapper
  - Wraps any `Agent` (`SimpleAgent` or `WorkflowAgent`) and exposes the standard `ModelClient` interface
  - `chat()` / `chat_streamed()` run the full Agent loop (multi-turn until tools stop); `chat_streamed()` adapts `AgentChunk` → `StreamChunk`
  - `generate()` / `generate_streamed()` pass through to the inner client unchanged (no loop)
  - `messages`, `mcp_client`, `system_message`, `last_thinking` delegate to the innermost `SimpleAgent`'s `model_client` (last agent in chain for `WorkflowAgent`)
  - Constructor: `AgenticModelClient(agent: Agent)` — accepts `SimpleAgent` or `WorkflowAgent`
  - Use anywhere a `ModelClient` is accepted to get agentic behaviour transparently

- **[aimu/agents/workflow_agent.py](aimu/agents/workflow_agent.py)**: `WorkflowAgent(Agent)` for chaining agents sequentially
  - Output of step N (accumulated `GENERATING` chunks) becomes task input to step N+1
  - `run(task, generate_kwargs)`: runs all agents in order, returns final string
  - `run_streamed(task, generate_kwargs)`: yields `WorkflowChunk(step, agent_name, iteration, phase: ContentType, content: str)`
  - `from_config(configs, client)`: builds from a list of dicts and a single `ModelClient`; the client is shared across steps — `messages` cleared and `system_message` set from each step's config before it runs
  - `agents: list[Agent]` — steps may be `SimpleAgent` or nested `WorkflowAgent` instances

- Agents use the public `chat()` / `chat_streamed()` API only — no changes to `ModelClient` subclasses

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
  - Caches best state — regressing mutations revert without re-evaluation
  - Saves each improvement to catalog when `catalog` and `prompt_name` are provided
- **[aimu/prompts/tuners/classification.py](aimu/prompts/tuners/classification.py)**: `ClassificationPromptTuner(PromptTuner)`
  - Binary YES/NO classification; prompt template must contain `{content}` placeholder
  - Data must have `content` (text) and `actual_class` (bool) columns
  - `classify_data(prompt, data)`: runs prompt on all rows, adds `predicted_class` column
  - `evaluate_results(data)`: returns `{accuracy, precision, recall}`

### Key Design Patterns

1. **Abstract Base Class**: All model clients inherit from `ModelClient` and implement abstract methods
2. **Optional Dependencies**: Graceful degradation when model backends aren't installed
3. **OpenAI-Compatible Format**: Message history uses `{"role": "...", "content": "..."}` format
4. **Tool Calling Abstraction**: Base class handles tool calls uniformly across providers
5. **Streaming Support**: All clients support both streaming and non-streaming generation; `chat_streamed()` and `generate_streamed()` yield `StreamChunk(phase, content)` named tuples — `phase` is a `StreamingContentType` (THINKING, TOOL_CALLING, GENERATING); `content` is `str` for THINKING/GENERATING and `dict {"name": ..., "response": ...}` for TOOL_CALLING. `StreamPhase` is an alias for `StreamingContentType` used in notebooks and user-facing code.
6. **Model Capability Flags**: `supports_tools` and `supports_thinking` are encoded directly on each `Model` enum member

## Important Implementation Notes

### Adding New Models

When adding new models to a client:
1. Add model enum member to the client's `Model` class (e.g., `OllamaModel`) with `(model_id, supports_tools, supports_thinking)` tuple
2. `TOOL_MODELS` and `THINKING_MODELS` lists are derived automatically from the enum flags — no manual update needed for Ollama/HF clients
3. For HuggingFace models, also specify the `ToolCallPrefix` format (XML, JSON_OBJECT, JSON_ARRAY, BRACKETED)
4. Test with `pytest tests/test_models.py --client=<client> --model=<MODEL_NAME>`

### Tool Calling Compatibility

- Tool calls use OpenAI format: `{"type": "function", "function": {"name": "...", "arguments": {...}}}`
- Some models (e.g., Llama 3.1) use `parameters` instead of `arguments` — this is normalized in `_handle_tool_calls()`
- Tool responses must be added to messages with role "tool" and include `tool_call_id`
- HuggingFace client uses `ToolCallPrefix` enum to detect and parse different tool call response formats per model

### Message History Management

- System message is automatically prepended to `messages` if set via the `system_message` property
- Message history persists across `chat()` calls on the same client instance
- Use `ConversationManager` to persist conversations across sessions
- Clear history by setting `model_client.messages = []`

### Thinking Models

Models with `supports_thinking=True` support extended reasoning:
- Reasoning process stored in `self.last_thinking`
- Streamed as `StreamChunk(phase=StreamPhase.THINKING, content=str)` chunks
- Pass `include_thinking=False` to `generate_streamed()` to suppress thinking chunks
- Thinking content may also be stored in the messages dict under a "thinking" key

### OpenAICompatClient Notes

- `OpenAICompatClient` uses the `openai` Python SDK with a configurable `base_url` — works with any server that speaks the OpenAI REST API
- Service-specific subclasses each define their own `Model` enum with service-appropriate model IDs and a default `base_url`
- Thinking model support uses `<think>...</think>` tag parsing (via `_split_thinking` and `_ThinkingParser` in [aimu/models/_thinking.py](aimu/models/_thinking.py)) since OpenAI-compat endpoints embed thinking in content rather than a dedicated field; `LlamaCppClient` uses the same shared utilities
- Pass `tools=openai.NOT_GIVEN` (not `tools=[]`) when no tools are available — some local servers reject an empty tools array
- Model ID format varies by service: LM Studio uses loaded model keys, Ollama uses `name:tag` format, HF Transformers Serve, vLLM, and SGLang use HuggingFace repo paths (e.g. `Qwen/Qwen3-8B`); llama-server uses GGUF filenames (the model field is ignored server-side, so these are capability-lookup only)
- Test with `pytest tests/test_models.py --client=lmstudio_openai` (or `ollama_openai`, `hf_openai`, `vllm_openai`, `llamaserver_openai`, `sglang_openai`)

### LlamaCppClient Notes

- `LlamaCppClient` loads a GGUF file in-process via `llama_cpp.Llama` — no external service or server required
- Constructor requires `model_path` (path to GGUF file); `model` enum specifies capability flags (`supports_tools`, `supports_thinking`)
- `n_gpu_layers=-1` offloads all layers to GPU (default); `n_gpu_layers=0` forces CPU-only inference
- `chat_format=None` (default) auto-detects from GGUF metadata; specify explicitly (e.g. `"chatml-function-calling"`) for older GGUFs that lack embedded templates
- llama-cpp-python returns plain Python dicts, not OpenAI SDK objects — all response access uses dict indexing (`response["choices"][0]["message"]["content"]`)
- Tool calling requires both `supports_tools=True` on the model enum AND a compatible `chat_format`; modern GGUFs auto-detect this
- Thinking support works the same as `OpenAICompatClient` — `<think>...</think>` tags in content, parsed by shared utilities in `_thinking.py`
- Test with `pytest tests/test_models.py --client=llamacpp --model-path=/path/to/model.gguf`

### Cloud Provider Client Notes

- **`OpenAIClient`** (`aimu[openai_compat]`): thin subclass of `OpenAICompatClient` pointing at `https://api.openai.com/v1`; reads `OPENAI_API_KEY`. Overrides `_update_generate_kwargs` to rename `max_tokens → max_completion_tokens` and force `temperature=1` for o-series models (o1/o3/o4 prefix). Lives in [aimu/models/openai_compat/openai_client.py](aimu/models/openai_compat/openai_client.py).
- **`AnthropicClient`** (`aimu[anthropic]`): full implementation using the native `anthropic` SDK; reads `ANTHROPIC_API_KEY`. `self.messages` is always stored in OpenAI format — conversion to Anthropic format (`_openai_messages_to_anthropic`) and tool format conversion (`_openai_tools_to_anthropic`) happen at call time. After `_handle_tool_calls()` assigns random IDs, `_patch_tool_ids()` replaces them with Anthropic's original `tool_use` block IDs so that tool results are correctly matched. Thinking is native (`thinking={"type": "enabled", "budget_tokens": N}`) — not `<think>` tag parsing.
- **`GeminiClient`** (`aimu[openai_compat]`): thin subclass of `OpenAICompatClient` pointing at `https://generativelanguage.googleapis.com/v1beta/openai/`; reads `GOOGLE_API_KEY`. Gemini 2.5 thinking models emit `<think>` tags on this endpoint, so `_ThinkingParser` works as-is. Lives in [aimu/models/openai_compat/gemini_client.py](aimu/models/openai_compat/gemini_client.py).
- Test with `pytest tests/test_models.py --client=openai` (or `anthropic`, `gemini`)

## Project Structure

```
aimu/
├── models/              # Model client implementations
│   ├── base_client.py   # Abstract base class (ModelClient, StreamChunk, StreamPhase, Model)
│   ├── _thinking.py     # Shared thinking utilities (_split_thinking, _ThinkingParser)
│   ├── ollama/          # Ollama client (OllamaClient, OllamaModel)
│   ├── hf/              # Hugging Face client (HuggingFaceClient, HuggingFaceModel, ToolCallPrefix)
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
│   └── llamacpp/        # llama-cpp-python client (requires llamacpp package)
│       └── llamacpp_client.py        # LlamaCppClient + LlamaCppModel
├── tools/               # MCP tools integration
│   ├── client.py        # MCPClient wrapper
│   └── mcp.py           # Example FastMCP server with built-in tools
├── agents/              # Agentic workflows
│   ├── base_agent.py    # Agent ABC + AgentChunk
│   ├── simple_agent.py  # SimpleAgent (agentic loop over ModelClient.chat())
│   ├── agentic_client.py # AgenticModelClient (ModelClient wrapper, accepts any Agent)
│   └── workflow_agent.py # WorkflowAgent + WorkflowChunk (sequential agent chaining)
├── history.py           # Conversation management (TinyDB)
├── memory/              # Semantic memory store (ChromaDB)
│   ├── store.py         # MemoryStore class
│   ├── mcp.py           # FastMCP server (search_memories, add_memories tools)
│   └── __main__.py      # Entrypoint for python -m aimu.memory
├── skills/              # Agent skill discovery and MCP server
│   ├── skill.py         # Skill dataclass (name, description, path, load_body())
│   ├── manager.py       # SkillManager (discovery, catalog_prompt, get_skill_body)
│   └── mcp.py           # build_skills_server() — FastMCP server from SkillManager
├── prompts/             # Prompt storage and management
│   ├── catalog.py       # PromptCatalog (SQLAlchemy, versioned, (name, model_id) keyed)
│   ├── tuner.py         # PromptTuner ABC (hill-climbing loop, generate/mutation kwargs)
│   ├── mcp.py           # FastMCP server for prompt catalog (python -m aimu.prompts.mcp)
│   └── tuners/
│       └── classification.py  # ClassificationPromptTuner (YES/NO, classify_data, evaluate_results)
└── paths.py             # Path configuration (root, tests, package, output, skills)

tests/                   # Pytest test suite
├── conftest.py          # Shared helpers: resolve_model_params, create_real_model_client
├── test_models.py       # Model client tests (generate, chat, streaming, tools, thinking)
├── test_history.py      # Conversation management tests
├── test_memory.py       # Semantic memory tests
├── test_tools.py        # MCP tools tests
├── test_agents.py       # Agent and Workflow tests (mock by default, real with --client)
├── test_skills.py       # Skill discovery and Agent skill integration tests
└── test_prompt_*.py     # Prompt catalog and tuning tests (mock by default, real with --client)

web/                     # Example chat UIs
├── streamlit_chatbot.py # Full-featured Streamlit chat interface with streaming
└── gradio_chatbot.py    # Full-featured Gradio chat interface with streaming

notebooks/               # Jupyter notebook demos
```

## Testing Strategy

- Tests use pytest fixtures and parameterization
- `conftest.py` adds `--client` and `--model` CLI options for targeted testing
- Tests can run against all clients or specific client/model combinations
- Model clients are auto-pulled/downloaded on first use
- MCP tools tested with mock FastMCP servers
