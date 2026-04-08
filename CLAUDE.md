# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMU (AI Model Utilities) is a Python package for working with language models, specifically designed for running models locally via Ollama, Hugging Face Transformers, or any OpenAI-compatible local serving framework, with support for cloud models through aisuite. The package provides unified interfaces for model interaction, MCP tool integration, conversation management, semantic memory storage, and prompt storage/tuning.

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
pip install -e '.[aisuite]'        # Cloud models (OpenAI, Anthropic, etc.)
pip install -e '.[openai_compat]'  # OpenAI-compatible local servers (LM Studio, HF Transformers Serve, vLLM, etc.)
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
pytest tests/test_models.py --client=aisuite
pytest tests/test_models.py --client=lmstudio_openai
pytest tests/test_models.py --client=ollama_openai
pytest tests/test_models.py --client=hf_openai
pytest tests/test_models.py --client=vllm
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
  - [aimu/models/aisuite/aisuite_client.py](aimu/models/aisuite/aisuite_client.py): `AisuiteClient` for cloud models (OpenAI, Anthropic, etc.)
  - [aimu/models/openai_compat/](aimu/models/openai_compat/): OpenAI-compatible REST API clients (require `openai_compat` extra):
    - `OpenAICompatClient` — generic base for any OpenAI-compatible server
    - `LMStudioOpenAIClient` — [LM Studio](https://lmstudio.ai/) (default: `localhost:1234`)
    - `OllamaOpenAIClient` — Ollama OpenAI-compat endpoint (default: `localhost:11434`)
    - `HFOpenAIClient` — HuggingFace Transformers Serve (default: `localhost:8000`)
    - `VLLMOpenAIClient` — vLLM (default: `localhost:8000`)
  - [aimu/models/llamacpp/llamacpp_client.py](aimu/models/llamacpp/llamacpp_client.py): `LlamaCppClient` for local GGUF models via llama-cpp-python (requires `llamacpp` extra):
    - Loads GGUF files in-process — no external service required
    - Constructor takes `model_path` (GGUF file path) plus `n_ctx`, `n_gpu_layers`, `chat_format`, `verbose`
    - GPU offloading via `n_gpu_layers=-1` (all layers); `n_gpu_layers=0` for CPU-only
    - Thinking model support via `<think>...</think>` tag parsing (same as OpenAI-compat clients)

- **Model Definitions**: Each client defines a `Model` enum (e.g., `OllamaModel`) where each member encodes `(model_id, supports_tools, supports_thinking)`. Ollama and HuggingFace clients also expose:
  - `TOOL_MODELS`: Derived list of models where `supports_tools=True`
  - `THINKING_MODELS`: Derived list of models where `supports_thinking=True`

- **Optional Dependencies**: [aimu/models/__init__.py](aimu/models/__init__.py) gracefully handles missing dependencies — clients only import if their dependencies are installed (flags: `HAS_OLLAMA`, `HAS_HF`, `HAS_AISUITE`, `HAS_OPENAI_COMPAT`, `HAS_LLAMACPP`).

### MCP Tools Integration

- **[aimu/tools/client.py](aimu/tools/client.py)**: `MCPClient` wraps FastMCP 2.0 client
  - Converts async FastMCP API to synchronous interface using an internal event loop
  - `get_tools()`: Returns tools in OpenAI function calling format
  - `call_tool()`: Synchronous tool execution
  - `list_tools()`: Returns available tool names
  - Integrates with ModelClient via `model_client.mcp_client` attribute

- **[aimu/tools/servers.py](aimu/tools/servers.py)**: Example FastMCP server with built-in tools:
  - `echo(echo_string)`: Returns input string
  - `get_current_data_and_time()`: Returns ISO format datetime

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

- **[aimu/memory.py](aimu/memory.py)**: `MemoryStore` class for semantic fact storage
  - Uses ChromaDB with cosine similarity for vector search
  - `store_fact(fact)`: Store a subject-predicate-object string
  - `retrieve_facts(topic, n_results)`: Vector semantic search
  - `delete_fact(fact)`: Remove exact fact
  - `__len__()`: Count stored facts
  - Can also run as an MCP server: `python -m aimu.memory`

### Agentic Workflows

- **[aimu/agents/agent.py](aimu/agents/agent.py)**: `Agent` class for autonomous, multi-round tool execution
  - Wraps any `ModelClient`; runs `chat()` in a loop until no tools are called
  - Stop condition: scans `model_client.messages` in reverse — if a `"tool"` role message is found before the last `"user"` message, the agent sends `continuation_prompt` and loops again
  - `run(task, generate_kwargs)`: synchronous agentic loop, returns final response string
  - `run_streamed(task, generate_kwargs)`: yields `AgentChunk(agent_name, iteration, phase, content)`
  - `from_config(config, model_client)`: factory from plain dict (keys: `name`, `system_message`, `max_iterations`, `continuation_prompt`)
  - `AgentChunk(NamedTuple)`: `(agent_name, iteration, phase: ContentType, content: str)`

- **[aimu/agents/workflow.py](aimu/agents/workflow.py)**: `Workflow` class for chaining agents sequentially
  - Output of step N (accumulated `GENERATING` chunks) becomes task input to step N+1
  - `run(task)`: runs all agents in order, returns final string
  - `run_streamed(task)`: yields `WorkflowChunk(step, agent_name, iteration, phase: ContentType, content: str)`
  - `from_config(configs, client_factory)`: factory from list of dicts; `client_factory(cfg)` must return a `ModelClient`

- **No changes to existing `ModelClient` or its subclasses** — agents use the public `chat()` / `chat_streamed()` API only

### Prompt Management

- **[aimu/prompts/catalog.py](aimu/prompts/catalog.py)**: `PromptCatalog` for versioned prompt storage
  - Uses SQLAlchemy ORM; `Prompt` model tracks id, parent_id, prompt, model_id, version, mutation_prompt, reasoning_prompt, metrics
  - `store_prompt()`, `retrieve_last()`, `retrieve_all()`, `delete_all()`, `retrieve_model_ids()`
- **[aimu/prompts/classification.py](aimu/prompts/classification.py)**: `ClassificationPromptTuner`
  - Hill-climbing optimization to achieve 100% classification accuracy
  - `tune_prompt(training_data)`: Main tuning loop
  - `classify_data()`, `evaluate_results()`, mutation prompt helpers
  - Thinking models automatically get higher `max_new_tokens` and `temperature`

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
- Service-specific subclasses (`LMStudioOpenAIClient`, `OllamaOpenAIClient`, `HFOpenAIClient`, `VLLMOpenAIClient`) each define their own `Model` enum with service-appropriate model IDs and a default `base_url`
- Thinking model support uses `<think>...</think>` tag parsing (via `_split_thinking` and `_ThinkingParser` in [aimu/models/_thinking.py](aimu/models/_thinking.py)) since OpenAI-compat endpoints embed thinking in content rather than a dedicated field; `LlamaCppClient` uses the same shared utilities
- Pass `tools=openai.NOT_GIVEN` (not `tools=[]`) when no tools are available — some local servers reject an empty tools array
- Model ID format varies by service: LM Studio uses loaded model keys, Ollama uses `name:tag` format, HF Transformers Serve and vLLM use HuggingFace repo paths (e.g. `Qwen/Qwen3-8B`)
- Test with `pytest tests/test_models.py --client=lmstudio_openai` (or `ollama_openai`, `hf_openai`, `vllm_openai`)

### LlamaCppClient Notes

- `LlamaCppClient` loads a GGUF file in-process via `llama_cpp.Llama` — no external service or server required
- Constructor requires `model_path` (path to GGUF file); `model` enum specifies capability flags (`supports_tools`, `supports_thinking`)
- `n_gpu_layers=-1` offloads all layers to GPU (default); `n_gpu_layers=0` forces CPU-only inference
- `chat_format=None` (default) auto-detects from GGUF metadata; specify explicitly (e.g. `"chatml-function-calling"`) for older GGUFs that lack embedded templates
- llama-cpp-python returns plain Python dicts, not OpenAI SDK objects — all response access uses dict indexing (`response["choices"][0]["message"]["content"]`)
- Tool calling requires both `supports_tools=True` on the model enum AND a compatible `chat_format`; modern GGUFs auto-detect this
- Thinking support works the same as `OpenAICompatClient` — `<think>...</think>` tags in content, parsed by shared utilities in `_thinking.py`
- Test with `pytest tests/test_models.py --client=llamacpp --model-path=/path/to/model.gguf`

### AisuiteClient Notes

- `AisuiteClient` defines `TOOL_MODELS` and `THINKING_MODELS` via the shared `classproperty` pattern, derived from `supports_tools` / `supports_thinking` flags on `AisuiteModel` enum members
- The client checks aisuite capabilities to decide whether to pass `max_tokens` vs `max_completion_tokens`

## Project Structure

```
aimu/
├── models/              # Model client implementations
│   ├── base_client.py   # Abstract base class (ModelClient, StreamChunk, StreamPhase, Model)
│   ├── _thinking.py     # Shared thinking utilities (_split_thinking, _ThinkingParser)
│   ├── ollama/          # Ollama client (OllamaClient, OllamaModel)
│   ├── hf/              # Hugging Face client (HuggingFaceClient, HuggingFaceModel, ToolCallPrefix)
│   ├── aisuite/         # Cloud models client (AisuiteClient, AisuiteModel)
│   ├── openai_compat/   # OpenAI-compatible clients (requires openai package)
│   │   ├── openai_compat_client.py   # OpenAICompatClient base
│   │   ├── lmstudio_openai_client.py # LMStudioOpenAIClient + LMStudioOpenAIModel
│   │   ├── ollama_openai_client.py   # OllamaOpenAIClient + OllamaOpenAIModel
│   │   ├── hf_openai_client.py       # HFOpenAIClient + HFOpenAIModel
│   │   └── vllm_openai_client.py     # VLLMOpenAIClient + VLLMOpenAIModel
│   └── llamacpp/        # llama-cpp-python client (requires llamacpp package)
│       └── llamacpp_client.py        # LlamaCppClient + LlamaCppModel
├── tools/               # MCP tools integration
│   ├── client.py        # MCPClient wrapper
│   └── servers.py       # Example FastMCP server with built-in tools
├── agents/              # Agentic workflows
│   ├── agent.py         # Agent class + AgentChunk (agentic loop over ModelClient.chat())
│   └── workflow.py      # Workflow class + WorkflowChunk (sequential agent chaining)
├── history.py           # Conversation management (TinyDB)
├── memory.py            # Semantic memory store (ChromaDB); also runnable as MCP server
├── prompts/             # Prompt storage and management
│   ├── catalog.py       # SQLAlchemy-based prompt catalog with versioning
│   └── classification.py # ClassificationPromptTuner with hill-climbing optimization
└── paths.py             # Path configuration (root, tests, package, output)

tests/                   # Pytest test suite
├── conftest.py          # Pytest configuration (--client, --model options)
├── test_models.py       # Model client tests (generate, chat, streaming, tools, thinking)
├── test_history.py      # Conversation management tests
├── test_memory.py       # Semantic memory tests
├── test_tools.py        # MCP tools tests
├── test_agents.py       # Agent and Workflow tests (mock ModelClient, no real backend needed)
└── test_prompt_*.py     # Prompt catalog and tuning tests

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
