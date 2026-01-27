# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIMU (AI Model Utilities) is a Python package for working with language models, specifically designed for running models locally via Ollama or Hugging Face Transformers, with support for cloud models through aisuite. The package provides unified interfaces for model interaction, MCP tool integration, conversation management, and prompt storage.

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
pip install -e '.[ollama]'      # Ollama only
pip install -e '.[hf]'          # Hugging Face only
pip install -e '.[aisuite]'     # Cloud models (OpenAI, Anthropic, etc.)
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
```

### Linting

The project uses Ruff for linting and formatting (configured in pyproject.toml with line length 120):
```bash
ruff check .
ruff format .
```

### Running the Chat UI

Launch the Streamlit example chat application:
```bash
streamlit run streamlit/chatbot_example.py
```

## Architecture

### Model Client Architecture

The codebase uses an abstract base class pattern for model clients:

- **[aimu/models/base_client.py](aimu/models/base_client.py)**: Defines `ModelClient` abstract base class with core methods:
  - `generate()` / `generate_streamed()`: Single-turn text generation
  - `chat()` / `chat_streamed()`: Multi-turn chat with message history
  - `_handle_tool_calls()`: Universal tool calling logic (MCP or Python functions)
  - Message history stored in `self.messages` (OpenAI-style format)
  - Optional `mcp_client` attribute for MCP tool integration

- **Model Implementations**:
  - [aimu/models/ollama/ollama_client.py](aimu/models/ollama/ollama_client.py): `OllamaClient` for local Ollama models
  - [aimu/models/hf/](aimu/models/hf/): `HuggingFaceClient` for local Transformers models
  - [aimu/models/aisuite/](aimu/models/aisuite/): `AisuiteClient` for cloud models (OpenAI, Anthropic, etc.)

- **Model Definitions**: Each client defines:
  - `MODELS`: Enum of supported models (e.g., `OllamaModel.LLAMA_3_1_8B`)
  - `TOOL_MODELS`: List of models supporting tool/function calling
  - `THINKING_MODELS`: List of models with extended reasoning capabilities

- **Optional Dependencies**: [aimu/models/__init__.py](aimu/models/__init__.py:5-37) gracefully handles missing dependencies - clients only import if their dependencies are installed.

### MCP Tools Integration

- **[aimu/tools/client.py](aimu/tools/client.py)**: `MCPClient` wraps FastMCP 2.0 client
  - Converts async FastMCP API to synchronous interface using event loop
  - `get_tools()`: Returns tools in OpenAI function calling format
  - `call_tool()`: Synchronous tool execution
  - Integrates with ModelClient via `model_client.mcp_client` attribute

- **Tool Calling Flow**:
  1. If model supports tools and `mcp_client` is set, tools are passed to model
  2. Model returns tool calls in response
  3. `_handle_tool_calls()` executes tools via MCP client
  4. Tool results added to message history
  5. Model called again with tool results to generate final response

### Conversation Management

- **[aimu/history.py](aimu/history.py)**: `ConversationManager` class (previously called "memory" module)
  - Uses TinyDB for JSON-based conversation storage
  - Tracks message history with timestamps
  - Supports loading last conversation or creating new ones
  - Integrates with ModelClient via `model_client.messages`

### Prompt Management

- **[aimu/prompts/catalog.py](aimu/prompts/catalog.py)**: `PromptCatalog` for versioned prompt storage
  - Uses SQLAlchemy for structured prompt storage
  - Supports prompt versioning and model-specific prompts
- **[aimu/prompts/classification.py](aimu/prompts/classification.py)**: Prompt tuning and classification utilities

### Key Design Patterns

1. **Abstract Base Class**: All model clients inherit from `ModelClient` and implement abstract methods
2. **Optional Dependencies**: Graceful degradation when model backends aren't installed
3. **OpenAI-Compatible Format**: Message history uses `{"role": "...", "content": "..."}` format
4. **Tool Calling Abstraction**: Base class handles tool calls uniformly across providers
5. **Streaming Support**: All clients support both streaming and non-streaming generation
6. **Model Capability Lists**: `TOOL_MODELS` and `THINKING_MODELS` define per-model capabilities

## Important Implementation Notes

### Adding New Models

When adding new models to a client:
1. Add model enum to the client's Model class (e.g., `OllamaModel`)
2. Add to `TOOL_MODELS` list if model supports function calling
3. Add to `THINKING_MODELS` list if model has extended reasoning capabilities
4. Test with both `pytest tests/test_models.py --client=<client> --model=<MODEL_NAME>`

### Tool Calling Compatibility

- Tool calls use OpenAI format: `{"type": "function", "function": {"name": "...", "arguments": {...}}}`
- Some models (e.g., Llama 3.1) use `parameters` instead of `arguments` - this is normalized in `_handle_tool_calls()`
- Tool responses must be added to messages with role "tool" and include `tool_call_id`

### Message History Management

- System message is automatically prepended to `messages` if set via `system_message` property
- Message history persists across `chat()` calls on the same client instance
- Use `ConversationManager` to persist conversations across sessions
- Clear history by setting `model_client.messages = []`

### Thinking Models

Models in `THINKING_MODELS` (e.g., GPT-OSS, DeepSeek-R1, QWen3) support extended reasoning:
- Set `self.thinking = True` in client initialization
- Model's reasoning process stored in `self.last_thinking`
- Final answer extracted separately from thinking process

## Project Structure

```
aimu/
├── models/              # Model client implementations
│   ├── base_client.py   # Abstract base class
│   ├── ollama/          # Ollama client
│   ├── hf/              # Hugging Face client
│   └── aisuite/         # Cloud models client
├── tools/               # MCP tools integration
│   ├── client.py        # MCPClient wrapper
│   └── servers.py       # MCP server utilities
├── history.py           # Conversation management (TinyDB)
├── prompts/             # Prompt storage and management
│   ├── catalog.py       # SQLAlchemy-based prompt catalog
│   └── classification.py # Prompt tuning utilities
└── paths.py             # Path utilities

tests/                   # Pytest test suite
├── conftest.py          # Pytest configuration (--client, --model options)
├── test_models.py       # Model client tests
├── test_history.py      # Conversation management tests
├── test_tools.py        # MCP tools tests
└── test_prompt_*.py     # Prompt management tests

streamlit/               # Example Streamlit chat UI
└── chatbot_example.py   # Full-featured chat interface

notebooks/               # Jupyter notebook demos
```

## Testing Strategy

- Tests use pytest fixtures and parameterization
- `conftest.py` adds `--client` and `--model` CLI options for targeted testing
- Tests can run against all clients or specific client/model combinations
- Model clients are auto-pulled/downloaded on first use
- MCP tools tested with mock FastMCP servers
