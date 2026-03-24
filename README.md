[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

A Python package containing easy to use tools for working with various language models and AI services. AIMU is specifically designed for running models locally, using Ollama or Hugging Face Transformers. However, it can also be used with cloud models (OpenAI, Anthropic, Google, etc.) with [aisuite](https://github.com/andrewyng/aisuite) support (in development).

## Features

-   **Model Clients**: Support for multiple AI model providers including:

    -   [Ollama](https://ollama.com/) (local models)
    -   [Hugging Face Transformers](https://huggingface.co/docs/transformers) (local models)
    -   [aisuite](https://github.com/andrewyng/aisuite) supported models (cloud and local models), including OpenAI (others coming)

-   **Thinking Models**: First-class support for extended reasoning models (e.g. DeepSeek-R1, Qwen3, GPT-OSS). Thinking is enabled automatically for supported models, with access to the reasoning traces.

-   **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities. Provides a simple(r) interface for [FastMCP 2.0](https://gofastmcp.com).

-   **Chat Conversation Storage/Management**: Chat conversation history management using [TinyDB](https://tinydb.readthedocs.io).

-   **Semantic Memory Storage**: Persistent fact memory using [ChromaDB](https://www.trychroma.com/). Facts are stored as natural-language subject-predicate-object strings (e.g. `"Paul works at Google"`) and retrieved by semantic topic (e.g. `"employment"`, `"family life"`).

-   **Prompt Storage/Management**: Prompt catalog for storing and versioning prompts using [SQLAlchemy](https://www.sqlalchemy.org/).

## Components

In addition to the AIMU package in the 'aimu' directory, the AIMU code repository includes:

-   Jupyter notebooks demonstrating key AIMU features.

-   A example chat client, built with [Streamlit](https://streamlit.io/), using AIMU Model Client, MCP tools support, and chat conversation management.

-   A full suite of Pytest tests.

## Installation

AIMU can be installed with Ollama support, Hugging Face Transformers support, and/or aisuite (cloud models) support.

For all features, run:

``` bash
pip install aimu[all]
```

Alternatively, for Ollama-only support:

``` bash
pip install aimu[ollama]
```

For Hugging Face Tranformers model support:

``` bash
pip install aimu[hf]
```

For aisuite models (e.g. OpenAI):

``` bash
pip install aimu[aisuite]
```

For accessing potentially gated models via Hugging Face, you'll need to get and store (locally) a [Hugging Face Hub access token](https://huggingface.co/docs/huggingface_hub/en/quick-start). Once you have a token, you can install it locally with:

``` bash
hf auth login
```

## Development

Once you've cloned the repository, run the following command to install all model dependencies:

``` bash
pip install -e '.[all]'
```

Additionally, run the following command to install development (testing, linting) and notebook dependencies:

``` bash
pip install -e '.[dev,notebooks]'
```

Alternatively, if you have [uv](https://docs.astral.sh/uv/) installed, you can get all model and development dependencies with:

``` bash
uv sync --all-extras
```

Using Pytest, tests can be run for a specific model client and/or model, using optional arguments:

``` bash
pytest tests\test_models.py --client=ollama --model=GPT_OSS_20B
```

## Usage

### Text Generation

``` python
from aimu.models import OllamaClient as ModelClient ## or HuggingFaceClient, or AisuiteClient

model_client = ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
response = model_client.generate("What is the capital of France?", {"temperature": 0.7})
```

### Chat

``` python
from aimu.models import OllamaClient as ModelClient

model_client = ModelClient(ModelClient.MODELS.LLAMA_3_1_8B)
response = model_client.chat("What is the capital of France?")
```

### Thinking Models

Models with extended reasoning capabilities (e.g. DeepSeek-R1, Qwen3, GPT-OSS) are identified by the `THINKING_MODELS` list on each client. Thinking is enabled automatically when one of these models is selected — no extra configuration required.

After generation, the model's reasoning trace is available in `last_thinking`:

``` python
from aimu.models import OllamaClient as ModelClient

model_client = ModelClient(ModelClient.MODELS.DEEPSEEK_R1_8B)
response = model_client.generate("What is the capital of France?")

print(model_client.last_thinking)  # reasoning trace
print(response)                     # final answer
```

During streamed generation via `generate_streamed()`, thinking tokens are yielded first followed by the response tokens as a single flat stream. For phase-separated streaming (thinking, tool calls, response), use `chat_streamed()` instead.

### Streamed Chat

`chat_streamed()` yields `StreamChunk(phase, content)` objects. The `phase` field is a `StreamPhase` enum:

| Phase | Content | Description |
|---|---|---|
| `StreamPhase.THINKING` | `str` | Reasoning token (thinking models only) |
| `StreamPhase.TOOL_CALLING` | `{"name": str, "response": str}` | Tool call and its result |
| `StreamPhase.GENERATING` | `str` | Final response token |
| `StreamPhase.DONE` | `""` | Stream complete |

``` python
from aimu.models import OllamaClient as ModelClient, StreamPhase

model_client = ModelClient(ModelClient.MODELS.LLAMA_3_1_8B)

for chunk in model_client.chat_streamed("What is the capital of France?"):
    if chunk.phase == StreamPhase.THINKING:
        print(f"[thinking] {chunk.content}", end="", flush=True)
    elif chunk.phase == StreamPhase.TOOL_CALLING:
        print(f"\n[tool: {chunk.content['name']}] {chunk.content['response']}")
    elif chunk.phase == StreamPhase.GENERATING:
        print(chunk.content, end="", flush=True)
```

### Chat UI (Streamlit)

``` bash
cd streamlit
streamlit run streamlit/chatbot_example.py
```

### MCP Tool Usage

``` python
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

mcp_client.call_tool("mytool", {"input": "hello world!"})
```

### MCP Tool Usage with ModelClient

``` python
from aimu.models import OllamaClient as ModelClient
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

model_client = ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
model_client.mcp_client = mcp_client

model_client.chat("use my tool please")
```

### Chat Conversation Storage/Management

``` python
from aimu.models import OllamaClient as ModelClient
from aimu.memory import ConversationManager

chat_manager = ConversationManager("conversations.json", use_last_conversation=True) # loads the last saved convesation

model_client = new ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
model_client.messages = chat_manager.messages

model_client.chat("What is the capital of France?")

chat_manager.update_conversation(model_client.messages) # store the updated conversation
```

### Semantic Memory Storage

``` python
from aimu.memory import MemoryStore

store = MemoryStore(persist_path="./memory_store")

store.store_fact("Paul works at Google")
store.store_fact("Paul is married to Sarah")
store.store_fact("Sarah is the sister of Emma")

store.retrieve_facts("work and employment")   # ["Paul works at Google", ...]
store.retrieve_facts("family relationships")  # ["Paul is married to Sarah", ...]
```

### Prompt Storage/Management

``` python
from aimu.prompts import PromptCatalog, Prompt

prompt_catalog = PromptCatalog("prompts.db")

prompt = Prompt("You are a helpful assistant", model_id="llama3.1:8b", version=1)
prompt_catalog.store_prompt(prompt)
```

## License

This project is licensed under the Apache 2.0 license.
