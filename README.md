[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

A Python package containing easy to use tools for working with various language models and AI services. AIMU is specifically designed for running models locally, using Ollama or Hugging Face Transformers. However, it can also be used with cloud models (OpenAI, Anthropic, Google, etc.) with [aisuite](https://github.com/andrewyng/aisuite) support (in development).

## Features

-   **Model Clients**: Support for multiple AI model providers including:

    -   [Ollama](https://ollama.com/) (local models)
    -   [Hugging Face Transformers](https://huggingface.co/docs/transformers) (local models)
    -   [aisuite](https://github.com/andrewyng/aisuite) supported models (cloud and local models), including OpenAI (others coming)

-   **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities. Provides a simple(r) interface for [FastMCP 2.0](https://gofastmcp.com).

-   **Chat Conversation (Memory) Storage/Management**: Chat conversation history management using [TinyDB](https://tinydb.readthedocs.io).

-   **Prompt Storage/Management**: Prompt catalog for storing and versioning prompts using [SQLAlchemy](https://www.sqlalchemy.org/).

## Components

------------------------------------------------------------------------

In addition to the AIMU package in the 'aimu' directory, the AIMU code repository includes:

-   Jupyter notebooks demonstrating key AIMU features.

-   A example chat client, built with [Streamlit](https://streamlit.io/), using AIMU Model Client, MCP tools support, and chat conversation management.

-   A full suite of Pytest tests.

## Installation

------------------------------------------------------------------------

AIMU can be installed with Ollama support, Hugging Face (Transformers) support, or both! For both, simply install the full package.

``` bash
pip install aimu
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

## Development

------------------------------------------------------------------------

Once you've cloned the repository, run the following command to install all dependencies:

``` bash
pip install -e '.[all]'
```

If you have [uv](https://docs.astral.sh/uv/) installed, you can get all dependencies with:

``` bash
uv sync --all-extras
```

## Usage

------------------------------------------------------------------------

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

### Prompt Storage/Management

``` python
from aimu.prompts import PromptCatalog, Prompt

prompt_catalog = PromptCatalog("prompts.db")

prompt = Prompt("You are a helpful assistant", model_id="llama3.1:8b", version=1)
prompt_catalog.store_prompt(prompt)
```

## License

------------------------------------------------------------------------

This project is licensed under the Apache 2.0 license.
