# AIMU - AI Model Utilities

A Python package containing tools for working with various language models and AI services.

## Features

- **Model Clients**: Support for multiple AI model providers including:
  - Ollama (local models)
  - Hugging Face Transformers (local models)
  - Cloud Providers, such as OpenAI, Anthropic, Google, AWS, etc. (via aisuite)

- **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities

- **Prompt Management**: SQLAlchemy-based prompt catalog for storing and versioning prompts

## Installation

```bash
pip install aimu
```

## Development

Once you've cloned the repository, run:

```bash
pip install -e .
```

For running tests:
```bash
pip install -e '.[dev]'
```

## Usage

### Basic Model Usage

```python
from aimu.models import OllamaClient, HuggingFaceClient

# Using Ollama
client = OllamaClient(OllamaClient.MODEL_LLAMA_3_1_8B)
response = client.generate("What is the capital of France?", {"temperature": 0.7})

# Using Hugging Face
hf_client = HuggingFaceClient(HuggingFaceClient.MODEL_LLAMA_3_1_8B)
response = hf_client.chat({"role": "user", "content": "Hello!"})
```

### Tool usage

```python
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

mcp_client.call_tool("mytool", {"input": "hello world!"})
```

### Tool usage with ModelClients
```python
from aimu.models import OllamaClient
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

model_client = OllamaClient(OllamaClient.MODEL_LLAMA_3_1_8B)

model_client.chat(
  {
    "role": "user",
    "content": "Use my tool please."
  },
  tools=mcp_client.get_tools()
)
```

### Prompt Management

```python
from aimu.prompts import PromptCatalog, Prompt

catalog = PromptCatalog("prompts.db")
prompt = Prompt("You are a helpful assistant", model_id="llama3.1:8b", version=1)
catalog.store_prompt(prompt)
```

## License

This project is licensed under the Apache 2.0 license.
