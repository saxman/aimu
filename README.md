# AIMU - AI Model Utilities

A Python package containing tools for working with various language models and AI services.

## Features

- **Model Clients**: Support for multiple AI model providers including:
  - Ollama (local models)
  - Hugging Face Transformers (local models)
  - OpenAI (via aisuite)
  - And more

- **Prompt Management**: SQLAlchemy-based prompt catalog for storing and versioning prompts

- **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities

## Installation

```bash
pip install -e .
```

For development:
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

### Prompt Management

```python
from aimu.prompts import PromptCatalog, Prompt

catalog = PromptCatalog("prompts.db")
prompt = Prompt("You are a helpful assistant", model_id="llama3.1:8b", version=1)
catalog.store_prompt(prompt)
```

## License

This project is licensed under the Apache 2.0 license.
