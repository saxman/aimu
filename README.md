# AIMU - AI Model Utilities

A Python package containing core AI model functionality, tools, and prompts for working with various language models and AI services.

## Features

- **Model Clients**: Support for multiple AI model providers including:
  - Ollama (local models)
  - Hugging Face Transformers
  - OpenAI (via aisuite)
  - And more

- **Prompt Management**: SQLAlchemy-based prompt catalog for storing and versioning prompts

- **MCP Tools**: Model Context Protocol (MCP) integration for enhanced AI capabilities

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
client = OllamaClient("llama3.1:8b")
response = client.generate("Hello, world!", {"temperature": 0.7})

# Using Hugging Face
hf_client = HuggingFaceClient("meta-llama/Meta-Llama-3.1-8B-Instruct")
response = hf_client.chat({"role": "user", "content": "Hello!"})
```

### Prompt Management

```python
from aimu.prompts import PromptCatalog, Prompt

catalog = PromptCatalog("prompts.db")
prompt = Prompt(prompt="You are a helpful assistant", model_id="llama3.1:8b", version=1)
catalog.store_prompt(prompt)
```

## License

This project is licensed under the same terms as the original genscai project.