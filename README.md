# AIMU - AI Model Utilities

A Python package containing tools for working with various language models and AI services.

## Features

- **Model Clients**: Support for multiple AI model providers including:
  - Ollama (local models)
  - Hugging Face Transformers (local models)
  - Cloud Providers, such as OpenAI, Anthropic, Google, AWS, etc. (via aisuite)

- **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities

- **Chat Conversation (Memory) Storage/Management**: TinyDB-based memory for chat conversations

- **Prompt Storage/Management**: SQLAlchemy-based prompt catalog for storing and versioning prompts

## Installation

AIMU can be installed with Ollama support, Hugging Face (Transformers) support, or both! For both, simply install the full package.

```bash
pip install aimu
```

Alternatively, for Ollama-only support:

```bash
pip install aimu[ollama]
```

Or for Hugging Face:

```bash
pip install aimu[hf]
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

### Text Generation

```python
from aimu.models import OllamaClient as ModelClient ## or HuggingFaceClient

model_client = ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
response = model_client.generate("What is the capital of France?", {"temperature": 0.7})
```

### Chat
```python
from aimu.models import OllamaClient as ModelClient

model_client = ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
model_client.messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
]

response = model_client.chat("What is the capital of France?")
```

### MCP Tool Usage

```python
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

mcp_client.call_tool("mytool", {"input": "hello world!"})
```

### MCP Tool Usage with ModelClients
```python
from aimu.models import OllamaClient as ModelClient
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

model_client = ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
model_client.mcp_client = mcp_client

model_client.chat(
  "use my tool please",
  tools=mcp_client.get_tools()
)
```

### Chat Conversation Storage/Management
```python
from aimu.models import OllamaClient as ModelClient
from aimu.memory import ConversationManager

chat_manager = ConversationManager("conversations.json", use_last_conversation=True) # loads the last saved convesation

model_client = new ModelClient(ModelClient.MODEL_LLAMA_3_1_8B)
model_client.messages = chat_manager.messages

model_client.chat("What is the capital of France?")

chat_manager.update_conversation(model_client.messages) # store the updated conversation
```

### Prompt Storage/Management

```python
from aimu.prompts import PromptCatalog, Prompt

catalog = PromptCatalog("prompts.db")
prompt = Prompt("You are a helpful assistant", model_id="llama3.1:8b", version=1)
catalog.store_prompt(prompt)
```

## License

This project is licensed under the Apache 2.0 license.
