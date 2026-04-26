[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

A Python package containing easy to use tools for working with various language models and AI services. AIMU is designed for running models locally via Ollama, Hugging Face Transformers, or any OpenAI-compatible local serving framework, and for cloud models via native provider SDKs (OpenAI, Anthropic, Google Gemini).

## Table of Contents

-   [Features](#features)
-   [Components](#components)
-   [Examples](#examples)
-   [Installation](#installation)
-   [Development](#development)
-   [Usage](#usage)
    -   [Model Clients](#model-clients)
    -   [Agents & Workflows](#agents--workflows)
    -   [MCP Tools](#mcp-tools)
    -   [Persistence](#persistence)
    -   [Prompt Management](#prompt-management)
-   [License](#license)

## Features

-   **Model Clients**: Support for multiple AI model providers including:

    -   [Ollama](https://ollama.com/) (local models, native API)
    -   [Hugging Face Transformers](https://huggingface.co/docs/transformers) (local models)
    -   [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (local GGUF models, in-process, no external service required)
    -   [Anthropic](https://www.anthropic.com/) Claude models via native `anthropic` SDK (`AnthropicClient`) - native thinking support
    -   Cloud and local servers via the `openai` SDK (`aimu[openai_compat]`):
        -   [OpenAI](https://platform.openai.com/) (`OpenAIClient`) - GPT-4o, GPT-4.1, o3, o4-mini, and more
        -   [Google Gemini](https://ai.google.dev/) (`GeminiClient`) - Gemini 2.0/2.5 via Google's OpenAI-compatible endpoint
        -   [LM Studio](https://lmstudio.ai/) (`LMStudioOpenAIClient`)
        -   [Ollama](https://ollama.com/) OpenAI-compat endpoint (`OllamaOpenAIClient`)
        -   [HuggingFace Transformers Serve](https://huggingface.co/docs/transformers/main/serving) (`HFOpenAIClient`)
        -   [vLLM](https://docs.vllm.ai/) (`VLLMOpenAIClient`)
        -   [llama.cpp llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) (`LlamaServerOpenAIClient`)
        -   [SGLang](https://docs.sglang.ai/) (`SGLangOpenAIClient`)
        -   Any OpenAI-compatible server (`OpenAICompatClient`)
    -   **Thinking Models**: First-class support for extended reasoning models (e.g. DeepSeek-R1, Qwen3, GPT-OSS). Enabled automatically for supported models, with access to reasoning traces via `last_thinking` and `StreamPhase.THINKING` stream chunks.

-   **Agents & Workflows**: Per Anthropic's taxonomy, AIMU separates autonomous agents from code-controlled workflows. All share a `Runner` base class with `run()` / `run_streamed()`.

    -   **Agents**: `SimpleAgent` runs an autonomous tool-calling loop until the model stops invoking tools. `SkillAgent` extends it with automatic skill injection. `AgenticModelClient` wraps a `SimpleAgent` behind the standard `BaseModelClient` interface, making agentic and single-turn clients interchangeable.
    -   **Workflows**: Four code-controlled patterns: `Chain` (prompt chaining), `Router` (classify and dispatch), `Parallel` (concurrent workers with optional aggregation), and `EvaluatorOptimizer` (generate → evaluate → revise loop).
    -   **Skills**: Filesystem-discovered `SKILL.md` files that inject instructions and tools into `SkillAgent` automatically. Skills are discovered from project and user directories (`.agents/skills/`, `.claude/skills/`).

-   **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities. Provides a simple(r) interface for [FastMCP 2.0](https://gofastmcp.com).

-   **Persistence**: Three complementary stores for persisting conversation and knowledge:

    -   **Conversation History** (`ConversationManager`): Persistent chat message history backed by [TinyDB](https://tinydb.readthedocs.io). Load the last conversation on startup and save updates after each turn.
    -   **Semantic Memory** (`SemanticMemoryStore`): Fact storage using [ChromaDB](https://www.trychroma.com/) vector embeddings. Store natural-language subject-predicate-object strings (e.g. `"Paul works at Google"`) and retrieve by semantic topic (e.g. `"employment"`, `"family life"`).
    -   **Document Memory** (`DocumentStore`): Path-based document store mirroring Anthropic's Managed Agents Memory API. Supports `write`, `read`, `edit`, `delete`, and full-text `search` on named paths (e.g. `/preferences.md`).

-   **Prompt Management**: Versioned prompt storage and automatic prompt optimization:

    -   **Prompt Storage**: Versioned prompt catalog backed by SQLite ([SQLAlchemy](https://www.sqlalchemy.org/)). Prompts are keyed by `(name, model_id)` and auto-versioned on each store.
    -   **Prompt Tuning**: Hill-climbing `PromptTuner` for automatic prompt optimization. Four concrete tuners: `ClassificationPromptTuner` (binary YES/NO), `MultiClassPromptTuner` (N-way), `ExtractionPromptTuner` (JSON field extraction), and `JudgedPromptTuner` (open-ended generation rated by a second LLM). Subclass `PromptTuner` to implement custom task types.

## Components

In addition to the AIMU package in the 'aimu' directory, the AIMU code repository includes:

-   Jupyter notebooks demonstrating key AIMU features.

-   Example chat clients in the `web/` directory, built with [Streamlit](https://streamlit.io/) and [Gradio](https://www.gradio.app/), using AIMU Model Client, MCP tools support, and chat conversation management.

-   A full suite of Pytest tests.

## Examples

The following Jupyter notebooks demonstrate key AIMU features:

| Notebook | Description |
|---|---|
| [01 - Model Client](notebooks/01%20-%20Model%20Client.ipynb) | Text generation, chat, streaming, and thinking models |
| [02 - MCP Tools](notebooks/02%20-%20MCP%20Tools.ipynb) | MCP tool integration with model clients |
| [03 - Prompt Management](notebooks/03%20-%20Prompt%20Management.ipynb) | Versioned prompt storage |
| [04 - Conversations](notebooks/04%20-%20Conversations.ipynb) | Persistent chat conversation management |
| [05 - Memory](notebooks/05%20-%20Memory.ipynb) | Semantic fact storage and retrieval |
| [06 - Agents](notebooks/06%20-%20Agents.ipynb) | SimpleAgent and AgenticModelClient |
| [07 - Agent Skills](notebooks/07%20-%20Agent%20Skills.ipynb) | Filesystem-discovered skill injection with SkillAgent |
| [08 - Agent Workflows](notebooks/08%20-%20Agent%20Workflows.ipynb) | Chain, Router, Parallel, and EvaluatorOptimizer patterns |
| [09 - Prompt Tuning](notebooks/03%20-%20Prompt%20Tuning.ipynb) | Prompt tuning |

## Installation

For all features, run:

``` bash
pip install aimu[all]
```

Or install only what you need:

``` bash
pip install aimu[ollama]        # Ollama (local models, native API)
pip install aimu[hf]            # Hugging Face Transformers (local models)
pip install aimu[anthropic]     # Anthropic Claude models
pip install aimu[openai_compat] # OpenAI, Google Gemini, and OpenAI-compatible local servers
pip install aimu[llamacpp]      # Local GGUF models via llama-cpp-python (no external service)
```

For gated Hugging Face models, you'll need a [Hugging Face Hub access token](https://huggingface.co/docs/huggingface_hub/en/quick-start):

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

### Model Clients

All clients implement the `BaseModelClient` abstract interface. `ModelClient` is a factory that automatically selects the right client from a model enum, so you can swap providers without changing call sites:

``` python
from aimu.models import ModelClient
from aimu.models.ollama.ollama_client import OllamaModel

client = ModelClient(OllamaModel.QWEN_3_8B)  # factory: picks OllamaClient automatically

response = client.generate("Summarise this text.", {"temperature": 0.7})  # stateless
response = client.chat("What is the capital of France?")                  # multi-turn
print(client.messages)  # full message history
```

You can also import the concrete client directly when you prefer explicit control:

``` python
from aimu.models.ollama import OllamaClient, OllamaModel

client = OllamaClient(OllamaModel.QWEN_3_8B)
```

Cloud and local server clients follow the same pattern; only the model enum (and optional kwargs) differ:

| Client | Extra | API key / notes |
|---|---|---|
| `OllamaClient` | `aimu[ollama]` | - |
| `HuggingFaceClient` | `aimu[hf]` | - |
| `LlamaCppClient` | `aimu[llamacpp]` | `model_path=` (GGUF file); no external service |
| `OpenAIClient` | `aimu[openai_compat]` | `OPENAI_API_KEY` |
| `AnthropicClient` | `aimu[anthropic]` | `ANTHROPIC_API_KEY` |
| `GeminiClient` | `aimu[openai_compat]` | `GOOGLE_API_KEY` |
| `LMStudioOpenAIClient` | `aimu[openai_compat]` | localhost:1234 |
| `OllamaOpenAIClient` | `aimu[openai_compat]` | localhost:11434 |
| `VLLMOpenAIClient` | `aimu[openai_compat]` | localhost:8000 |
| `LlamaServerOpenAIClient` | `aimu[openai_compat]` | localhost:8080 |
| `SGLangOpenAIClient` | `aimu[openai_compat]` | localhost:30000 |

**Streaming**: `chat(..., stream=True)` yields `StreamChunk` objects tagged by phase:

| `chunk.phase` | `chunk.content` type | Description |
|---|---|---|
| `StreamPhase.THINKING` | `str` | Reasoning token (thinking models only) |
| `StreamPhase.TOOL_CALLING` | `dict` `{"name": str, "response": str}` | Tool call and result |
| `StreamPhase.GENERATING` | `str` | Final response token |

**Thinking models**: extended reasoning (e.g. DeepSeek-R1, Qwen3, GPT-OSS) is enabled automatically for supported models. The reasoning trace is available in `client.last_thinking` after generation, or as `StreamPhase.THINKING` chunks during streaming.

**Chat UIs**: full-featured UIs with streaming, tool calls, and conversation persistence: `streamlit run web/streamlit_chatbot.py` ([Streamlit](web/streamlit_chatbot.py)) or `python web/gradio_chatbot.py` ([Gradio](web/gradio_chatbot.py)).

See [01 - Model Client](notebooks/01%20-%20Model%20Client.ipynb) for detailed examples.

### Agents & Workflows

`SimpleAgent` wraps a `ModelClient` and runs a tool-calling loop until the model stops invoking tools:

``` python
from aimu.models.ollama import OllamaClient, OllamaModel
from aimu.tools import MCPClient
from aimu.agents import SimpleAgent

client = OllamaClient(OllamaModel.QWEN_3_8B)
client.mcp_client = MCPClient({"mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}}})

agent = SimpleAgent.from_config(
    {"name": "researcher", "system_message": "Use tools to answer.", "max_iterations": 8},
    client,
)
result = agent.run("Find all log files modified today and summarise the errors.")
```

`SkillAgent` extends `SimpleAgent` with automatic discovery and injection of `SKILL.md` skill files:

``` python
from aimu.agents import SkillAgent

agent = SkillAgent(client, name="assistant")  # discovers skills from .agents/skills/ and .claude/skills/
result = agent.run("Use the pdf-processing skill to extract pages from report.pdf")
```

Workflow patterns have code-controlled flow. `Chain` sequences agents so each step's output becomes the next step's input:

``` python
from aimu.agents import Chain

chain = Chain.from_config(
    [
        {"name": "planner",   "system_message": "Break the task into steps.", "max_iterations": 3},
        {"name": "executor",  "system_message": "Execute each step using tools.", "max_iterations": 10},
        {"name": "formatter", "system_message": "Format the results clearly.", "max_iterations": 1},
    ],
    client,
)
result = chain.run("Research the top Python web frameworks.")
```

Every `Runner` exposes `.messages`, a `dict[str, list[dict]]` mapping each agent name to its message history snapshot after a run.

See [06 - Agents](notebooks/06%20-%20Agents.ipynb), [07 - Agent Skills](notebooks/07%20-%20Agent%20Skills.ipynb), and [08 - Agent Workflows](notebooks/08%20-%20Agent%20Workflows.ipynb).

### MCP Tools

`MCPClient` wraps a FastMCP 2.0 server and integrates with any `ModelClient` via `model_client.mcp_client`:

``` python
from aimu.models import ModelClient
from aimu.models.ollama.ollama_client import OllamaModel
from aimu.tools import MCPClient

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})

# Use standalone
mcp_client.call_tool("mytool", {"input": "hello world!"})

# Or attach to a model client; tools are passed to the model automatically
model_client = ModelClient(OllamaModel.QWEN_3_8B)
model_client.mcp_client = mcp_client
model_client.chat("use my tool please")
```

See [02 - MCP Tools](notebooks/02%20-%20MCP%20Tools.ipynb).

### Persistence

**Conversation history**: `ConversationManager` persists chat message sequences across sessions:

``` python
from aimu.models import ModelClient
from aimu.models.ollama.ollama_client import OllamaModel
from aimu.history import ConversationManager

manager = ConversationManager("conversations.json", use_last_conversation=True)
model_client = ModelClient(OllamaModel.QWEN_3_8B)
model_client.messages = manager.messages

model_client.chat("What is the capital of France?")
manager.update_conversation(model_client.messages)
```

**Semantic memory**: `SemanticMemoryStore` stores and retrieves facts by semantic similarity:

``` python
from aimu.memory import SemanticMemoryStore

store = SemanticMemoryStore(persist_path="./memory_store")
store.store("Paul works at Google")
store.search("employment")                    # ["Paul works at Google"]
store.search("employment", max_distance=0.4)  # only close matches
```

**Document memory**: `DocumentStore` is a path-keyed document store mirroring Anthropic's Managed Agents Memory API:

``` python
from aimu.memory import DocumentStore

store = DocumentStore(persist_path="./doc_store")
store.write("/preferences.md", "Always use concise responses.")
store.edit("/preferences.md", "concise", "detailed")
store.search_full_text("detailed")
```

See [04 - Conversations](notebooks/04%20-%20Conversations.ipynb) and [05 - Memory](notebooks/05%20-%20Memory.ipynb).

### Prompt Management

**Prompt catalog**: `PromptCatalog` stores versioned prompts keyed by `(name, model_id)`:

``` python
from aimu.prompts import PromptCatalog, Prompt

with PromptCatalog("prompts.db") as catalog:
    prompt = Prompt(name="summarizer", prompt="Summarize the following: {content}", model_id="llama3.1:8b")
    catalog.store_prompt(prompt)  # version and created_at assigned automatically

    latest = catalog.retrieve_last("summarizer", "llama3.1:8b")
    print(f"v{latest.version}: {latest.prompt}")
```

**Prompt tuning**: `PromptTuner` runs a hill-climbing loop to automatically improve a prompt against labelled data. Pass a DataFrame with `content` and `actual_class` columns:

``` python
import pandas as pd
from aimu.prompts import ClassificationPromptTuner

tuner = ClassificationPromptTuner(model_client=client)
df = pd.DataFrame({
    "content": ["LLMs are transforming AI.", "The recipe calls for flour.", ...],
    "actual_class": [True, False, ...],
})
best_prompt = tuner.tune(df, initial_prompt="Is this about AI? Reply [YES] or [NO]. Content: {content}")
```

`MultiClassPromptTuner`, `ExtractionPromptTuner`, and `JudgedPromptTuner` follow the same pattern. Subclass `PromptTuner` and implement `apply_prompt`, `evaluate`, and `mutation_prompt` for custom task types.

See [03 - Prompts](notebooks/03%20-%20Prompts.ipynb).

## License

This project is licensed under the Apache 2.0 license.
