[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

A Python package containing easy to use tools for working with various language models and AI services. AIMU is designed for running models locally via Ollama, Hugging Face Transformers, or any OpenAI-compatible local serving framework, and for cloud models via native provider SDKs (OpenAI, Anthropic, Google Gemini).

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

-   **Thinking Models**: First-class support for extended reasoning models (e.g. DeepSeek-R1, Qwen3, GPT-OSS). Thinking is enabled automatically for supported models, with access to the reasoning traces.

-   **Agentic Workflows**: `SimpleAgent` and `WorkflowAgent` classes for autonomous, tool-driven task execution, both inheriting from a shared `Agent` ABC. Agents loop over tool calls until the task is complete; workflows chain agents sequentially. `AgenticModelClient` wraps any `Agent` behind the standard `ModelClient` interface, making agentic and single-turn clients interchangeable anywhere a `ModelClient` is accepted.

-   **MCP Tools**: Model Context Protocol (MCP) client for enhancing AI capabilities. Provides a simple(r) interface for [FastMCP 2.0](https://gofastmcp.com).

-   **Chat Conversation Storage/Management**: Chat conversation history management using [TinyDB](https://tinydb.readthedocs.io).

-   **Semantic Memory Storage**: Persistent fact memory using [ChromaDB](https://www.trychroma.com/). Facts are stored as natural-language subject-predicate-object strings (e.g. `"Paul works at Google"`) and retrieved by semantic topic (e.g. `"employment"`, `"family life"`).

-   **Agent Skills**: Filesystem-discovered skill definitions that inject instructions and tools into agents automatically. Skills are YAML-fronted Markdown files discovered from project and user directories.

-   **Prompt Storage/Management**: Versioned prompt catalog backed by SQLite ([SQLAlchemy](https://www.sqlalchemy.org/)), plus a hill-climbing `PromptTuner` for automatic prompt optimization. A `ClassificationPromptTuner` is included for binary YES/NO classification tasks.

## Components

In addition to the AIMU package in the 'aimu' directory, the AIMU code repository includes:

-   Jupyter notebooks demonstrating key AIMU features.

-   Example chat clients in the `web/` directory, built with [Streamlit](https://streamlit.io/) and [Gradio](https://www.gradio.app/), using AIMU Model Client, MCP tools support, and chat conversation management.

-   A full suite of Pytest tests.

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

### Text Generation

``` python
from aimu.models import OllamaClient as ModelClient ## or HuggingFaceClient, or OpenAiCompatClient

model_client = ModelClient(ModelClient.MODELS.QWEN_3_5_9B)
response = model_client.generate("What is the capital of France?", {"temperature": 0.7})
```

### Chat

``` python
from aimu.models import OllamaClient as ModelClient

model_client = ModelClient(ModelClient.MODELS.QWEN_3_5_9B)
response = model_client.chat("What is the capital of France?")

print(model_client.messages)
```

### Thinking Models

Models with extended reasoning capabilities (e.g. DeepSeek-R1, Qwen3, GPT-OSS) are identified by the `THINKING_MODELS` list on each client. Thinking is enabled automatically when one of these models is selected.

After generation, the model's reasoning trace is available in `last_thinking`:

``` python
from aimu.models import OllamaClient as ModelClient

model_client = ModelClient(ModelClient.MODELS.DEEPSEEK_R1_8B)
response = model_client.generate("What is the capital of France?")

print(model_client.last_thinking)  # reasoning trace
print(response)                    # final answer
```

During streamed generation via `generate_streamed()`, thinking tokens are yielded first followed by the response tokens as a single flat stream. For phase-separated streaming (thinking, tool calls, response), use `chat_streamed()` instead.

### Streamed Chat

`chat_streamed()` yields `StreamChunk` objects. Each chunk carries its own type:

| `chunk.phase` | `chunk.content` type | Description |
|---|---|---|
| `StreamPhase.THINKING` | `str` | Reasoning token (thinking models only) |
| `StreamPhase.TOOL_CALLING` | `dict` `{"name": str, "response": str}` | Tool call and its result |
| `StreamPhase.GENERATING` | `str` | Final response token |

``` python
from aimu.models import OllamaClient as ModelClient, StreamPhase

model_client = ModelClient(ModelClient.MODELS.QWEN_3_5_9B)
last_phase = None

for chunk in model_client.chat_streamed("What is the capital of France?"):
    if last_phase != chunk.phase:
        print(f"--- {chunk.phase} ---")
        last_phase = chunk.phase

    print(chunk.content, end="", flush=True)
```

### Cloud Model Providers

`OpenAIClient`, `AnthropicClient`, and `GeminiClient` connect to cloud APIs using each provider's native SDK or OpenAI-compatible endpoint. API keys are read from environment variables (or a `.env` file).

``` python
from aimu.models import OpenAIClient, OpenAIModel

client = OpenAIClient(OpenAIModel.GPT_4O_MINI)
response = client.chat("What is the capital of France?")
```

``` python
from aimu.models import AnthropicClient, AnthropicModel

client = AnthropicClient(AnthropicModel.CLAUDE_SONNET_4_6)
response = client.chat("What is the capital of France?")
```

``` python
from aimu.models import GeminiClient, GeminiModel

client = GeminiClient(GeminiModel.GEMINI_2_0_FLASH)
response = client.chat("What is the capital of France?")
```

All three support the full `ModelClient` API including streaming, tool calling, and thinking models (`AnthropicModel.CLAUDE_SONNET_4_6`, `GeminiModel.GEMINI_2_5_PRO`).

Required environment variables:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google Gemini: `GOOGLE_API_KEY`

### OpenAI-Compatible Local Servers

Use any of the service-specific clients to connect to a local server that speaks the OpenAI REST API. Each client uses service-appropriate default URLs and model IDs:

``` python
from aimu.models import LMStudioOpenAIClient, LMStudioOpenAIModel

# Connects to http://localhost:1234/v1 by default
client = LMStudioOpenAIClient(LMStudioOpenAIModel.QWEN_3_8B)
response = client.chat("What is the capital of France?")
```

``` python
from aimu.models import OllamaOpenAIClient, OllamaOpenAIModel

# Connects to Ollama's OpenAI-compat endpoint at http://localhost:11434/v1
client = OllamaOpenAIClient(OllamaOpenAIModel.QWEN_3_8B)
response = client.chat("What is the capital of France?")
```

``` python
from aimu.models import LlamaServerOpenAIClient, LlamaServerOpenAIModel

# Connects to llama-server at http://localhost:8080/v1 by default
# Start with: llama-server -m /path/to/model.gguf --port 8080
client = LlamaServerOpenAIClient(LlamaServerOpenAIModel.QWEN_3_8B)
response = client.chat("What is the capital of France?")
```

``` python
from aimu.models import SGLangOpenAIClient, SGLangOpenAIModel

# Connects to SGLang at http://localhost:30000/v1 by default
# Start with: python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000
client = SGLangOpenAIClient(SGLangOpenAIModel.QWEN_3_8B)
response = client.chat("What is the capital of France?")
```

For a custom server or model not in the enum, use `OpenAICompatClient` directly:

``` python
from aimu.models import OpenAICompatClient
from aimu.models.openai_compat import OllamaOpenAIModel

client = OpenAICompatClient(OllamaOpenAIModel.QWEN_3_8B, base_url="http://myserver:8080/v1")
```

All OpenAI-compatible clients support the full `ModelClient` API. Streaming, tool calling, thinking models, and MCP tools work identically to the other clients.

### Local GGUF Models (llama-cpp-python)

`LlamaCppClient` runs GGUF models directly in-process. Ollama, LM Studio, or another service are not required. Pass the path to any GGUF file and a `LlamaCppModel` enum value that describes the model's capabilities:

``` python
from aimu.models.llamacpp import LlamaCppClient, LlamaCppModel

client = LlamaCppClient(LlamaCppModel.QWEN_3_4B, model_path="/path/to/qwen3-4b.Q4_K_M.gguf")
response = client.chat("What is the capital of France?")
```

GPU offloading is enabled by default (`n_gpu_layers=-1`). To run on CPU only, pass `n_gpu_layers=0`. The context window defaults to 4096 tokens; increase with `n_ctx`:

``` python
client = LlamaCppClient(
    LlamaCppModel.QWEN_3_4B,
    model_path="/path/to/model.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,  # offload all layers to GPU
)
```

All standard `ModelClient` features work: Streaming, tool calling, thinking models, and MCP tools.

### Chat UI (Streamlit)

A full-featured chat UI with model/client selection, streaming, thinking model support, MCP tool calls, and conversation persistence.

``` bash
streamlit run web/streamlit_chatbot.py
```

### Chat UI (Gradio)

A full-featured chat UI equivalent to the Streamlit example above.

``` bash
python web/gradio_chatbot.py
```

### Agentic Workflows

`SimpleAgent` wraps a `ModelClient` and runs a tool-calling loop until the model produces a response without invoking any tools.

``` python
from aimu.models.ollama import OllamaClient, OllamaModel
from aimu.tools import MCPClient
from aimu.agents import SimpleAgent

client = OllamaClient(OllamaModel.QWEN_3_8B)
client.mcp_client = MCPClient({"mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}}})

agent = SimpleAgent(client, name="assistant", max_iterations=10)
result = agent.run("Find all log files modified today and summarise the errors.")
```

Agents are configurable from a plain dict, making them easy to embed in larger systems:

``` python
agent = SimpleAgent.from_config(
    {"name": "researcher", "system_message": "Use tools to answer.", "max_iterations": 8},
    client,
)
```

`WorkflowAgent` chains agents sequentially. The output of each step becomes the input to the next. Pass a single `ModelClient` — it is shared across steps, with `messages` cleared and `system_message` applied from each step's config before it runs:

``` python
from aimu.agents import WorkflowAgent

client = OllamaClient(OllamaModel.QWEN_3_8B)
client.mcp_client = MCPClient({"mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}}})

wf = WorkflowAgent.from_config(
    [
        {"name": "planner",   "system_message": "Break the task into steps.", "max_iterations": 3},
        {"name": "executor",  "system_message": "Execute each step using tools.", "max_iterations": 10},
        {"name": "formatter", "system_message": "Format the results clearly.", "max_iterations": 1},
    ],
    client,
)
result = wf.run("Research the top Python web frameworks.")
```

Both `SimpleAgent` and `WorkflowAgent` inherit from the `Agent` ABC and support streaming via `run_streamed()`, which yields `AgentChunk` / `WorkflowChunk` objects tagged with agent name, iteration, and `StreamPhase`.

### AgenticModelClient

`AgenticModelClient` wraps any `Agent` (a `SimpleAgent` or a `WorkflowAgent`) behind the standard `ModelClient` interface. Use it anywhere a `ModelClient` is accepted - web UIs, conversation managers, etc. - to get the full agentic loop transparently:

``` python
from aimu.models.ollama import OllamaClient, OllamaModel
from aimu.tools import MCPClient
from aimu.agents import SimpleAgent, WorkflowAgent, AgenticModelClient

inner = OllamaClient(OllamaModel.QWEN_3_8B)
inner.mcp_client = MCPClient({"mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}}})

# Single-turn client
client = inner

# Agentic client - same interface, loops until tools stop being called
client = AgenticModelClient(SimpleAgent(inner, max_iterations=10))

# Or wrap an entire pipeline behind the same interface
client = AgenticModelClient(WorkflowAgent.from_config(configs, inner))

# All three work identically here:
response = client.chat("Find all log files modified today and summarise the errors.")
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

model_client = ModelClient(ModelClient.MODELS.QWEN_3_5_9B)
model_client.mcp_client = mcp_client

model_client.chat("use my tool please")
```

### Chat Conversation Storage/Management

``` python
from aimu.models import OllamaClient as ModelClient
from aimu.history import ConversationManager

chat_manager = ConversationManager("conversations.json", use_last_conversation=True) # loads the last saved conversation

model_client = ModelClient(ModelClient.MODELS.QWEN_3_5_9B)
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

### Agent Skills

Skills are SKILL.md files discovered from `.agents/skills/` or `.claude/skills/` directories (project-level overrides user-level). Pass `skill_dirs` or `use_skills=True` to `Agent.from_config()` to enable them:

``` python
from aimu.agents import SimpleAgent
from aimu.skills import SkillManager

manager = SkillManager(skill_dirs=["./skills"])
agent = SimpleAgent.from_config({"name": "assistant", "skill_dirs": ["./skills"]}, client)
result = agent.run("Use the pdf-processing skill to extract pages from report.pdf")
```

Each skill directory contains a `SKILL.md` with YAML frontmatter (`name`, `description`) and optional `scripts/*.py` files that are registered as callable tools.

### Prompt Storage/Management

``` python
from aimu.prompts import PromptCatalog, Prompt

with PromptCatalog("prompts.db") as catalog:
    prompt = Prompt(name="summarizer", prompt="Summarize the following: {content}", model_id="llama3.1:8b")
    catalog.store_prompt(prompt)  # version and created_at assigned automatically

    latest = catalog.retrieve_last("summarizer", "llama3.1:8b")
    print(f"v{latest.version}: {latest.prompt}")
```

### Prompt Tuning

`ClassificationPromptTuner` uses hill-climbing to iteratively improve a prompt for binary YES/NO classification:

``` python
import pandas as pd
from aimu.models import OllamaClient as ModelClient
from aimu.prompts import ClassificationPromptTuner

model_client = ModelClient(ModelClient.MODELS.QWEN_3_5_9B)
tuner = ClassificationPromptTuner(model_client=model_client)

df = pd.DataFrame({
    "content": ["LLMs are transforming AI.", "The recipe calls for flour.", ...],
    "actual_class": [True, False, ...],
})

best_prompt = tuner.tune(
    df,
    initial_prompt="Is this about AI? Reply [YES] or [NO]. Content: {content}",
    max_iterations=10,
)
```

The tuner calls `apply_prompt` → `evaluate` → `mutation_prompt` in a loop, reverting on regression and saving each improvement to an optional `PromptCatalog`. Subclass `PromptTuner` to implement custom task types.

## License

This project is licensed under the Apache 2.0 license.
