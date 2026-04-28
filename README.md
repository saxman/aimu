[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

AIMU is a Python library for building LLM-powered applications with a consistent interface across local and cloud providers. It separates autonomous agents from code-controlled workflows, and treats agents as composable units that can be used anywhere a simple model is accepted. MCP tool integration is structural (not a plugin), semantic and/or document memory can be dropped in, and prompt management and tuning make it easy to optimize prompts for concrete use cases.

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
    -   [Evaluations](#evaluations)
-   [License](#license)

## Features

-   **Model Clients**: A factory selects the right provider client from the model enum, so you can swap providers without changing call sites. Streaming output is typed by phase (thinking, tool calling, response generation), making it straightforward to build UIs or observability on top. All clients support extended reasoning models (e.g. DeepSeek-R1, Qwen3, GPT-OSS) with reasoning traces available in both streaming and non-streaming modes.

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

-   **Agents & Workflows**: Per Anthropic's taxonomy, AIMU separates autonomous agents from code-controlled workflows. Agents expose the same interface as plain model clients, so they can be used as drop-in replacements anywhere a client is accepted, enabling recursive composition of agents within workflows and workflows within agents.

    -   **Agents**: `SimpleAgent` runs an autonomous tool-calling loop until the model stops invoking tools. `SkillAgent` extends it with automatic skill injection. `AgenticModelClient` wraps a `SimpleAgent` behind the standard model client interface, making agentic and single-turn clients interchangeable.
    -   **Workflows**: Four code-controlled patterns: `Chain` (prompt chaining), `Router` (classify and dispatch), `Parallel` (concurrent workers with optional aggregation), and `EvaluatorOptimizer` (generate → evaluate → revise loop).
    -   **Example Agents**: Ready-to-use orchestrator agents in `aimu.agents.examples` that demonstrate the orchestrator + worker tools pattern: `ResearchReportAgent`, `CodeReviewAgent`, and `ContentCreationAgent`. Each coordinates worker sub-agents via MCP tools, letting the LLM decide how to use them.
    -   **Skills**: Filesystem-discovered `SKILL.md` files that inject instructions and tools into `SkillAgent` automatically. Skills are discovered from project and user directories (`.agents/skills/`, `.claude/skills/`).

-   **MCP Tools**: MCP tool integration is built into the base model client as a first-class attribute, not a plugin. Attach an `MCPClient` to any model client and tools are passed to the model automatically. Provides a simpler interface for [FastMCP 2.0](https://gofastmcp.com).

-   **Prompt Management**: Versioned prompt storage and automatic prompt optimization:

    -   **Prompt Storage**: Versioned prompt catalog backed by SQLite ([SQLAlchemy](https://www.sqlalchemy.org/)). Prompts are keyed by `(name, model_id)` and auto-versioned on each store.
    -   **Prompt Tuning**: Hill-climbing `PromptTuner` for automatic prompt optimization against labelled data, without ML machinery. Four concrete tuners: `ClassificationPromptTuner` (binary YES/NO), `MultiClassPromptTuner` (N-way), `ExtractionPromptTuner` (JSON field extraction), and `JudgedPromptTuner` (open-ended generation rated per-row by a pluggable `Scorer` — built-in `LLMJudgeScorer` or `DeepEvalScorer` for DeepEval metrics). Subclass `PromptTuner` to implement custom task types.

-   **Evaluations** (`aimu[deepeval]`): `DeepEvalModel` is a thin adapter that wraps any AIMU `ModelClient` to satisfy [DeepEval](https://deepeval.com/)'s `DeepEvalBaseLLM` interface. Pass it as the `model=` argument to any DeepEval metric — `GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, and others — to use a local Ollama model, HuggingFace model, or any cloud provider as your evaluation judge. The same metrics also drop directly into `JudgedPromptTuner` via `DeepEvalScorer` to drive prompt optimization.

-   **Persistence**: Three complementary stores for persisting conversation and knowledge:

    -   **Conversation History** (`ConversationManager`): Persistent chat message history backed by [TinyDB](https://tinydb.readthedocs.io). Load the last conversation on startup and save updates after each turn.
    -   **Semantic Memory** (`SemanticMemoryStore`): Fact storage using [ChromaDB](https://www.trychroma.com/) vector embeddings. Store natural-language subject-predicate-object strings (e.g. `"Paul works at Google"`) and retrieve by semantic topic (e.g. `"employment"`, `"family life"`).
    -   **Document Memory** (`DocumentStore`): Path-based document store mirroring Anthropic's Managed Agents Memory API. Supports `write`, `read`, `edit`, `delete`, and full-text `search` on named paths (e.g. `/preferences.md`).

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
| [04 - Prompt Tuning](notebooks/04%20-%20Prompt%20Tuning.ipynb) | ClassificationPromptTuner, MultiClassPromptTuner, ExtractionPromptTuner, JudgedPromptTuner |
| [05 - Conversations](notebooks/05%20-%20Conversations.ipynb) | Persistent chat conversation management |
| [06 - Memory](notebooks/06%20-%20Memory.ipynb) | Semantic fact storage and retrieval |
| [07 - Agents](notebooks/07%20-%20Agents.ipynb) | SimpleAgent and AgenticModelClient |
| [08 - Agent Skills](notebooks/08%20-%20Agent%20Skills.ipynb) | Filesystem-discovered skill injection with SkillAgent |
| [09 - Agent Workflows](notebooks/09%20-%20Agent%20Workflows.ipynb) | Chain, Router, Parallel, and EvaluatorOptimizer patterns |
| [10 - Agent Examples](notebooks/10%20-%20Agent%20Examples.ipynb) | `ResearchReportAgent`, `CodeReviewAgent`, `ContentCreationAgent` — orchestrator + worker tools pattern |
| [11 - Evaluations](notebooks/11%20-%20Evaluations.ipynb) | DeepEval integration: GEval, AnswerRelevancy, Faithfulness, and batch evaluation |

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
pip install aimu[deepeval]      # DeepEval evaluation metrics (GEval, AnswerRelevancy, Faithfulness, etc.)
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

Every `Runner` exposes `run(task, stream=False)` and `.messages`. Pass `stream=True` to get an `AgentChunk` iterator instead of a string.

**Example agents** in `aimu.agents.examples` wire up an orchestrator with worker sub-agents as MCP tools — the LLM coordinates them autonomously:

``` python
from aimu.models.ollama import OllamaClient
from aimu.agents.examples import ResearchReportAgent

client = OllamaClient(OllamaClient.MODELS.QWEN_3_8B)
agent = ResearchReportAgent(client)
report = agent.run("What is retrieval-augmented generation?")

for chunk in agent.run("Explain transformer attention", stream=True):
    if chunk.phase == StreamPhase.GENERATING:
        print(chunk.content, end="", flush=True)
```

See [07 - Agents](notebooks/07%20-%20Agents.ipynb), [08 - Agent Skills](notebooks/08%20-%20Agent%20Skills.ipynb), [09 - Agent Workflows](notebooks/09%20-%20Agent%20Workflows.ipynb), and [10 - Agent Examples](notebooks/10%20-%20Agent%20Examples.ipynb) for the example agents.

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

See [05 - Conversations](notebooks/05%20-%20Conversations.ipynb) and [06 - Memory](notebooks/06%20-%20Memory.ipynb).

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

`MultiClassPromptTuner` and `ExtractionPromptTuner` follow the same pattern. `JudgedPromptTuner` handles open-ended generation by delegating per-row scoring to a `Scorer`:

``` python
from aimu.prompts import JudgedPromptTuner, LLMJudgeScorer

# Built-in: a separate judge model rates each output 1-10
tuner = JudgedPromptTuner(
    model_client=writer_client,
    scorer=LLMJudgeScorer(judge_client, criteria="Concise, accurate, plain English."),
)

# Or drive the same loop with DeepEval metrics (aimu[deepeval])
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from aimu.evals import DeepEvalModel, DeepEvalScorer

geval = GEval(
    name="quality",
    criteria="Concise, accurate, plain English.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=DeepEvalModel(judge_client),
)
tuner = JudgedPromptTuner(model_client=writer_client, scorer=DeepEvalScorer([geval]))
```

Subclass `PromptTuner` and implement `apply_prompt`, `evaluate`, and `mutation_prompt` for custom task types.

See [03 - Prompt Management](notebooks/03%20-%20Prompt%20Management.ipynb) and [04 - Prompt Tuning](notebooks/04%20-%20Prompt%20Tuning.ipynb).

### Evaluations

`DeepEvalModel` wraps any AIMU `ModelClient` as a DeepEval judge. Pass it to any DeepEval metric via the `model=` argument:

``` python
from aimu.models.ollama import OllamaClient
from aimu.models.ollama.ollama_client import OllamaModel
from aimu.evals import DeepEvalModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

judge = DeepEvalModel(OllamaClient(OllamaModel.QWEN_3_8B))

metric = GEval(
    name="Correctness",
    criteria="Is the actual output factually correct and directly responsive to the question?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.5,
)

test_case = LLMTestCase(input="What is the capital of France?", actual_output="Paris.")
metric.measure(test_case)
print(metric.score, metric.reason)
```

Any AIMU client — Ollama, HuggingFace, Anthropic, OpenAI, or any OpenAI-compatible server — can be used as the judge. Swap in a stronger cloud model for more reliable judgments on complex tasks:

``` python
from aimu.models.anthropic import AnthropicClient
from aimu.models.anthropic.anthropic_client import AnthropicModel

judge = DeepEvalModel(AnthropicClient(AnthropicModel.CLAUDE_SONNET_4))
```

See [11 - Evaluations](notebooks/11%20-%20Evaluations.ipynb) for GEval, AnswerRelevancy, Faithfulness, and batch evaluation examples.

## License

This project is licensed under the Apache 2.0 license.
