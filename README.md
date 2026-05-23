[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

AIMU is a Python library for building LLM-powered applications with a consistent interface across local and cloud providers. It separates autonomous agents from code-controlled workflows, and treats agents as composable units that can be used anywhere a simple model is accepted. Tool integration is structural (not a plugin), semantic and/or document memory can be dropped in, and prompt management and tuning make it easy to optimize prompts for concrete use cases.

## Table of Contents

-   [Features](#features)
-   [Components](#components)
-   [Examples](#examples)
-   [Installation](#installation)
-   [Development](#development)
-   [Usage](#usage)
    -   [Quick Start](#quick-start)
    -   [Model Clients](#model-clients)
    -   [Agents & Workflows](#agents--workflows)
    -   [Tools](#tools)
    -   [Persistence](#persistence)
    -   [Prompt Management](#prompt-management)
    -   [Evaluations](#evaluations)
    -   [Benchmarking](#benchmarking)
-   [License](#license)

## Features

-   **Model Clients**: A single `ModelClient` factory selects the right provider client from a `Model` enum member or a `"provider:model_id"` string, so you can swap providers without changing call sites. Streaming output is typed by phase (thinking, tool calling, response generation), making it straightforward to build UIs or observability on top. All clients support extended reasoning models (e.g. DeepSeek-R1, Qwen3, GPT-OSS) with reasoning traces available in both streaming and non-streaming modes.

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

-   **Agents & Workflows**: Per Anthropic's taxonomy, AIMU separates autonomous agents from code-controlled workflows. Agents expose the same interface as plain model clients via `agent.as_model_client()`, so they can be used as drop-in replacements anywhere a client is accepted, enabling recursive composition of agents within workflows and workflows within agents.

    -   **Agents**: `Agent` runs an autonomous tool-calling loop until the model stops invoking tools. `SkillAgent` extends it with automatic skill injection. `agent.as_model_client()` returns a `BaseModelClient` view that runs the full loop on every `chat()`.
    -   **Workflows**: Four code-controlled patterns: `Chain` (prompt chaining), `Router` (classify and dispatch), `Parallel` (concurrent workers with optional aggregation), and `EvaluatorOptimizer` (generate → evaluate → revise loop). Each exposes a one-line `.of(client, ...)` factory.
    -   **Orchestrator Agents**: `OrchestratorAgent` (exported from `aimu.agents`) is the base for the orchestrator-with-worker-tools pattern. Subclass and call `self._init_orchestrator(...)` for custom orchestrators, or use `OrchestratorAgent.assemble(client, system, workers=[...])` to wire a list of worker agents into a runnable orchestrator without subclassing. Ready-to-use subclasses in `aimu.agents.examples`: `ResearchReportAgent`, `CodeReviewAgent`, and `ContentCreationAgent`. `ResearchReportAgent` accepts an optional `worker_tools=` list to give workers live tool access (e.g. web search), automatically enabling concurrent worker dispatch.
    -   **Skills**: Filesystem-discovered `SKILL.md` files that inject instructions and tools into `SkillAgent` automatically. Skills are auto-discovered from `.agents/skills/`, `.claude/skills/`, `~/.agents/skills/`, and `~/.claude/skills/` (project-level wins on name collision); pass explicit `skill_dirs` to override. Malformed `SKILL.md` raises `SkillLoadError` — no silent skipping.

-   **Tools**: Two routes for giving agents tools. Decorate plain Python functions with `@tool` to register them in-process — the decorator builds an OpenAI-format spec from the function's signature and docstring, and raises `ToolSignatureError` on unsupported signatures (`*args`/`**kwargs`, unhinted required params). For cross-process or shared tool catalogs, attach an `MCPClient` to any model client; AIMU provides a simpler synchronous interface for [FastMCP 2.0](https://gofastmcp.com) with explicit `MCPConnectionError` on failure and a `.ping()` health check. Both routes can be combined on the same client. `aimu.tools.builtin` ships ready-made `@tool` functions grouped by domain (`builtin.web`, `builtin.fs`, `builtin.compute`, `builtin.misc`) that are also registered on the bundled FastMCP server.

-   **Prompt Management**: Versioned prompt storage and automatic prompt optimization:

    -   **Prompt Storage**: Versioned prompt catalog backed by SQLite ([SQLAlchemy](https://www.sqlalchemy.org/)). Prompts are keyed by `(name, model_id)` and auto-versioned on each store.
    -   **Prompt Tuning**: Hill-climbing `PromptTuner` for automatic prompt optimization against labelled data, without ML machinery. Four concrete tuners: `ClassificationPromptTuner` (binary YES/NO), `MultiClassPromptTuner` (N-way), `ExtractionPromptTuner` (JSON field extraction), and `JudgedPromptTuner` (open-ended generation rated per-row by a pluggable `Scorer` — built-in `LLMJudgeScorer` or `DeepEvalScorer` for DeepEval metrics). Subclass `PromptTuner` to implement custom task types.

-   **Evaluations & Benchmarking**:

    -   **DeepEval integration** (`aimu[deepeval]`): `DeepEvalModel` is a thin adapter that wraps any AIMU `ModelClient` to satisfy [DeepEval](https://deepeval.com/)'s `DeepEvalBaseLLM` interface. Pass it as the `model=` argument to any DeepEval metric — `GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, and others — to use a local Ollama model, HuggingFace model, or any cloud provider as your evaluation judge. The same metrics also drop directly into `JudgedPromptTuner` via `DeepEvalScorer` to drive prompt optimization.
    -   **Benchmarking**: `Benchmark` runs the same prompt and dataset across multiple `BaseModelClient` instances and aggregates per-row scores into a comparison DataFrame. Because rows are driven through `chat()` (with `client.reset()` between rows), the harness works uniformly for plain `ModelClient` instances and `agent.as_model_client()` views of an `Agent`. Scoring is delegated to any `Scorer` (`LLMJudgeScorer`, `DeepEvalScorer`, or custom). Results export to CSV / JSON and persist to `PromptCatalog` with auto-versioning.

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
| [02 - Vision](notebooks/02%20-%20Vision.ipynb) | Image input via the `images=` kwarg on `chat()`; OpenAI, Anthropic, Gemini, and Ollama |
| [03 - Tools](notebooks/03%20-%20Tools.ipynb) | `@tool` decorator, built-in tool groups, and MCPClient |
| [04 - Prompt Management](notebooks/04%20-%20Prompt%20Management.ipynb) | Versioned prompt storage |
| [05 - Prompt Tuning](notebooks/05%20-%20Prompt%20Tuning.ipynb) | ClassificationPromptTuner, MultiClassPromptTuner, ExtractionPromptTuner, JudgedPromptTuner |
| [06 - Conversations](notebooks/06%20-%20Conversations.ipynb) | Persistent chat conversation management |
| [07 - Memory](notebooks/07%20-%20Memory.ipynb) | Semantic fact storage and retrieval |
| [08 - Agents](notebooks/08%20-%20Agents.ipynb) | `Agent` and `agent.as_model_client()` |
| [09 - Agent Skills](notebooks/09%20-%20Agent%20Skills.ipynb) | Filesystem-discovered skill injection with SkillAgent |
| [10 - Agent Workflows](notebooks/10%20-%20Agent%20Workflows.ipynb) | Chain, Router, Parallel, and EvaluatorOptimizer patterns (with `.of()` factories) |
| [11 - Agent Examples](notebooks/11%20-%20Agent%20Examples.ipynb) | `ResearchReportAgent`, `CodeReviewAgent`, `ContentCreationAgent` — orchestrator + worker tools pattern |
| [12 - Evaluations](notebooks/12%20-%20Evaluations.ipynb) | DeepEval integration: GEval, AnswerRelevancy, Faithfulness, and batch evaluation |
| [13 - Benchmarking](notebooks/13%20-%20Benchmarking.ipynb) | Compare model clients (and agentic workflows) on a shared task; export to CSV/JSON/PromptCatalog |

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

### Quick Start

Three top-level entry points cover the most common patterns:

``` python
import aimu

# One-shot — build a client, send one message, return the response.
text = aimu.chat("Summarise this article.", model="anthropic:claude-sonnet-4-6")

# Multi-turn — build a reusable client.
client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
client.chat("Hi there")
client.chat("What did I just say?")          # message history is preserved

# Streaming
for chunk in aimu.chat("Tell me a story", model="ollama:qwen3.5:9b", stream=True):
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

Model strings have the form `"provider:model_id"`. Providers: `ollama`, `hf`, `anthropic`, `openai`, `gemini`, `lmstudio`, `ollama-openai`, `hf-openai`, `vllm`, `llamaserver`, `sglang`, `llamacpp`. Use a `Model` enum member when you want IDE autocomplete and type checking; use a string for quick scripts.

### Model Clients

All clients implement the `BaseModelClient` abstract interface. `ModelClient` is the public factory:

``` python
from aimu.models import ModelClient, OllamaModel

client = ModelClient(OllamaModel.QWEN_3_5_9B)           # enum form
client = ModelClient("ollama:qwen3.5:9b")               # string form
client = ModelClient("ollama:qwen3.5:9b", system_message="You are concise.")

response = client.generate("Summarise this text.", {"temperature": 0.7})  # stateless
response = client.chat("What is the capital of France?")                  # multi-turn
print(client.messages)  # full message history
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

**Streaming**: `chat(..., stream=True)` yields `StreamChunk` objects tagged by phase. Each chunk also carries the agent name (`agent`, `None` for plain client chats) and the loop iteration (`iteration`, `0` for plain chats):

| `chunk.phase` | `chunk.content` type | Description |
|---|---|---|
| `StreamPhase.THINKING` | `str` | Reasoning token (thinking models only) |
| `StreamPhase.TOOL_CALLING` | `dict` `{"name": str, "response": str}` | Tool call and result |
| `StreamPhase.GENERATING` | `str` | Final response token |

Use `chunk.is_text()` to filter text phases (`THINKING` or `GENERATING`) and `chunk.is_tool_call()` for tool-call dicts.

**Stream filtering**: pass `include=[...]` to drop unwanted phases — e.g. `client.chat("hi", stream=True, include=["generating"])` yields only response tokens. Accepts `StreamPhase` members or their string equivalents (`"thinking"`, `"tool_calling"`, `"generating"`, `"done"`).

**Thinking models**: extended reasoning (e.g. DeepSeek-R1, Qwen3, GPT-OSS) is enabled automatically for supported models. The reasoning trace is available in `client.last_thinking` after generation, or as `StreamPhase.THINKING` chunks during streaming.

**System message immutability**: `system_message` becomes immutable once the conversation starts (first `chat()`). The setter raises `RuntimeError` afterward. Call `client.reset()` to clear messages and unlock the setter; by default `reset()` preserves the existing system message:

``` python
client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
client.chat("Hi")
client.system_message = "Be verbose."        # ❌ RuntimeError
client.reset()                                # ✅ messages cleared; system_message preserved
client.system_message = "Be verbose."        # ✅ unlocked
client.reset(system_message=None)             # explicit clear
client.reset(system_message="New role.")      # explicit replace
```

**Chat UIs**: full-featured UIs with streaming, tool calls, and conversation persistence: `streamlit run web/streamlit_chatbot.py` ([Streamlit](web/streamlit_chatbot.py)) or `python web/gradio_chatbot.py` ([Gradio](web/gradio_chatbot.py)).

See [01 - Model Client](notebooks/01%20-%20Model%20Client.ipynb) for detailed examples.

**Vision**: pass images alongside the user prompt via the `images=` kwarg on `chat()`. Each item may be a file path string, `pathlib.Path`, raw `bytes`, an `http(s)://` URL, or a `data:image/...;base64,...` URL. AIMU normalizes inputs to OpenAI content blocks internally and adapts them per-provider (Anthropic native `image` blocks, Ollama message-level `images=` field, HuggingFace `AutoProcessor`).

``` python
import aimu

client = aimu.client("openai:gpt-4o-mini")
client.chat("What's in this image?", images=["./cat.jpg"])
client.chat("Compare these.", images=["a.png", b"\\x89PNG..."])  # multiple images
```

Vision is gated by a `supports_vision` flag on each `Model` enum member (mirroring `supports_tools` and `supports_thinking`); the corresponding `VISION_MODELS` classproperty lists them. Passing `images=` to a non-vision model raises `ValueError` up front. Vision models include OpenAI GPT-4o / GPT-4.1 / o3 / o4-mini, Anthropic Claude 4.x, Google Gemini 1.5+ and 2.x, Ollama Gemma 3 and Gemma 4, and HuggingFace Gemma 3 / Gemma 4. See [02 - Vision](notebooks/02%20-%20Vision.ipynb).

### Agents & Workflows

`Agent` wraps a `ModelClient` and runs a tool-calling loop until the model stops invoking tools. `system_message` is the second positional argument; `name` is optional (auto-derived when unset):

``` python
import aimu
from aimu.agents import Agent
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

client = aimu.client("ollama:qwen3.5:9b")
agent = Agent(client, "You are a helpful assistant.", tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

To drop an agent into an API that expects a `ModelClient` (e.g. a chat UI), call `agent.as_model_client()`:

``` python
chat_view = agent.as_model_client()        # BaseModelClient view; chat() runs the full loop
chat_view.chat("Continue the conversation.")
```

For tools that live in a separate process — or to share a tool catalog across many agents — attach a `FastMCP`-backed `MCPClient` instead. Python `@tool` functions and MCP tools can be combined on the same agent:

``` python
from aimu.tools import MCPClient

client.mcp_client = MCPClient({"mcpServers": {"mytools": {"command": "python", "args": ["tools.py"]}}})

agent = Agent.from_config(
    {"name": "researcher", "system_message": "Use tools to answer.", "max_iterations": 8},
    client,
)
result = agent.run("Find all log files modified today and summarise the errors.")
```

`SkillAgent` extends `Agent` with automatic discovery and injection of `SKILL.md` skill files:

``` python
from aimu.agents import SkillAgent

agent = SkillAgent(client, "You are a helpful assistant.")  # auto-discovers skills from .agents/skills/, .claude/skills/, ~/.agents/skills/, ~/.claude/skills/
result = agent.run("Use the pdf-processing skill to extract pages from report.pdf")
```

Workflow patterns have code-controlled flow. Each provides a one-line `.of(client, ...)` factory. `Chain` sequences agents so each step's output becomes the next step's input:

``` python
from aimu.agents import Chain

chain = Chain.of(client, [
    "Break the task into clear steps.",
    "Execute each step using available tools.",
    "Format the result clearly.",
])
result = chain.run("Research the top Python web frameworks.")
```

`Router.of()` builds a classifier-and-dispatch workflow, and `Parallel.of()` builds a fan-out-with-optional-aggregator:

``` python
from aimu.agents import Router, Parallel

router = Router.of(
    client,
    classifier_prompt="Classify as code, writing, or math. Reply with only the category.",
    handlers={
        "code":    Agent(client, "You are a coder.",        name="coder"),
        "writing": Agent(client, "You are a writer.",       name="writer"),
        "math":    Agent(client, "You are a mathematician.", name="mathematician"),
    },
)

parallel = Parallel.of(
    client,
    worker_prompts=[
        "Analyze for security issues.",
        "Analyze for performance issues.",
        "Analyze for readability issues.",
    ],
    aggregator_prompt="Synthesize the perspectives into one concise review.",
)
```

Every `Runner` exposes `run(task, stream=False)` and `.messages`. Pass `stream=True` to get a `StreamChunk` iterator instead of a string.

**Example agents** in `aimu.agents.examples` wire up an orchestrator with worker sub-agents as tools — the LLM coordinates them autonomously:

``` python
import aimu
from aimu.agents.examples import ResearchReportAgent
from aimu.models import StreamPhase

client = aimu.client("ollama:qwen3.5:9b")
agent = ResearchReportAgent(client)
report = agent.run("What is retrieval-augmented generation?")

for chunk in agent.run("Explain transformer attention", stream=True):
    if chunk.phase == StreamPhase.GENERATING:
        print(chunk.content, end="", flush=True)
```

Pass `worker_tools=` to give workers live tool access (e.g. web search via `aimu.tools.builtin`). The orchestrator will automatically dispatch all workers concurrently when the model returns multiple tool calls in one response:

``` python
from aimu.tools import builtin

agent = ResearchReportAgent(client, worker_tools=builtin.web)   # all web tools
report = agent.run("What is retrieval-augmented generation?")
```

Build your own orchestrator either with `OrchestratorAgent.assemble(...)` (no subclassing — auto-wraps each worker as a `@tool`) or by subclassing `OrchestratorAgent` and calling `self._init_orchestrator(...)` for full control over the dispatch tools:

``` python
from aimu.agents import Agent, OrchestratorAgent

extractor = Agent(client, "Extract the key facts from the text.", name="extractor")
writer    = Agent(client, "Write a concise summary from the extracted facts.", name="writer")

summary = OrchestratorAgent.assemble(
    client,
    "Extract facts then write a summary.",
    workers=[extractor, writer],
)
print(summary.run("..."))
```

For full control over each dispatch tool (e.g. custom descriptions, extra arguments) subclass `OrchestratorAgent` and decorate worker-dispatch functions with `@tool`:

``` python
from aimu.agents import Agent, OrchestratorAgent
from aimu.models import ModelClient
from aimu.tools import tool

class SummaryAgent(OrchestratorAgent):
    def __init__(self, model_client):
        extractor = Agent(ModelClient(model_client.model), "Extract the key facts from the text.", name="extractor")
        writer    = Agent(ModelClient(model_client.model), "Write a concise summary from the extracted facts.", name="writer")

        @tool
        def extract_facts(text: str) -> str:
            """Extract key facts from a block of text."""
            return extractor.run(text)

        @tool
        def write_summary(facts: str) -> str:
            """Write a summary from extracted facts."""
            return writer.run(facts)

        self._init_orchestrator(
            model_client,
            name="summary-agent",
            system_message="Extract facts then write a summary.",
            tools=[extract_facts, write_summary],
        )
```

See [08 - Agents](notebooks/08%20-%20Agents.ipynb), [09 - Agent Skills](notebooks/09%20-%20Agent%20Skills.ipynb), [10 - Agent Workflows](notebooks/10%20-%20Agent%20Workflows.ipynb), and [11 - Agent Examples](notebooks/11%20-%20Agent%20Examples.ipynb) for the example agents.

### Tools

Two routes, combinable on the same client. The distinction is *where the tool's code runs* relative to your agent:

-   **In-process**: the tool is a Python function that runs in the same Python interpreter as the agent. Dispatch is a direct function call: no serialization, no transport, no separate process to start. Best when you own the tool's code, its dependencies are already in your agent's environment, and you want the lowest possible call overhead.
-   **Cross-process**: the tool runs in a separate process, a subprocess is launched on demand, a long-running server, or a remote machine, and is reached over the [Model Context Protocol](https://modelcontextprotocol.io). Arguments and results cross the process boundary as JSON. Best when the tool is written in another language, ships as a standalone binary, is shared across many agents or users, has conflicting dependencies you want to isolate, or needs sandboxing away from your agent.

**`@tool` decorator** (in-process). The decorator inspects the function's signature and docstring to build an OpenAI-format spec. It raises `ToolSignatureError` at decoration time on unsupported signatures (`*args` / `**kwargs`, parameters with no type hint *and* no default). `Optional[T]` and `T | None` unwrap to the inner type. Hand a list of decorated functions to an `Agent` via `tools=`, or set `client.tools = [...]` directly:

``` python
import aimu
from aimu.agents import Agent
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

`aimu.tools.builtin` ships ready-made `@tool` functions grouped by domain — pass a group directly to `tools=`:

``` python
from aimu.tools import builtin

# Domain-specific subsets
builtin.web      # get_weather, get_webpage, search, wikipedia
builtin.fs       # list_directory, read_file
builtin.compute  # calculate
builtin.misc     # echo, get_current_date_and_time

# Or everything
builtin.ALL_TOOLS

agent = Agent(client, tools=builtin.web + builtin.fs)
```

**`MCPClient`** (cross-process). Wrap a FastMCP 2.0 server and assign to `model_client.mcp_client`. Connection failures raise `MCPConnectionError`; use `.ping()` to verify the connection is alive:

``` python
from aimu.tools import MCPClient, MCPConnectionError

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    }
})
mcp_client.ping()   # raises MCPConnectionError if dead

# Use standalone
mcp_client.call_tool("mytool", {"input": "hello world!"})

# Or attach to a model client; tools are passed to the model automatically
model_client.mcp_client = mcp_client
```

The same built-in tool set is also registered on the FastMCP server in [aimu/tools/mcp.py](aimu/tools/mcp.py) (`python -m aimu.tools.mcp` to run it).

See [03 - Tools](notebooks/03%20-%20Tools.ipynb).

### Persistence

**Conversation history**: `ConversationManager` persists chat message sequences across sessions:

``` python
import aimu
from aimu.history import ConversationManager

manager = ConversationManager("conversations.json", use_last_conversation=True)
client = aimu.client("ollama:qwen3.5:9b")
client.messages = manager.messages

client.chat("What is the capital of France?")
manager.update_conversation(client.messages)
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

See [06 - Conversations](notebooks/06%20-%20Conversations.ipynb) and [07 - Memory](notebooks/07%20-%20Memory.ipynb).

### Prompt Management

**Prompt catalog**: `PromptCatalog` stores versioned prompts keyed by `(name, model_id)`:

``` python
from aimu.prompts import PromptCatalog, Prompt

with PromptCatalog("prompts.db") as catalog:
    prompt = Prompt(name="summarizer", prompt="Summarize the following: {content}", model_id="qwen3.5:9b")
    catalog.store_prompt(prompt)  # version and created_at assigned automatically

    latest = catalog.retrieve_last("summarizer", "qwen3.5:9b")
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

See [04 - Prompt Management](notebooks/04%20-%20Prompt%20Management.ipynb) and [05 - Prompt Tuning](notebooks/05%20-%20Prompt%20Tuning.ipynb).

### Evaluations

`DeepEvalModel` wraps any AIMU `ModelClient` as a DeepEval judge. Pass it to any DeepEval metric via the `model=` argument:

``` python
import aimu
from aimu.evals import DeepEvalModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

judge = DeepEvalModel(aimu.client("ollama:qwen3.5:9b"))

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
judge = DeepEvalModel(aimu.client("anthropic:claude-sonnet-4-6"))
```

See [12 - Evaluations](notebooks/12%20-%20Evaluations.ipynb) for GEval, AnswerRelevancy, Faithfulness, and batch evaluation examples.

### Benchmarking

`Benchmark` runs one prompt and dataset across multiple model clients and returns aggregate metrics for direct comparison. Rows are driven through `chat()` with `client.reset()` between rows (which preserves the system message and unlocks it for the next chat), so the harness handles plain `ModelClient` instances and `agent.as_model_client()` views of an `Agent` uniformly:

``` python
import aimu
import pandas as pd
from aimu.agents import Agent
from aimu.evals import Benchmark
from aimu.prompts.tuners.scorers import LLMJudgeScorer

df = pd.DataFrame({"content": ["Summarise the water cycle.", "Explain photosynthesis briefly."]})
prompt = "Answer in one sentence: {content}"

judge = aimu.client("ollama:qwen3.5:9b")
scorer = LLMJudgeScorer(judge, criteria="Concise, accurate, plain English.")

bench = Benchmark(prompt=prompt, data=df, scorer=scorer, pass_threshold=0.7)
results = bench.run({
    "qwen3.5-9b":           aimu.client("ollama:qwen3.5:9b"),
    "gemma4-e4b":           aimu.client("ollama:gemma4:e4b"),
    "qwen3.5-9b (agentic)": Agent(aimu.client("ollama:qwen3.5:9b")).as_model_client(),
})

print(results.metrics)            # rows = client names, cols = score / pass_rate
results.to_csv("bench.csv")
results.to_json("bench.json")
results.to_catalog(catalog, prompt_name="summary-bench")  # persists per-client metrics, auto-versioned
```

Swap `LLMJudgeScorer` for `DeepEvalScorer([metric, ...])` to score with any DeepEval metric (`GEval`, `AnswerRelevancyMetric`, etc.).

See [13 - Benchmarking](notebooks/13%20-%20Benchmarking.ipynb) for the end-to-end workflow including agentic comparisons and DeepEval-backed scoring.

## License

This project is licensed under the Apache 2.0 license.
