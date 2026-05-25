[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

AIMU is a Python library for LLM-powered applications. It gives you a single interface across providers, autonomous agents and code-controlled workflows, and small composable utilities for tools, memory, prompt tuning, and benchmarking, all in plain Python that is apparent and easy to use.

Common tasks are one-liners: `aimu.chat("hi", model="...")`, `Agent(client, tools=[fn])`, `Chain.from_client(client, [step1, step2])`. Composition happens by passing objects to constructors. Conversation state is a `list[dict]` you can print and edit. Provider-specific details adapt at request time and never leak into your code.

📘 **[Read the docs](https://saxman.github.io/aimu/)** for tutorials, how-to guides, full API reference, and design explanations.

## Key features

### Models

- One client interface for Ollama, HuggingFace, llama-cpp, the Claude API, OpenAI, Gemini, and any OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve). Swap with a string change: `"provider:model_id"`.
- Reasoning, tool calling, and vision input work identically across every provider. Reasoning models surface their tokens as `StreamingContentType.THINKING` chunks via the same API.
- Typed streaming: `StreamChunk(phase, content, agent, iteration)` flows through `client.chat()`, `Agent.run()`, and every workflow. Filter with `include=["generating"]`.

### Agents and workflows

- `Agent` runs an autonomous tool-using loop until the model stops calling tools.
- Four code-controlled workflow patterns: `Chain.from_client(...)`, `Router.from_client(...)`, `Parallel.from_client(...)`, `EvaluatorOptimizer(...)`. Compose freely. Workflows accept agents as steps; agents accept workflows as tools via `as_model_client()`.
- `agent.as_model_client()` makes any agent a drop-in `BaseModelClient`, so agentic and non-agentic clients are interchangeable.

### Tools

- `@tool` on any plain Python function. Type hints + docstring become the spec.
- `MCPClient` for cross-process FastMCP tools. Combine with `@tool` on the same agent.
- Built-in tool groups ready to pass to `tools=`: `builtin.web`, `builtin.fs`, `builtin.compute`, `builtin.misc`.
- Filesystem-discovered `SKILL.md` files auto-inject into a `SkillAgent` (same format Claude Code uses).

### Memory and persistence

- `SemanticMemoryStore` (ChromaDB vector search), `DocumentStore` (path-keyed, drop-in compatible with the Claude memory tool API), `ConversationManager` (TinyDB chat history). All implement the same `MemoryStore` interface.

### Prompts and evaluation

- Hill-climbing `PromptTuner` for automatic prompt optimisation against labelled data. Four concrete tuners: classification, multi-class, extraction, judged-generation.
- `Benchmark` runs one prompt across multiple clients (plain or agentic, mixed providers) and returns a comparison DataFrame. DeepEval metrics plug in as `Scorer`s.

## Examples

```python
import aimu

# One-shot
text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

# Multi-turn
client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
client.chat("Hi there")
client.chat("What did I just say?")     # history preserved
```

**Streaming with phase filtering.** Drop unwanted phases (thinking, tool calls) with `include=`:

```python
for chunk in client.chat("Tell me a story", stream=True, include=["generating"]):
    print(chunk.content, end="", flush=True)
```

**An agent with a tool.** `@tool` works on any plain function:

```python
from aimu.agents import Agent
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

**A code-controlled workflow.** `Chain.from_client()` builds a multi-step pipeline in one line. Each step's output becomes the next step's input:

```python
from aimu.agents import Chain

chain = Chain.from_client(client, [
    "Break the task into clear steps.",
    "Execute each step using available tools.",
    "Polish the result into a single paragraph.",
])
result = chain.run("Research the top Python web frameworks.")
```

**Vision input.** Uniform across every vision-capable provider:

```python
client = aimu.client("openai:gpt-4o-mini")     # or anthropic, gemini, ollama, hf
client.chat("What's in this image?", images=["./cat.jpg"])
```

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]`, `aimu[hf]`, `aimu[llamacpp]`. See [installation in the docs](https://saxman.github.io/aimu/tutorials/01-getting-started/) for the full list of extras.

## Documentation

| | |
|---|---|
| 📘 [Tutorials](https://saxman.github.io/aimu/tutorials/) | Hand-held walkthroughs. Install to first agent in 15 mins |
| 🛠️ [How-to guides](https://saxman.github.io/aimu/how-to/) | Task-oriented recipes (switch providers, write a tool, stream output, benchmark models, ...) |
| 📚 [Reference](https://saxman.github.io/aimu/reference/) | Auto-generated API docs, capability matrices, environment variables, CLI |
| 💡 [Explanation](https://saxman.github.io/aimu/explanation/) | The *why*: architecture, design principles, agents vs workflows |

## Notebooks

The [`notebooks/`](notebooks/) directory ships interactive demos for every subsystem:

| Notebook | Description |
|---|---|
| [01 - Model Client](notebooks/01%20-%20Model%20Client.ipynb) | Text generation, chat, streaming, thinking models |
| [02 - Vision](notebooks/02%20-%20Vision.ipynb) | Image input via `images=` on `chat()` |
| [03 - Tools](notebooks/03%20-%20Tools.ipynb) | `@tool` decorator, built-in tool groups, MCPClient |
| [04 - Prompt Management](notebooks/04%20-%20Prompt%20Management.ipynb) | Versioned prompt storage |
| [05 - Prompt Tuning](notebooks/05%20-%20Prompt%20Tuning.ipynb) | Classification, multi-class, extraction, judged tuners |
| [06 - Conversations](notebooks/06%20-%20Conversations.ipynb) | Persistent chat history |
| [07 - Memory](notebooks/07%20-%20Memory.ipynb) | Semantic fact storage and retrieval |
| [08 - Agents](notebooks/08%20-%20Agents.ipynb) | `Agent` and `agent.as_model_client()` |
| [09 - Agent Skills](notebooks/09%20-%20Agent%20Skills.ipynb) | Filesystem-discovered skill injection |
| [10 - Workflows](notebooks/10%20-%20Workflows.ipynb) | Chain, Router, Parallel, EvaluatorOptimizer, PlanExecuteEvaluator |
| [11 - Prebuilt Agents](notebooks/11%20-%20Prebuilt%20Agents.ipynb) | Orchestrator + worker tools pattern |
| [12 - Evaluations](notebooks/12%20-%20Evaluations.ipynb) | DeepEval integration |
| [13 - Benchmarking](notebooks/13%20-%20Benchmarking.ipynb) | Multi-model comparison harness |

## Design principles

AIMU is small and stays small. Six principles shape the API: plain Python, plain data (OpenAI message dicts only), composability through uniform interfaces, progressive disclosure, direct paths for common tasks, and apparent failures. The reasoning behind each, and the patterns each one excludes, lives on the [design principles](https://saxman.github.io/aimu/explanation/design-principles/) page.

## Contributing

See the [contributing guide](https://saxman.github.io/aimu/contributing/) for dev setup, testing, lint, and PR conventions.

## License

Apache 2.0.
