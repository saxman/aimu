[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

A lightweight Python library for building LLM-powered applications with one interface across local and cloud providers. AIMU separates autonomous agents from code-controlled workflows, treats agents as drop-in model clients, and stays small on purpose — plain Python, OpenAI message dicts as the only data model, no framework primitives.

📘 **[Read the docs](https://saxman.github.io/aimu/)** for tutorials, how-to guides, full API reference, and design explanations.

## What makes AIMU different

AIMU is deliberately the (?) smallest library in its category. It refuses the framework primitives that bloat other model and agent framework in favour of plain Python and direct method calls. A few principles guiding the design of AIMU.

- **Plain Python over framework primitives.**
- **Simple, unified data model using (OpenAI) message dicts.**
- **Composable agent workflow patterns.**
- **Agents are drop-in model clients for chat.**

The full "what we deliberately don't do" list lives in the [design principles](https://saxman.github.io/aimu/explanation/design-principles/) page.

## Features

- **Uniform interface across every provider.** Thinking, tool calling, and vision input work identically whether you point AIMU at Ollama, HuggingFace, llama-cpp, the Claude API, OpenAI, Gemini, or any OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve). Swap with a string change: `"provider:model_id"`.
- **Reasoning models, universally.** DeepSeek-R1, Qwen3, GPT-OSS, Magistral, Gemini 2.5, Claude 4.x — every thinking-capable model surfaces reasoning as `StreamPhase.THINKING` chunks via the same API.
- **Typed streaming, one chunk type everywhere.** `StreamChunk(phase, content, agent, iteration)` flows through `client.chat()`, `Agent.run()`, and every workflow. Filter phases with `include=["generating"]`; no callbacks, no event subscriptions.
- **Composable agents.** `Agent` runs the loop. `agent.as_model_client()` makes it interchangeable with any plain client. Workflows accept agents as steps. Agents can be orchestrators dispatching to other agents via tool calls.
- **Four workflow patterns when you want the flow fixed.** `Chain.of(...)`, `Router.of(...)`, `Parallel.of(...)`, and `EvaluatorOptimizer(...)` — the documented code-controlled agentic patterns, implemented directly. Use `Agent` when you want the model to direct the flow; use these when you want the code to. Compose freely.
- **Tools as plain Python.** `@tool` on any function — type hints + docstring become the spec. No Pydantic, no `BaseTool` subclass. Pair with `MCPClient` for cross-process FastMCP tools on the same agent.
- **Skills compatible with Claude Code.** Filesystem-discovered `SKILL.md` files auto-inject into a `SkillAgent`. Same format Claude Code uses; AIMU reads from `.agents/skills/`, `.claude/skills/`, and the `~/` variants.
- **Three memory stores.** Semantic facts (ChromaDB), path-based documents (`DocumentStore` — drop-in compatible with the Claude memory tool API), and conversation history (TinyDB). All three implement the same `MemoryStore` abstract interface.
- **Hill-climbing prompt tuner, no ML machinery.** Four concrete tuners (classification, multi-class, extraction, judged-generation) optimise prompts against labelled data using just a model client and a `Scorer`.
- **Multi-model benchmarks.** `Benchmark` runs one prompt over multiple clients — plain or agentic, mixed providers, mixed local-and-cloud — and returns a comparison DataFrame. DeepEval metrics plug in as scorers.

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]`, `aimu[hf]`, `aimu[llamacpp]`. See [installation in the docs](https://saxman.github.io/aimu/tutorials/01-getting-started/) for the full list of extras.

## Quick start

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

**A code-controlled workflow.** `Chain.of()` builds a multi-step pipeline in one line — each step's output becomes the next step's input:

```python
from aimu.agents import Chain

chain = Chain.of(client, [
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

That's the full mental model: `aimu.chat()` for one-shots, `aimu.client()` for conversations, `Agent` for tool-using loops, workflows for code-controlled patterns, and `"provider:model_id"` strings to swap backends. The [tutorials](https://saxman.github.io/aimu/tutorials/) take you from here to a working multi-agent system in 50 minutes.

## Documentation

| | |
|---|---|
| 📘 [Tutorials](https://saxman.github.io/aimu/tutorials/) | Hand-held walkthroughs — install to first agent in 15 min |
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
| [10 - Agent Workflows](notebooks/10%20-%20Agent%20Workflows.ipynb) | Chain, Router, Parallel, EvaluatorOptimizer |
| [11 - Agent Examples](notebooks/11%20-%20Agent%20Examples.ipynb) | Orchestrator + worker tools pattern |
| [12 - Evaluations](notebooks/12%20-%20Evaluations.ipynb) | DeepEval integration |
| [13 - Benchmarking](notebooks/13%20-%20Benchmarking.ipynb) | Multi-model comparison harness |

## Contributing

See the [contributing guide](https://saxman.github.io/aimu/contributing/) for dev setup, testing, lint, and PR conventions.

## License

Apache 2.0.
