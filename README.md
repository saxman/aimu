[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AIMU - AI Model Utilities

A lightweight Python library for building LLM-powered applications with a consistent interface across local and cloud providers. AIMU separates autonomous agents from code-controlled workflows, treats agents as composable units that can be used anywhere a plain model client is accepted, and stays small on purpose — plain Python, OpenAI message dicts as the only data model, no framework primitives.

📘 **[Read the docs](https://saxman.github.io/aimu/)** for tutorials, how-to guides, full API reference, and design explanations.

## Features

- **Provider-agnostic model clients** — Ollama, HuggingFace, llama-cpp, Anthropic, OpenAI, Gemini, plus every OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve). One factory, swap with a string change.
- **Agents and workflows** per Anthropic's *Building Effective Agents* taxonomy: `Agent` for autonomous tool-using loops; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` for code-controlled patterns; `OrchestratorAgent` for the dispatch-to-workers pattern.
- **Tools** via the `@tool` decorator (in-process Python) or `MCPClient` (cross-process FastMCP). Both routes coexist on the same client.
- **Skills** — filesystem-discovered `SKILL.md` files auto-injected into a `SkillAgent`.
- **Memory** — semantic facts (ChromaDB), path-based documents (Anthropic Memory API), and conversation history (TinyDB).
- **Prompt management** — versioned SQLite catalog plus a hill-climbing tuner with classification, multi-class, extraction, and judged variants.
- **Evaluation** — DeepEval integration and a multi-model benchmark harness with CSV / JSON / catalog export.

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]`, `aimu[hf]`, `aimu[llamacpp]`. See [installation in the docs](https://saxman.github.io/aimu/tutorials/01-getting-started/) for the full list.

## Quick start

```python
import aimu

# One-shot
text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

# Multi-turn
client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
client.chat("Hi there")
client.chat("What did I just say?")     # history preserved

# Agent with a tool
from aimu.agents import Agent
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

That's the full mental model: `aimu.chat()` for one-shots, `aimu.client()` for conversations, `Agent` for tool-using loops, and `"provider:model_id"` strings to swap backends.

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
