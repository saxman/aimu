# AIMU

**AI Model Utilities** — a lightweight Python library for building LLM-powered applications with a consistent interface across local and cloud providers.

AIMU separates autonomous agents from code-controlled workflows, and treats agents as composable units that can be used anywhere a plain model client is accepted. Tool integration is structural (not a plugin), semantic and document memory can be dropped in, and a prompt-tuning loop optimises prompts against labelled data without ML machinery.

---

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]`, `aimu[hf]`, `aimu[llamacpp]`.

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

That's the full mental model: a `chat()` function for one-shots, a `client()` factory for conversations, and `provider:model_id` strings to swap backends.

## Where to next

<div class="grid cards" markdown>

-   :material-school: **[Tutorials](tutorials/index.md)**

    Hands-on walkthroughs. Start here if you're new — install to first working agent in 15 minutes.

-   :material-tools: **[How-to guides](how-to/index.md)**

    Task-oriented recipes. "How do I swap providers / write a tool / stream output / benchmark models?"

-   :material-book-open-variant: **[Reference](reference/index.md)**

    The full API surface, capability matrices, environment variables, and CLI commands.

-   :material-lightbulb: **[Explanation](explanation/index.md)**

    The *why*. Architecture, design principles, the agent/workflow taxonomy, and what AIMU deliberately doesn't do.

</div>

## What's in the box

- **Provider-agnostic clients** — Ollama, HuggingFace, llama-cpp, Anthropic, OpenAI, Gemini, plus every OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve).
- **Agents and workflows** — `Agent` for autonomous tool-using loops; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` for code-controlled patterns from Anthropic's *Building Effective Agents*.
- **Tools** — `@tool` decorator for plain Python functions, plus a synchronous `MCPClient` wrapper for cross-process tools.
- **Skills** — filesystem-discovered `SKILL.md` files that auto-inject capabilities into a `SkillAgent`.
- **Memory** — semantic facts (ChromaDB), path-based documents (Anthropic Memory API), and conversation history (TinyDB).
- **Prompt management** — versioned SQLite catalog plus a hill-climbing tuner with classification, multi-class, extraction, and judged variants.
- **Evaluation** — DeepEval integration and a multi-model benchmark harness with CSV / JSON / catalog export.

## Notebooks

The [`notebooks/`](https://github.com/saxman/aimu/tree/main/notebooks) directory ships 13 runnable demos covering every subsystem end-to-end. The [README](https://github.com/saxman/aimu#examples) lists each one.
