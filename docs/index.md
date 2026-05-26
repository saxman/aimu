# AIMU

**AI Modeling Utilities** — a lightweight Python library for building AI-powered applications with a consistent, provider-agnostic interface across text, images, and audio.

Language models are the primary building block, with the same interface extending to image generation and audio processing. AIMU separates autonomous agents from code-controlled workflows, and treats agents as composable units that can be used anywhere a plain model client is accepted. Tool integration is structural (not a plugin), semantic and document memory can be dropped in, and a prompt-tuning loop optimises prompts against labelled data without ML machinery.

---

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]`, `aimu[hf]` (text + HF `diffusers` image generation), `aimu[google]` (Google Nano Banana image generation), `aimu[llamacpp]`.

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
- **Text-to-image** — `aimu.image_client()` and `aimu.generate_image()` parallel the text surface. HuggingFace `diffusers` for local generation (SD 1.5 / SDXL / SD 3.5 / FLUX), Google Nano Banana for cloud. Drops into any chat agent via the built-in `generate_image` tool. See [how-to: generate images](how-to/generate-images.md).
- **Agents and workflows** — `Agent` for autonomous tool-using loops; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` for code-controlled patterns from Anthropic's *Building Effective Agents*.
- **Tools** — `@tool` decorator for plain Python functions, plus a synchronous `MCPClient` wrapper for cross-process tools.
- **Skills** — filesystem-discovered `SKILL.md` files that auto-inject capabilities into a `SkillAgent`.
- **Memory** — semantic facts (ChromaDB), path-based documents (Anthropic Memory API), and conversation history (TinyDB).
- **Prompt management** — versioned SQLite catalog plus a hill-climbing tuner with classification, multi-class, extraction, and judged variants.
- **Evaluation** — DeepEval integration and a multi-model benchmark harness with CSV / JSON / catalog export.
- **Optional async surface** — `aimu.aio` mirrors the whole sync API (same class names, one-import-away). `Parallel` and `concurrent_tool_calls` use `asyncio.TaskGroup` for structured concurrency. See [async design](explanation/async-design.md).

## Notebooks

The [`notebooks/`](https://github.com/saxman/aimu/tree/main/notebooks) directory ships 15 runnable demos covering every subsystem end-to-end, including a dedicated `14 - Async` notebook for the `aimu.aio` surface and `15 - Image Generation` for text-to-image. The [README](https://github.com/saxman/aimu#examples) lists each one.
