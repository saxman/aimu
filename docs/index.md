<div align="center" markdown>

![AIMU](assets/aimu-horizontal-light.svg#only-light){ width="400" }
![AIMU](assets/aimu-horizontal-dark.svg#only-dark){ width="400" }

</div>

# AIMU

**AI Modeling Utilities**: a lightweight Python library for building AI-powered applications with a consistent, provider-agnostic interface across text, images, audio, and speech.

Language models are the primary building block, with the same interface extending to image generation, audio generation, and text-to-speech. AIMU separates autonomous agents from code-controlled workflows, and treats agents as composable units that can be used anywhere a plain model client is accepted. Tool integration is structural (not a plugin), semantic and document memory can be dropped in, and a prompt-tuning loop optimises prompts against labelled data without ML machinery.

---

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]` (also enables OpenAI TTS), `aimu[hf]` (text + HF image + audio + TTS), `aimu[google]` (Google Nano Banana image), `aimu[llamacpp]`.

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

    Hands-on walkthroughs. Start here if you're new: install to first working agent in 15 minutes.

-   :material-tools: **[How-to guides](how-to/index.md)**

    Task-oriented recipes. "How do I swap providers / write a tool / stream output / benchmark models?"

-   :material-book-open-variant: **[Reference](reference/index.md)**

    The full API surface, capability matrices, environment variables, and CLI commands.

-   :material-lightbulb: **[Explanation](explanation/index.md)**

    The *why*. Architecture, design principles, the agent/workflow taxonomy, and what AIMU deliberately doesn't do.

</div>

## What's in the box

- **Provider-agnostic clients**: Ollama, HuggingFace, llama-cpp, Anthropic, OpenAI, Gemini, plus every OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve).
- **Text-to-image and image-to-image**: `aimu.image_client()` and `aimu.generate_image()` parallel the text surface. HuggingFace `diffusers` for local generation (SD 1.5 / SDXL / SD 3.5 / FLUX 1 / FLUX 2 Klein), Google Nano Banana for cloud. Pass `reference_image=` to any `generate()` call for img2img. Drops into any chat agent via the built-in `generate_image` tool. See [how-to: generate images](how-to/generate-images.md).
- **Text-to-audio**: `aimu.audio_client()` and `aimu.generate_audio()` for music and sound generation (not TTS). HuggingFace MusicGen, AudioLDM2, and Stable Audio Open. See [how-to: generate audio](how-to/generate-audio.md).
- **Text-to-speech**: `aimu.speech_client()` and `aimu.generate_speech()` for TTS. HuggingFace MMS-TTS/BARK locally; OpenAI tts-1/tts-1-hd in the cloud. Live sentence-by-sentence narration in the Streamlit chatbot. See [how-to: generate speech](how-to/generate-speech.md).
- **Agents and workflows**: `Agent` for autonomous tool-using loops; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` for code-controlled patterns from Anthropic's *Building Effective Agents*.
- **Tools**: `@tool` decorator for plain Python functions, plus a synchronous `MCPClient` wrapper for cross-process tools.
- **Skills**: filesystem-discovered `SKILL.md` files that auto-inject capabilities into a `SkillAgent`.
- **Memory**: semantic facts (ChromaDB), path-based documents (Anthropic Memory API), and conversation history (TinyDB).
- **Prompt management**: versioned SQLite catalog plus a hill-climbing tuner with classification, multi-class, extraction, and judged variants.
- **Evaluation**: DeepEval integration and a multi-model benchmark harness with CSV / JSON / catalog export.
- **Optional async surface**: `aimu.aio` mirrors the whole sync API (same class names, one-import-away). `Parallel` and `concurrent_tool_calls` use `asyncio.TaskGroup` for structured concurrency. See [async design](explanation/async-design.md).

## Examples

The [`examples/`](https://github.com/saxman/aimu/tree/main/examples) directory ships larger, real-world programs organized by theme: `text-refinement/` and `image-refinement/` (the same generate → judge → refine loop in two modalities, each implemented as a code loop, an `Agent`, an `EvaluatorOptimizer` workflow, and simulated annealing), `news-summarizer/` (one task solved with `Agent`, `Chain`, `Parallel`, and `OrchestratorAgent`), and `skills/` (demo skills for `SkillAgent` discovery). See the [examples overview](examples.md).

## Notebooks

The [`notebooks/`](https://github.com/saxman/aimu/tree/main/notebooks) directory ships 22 runnable demos ordered to build up incrementally, from `01 - Model Client`, `03 - Structured Output`, and `06 - Tools` through `07 - Agents`, `11 - Embeddings`, `13 - RAG`, and the generative-modality and `22 - Async` notebooks. The numbered filenames are self-describing; browse the directory to run them.

## Web apps

The [`examples/web/`](https://github.com/saxman/aimu/tree/main/examples/web) directory ships two Streamlit chat applications (install the UI stack with `pip install aimu[web]`). `streamlit_chatbot_basic.py` (~70 lines) is a minimal showcase (provider/model selector, streaming chat, built-in tools) illustrating how little code a working AIMU chatbot takes. `streamlit_chatbot.py` is a full-featured version that adds image generation, audio generation, speech narration (live sentence-by-sentence TTS as the model streams), agentic mode, thinking display, and generation sliders; it's intended as an extensible starting point for more sophisticated apps. A Gradio variant is also included. The [personal-assistant](https://github.com/saxman/aimu/tree/main/examples/personal-assistant) example also ships a WebSocket front end (`web_assistant.py`) that streams replies and pushes proactive messages to the browser.
