<div align="center" markdown>

![AIMU](assets/aimu-horizontal-light.svg#only-light){ width="400" }
![AIMU](assets/aimu-horizontal-dark.svg#only-dark){ width="400" }

</div>

# AIMU

**AI Modeling Utilities**: a Python library for finding out what generative AI models and systems are actually capable of, and understanding how they work while you do it.

AIMU gives you one provider-agnostic interface across text, images, audio, and speech, so trying a task on a different model is a string change and the difference you observe is the models' and not your harness's. Language models are the primary building block, with the same interface extending to image generation, audio generation, and text-to-speech. Agents and code-controlled workflows are separated but interchangeable, tool integration is structural rather than a plugin, semantic and document memory drop in, and a prompt-tuning loop optimises prompts against labelled data without ML machinery.

It is all plain Python you can read, so when a run surprises you, you can tell whether the surprise was the model's or the library's. See [design principles](explanation/design-principles.md) for why that constraint drives everything else.

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

New here? Work through it in this order: the [tutorials](tutorials/index.md) get you to a working
agent in about 15 minutes, the
[notebooks](https://github.com/saxman/aimu/tree/main/notebooks) add one subsystem at a time, and the
[examples](examples.md) solve the same task several ways so you can compare approaches rather than
just read about them. Already oriented? [Compare models](how-to/compare-models.md) is the shortest
path to seeing what a model can actually do.

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

- **Provider-agnostic clients**: Ollama, HuggingFace, llama-cpp, Anthropic, OpenAI, Gemini, plus every OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve, oMLX).
- **MLX on Apple Silicon**: MLX-optimized weights run through `omlx`, LM Studio's MLX engine, or Ollama 0.19+ (which picks MLX automatically on Apple Silicon). See [how-to: switch providers](how-to/switch-providers.md#run-mlx-models-on-apple-silicon).
- **Text-to-image and image-to-image**: `aimu.image_client()` and `aimu.generate_image()` parallel the text surface. HuggingFace `diffusers` for local generation (SD 1.5 / SDXL / SD 3.5 / FLUX 1 / FLUX 2 Klein), Google Nano Banana for cloud. Pass `reference_image=` to any `generate()` call for img2img. Drops into any chat agent via the built-in `generate_image` tool. See [how-to: generate images](how-to/generate-images.md).
- **Text-to-audio**: `aimu.audio_client()` and `aimu.generate_audio()` for music and sound generation (not TTS). HuggingFace MusicGen, AudioLDM2, and Stable Audio Open. See [how-to: generate audio](how-to/generate-audio.md).
- **Text-to-speech**: `aimu.speech_client()` and `aimu.generate_speech()` for TTS. HuggingFace MMS-TTS/BARK locally; OpenAI tts-1/tts-1-hd in the cloud. Live sentence-by-sentence narration in the Streamlit chatbot. See [how-to: generate speech](how-to/generate-speech.md).
- **Agents and workflows**: `Agent` for autonomous tool-using loops; `Chain` / `Router` / `Parallel` / `EvaluatorOptimizer` for code-controlled patterns from Anthropic's *Building Effective Agents*.
- **Tools**: `@tool` decorator for plain Python functions, plus a synchronous `MCPClient` wrapper for cross-process tools.
- **Skills**: filesystem-discovered `SKILL.md` files that auto-inject capabilities into a `SkillAgent`.
- **Memory**: semantic facts (ChromaDB, `aimu[memory]`), path-based documents (Anthropic Memory API, no extra needed), and conversation history (TinyDB).
- **Prompt management**: versioned SQLite catalog (`aimu[prompts]`) plus a hill-climbing tuner (`aimu[tuning]`) with classification, multi-class, extraction, and judged variants.
- **Evaluation**: DeepEval integration and a multi-model benchmark harness with CSV / JSON / catalog export.
- **Optional async surface**: `aimu.aio` mirrors the whole sync API (same class names, one-import-away). `Parallel` and `concurrent_tool_calls` use `asyncio.TaskGroup` for structured concurrency. See [async design](explanation/async-design.md).

## Examples

The [`examples/`](https://github.com/saxman/aimu/tree/main/examples) directory ships larger, real-world programs organized by theme: `text-refinement/` and `image-refinement/` (the same generate → judge → refine loop in two modalities, each implemented as a code loop, an `Agent`, an `EvaluatorOptimizer` workflow, and simulated annealing), `news-summarizer/` (one task solved with `Agent`, `Chain`, `Parallel`, and `OrchestratorAgent`), and `skills/` (demo skills for `SkillAgent` discovery). See the [examples overview](examples.md).

## Notebooks

The [`notebooks/`](https://github.com/saxman/aimu/tree/main/notebooks) directory ships 26 runnable demos ordered to build up incrementally, from `01-model-client`, `03-structured-output`, and `06-tools` through `07-agents`, `11-embeddings`, `13-rag`, and the generative-modality and `22-async` notebooks. They are authored as plain-text [Quarto](https://quarto.org) `.qmd` files (markdown with executable `python` cells); the numbered filenames are self-describing, so browse the directory to read or run them.

## Web apps

The [`examples/web/`](https://github.com/saxman/aimu/tree/main/examples/web) directory ships two Streamlit chat applications (install the UI stack with `pip install aimu[web]`). `streamlit_chatbot_basic.py` (~70 lines) is a minimal showcase (provider/model selector, streaming chat, built-in tools) illustrating how little code a working AIMU chatbot takes. `streamlit_chatbot.py` is a full-featured version that adds image generation, audio generation, speech narration (live sentence-by-sentence TTS as the model streams), agentic mode, thinking display, and generation sliders; it's intended as an extensible starting point for more sophisticated apps. A Gradio variant is also included. The [personal-assistant](https://github.com/saxman/aimu/tree/main/examples/personal-assistant) example also ships a WebSocket front end (`web_assistant.py`) that streams replies and pushes proactive messages to the browser.
