<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/aimu-horizontal-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/aimu-horizontal-light.png">
  <img alt="AIMU" src="docs/assets/aimu-horizontal-light.png" width="480">
</picture>

**Simple, composable AI for Python, local or in the cloud.**

[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Docs](https://saxman.github.io/aimu/) · [Tutorials](https://saxman.github.io/aimu/tutorials/) · [How-to](https://saxman.github.io/aimu/how-to/) · [Reference](https://saxman.github.io/aimu/reference/) · [Notebooks](notebooks/)

</div>

AIMU is a Python library for AI-powered applications, with language models as the primary building block. It gives you a single provider-agnostic interface across text, images, audio, and speech; autonomous agents and code-controlled workflows; and small composable utilities for tools, memory, prompt tuning, evaluations, and benchmarking. All of these features in plain Python that is apparent and easy to use.

Whether you need vision input, autonomous tool use, image generation, audio generation, or text-to-speech, the call is one line:
```python
aimu.chat("What's in this photo?", model="...", images=["photo.jpg"])

aimu.agent("...", tools=builtin.web).run("Search the web and summarize today's AI news")

aimu.generate_image("a watercolor fox in a snowy forest", model="...")
aimu.generate_audio("a lo-fi hip-hop beat with soft piano", model="...")
aimu.generate_speech("Hello, world!", model="...")
```
Composition happens by passing objects to constructors. Conversation state is a `list[dict]` you can print and edit. Provider-specific details adapt at request time and never leak into your code.

## Key features

### Language models

- One client interface for Ollama, HuggingFace, llama-cpp, the Claude API, OpenAI, Gemini, and any OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve). Swap with a string change: `"provider:model_id"`.
- Reasoning, tool calling, and vision input work identically across every provider. Reasoning models surface their tokens as `StreamingContentType.THINKING` chunks via the same API.
- Typed streaming: `StreamChunk(phase, content, agent, iteration)` flows through `client.chat()`, `Agent.run()`, and every workflow. Filter with `include=["generating"]`.

### Image and audio generation

- Consistent APIs for text-to-image (`aimu.image_client()` / `aimu.generate_image()`) and text-to-audio (`aimu.audio_client()` / `aimu.generate_audio()`), mirroring the text client interface.
- For images: HuggingFace `diffusers` locally (SD 1.5 / SDXL / SD 3.5 / FLUX 1 dev & schnell / FLUX 2 Klein 4B & 9B) and Google Nano Banana via the cloud API. Pass `reference_image=` to any `generate()` call for image-to-image workflows.
- For audio (music and sound): HuggingFace with MusicGen small/medium/large (32 kHz), AudioLDM2 (16 kHz), and Stable Audio Open (44.1 kHz stereo).
- Drop image and audio generation into any chat agent via the built-in `generate_image` and `generate_audio` tools.

### Speech

- `aimu.speech_client()` / `aimu.generate_speech()` for text-to-speech. HuggingFace locally (SpeechT5, MMS-TTS, BARK); OpenAI (`tts-1`, `tts-1-hd`) in the cloud.
- Drop TTS into any agent via the built-in `generate_speech` tool; bind a specific voice with `make_speech_tool(client, voice=...)`.
- Speech-to-text (transcription) is planned as a parallel `aimu.transcription_client()` surface.

### Agents and workflows

- `Agent` runs an autonomous tool-using loop until the model stops calling tools.
- `OrchestrationAgent` interface/pattern for coordinating sub-agent work, and three pre-built agents (`CodeReviewAgent`, `ContentCreationAgent`, and `ResearchReportAgent`).
- Four code-controlled workflow patterns: `Chain.from_client(...)`, `Router.from_client(...)`, `Parallel.from_client(...)`, `EvaluatorOptimizer(...)`. Compose freely. Workflows accept agents as steps; agents accept workflows as tools via `as_model_client()`.
- `agent.as_model_client()` makes any agent a drop-in `BaseModelClient`, so agentic and non-agentic clients are interchangeable.

### Tools

- `@tool` on any plain Python function. Type hints + docstring become the spec.
- Per-call tool override: pass `tools=` to `client.chat()` or `Agent.run()` to swap the tool set for one call (or `tools=[]` to disable), without mutating the client's configured tools.
- `MCPClient` for cross-process FastMCP tools. `mcp.as_tools()` turns a server's tools into `@tool`-style callables you add to `tools=` — one registry, mix freely with `@tool` functions (`tools=builtin.web + mcp.as_tools()`).
- Built-in tool groups ready to pass to `tools=`: `builtin.web`, `builtin.fs`, `builtin.compute`, `builtin.misc`, `builtin.image`, `builtin.audio`, `builtin.speech`. `builtin.make_tools(client, ..., python_sandbox=False)` assembles the full tool list with optional wiring for image/vision/audio/speech/memory/sandbox.
- `execute_python` sandboxed Python REPL in `builtin.compute`: run code, capture stdout and last expression. Opt-in only (`tools=builtin.compute` or `make_tools(python_sandbox=True)`).
- Filesystem-discovered `SKILL.md` files auto-inject into a `SkillAgent` (same format Claude Code uses).

### Memory and persistence

- `SemanticMemoryStore`, `DocumentStore`, and `ConversationManager` all implement the same `MemoryStore` interface and are interchangeable wherever one is accepted.
- `SemanticMemoryStore` - ChromaDB vector search for semantic fact storage and retrieval.
- `DocumentStore` - path-keyed document storage, drop-in compatible with the Claude memory tool API.
- `ConversationManager` - TinyDB-backed chat history that persists across sessions. Integrates directly with any `ModelClient` via `client.messages`.
- `make_memory_tools(store)` returns `store_memory`, `search_memories`, and `list_memories` as `@tool` functions for direct in-process agent use. Pass the result to `Agent(client, tools=...)` or include it via `make_tools(..., memory_store=store)`. For cross-process or multi-agent memory, use the FastMCP servers in `aimu.memory.mcp` / `aimu.memory.document_mcp`.

### Output utilities

- `parse_json_response(text, schema=None)` — extract JSON from any LLM response string (raw, fenced block, or `{…}` substring). Pass a dataclass or Pydantic v2 model to coerce into a typed object.
- `generate_json(client, prompt, schema=None, *, retries=2)` — generate then parse in one call, retrying automatically on malformed output.
- `extract_tool_calls(messages)` — turn an OpenAI-format message list into a clean `list[dict]` of `{iteration, tool, arguments, result}` records. Works on `agent.model_client.messages` or any `client.messages`.
- `runner.restore(messages)` — restore an `Agent`, `EvaluatorOptimizer`, or `Chain` from a saved message list for resuming after failure. Handles the system-message de-duplication automatically.
- HuggingFace model weight caching — all four modality clients (text, image, audio, speech) share a module-level registry; a second instance for the same model reuses weights. Call `aimu.clear_hf_cache()` to free VRAM. Same pattern for `LlamaCppClient` via `aimu.clear_llamacpp_cache()`.

### Prompts and evaluation

- Hill-climbing `PromptTuner` for automatic prompt optimisation against labelled data. Four concrete tuners: classification, multi-class, extraction, judged-generation.
- `Benchmark` runs one prompt across multiple clients (plain or agentic, mixed providers) and returns a comparison DataFrame. DeepEval metrics plug in as `Scorer`s.

### Async (optional)

- `aimu.aio` mirrors the entire public surface — same class names, one import switches paradigms. The sync ladder is unchanged; async is strictly opt-in.
- `aio.Parallel` and `concurrent_tool_calls=True` use `asyncio.TaskGroup` for structured concurrency: sibling cancellation on first failure, `ExceptionGroup` aggregation.
- Same `@tool`-decorated functions work on both surfaces. `async def` tools are auto-detected and awaited; sync (CPU-bound) tools are routed through `asyncio.to_thread` so the event loop stays free.
- Native async providers: Anthropic, OpenAI, Gemini, Ollama, every OpenAI-compatible endpoint. In-process providers (HuggingFace, LlamaCpp) wrap an existing sync client so model weights load only once.

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

**Default model.** Omit `model=` and AIMU resolves one for you: it reads `AIMU_LANGUAGE_MODEL` (`"provider:model_id"`), else auto-selects an already-available local model (a running Ollama server, a cached HuggingFace model, or a local OpenAI-compatible server). A cloud provider is never auto-selected and weights are never downloaded implicitly.

```python
reply = aimu.chat("Hello")                # uses AIMU_LANGUAGE_MODEL or a discovered local model
client = aimu.client(system="Be brief.")  # same resolution
```

Image, audio, and speech read `AIMU_IMAGE_MODEL` / `AIMU_AUDIO_MODEL` / `AIMU_SPEECH_MODEL` respectively.

**Streaming with phase filtering.** Drop unwanted phases (thinking, tool calls) with `include=`:

```python
for chunk in client.chat("Tell me a story", stream=True, include=["generating"]):
    print(chunk.content, end="", flush=True)
```

**Tools for models and agents.** `@tool` works on any plain function:

```python
from aimu.tools import tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = aimu.agent("ollama:qwen3.5:9b", tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))
```

**Code-controlled workflows.** AIMU supports several workflow patterns: chaining, routing, parallelization, and evaluation loops. `Chain.from_client()`, for example, executes a series of LLM calls using a shared client and a list of per-step instructions:

```python
from aimu.agents import Chain

chain = Chain.from_client(client, [
    "Break the task into clear steps.",
    "Execute each step using available tools.",
    "Polish the result into a single paragraph.",
])
result = chain.run("Research the top Python web frameworks.")
```

**Vision input.** Uniform across every vision-capable provider — on stateful `chat()` or stateless one-shot `generate()`:

```python
client = aimu.client("openai:gpt-4o-mini")
client.chat("What's in this image?", images=["./cat.jpg"])      # multi-turn, keeps history
client.generate("Caption this image.", images=["./cat.jpg"])    # one-shot, no history
```

**Image generation.** Same `provider:model_id` shape, parallel factory. Pass `reference_image=` for image-to-image:

```python
# One-shot, local HuggingFace diffusers
path = aimu.generate_image(
    "a watercolor of a fox in a snowy forest",
    model="hf:runwayml/stable-diffusion-v1-5",
)

# Reuse loaded weights across calls
client = aimu.image_client("hf:stabilityai/stable-diffusion-xl-base-1.0")
img = client.generate("a cyberpunk city skyline at dusk")

# Image-to-image: steer generation from a reference image
img = client.generate("a cyberpunk version", reference_image="./photo.jpg", strength=0.7)

# FLUX.2 Klein — 4-step distilled, native img2img (no separate strength param)
client = aimu.image_client("hf:black-forest-labs/FLUX.2-klein-4B")
img = client.generate("a cat in a sunlit garden")
img = client.generate("add snow", reference_image="./cat.jpg")  # img2img
```


**Audio generation.** Same `provider:model_id` shape, parallel factory:

```python
# One-shot — returns (sample_rate, np.ndarray) by default
sr, audio = aimu.generate_audio(
    "a lo-fi hip-hop beat with soft piano",
    model="hf:facebook/musicgen-small",
    duration_s=5.0,
)

# Save directly to WAV
path = aimu.generate_audio("ambient ocean waves", model="hf:facebook/musicgen-small", format="path")
```

**Speech generation (TTS).** Same `provider:model_id` shape:

```python
# Save spoken WAV — OpenAI cloud (requires OPENAI_API_KEY)
path = aimu.generate_speech("Hello, world!", model="openai:tts-1")

# Local HuggingFace TTS — returns (sample_rate, np.ndarray)
sr, audio = aimu.generate_speech("Hello!", model="hf:facebook/mms-tts-eng", format="numpy")

# Reuse a client across calls — weights load once
client = aimu.speech_client("openai:tts-1-hd")
path = client.generate("Good morning.", voice="nova", format="path")
```

**Vision and image generation together.** A vision-capable agent with `generate_image` as a tool can perceive and create in the same run:

```python
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(aimu.client("anthropic:claude-sonnet-4-6"), tools=[builtin.generate_image])
agent.run("Describe the scene in this photo, then generate a watercolor painting of it.", images=["photo.jpg"])
```

**Memory-capable agents.** `make_memory_tools(store)` gives any agent `store_memory`, `search_memories`, and `list_memories` as in-process tools. The store is always explicit — you control whether it's ephemeral or on-disk, and which backend it uses:

```python
from aimu.agents import Agent
from aimu.memory import SemanticMemoryStore
from aimu.tools.builtin import make_memory_tools

store = SemanticMemoryStore(persist_path="./.memory")
agent = Agent(aimu.client("anthropic:claude-sonnet-4-6"), tools=make_memory_tools(store))
agent.run("Remember that the meeting is on Friday at 2pm.")
agent.run("When is the meeting?")  # retrieves from memory
```

**Async (opt-in).** Same names, one import away:

```python
import asyncio
from aimu import aio

async def main():
    client = aio.client("anthropic:claude-sonnet-4-6")
    agent = aio.Agent(client, tools=[my_async_tool])
    reply = await agent.run("Hello")

    # asyncio.TaskGroup-backed Parallel — true coroutine concurrency
    parallel = aio.Parallel.from_client(client, worker_prompts=[...], aggregator_prompt="...")
    result = await parallel.run("topic")

asyncio.run(main())
```

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]` (also enables OpenAI TTS speech), `aimu[hf]` (text + HuggingFace `diffusers` image + HuggingFace audio + HuggingFace TTS speech), `aimu[google]` (Nano Banana image generation), `aimu[llamacpp]`. See [installation in the docs](https://saxman.github.io/aimu/tutorials/01-getting-started/) for the full list of extras.

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
| [02 - Vision](notebooks/02%20-%20Vision.ipynb) | Image input via `images=` on `chat()` and one-shot `generate()` |
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
| [14 - Async](notebooks/14%20-%20Async.ipynb) | `aimu.aio` surface end-to-end: chat, streaming, async tools, `asyncio.TaskGroup`-backed `Parallel`, async `MCPClient`, in-process provider wrapping |
| [15 - Image Generation](notebooks/15%20-%20Image%20Generation.ipynb) | `aimu.image_client()` / `aimu.generate_image()` with HuggingFace `diffusers` and Google Nano Banana, plus the built-in `generate_image` agent tool |
| [16 - Audio Generation](notebooks/16%20-%20Audio%20Generation.ipynb) | `aimu.audio_client()` / `aimu.generate_audio()` with MusicGen, AudioLDM2, and Stable Audio Open, plus streaming and the built-in `generate_audio` agent tool |
| [17 - Speech](notebooks/17%20-%20Speech.ipynb) | TTS with HuggingFace (SpeechT5, MMS-TTS, BARK) and OpenAI (tts-1/tts-1-hd); `generate_speech` agent tool; Streamlit live narration; STT placeholder |

## Web apps

The [`web/`](web/) directory ships two Streamlit chat applications that demonstrate AIMU in action:

| App | Description |
|---|---|
| [streamlit_chatbot_basic.py](web/streamlit_chatbot_basic.py) | ~70-line showcase — provider/model selector, streaming chat, built-in tools. Start here. |
| [streamlit_chatbot.py](web/streamlit_chatbot.py) | Full-featured — image/audio/speech generation, agentic mode, thinking display, generation sliders, live TTS narration. Extensible foundation. |

```bash
streamlit run web/streamlit_chatbot.py         # full-featured Streamlit demo (agents, tools, images, audio, speech narration, etc.)
streamlit run web/streamlit_chatbot_basic.py   # basic Streamlit demo app
python web/gradio_chatbot_basic.py             # basic Gradio demo app
```

## Design principles

AIMU is small and stays small. Six principles shape the API: plain Python, plain data (OpenAI message dicts only), composability through uniform interfaces, progressive disclosure, direct paths for common tasks, and apparent failures. The reasoning behind each, and the patterns each one excludes, lives on the [design principles](https://saxman.github.io/aimu/explanation/design-principles/) page.

A curated model catalog, capturing model capabilities and nuances, is part of that design: every `"provider:model_id"` string must name a model AIMU ships a spec for. An unknown id raises rather than running with guessed capabilities. To use a one-off custom model, build the spec and pass it directly (`aimu.image_client(HuggingFaceImageSpec(...))`).

## Contributing

See the [contributing guide](https://saxman.github.io/aimu/contributing/) for dev setup, testing, lint, and PR conventions.

## License

Apache 2.0.
