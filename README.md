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
- Token usage is surfaced as `client.last_usage` (`{"input_tokens", "output_tokens", "total_tokens"}`) after each non-streaming call.
- Structured output: pass `schema=` (a dataclass or Pydantic model) to `chat()` / `generate()` to get a typed object back. Native provider enforcement (OpenAI `response_format`, Ollama `format=`, Anthropic forced-tool) where available, with a prompt-and-parse fallback elsewhere.
- `aimu.embedding_client()` / `aimu.embed()` for text embeddings. OpenAI and Ollama, plus local HuggingFace `sentence-transformers`. Single string → one vector; list → list of vectors.

### Image and audio generation

- Consistent APIs for text-to-image (`aimu.image_client()` / `aimu.generate_image()`) and text-to-audio (`aimu.audio_client()` / `aimu.generate_audio()`), mirroring the text client interface.
- For images: HuggingFace `diffusers` locally (SD 1.5 / SDXL / SD 3.5 / FLUX 1 dev & schnell / FLUX 2 Klein 4B & 9B) and Google Nano Banana via the cloud API. Pass `reference_image=` to any `generate()` call for image-to-image workflows.
- For audio (music and sound): HuggingFace with MusicGen small/medium/large (32 kHz), AudioLDM2 (16 kHz), and Stable Audio Open (44.1 kHz stereo).
- Drop image and audio generation into any chat agent via the built-in `generate_image` and `generate_audio` tools.

### Speech and transcription

- `aimu.speech_client()` / `aimu.generate_speech()` for text-to-speech. HuggingFace locally (SpeechT5, MMS-TTS, BARK); OpenAI (`tts-1`, `tts-1-hd`) in the cloud.
- Drop TTS into any agent via the built-in `generate_speech` tool; bind a specific voice with `make_speech_tool(client, voice=...)`.
- `aimu.transcription_client()` / `aimu.transcribe()` for speech-to-text. OpenAI (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe) in the cloud; HuggingFace Whisper family locally. Pass `response_format="verbose_json"` for timed segments.
- Drop transcription into any agent via the built-in `transcribe_audio` tool; bind a specific client with `make_transcription_tool(client)`.

### Agents and workflows

- `Agent` runs an autonomous tool-using loop until the model stops calling tools.
- `OrchestrationAgent` interface/pattern for coordinating sub-agent work, and three pre-built agents (`CodeReviewAgent`, `ContentCreationAgent`, and `ResearchReportAgent`).
- Four code-controlled workflow patterns: `Chain.from_client(...)`, `Router.from_client(...)`, `Parallel.from_client(...)`, `EvaluatorOptimizer(...)`. Compose freely. Workflows accept agents as steps; agents accept workflows as tools via `as_model_client()`.
- `agent.as_model_client()` makes any agent a drop-in `BaseModelClient`, so agentic and non-agentic clients are interchangeable.

### Tools

- `@tool` on any plain Python function. Type hints + docstring become the spec.
- Per-call tool override: pass `tools=` to `client.chat()` or `Agent.run()` to swap the tool set for one call (or `tools=[]` to disable), without mutating the client's configured tools.
- `MCPClient` for cross-process FastMCP tools. `mcp.as_tools()` turns a server's tools into `@tool`-style callables you add to `tools=` — one registry, mix freely with `@tool` functions (`tools=builtin.web + mcp.as_tools()`).
- Built-in tool groups ready to pass to `tools=`: `builtin.web`, `builtin.fs`, `builtin.compute`, `builtin.misc`, `builtin.image`, `builtin.audio`, `builtin.speech`, `builtin.transcription`. `builtin.make_tools(client, ..., python_sandbox=False)` assembles the full tool list with optional wiring for image/vision/audio/speech/transcription/memory/sandbox.
- `execute_python` sandboxed Python REPL in `builtin.compute`: run code, capture stdout and last expression. Opt-in only (`tools=builtin.compute` or `make_tools(python_sandbox=True)`).
- Filesystem-discovered `SKILL.md` files auto-inject into a `SkillAgent` (same format Claude Code uses).

### Memory and persistence

- `SemanticMemoryStore`, `DocumentStore`, and `ConversationManager` all implement the same `MemoryStore` interface and are interchangeable wherever one is accepted.
- `SemanticMemoryStore` - ChromaDB vector search for semantic fact storage and retrieval. Pass `embedding_client=aimu.embedding_client(...)` to use your own embedding model instead of ChromaDB's default.
- `DocumentStore` - path-keyed document storage, drop-in compatible with the Claude memory tool API.
- `ConversationManager` - TinyDB-backed chat history that persists across sessions. Integrates directly with any `ModelClient` via `client.messages`.
- `make_memory_tools(store)` returns `store_memory`, `search_memories`, and `list_memories` as `@tool` functions for direct in-process agent use. Pass the result to `Agent(client, tools=...)` or include it via `make_tools(..., memory_store=store)`. For cross-process or multi-agent memory, use the FastMCP servers in `aimu.memory.mcp` / `aimu.memory.document_mcp`.
- `aimu.rag` — retrieval-augmented generation as plain functions over `MemoryStore`: `split_text` (recursive, token-aware), `ingest`, `retrieve`, `format_context`, and optional cross-encoder `rerank`. `make_retrieval_tool(store)` exposes retrieval as an agent tool. No retriever/splitter/loader class hierarchy.

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

### Chat

One-shot with `aimu.chat()`, multi-turn with `aimu.client()`, and streaming with phase filtering (thinking, tool usage, generation) via `include=`. Omit `model=` and AIMU resolves one for you by reading `AIMU_[LANGUAGE|IMAGE|AUDIO|SPEECH|TRANSCRIPTION|EMBEDDING]_MODEL` or auto-selects an available local model (LANGUAGE only, via a running Ollama server, a cached HuggingFace model, or a local OpenAI-compatible server).

```python
import aimu

# One-shot
text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

# Multi-turn — history preserved across calls
client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
client.chat("Hi there")
client.chat("What did I just say?")

# Default model — resolves AIMU_LANGUAGE_MODEL or a discovered local model
reply = aimu.chat("Hello")
client = aimu.client(system="Be brief.")

# Streaming — drop unwanted phases (thinking, tool calls) with include=
for chunk in client.chat("Tell me a story", stream=True, include=["generating"]):
    print(chunk.content, end="", flush=True)
```

### Tools and workflows

`@aimu.tool` turns any plain function into a tool (type hints + docstring become the spec). `Chain.from_client()` runs a series of LLM calls over a shared client with per-step instructions; `Router`, `Parallel`, and `EvaluatorOptimizer` follow the same shape.

```python
import aimu
from aimu.agents import Chain

@aimu.tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())

agent = aimu.agent("ollama:qwen3.5:9b", tools=[letter_counter])
print(agent.run("How many r's in strawberry?"))

chain = Chain.from_client(agent.model_client, [
    "Break the task into clear steps.",
    "Execute each step using available tools.",
    "Polish the result into a single paragraph.",
])
result = chain.run("Research the top Python web frameworks.")
```

### Vision and audio input

Pass `images=` or `audio=` to any vision- or audio-capable text model, on stateful `chat()` or stateless one-shot `generate()`. Both accept a file path (`str` or `pathlib.Path`), raw `bytes`, a `data:...;base64,...` URL, or an `https://` URL. Audio providers: OpenAI (GPT-4o, GPT-4.1 series), Gemini (2.0/2.5), HuggingFace Gemma 4 / Nemotron-H-8B.

```python
client = aimu.client("openai:gpt-4o")

# Vision
client.chat("What's in this image?", images=["./cat.jpg"])      # multi-turn, keeps history
client.generate("Caption this image.", images=["./cat.jpg"])    # one-shot, no history

# Audio
client.chat("Transcribe this clip.", audio=["./interview.wav"])           # multi-turn
client.generate("What language is spoken here?", audio=["./clip.mp3"])    # one-shot
```

### Image, audio, and speech generation

Each modality has a parallel factory (`image_client` / `audio_client` / `speech_client`) and one-shot helper (`generate_image` / `generate_audio` / `generate_speech`), all using the same `provider:model_id` shape. Pass `reference_image=` to any image `generate()` for image-to-image.

```python
import aimu

# Image — local HuggingFace diffusers, with image-to-image
path = aimu.generate_image("a watercolor of a fox in a snowy forest", model="hf:runwayml/stable-diffusion-v1-5")
client = aimu.image_client("hf:stabilityai/stable-diffusion-xl-base-1.0")  # reuse loaded weights
img = client.generate("a cyberpunk city skyline at dusk")
img = client.generate("a cyberpunk version", reference_image="./photo.jpg", strength=0.7)

# FLUX.2 Klein — 4-step distilled, native img2img (no separate strength param)
klein = aimu.image_client("hf:black-forest-labs/FLUX.2-klein-4B")
img = klein.generate("add snow", reference_image="./cat.jpg")

# Audio (music and sound) — returns (sample_rate, np.ndarray) by default
sr, audio = aimu.generate_audio("a lo-fi hip-hop beat with soft piano", model="hf:facebook/musicgen-small", duration_s=5.0)
path = aimu.generate_audio("ambient ocean waves", model="hf:facebook/musicgen-small", format="path")

# Speech (TTS) — OpenAI cloud or local HuggingFace
path = aimu.generate_speech("Hello, world!", model="openai:tts-1")
sr, audio = aimu.generate_speech("Hello!", model="hf:facebook/mms-tts-eng", format="numpy")
tts = aimu.speech_client("openai:tts-1-hd")  # reuse a client across calls
path = tts.generate("Good morning.", voice="nova", format="path")

# Speech-to-text (transcription) — OpenAI cloud or local HuggingFace
text = aimu.transcribe("./clip.wav", model="openai:whisper-1")
stt = aimu.transcription_client("hf:openai/whisper-tiny")
text = stt.transcribe("./clip.wav")
```

### Structured output

Pass `schema=` (a dataclass or Pydantic model) to `chat()` / `generate()` to get a typed object back. Native enforcement on capable models (OpenAI / Ollama / Anthropic), with a prompt-and-parse fallback otherwise.

```python
import aimu
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

person = aimu.client("openai:gpt-4.1").chat("Extract: Ada Lovelace, 36", schema=Person)
# Person(name="Ada Lovelace", age=36)
```

### Embeddings and RAG

`aimu.embed()` / `aimu.embedding_client()` map text to vectors (single string → one vector, list → list of vectors). RAG is plain functions over a `MemoryStore`: chunk with `ingest`, fetch with `retrieve`, ground with `format_context`.

```python
import aimu
from aimu.memory import SemanticMemoryStore
from aimu.rag import ingest, retrieve, format_context

vector = aimu.embed("the quick brown fox", model="openai:text-embedding-3-small")  # list[float]
embedder = aimu.embedding_client("hf:BAAI/bge-small-en-v1.5")  # local sentence-transformers
vectors = embedder.embed(["alpha", "beta"])                    # list[list[float]]

store = SemanticMemoryStore(embedding_client=embedder)  # use a chosen embedding model
ingest(store, my_documents, chunk_size=800, chunk_overlap=100)

question = "What is AIMU's design philosophy?"
context = format_context(retrieve(store, question, n_results=5))
answer = aimu.chat(f"Context:\n{context}\n\nQuestion: {question}")
```

### Composed agents

Agents combine perception, generation, and memory in a single run. A vision-capable agent can take `generate_image` as a tool; `make_memory_tools(store)` adds `store_memory`, `search_memories`, and `list_memories` over an explicit (ephemeral or on-disk) store.

```python
from aimu.agents import Agent
from aimu.tools import builtin
from aimu.tools.builtin import make_memory_tools
from aimu.memory import SemanticMemoryStore

# Perceive and create in one run
agent = Agent(aimu.client("anthropic:claude-sonnet-4-6"), tools=[builtin.generate_image])
agent.run("Describe the scene in this photo, then generate a watercolor painting of it.", images=["photo.jpg"])

# Memory across turns
store = SemanticMemoryStore(persist_path="./.memory")
agent = Agent(aimu.client("anthropic:claude-sonnet-4-6"), tools=make_memory_tools(store))
agent.run("Remember that the meeting is on Friday at 2pm.")
agent.run("When is the meeting?")
```

### Async

`aimu.aio` mirrors the entire public surface — same class names, one import switches paradigms. `aio.Parallel` and `concurrent_tool_calls=True` use `asyncio.TaskGroup` for true coroutine concurrency.

```python
import asyncio
from aimu import aio

async def main():
    client = aio.client("anthropic:claude-sonnet-4-6")
    agent = aio.Agent(client, tools=[my_async_tool])
    reply = await agent.run("Hello")

    parallel = aio.Parallel.from_client(client, worker_prompts=[...], aggregator_prompt="...")
    result = await parallel.run("topic")

asyncio.run(main())
```

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]` (also enables OpenAI TTS speech and transcription STT), `aimu[hf]` (text + HuggingFace `diffusers` image + HuggingFace audio + HuggingFace TTS speech), `aimu[google]` (Nano Banana image generation), `aimu[llamacpp]`. See [installation in the docs](https://saxman.github.io/aimu/tutorials/01-getting-started/) for the full list of extras.

## Documentation

- 📘 [Tutorials](https://saxman.github.io/aimu/tutorials/): Hand-held walkthroughs. Install to first agent in 15 mins
- 🛠️ [How-to guides](https://saxman.github.io/aimu/how-to/): Task-oriented recipes (switch providers, write a tool, stream output, benchmark models, ...)
- 📚 [Reference](https://saxman.github.io/aimu/reference/): Auto-generated API docs, capability matrices, environment variables, CLI
- 💡 [Explanation](https://saxman.github.io/aimu/explanation/): The *why*: architecture, design principles, agents vs workflows

## Notebooks

The [`notebooks/`](notebooks/) directory ships interactive demos for every subsystem:

- [01 - Model Client](notebooks/01%20-%20Model%20Client.ipynb): Text generation, chat, streaming, thinking models
- [02 - Conversations](notebooks/02%20-%20Conversations.ipynb): Persistent multi-turn chat history (`ConversationManager`)
- [03 - Structured Output](notebooks/03%20-%20Structured%20Output.ipynb): Typed responses via `schema=` on `chat()` / `generate()`; native enforcement (OpenAI / Ollama / Anthropic) with a prompt-and-parse fallback
- [04 - Vision](notebooks/04%20-%20Vision.ipynb): Image input via `images=` on `chat()` and one-shot `generate()`
- [05 - Audio Input](notebooks/05%20-%20Audio%20Input.ipynb): Audio input via `audio=` on `chat()` and `generate()`; model selection; accepted formats; async surface
- [06 - Tools](notebooks/06%20-%20Tools.ipynb): `@tool` decorator, built-in tool groups, MCPClient
- [07 - Agents](notebooks/07%20-%20Agents.ipynb): `Agent` and `agent.as_model_client()`
- [08 - Agent Skills](notebooks/08%20-%20Agent%20Skills.ipynb): Filesystem-discovered skill injection
- [09 - Workflows](notebooks/09%20-%20Workflows.ipynb): Chain, Router, Parallel, EvaluatorOptimizer, PlanExecuteEvaluator
- [10 - Prebuilt Agents](notebooks/10%20-%20Prebuilt%20Agents.ipynb): Orchestrator + worker tools pattern
- [11 - Embeddings](notebooks/11%20-%20Embeddings.ipynb): Text embeddings via `embedding_client()` / `embed()`; OpenAI, Ollama, and local HuggingFace providers; cosine similarity; pluggable into `SemanticMemoryStore`; async surface
- [12 - Memory](notebooks/12%20-%20Memory.ipynb): Semantic fact storage and retrieval
- [13 - RAG](notebooks/13%20-%20RAG.ipynb): Retrieval-augmented generation with `aimu.rag`: `split_text` chunking, `ingest` / `retrieve` / `format_context`, reranking, and `make_retrieval_tool` for agents
- [14 - Prompt Management](notebooks/14%20-%20Prompt%20Management.ipynb): Versioned prompt storage
- [15 - Prompt Tuning](notebooks/15%20-%20Prompt%20Tuning.ipynb): Classification, multi-class, extraction, judged tuners
- [16 - Evaluations](notebooks/16%20-%20Evaluations.ipynb): DeepEval integration
- [17 - Benchmarking](notebooks/17%20-%20Benchmarking.ipynb): Multi-model comparison harness
- [18 - Image Generation](notebooks/18%20-%20Image%20Generation.ipynb): `aimu.image_client()` / `aimu.generate_image()` with HuggingFace `diffusers` and Google Nano Banana, plus the built-in `generate_image` agent tool
- [19 - Audio Generation](notebooks/19%20-%20Audio%20Generation.ipynb): `aimu.audio_client()` / `aimu.generate_audio()` with MusicGen, AudioLDM2, and Stable Audio Open, plus streaming and the built-in `generate_audio` agent tool
- [20 - Speech](notebooks/20%20-%20Speech.ipynb): TTS with HuggingFace (SpeechT5, MMS-TTS, BARK) and OpenAI (tts-1/tts-1-hd); `generate_speech` agent tool; Streamlit live narration (speech-to-text is notebook 21)
- [21 - Transcription](notebooks/21%20-%20Transcription.ipynb): Speech-to-text via `transcription_client()` / `transcribe()`; OpenAI and HuggingFace providers; timestamps; async surface
- [22 - Async](notebooks/22%20-%20Async.ipynb): `aimu.aio` surface end-to-end: chat, streaming, async tools, `asyncio.TaskGroup`-backed `Parallel`, async `MCPClient`, in-process provider wrapping

## Web apps

The [`web/`](web/) directory ships chat applications that demonstrate AIMU in action:

- [streamlit_chatbot_basic.py](web/streamlit_chatbot_basic.py): ~70-line showcase. Provider/model selector, streaming chat, built-in tools. Start here.
- [streamlit_chatbot.py](web/streamlit_chatbot.py): Full-featured. Image/audio/speech generation, agentic mode, thinking display, generation sliders, live TTS narration. Extensible foundation.
- [gradio_chatbot_basic.py](web/gradio_chatbot_basic.py): Basic Gradio chat interface with streaming.

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
