<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/aimu-horizontal-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/aimu-horizontal-light.png">
  <img alt="AIMU" src="docs/assets/aimu-horizontal-light.png" width="480">
</picture>

**Simple, composable AI for Python, local or in the cloud.**

[![PyPI](https://img.shields.io/pypi/v/aimu)](https://pypi.org/project/aimu/) ![GitHub License](https://img.shields.io/github/license/saxman/genscai) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Faimu%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Docs](https://saxman.github.io/aimu/) · [Tutorials](https://saxman.github.io/aimu/tutorials/) · [How-to](https://saxman.github.io/aimu/how-to/) · [Reference](https://saxman.github.io/aimu/reference/) · [Examples](examples/) · [Notebooks](notebooks/)

</div>

AIMU is a Python library for AI-powered applications, with language models as the primary building block. It gives you a single provider-agnostic interface across text, images, audio, and speech, autonomous agents and code-controlled workflows, and small composable utilities for tools, memory, prompt tuning, evaluations, and benchmarking. All of these features in plain Python that is apparent and easy to use.

Whether you need vision input, autonomous tool use, image generation, audio generation, or text-to-speech, the call is one line:
```python
aimu.chat("What's in this photo?", model="...", images=["photo.jpg"])

aimu.agent("...", tools=builtin.web).run("Search the web and summarize today's AI news")

aimu.generate_image("a watercolor fox in a snowy forest", model="...")
aimu.generate_audio("a lo-fi hip-hop beat with soft piano", model="...")
aimu.generate_speech("Hello, world!", model="...")
```
Composition happens by passing objects to constructors. Conversation state is a `list[dict]` you can print and edit. Provider-specific details adapt at request time and never leak into your code.

## Why AIMU

AIMU is compact, direct, and easy to understand, by design. Six principles shape the API: plain Python, plain data (OpenAI message dicts only), composability through uniform interfaces, progressive disclosure of capabilities, direct paths for common tasks, and apparent failures. The reasoning behind each, and the patterns each one excludes, lives on the [design principles](https://saxman.github.io/aimu/explanation/design-principles/) page.

A curated model catalog, capturing model capabilities and nuances, is part of that design: every `"provider:model_id"` string must name a model AIMU ships a spec for. An unknown id raises rather than running with guessed capabilities. To use a one-off custom model, build the spec and pass it directly (`aimu.image_client(HuggingFaceImageSpec(...))`).

## Install

```bash
pip install aimu[all]
```

Or pick the providers you need: `aimu[ollama]`, `aimu[anthropic]`, `aimu[openai_compat]` (also enables OpenAI TTS speech and transcription STT), `aimu[hf]` (text + HuggingFace `diffusers` image + HuggingFace audio + HuggingFace TTS speech), `aimu[google]` (Nano Banana image generation), `aimu[llamacpp]`. See [installation in the docs](https://saxman.github.io/aimu/tutorials/01-getting-started/) for the full list of extras.

## Key features

### Language models

- **[One provider-agnostic client](https://saxman.github.io/aimu/how-to/switch-providers/).** A single interface covers Ollama, HuggingFace, llama-cpp, Claude, OpenAI, Gemini, and any OpenAI-compatible local server (LM Studio, vLLM, SGLang, llama-server, HF Transformers Serve), and you swap providers with a `"provider:model_id"` string change.
- **[Reasoning, tools, and vision everywhere](https://saxman.github.io/aimu/reference/model-matrix/).** These capabilities work identically across every provider, and reasoning models surface their thinking as a distinct stream phase through the same API.
- **[Typed streaming](https://saxman.github.io/aimu/how-to/stream-output/).** A streamed response arrives as `StreamChunk` objects tagged by phase, so reasoning, tool calls, and generated text are each labelled and you keep only the phases you want with `include=`. The same chunk type flows through `client.chat()`, `Agent.run()`, and every workflow, so one rendering loop handles them all.
- **[Structured output](https://saxman.github.io/aimu/how-to/use-structured-output/).** Pass `schema=` (a dataclass or Pydantic model) to `chat()` or `generate()` to get a validated, typed object back, with native enforcement on OpenAI, Ollama, and Anthropic and a prompt-and-parse fallback (`parse_json_response`, `generate_json`) elsewhere.
- **[Embeddings](https://saxman.github.io/aimu/how-to/use-embeddings/).** Map text to vectors with `aimu.embedding_client()` / `aimu.embed()` over OpenAI, Ollama, and local HuggingFace `sentence-transformers`, where a string returns one vector and a list returns a list of vectors.
- **[Local weight reuse](https://saxman.github.io/aimu/reference/api/models/).** HuggingFace clients across every modality (text, image, audio, speech) and llama-cpp share loaded weights through a process-level registry, so a second client for the same model skips the load, and `aimu.clear_hf_cache()` / `aimu.clear_llamacpp_cache()` free VRAM on demand.

### Generative modalities

- **[Image generation](https://saxman.github.io/aimu/how-to/generate-images/).** Create images with `aimu.image_client()` / `aimu.generate_image()` using HuggingFace `diffusers` locally (SD 1.5, SDXL, SD 3.5, FLUX) and Google Nano Banana in the cloud, and pass `reference_image=` to any `generate()` for image-to-image.
- **[Audio and speech synthesis](https://saxman.github.io/aimu/how-to/generate-audio/).** Generate music and sound with `generate_audio` (MusicGen, AudioLDM2, Stable Audio Open) and [spoken audio](https://saxman.github.io/aimu/how-to/generate-speech/) with `generate_speech` (HuggingFace SpeechT5, MMS-TTS, BARK, plus OpenAI `tts-1`).
- **[Transcription](https://saxman.github.io/aimu/how-to/transcribe-audio/).** Convert speech to text with `aimu.transcribe()` over OpenAI (whisper-1, gpt-4o-transcribe) and the local HuggingFace Whisper family, requesting `response_format="verbose_json"` for timed segments.

### Agents and workflows

- **[Autonomous agents](https://saxman.github.io/aimu/tutorials/02-first-agent-with-tools/).** An `Agent` runs a tool-using loop until the model stops calling tools, while `OrchestratorAgent` coordinates sub-agents and ships three prebuilt variants (`CodeReviewAgent`, `ContentCreationAgent`, `ResearchReportAgent`).
- **[Code-controlled workflows](https://saxman.github.io/aimu/tutorials/03-workflows/).** Build pipelines from `Chain`, `Router`, `Parallel`, and `EvaluatorOptimizer`, each with a `from_client()` factory, then compose them freely: workflows take agents as steps, and `runner.as_tool()` hands any agent or workflow to another agent as a tool.
- **[Interchangeable clients](https://saxman.github.io/aimu/how-to/benchmark-models/#compare-with-an-agentic-client).** Calling `agent.as_model_client()` makes any agent a drop-in `BaseModelClient`, so agentic and non-agentic clients substitute for each other.
- **[A2A interop](https://saxman.github.io/aimu/how-to/connect-agents-a2a/).** With the optional `a2a` extra, `serve_a2a(runner)` exposes any agent or workflow over the Agent2Agent protocol, and `RemoteAgent.connect(url)` consumes a remote agent as a local `Runner`, so it composes (as a tool, a workflow step, or an orchestrator worker) exactly like a local one. The agent-level analog of MCP for tools.
- **[Tracing](https://saxman.github.io/aimu/how-to/stream-output/#trace-a-run).** See exactly what a run did: `extract_tool_calls(messages)` returns a clean list of `{iteration, tool, arguments, result}` records from any message history, and `pretty_print(stream)` renders a live run to the console with reasoning, tool calls, and output labelled.
- **[Resumable runs](https://saxman.github.io/aimu/how-to/using-llms-inside-tools/).** Persist a run's message history and rebuild its state later with `runner.restore(messages)`, which handles system-message de-duplication so an `Agent`, `EvaluatorOptimizer`, or `Chain` can survive a crash or restart.

### Tools

- **[Plain-function tools](https://saxman.github.io/aimu/how-to/add-custom-tool/).** Turn any plain function into a tool with `@tool`, where type hints and the docstring become the spec, and pass `tools=` to `chat()` or `Agent.run()` to override the tool set for one call (or `tools=[]` to disable).
- **[Context across tool calls](https://saxman.github.io/aimu/how-to/add-custom-tool/#share-state-across-tool-calls-toolcontext).** Hand an agent a `deps=` object and any tool reads or updates it through a `ctx: ToolContext[Deps]` parameter, so shared state (a store, a cache, config) carries across tool calls within a run without module globals, and the parameter stays hidden from the model-facing schema.
- **[Pre-built tools](https://saxman.github.io/aimu/reference/api/tools/).** A library of ready-to-use tools ships in the box: web search and fetch, filesystem reads, a calculator, an opt-in sandboxed Python REPL, and the generative-modality tools, grouped as `builtin.web` / `fs` / `compute` / `misc` / `image` / `audio` / `speech` / `transcription` to pass straight to `tools=`.
- **[MCP integration](https://saxman.github.io/aimu/how-to/use-mcp-tools/).** Bring cross-process FastMCP tools into the same registry with `MCPClient` and `mcp.as_tools()`, mixing them freely with `@tool` functions (`tools=builtin.web + mcp.as_tools()`).
- **[Agent skills](https://saxman.github.io/aimu/how-to/use-skills/).** A `SkillAgent` discovers `SKILL.md` files on the filesystem (the same format Claude Code uses) and injects their instructions and tools on demand, so you extend an agent by dropping in a folder instead of changing code.

### Memory and persistence

- **[Uniform store interface](https://saxman.github.io/aimu/reference/api/memory/).** The `SemanticMemoryStore`, `DocumentStore`, and `ConversationManager` classes all implement `MemoryStore` and are interchangeable wherever one is accepted.
- **[Semantic memory](https://saxman.github.io/aimu/how-to/use-semantic-memory/).** ChromaDB vector search backs fact storage and retrieval, and you can pass `embedding_client=` to use your own embedding model instead of ChromaDB's default.
- **[Document and conversation stores](https://saxman.github.io/aimu/how-to/use-document-memory/).** A path-keyed `DocumentStore` (drop-in compatible with the Claude memory tool API) and a TinyDB-backed `ConversationManager` persist documents and chat history across sessions.
- **[Memory tools](https://saxman.github.io/aimu/how-to/use-semantic-memory/).** Calling `make_memory_tools(store)` exposes store, search, and list as agent tools, while FastMCP servers in `aimu.memory.mcp` / `aimu.memory.document_mcp` cover cross-process and multi-agent use.
- **[Retrieval-augmented generation](https://saxman.github.io/aimu/how-to/use-rag/).** The `aimu.rag` module provides plain functions over `MemoryStore` (`split_text`, `ingest`, `retrieve`, `format_context`, and optional cross-encoder `rerank`) with no retriever, splitter, or loader class hierarchy.

### Prompts and evaluation

- **[Prompt tuning](https://saxman.github.io/aimu/how-to/tune-prompts/).** A hill-climbing `PromptTuner` optimises prompts against labelled data, with four concrete tuners for classification, multi-class, extraction, and judged-generation.
- **[Benchmarking](https://saxman.github.io/aimu/how-to/benchmark-models/).** The `Benchmark` harness runs one prompt across multiple clients (plain or agentic, mixed providers) and returns a comparison DataFrame, and DeepEval metrics plug in as `Scorer`s.

### Async (optional)

- **[Full async mirror](https://saxman.github.io/aimu/how-to/use-async/).** The `aimu.aio` namespace mirrors the entire public surface with the same class names, so one import switches paradigms while the sync ladder stays unchanged, and the same `@tool` functions work on both surfaces.
- **[Structured concurrency](https://saxman.github.io/aimu/explanation/async-design/).** Both `aio.Parallel` and `concurrent_tool_calls=True` use `asyncio.TaskGroup` for sibling cancellation and `ExceptionGroup` aggregation, with native async providers throughout and in-process providers (HuggingFace, LlamaCpp) wrapping a sync client so weights load once.

## Example usage

### Chat

One-shot with `aimu.chat()`, multi-turn with `aimu.client()`, and streaming with phase filtering (thinking, tool usage, generation) via `include=`. Omit `model=` and AIMU resolves one for you by reading environment variables or auto-selects an available local model (LLMs only) via a running Ollama server, a cached HuggingFace model, or a local OpenAI-compatible server.

```python
import aimu

# One-shot
text = aimu.chat("Hello", model="anthropic:claude-sonnet-4-6")

# Multi-turn (history preserved across calls)
client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")
client.chat("Hi there")
client.chat("What did I just say?")

# Default model: resolves AIMU_LANGUAGE_MODEL or a discovered local model
reply = aimu.chat("Hello")
client = aimu.client(system="Be brief.")

# Streaming: drop unwanted phases (thinking, tool calls) with include=
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

Tools can reach shared state through an injected `ToolContext` (no globals), a critic can return a
typed verdict instead of a magic string, and `pretty_print` renders a streamed run:

```python
import aimu
from dataclasses import dataclass, field
from pydantic import BaseModel
from aimu import ToolContext
from aimu.agents import Agent, EvaluatorOptimizer

@dataclass
class Deps:
    seen: dict = field(default_factory=dict)

@aimu.tool
def remember(ctx: ToolContext[Deps], key: str, value: str) -> str:
    """Store a value under a key."""   # the model only sees key + value; ctx is injected
    ctx.deps.seen[key] = value
    return "ok"

writer = Agent(aimu.client("ollama:qwen3.5:9b"), tools=[remember], deps=Deps())
aimu.pretty_print(writer.run("Remember the sky is blue, then summarize.", stream=True))

class Verdict(BaseModel):
    passed: bool
    feedback: str = ""

critic = Agent(aimu.client("ollama:qwen3.5:9b"), "Judge the draft; set passed and feedback.")
review = EvaluatorOptimizer(generator=writer, evaluator=critic, verdict_schema=Verdict)
print(review.run("Explain gradient descent."))
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
# Image: local HuggingFace diffusers, with image-to-image
path = aimu.generate_image("a watercolor of a fox in a snowy forest", model="hf:runwayml/stable-diffusion-v1-5")

client = aimu.image_client("hf:stabilityai/stable-diffusion-xl-base-1.0")  # reuse loaded weights
img = client.generate("a cyberpunk city skyline at dusk")
img = client.generate("a cyberpunk version", reference_image="./photo.jpg", strength=0.7)
```

```python
# Audio (music and sound): returns (sample_rate, np.ndarray) by default
sr, audio = aimu.generate_audio("a lo-fi hip-hop beat with soft piano", model="hf:facebook/musicgen-small", duration_s=5.0)
path = aimu.generate_audio("ambient ocean waves", model="hf:facebook/musicgen-small", format="path")
```

```python
# Speech synthesis (text-to-speech)
path = aimu.generate_speech("Hello, world!", model="openai:tts-1")
sr, audio = aimu.generate_speech("Hello!", model="hf:facebook/mms-tts-eng", format="numpy")
tts = aimu.speech_client("openai:tts-1-hd")  # reuse a client across calls
path = tts.generate("Good morning.", voice="nova", format="path")
```

```python
# Transcription (speech-to-text)
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

### Composable agents

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

`aimu.aio` mirrors the entire public surface with the same class names, so one import switches paradigms. `aio.Parallel` and `concurrent_tool_calls=True` use `asyncio.TaskGroup` for true coroutine concurrency.

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

## Resources

### Documentation

- 📘 [Tutorials](https://saxman.github.io/aimu/tutorials/): Hand-held walkthroughs. Install to first agent in 15 mins
- 🛠️ [How-to guides](https://saxman.github.io/aimu/how-to/): Task-oriented recipes (switch providers, write a tool, stream output, benchmark models, ...)
- 📚 [Reference](https://saxman.github.io/aimu/reference/): Auto-generated API docs, capability matrices, environment variables, CLI
- 💡 [Explanation](https://saxman.github.io/aimu/explanation/): The *why*: architecture, design principles, agents vs workflows

### Notebooks

The [`notebooks/`](notebooks/) directory ships one runnable demo per subsystem, numbered `01`–`23` and ordered to build up incrementally, from the model client, tools, and agents through workflows, memory, RAG, the generative modalities, the async surface, and agent composition / A2A interop. The filenames are self-describing; open the directory to browse and run them.

### Examples

The [`examples/`](examples/) directory ships larger, real-world programs organized by theme (each has its own README):

- [text-refinement/](examples/text-refinement/): A generate → judge → refine loop over **text**, implemented four ways (code loop, `Agent`, `EvaluatorOptimizer`, simulated annealing). GPU-free, Ollama-only.
- [image-refinement/](examples/image-refinement/): The same loop over **images** (diffusion + vision evaluator), five variants including img2img. Needs `aimu[hf]` and a GPU.
- [news-summarizer/](examples/news-summarizer/): One task (*summarize recent AI news*) solved with `Agent`, `Chain`, `Parallel`, and `OrchestratorAgent`, selected via `--method`.
- [skills/](examples/skills/): Demo `SKILL.md` skills (`haiku-poet`, `unit-converter`) for `SkillAgent` discovery, exposed as `aimu.paths.skills`.

### Web apps

The [`web/`](web/) directory ships chat applications that demonstrate AIMU in action:

- [streamlit_chatbot_basic.py](web/streamlit_chatbot_basic.py): ~70-line showcase. Provider/model selector, streaming chat, built-in tools. Start here.
- [streamlit_chatbot.py](web/streamlit_chatbot.py): Full-featured. Image/audio/speech generation, agentic mode, thinking display, generation sliders, live TTS narration. Extensible foundation.
- [gradio_chatbot_basic.py](web/gradio_chatbot_basic.py): Basic Gradio chat interface with streaming.

```bash
streamlit run web/streamlit_chatbot.py         # full-featured Streamlit demo (agents, tools, images, audio, speech narration, etc.)
streamlit run web/streamlit_chatbot_basic.py   # basic Streamlit demo app
python web/gradio_chatbot_basic.py             # basic Gradio demo app
```

## Contributing

See the [contributing guide](https://saxman.github.io/aimu/contributing/) for dev setup, testing, lint, and PR conventions.

## License

Apache 2.0. See [LICENSE](LICENSE).
