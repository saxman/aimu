# Environment variables

AIMU reads a small set of environment variables. All are loaded via [`python-dotenv`](https://github.com/theskumar/python-dotenv): a `.env` file in your working directory is picked up automatically by the clients that need keys.

## API keys

| Variable | Used by | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | `AnthropicClient` | Yes (for `anthropic:*` models) |
| `OPENAI_API_KEY` | `OpenAIClient` | Yes (for `openai:*` models) |
| `GOOGLE_API_KEY` | `GeminiClient` (text) and `GeminiImageClient` (Nano Banana) | Yes (for `gemini:*` models) |

If missing, the text clients construct successfully but the first request raises an authentication error from the underlying SDK. `GeminiImageClient` raises `RuntimeError` at construction time with an actionable message instead.

## Tool endpoints

| Variable | Used by | Default |
|---|---|---|
| `SEARXNG_BASE_URL` | `aimu.tools.builtin.web_search` | `http://localhost:8080` |

## Default model selection

These set the default `model=` when the argument is omitted, for both the top-level helpers and the built-in tools' lazy singletons. Each takes a `"provider:model_id"` string and is read through `aimu.models._internal.model_defaults`.

**Text is the only modality with a fallback.**

| Variable | Used by | When unset |
|---|---|---|
| `AIMU_LANGUAGE_MODEL` | `aimu.chat()` / `aimu.client()` / `aimu.agent()`, their `aimu.aio` equivalents, and the `python -m` entry points that take `--model` | Probes for an already-available local model; raises if none |
| `AIMU_IMAGE_MODEL` | `aimu.image_client()` / `aimu.generate_image()`, `builtin.generate_image` | Raises `ValueError` |
| `AIMU_AUDIO_MODEL` | `aimu.audio_client()` / `aimu.generate_audio()`, `builtin.generate_audio` | Raises `ValueError` |
| `AIMU_SPEECH_MODEL` | `aimu.speech_client()` / `aimu.generate_speech()`, `builtin.generate_speech` | Raises `ValueError` |
| `AIMU_TRANSCRIPTION_MODEL` | `aimu.transcription_client()` / `aimu.transcribe()` | Raises `ValueError` |
| `AIMU_EMBEDDING_MODEL` | `aimu.embedding_client()` / `aimu.embed()` | Raises `ValueError` |

### Text: `AIMU_LANGUAGE_MODEL`

Resolution order is env var → local discovery → error:

1. `AIMU_LANGUAGE_MODEL`, if set. It is validated immediately with `resolve_model()`, so a typo'd id raises at resolution time rather than at the first request.
2. Otherwise the first *already-available* local model: a running Ollama server → the local HuggingFace cache → a reachable local OpenAI-compatible server, preferring a tool-capable one. The pick is logged at `WARNING`, so an implicit default is never silent.
3. Otherwise `ValueError`, naming the installed text providers.

A cloud provider is never auto-selected and weights are never downloaded, so the fallback can only pick something already on the machine. Set the variable to make the choice deterministic:

```ini
AIMU_LANGUAGE_MODEL=ollama:qwen3.5:9b
```

The value is the **full model string**, so the `@endpoint` and ad-hoc `;flags` forms work here exactly as they do in an explicit `model=` argument:

```ini
AIMU_LANGUAGE_MODEL=ollama:qwen3.5:9b@http://gpu-box:11434
AIMU_LANGUAGE_MODEL=lmstudio:some-local-build@http://box:1234;tools,vision
```

Note this pins only the *client*. Local discovery stays endpoint-blind (see [switch-providers.md](../how-to/switch-providers.md)), so export `OLLAMA_HOST` as well when a remote server should also be considered by `available_text_models()`.

`resolve_default_text_model_enum()` is the exception, and it says so: it returns a `Model` enum member, which can carry neither an endpoint nor ad-hoc flags, so it rejects both by name rather than reporting a valid id as unknown. Use `resolve_default_text_model()` for those: it is the same resolution, returned as the string, so what comes back still carries the endpoint the env var set. That is the one to reach for when a host has to build a *second* client on the same default the first one got, since a resolved enum no longer knows which endpoint it came from.

The async surface skips the HuggingFace-cache probe (an `hf:` default would mean wrapping a sync in-process client), so it resolves from Ollama and local OpenAI-compatible servers only.

### The other modalities

Image, audio, speech, transcription, and embedding have **no fallback**: unset and no `model=` means the helper raises. That is deliberate, not an omission -- a wrong silent pick is expensive here (an image-style swing, or a persisted vector store corrupted by a mismatched embedder), and AIMU never downloads weights implicitly. The error lists any locally available models for that modality, so making the choice explicit is easy.

Accepted values follow each modality's client:

- `AIMU_IMAGE_MODEL`: `"hf:<repo>"` or `"gemini:<id_or_alias>"` (e.g. `gemini:nano-banana`).
- `AIMU_AUDIO_MODEL`: `"hf:<repo>"`.
- `AIMU_SPEECH_MODEL`: `"openai:<model_id>"` (needs `OPENAI_API_KEY`) or `"hf:<repo_id>"`.
- `AIMU_TRANSCRIPTION_MODEL`: `"openai:<model_id>"` or `"hf:<repo_id>"`.
- `AIMU_EMBEDDING_MODEL`: `"openai:<model_id>"` (needs `OPENAI_API_KEY`), `"ollama:<model_id>"`, or `"hf:<repo_id>"` (the `[hf]` extra).

The built-in tools construct their client lazily on first call from these same variables. To bind a specific client to one agent instead of using the process-wide singleton, build the tool yourself with `make_image_tool(client)` / `make_audio_tool(client)` / `make_speech_tool(client)`.

## MCP server storage paths

| Variable | Used by | Default |
|---|---|---|
| `MEMORY_STORE_PATH` | `python -m aimu.memory.mcp` (SemanticMemoryStore server) | None (in-memory) |
| `DOCUMENT_STORE_PATH` | `python -m aimu.memory.document_mcp` (DocumentStore server) | None (in-memory) |
| `PROMPT_CATALOG_PATH` | `python -m aimu.prompts.mcp` (PromptCatalog server) | `prompts.db` in cwd |

When unset, MCP servers run with ephemeral state: fine for tests, not for production.

## `.env` file example

```ini
# Cloud API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Default text model. Pins what aimu.chat() / aimu.client() / aimu.agent() use when
# model= is omitted; without it AIMU picks whatever local model is already available.
AIMU_LANGUAGE_MODEL=ollama:qwen3.5:9b

# Local search (SearXNG)
SEARXNG_BASE_URL=http://localhost:8080

# The other modality defaults. No fallback: unset means the helper (and the matching
# built-in tool) raises rather than guessing.
AIMU_IMAGE_MODEL=gemini:nano-banana
AIMU_AUDIO_MODEL=hf:facebook/musicgen-small
AIMU_SPEECH_MODEL=hf:microsoft/speecht5_tts
AIMU_TRANSCRIPTION_MODEL=openai:whisper-1
AIMU_EMBEDDING_MODEL=openai:text-embedding-3-small

# MCP server storage
MEMORY_STORE_PATH=./.aimu/memory
DOCUMENT_STORE_PATH=./.aimu/docs
PROMPT_CATALOG_PATH=./.aimu/prompts.db
```

Place it in your project root. Don't commit it; add `.env` to `.gitignore`.

## See also

- [Provider matrix](provider-matrix.md): which provider needs which key
- [CLI](cli.md): runnable `python -m` entry points that read these vars
