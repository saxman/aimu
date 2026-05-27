# Environment variables

AIMU reads a small set of environment variables. All are loaded via [`python-dotenv`](https://github.com/theskumar/python-dotenv) — a `.env` file in your working directory is picked up automatically by the clients that need keys.

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
| `AIMU_IMAGE_MODEL` | `aimu.tools.builtin.generate_image` (lazy singleton) | `hf:stabilityai/stable-diffusion-xl-base-1.0` |

The built-in `generate_image` tool constructs its image client lazily on first call, picking the provider and model from `AIMU_IMAGE_MODEL`. Accepts any string supported by `aimu.image_client()` — `"hf:<repo>"` or `"gemini:<id_or_alias>"`. Override per-agent by building your own tool with `make_image_tool(client)` instead of using the singleton.

## MCP server storage paths

| Variable | Used by | Default |
|---|---|---|
| `MEMORY_STORE_PATH` | `python -m aimu.memory.mcp` (SemanticMemoryStore server) | None (in-memory) |
| `DOCUMENT_STORE_PATH` | `python -m aimu.memory.document_mcp` (DocumentStore server) | None (in-memory) |
| `PROMPT_CATALOG_PATH` | `python -m aimu.prompts.mcp` (PromptCatalog server) | `prompts.db` in cwd |

When unset, MCP servers run with ephemeral state — fine for tests, not for production.

## `.env` file example

```ini
# Cloud API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Local search (SearXNG)
SEARXNG_BASE_URL=http://localhost:8080

# Image generation default (used by the built-in generate_image tool)
AIMU_IMAGE_MODEL=gemini:nano-banana

# MCP server storage
MEMORY_STORE_PATH=./.aimu/memory
DOCUMENT_STORE_PATH=./.aimu/docs
PROMPT_CATALOG_PATH=./.aimu/prompts.db
```

Place it in your project root. Don't commit it — add `.env` to `.gitignore`.

## See also

- [Provider matrix](provider-matrix.md) — which provider needs which key
- [CLI](cli.md) — runnable `python -m` entry points that read these vars
