# Embed text

Turn text into fixed-length vectors with `embedding_client().embed()` or the one-shot
`aimu.embed()`. Embeddings power semantic similarity, clustering, retrieval, and
[semantic memory](use-semantic-memory.md).

## Basic usage

```python
import aimu

# One-shot — a single string returns one vector (list[float])
vector = aimu.embed("The quick brown fox", model="openai:text-embedding-3-small")

# Reusable client (better for many inputs)
client = aimu.embedding_client("openai:text-embedding-3-small")
vector = client.embed("hello")               # -> list[float]
vectors = client.embed(["hello", "world"])   # -> list[list[float]]
```

A single `str` returns one vector; a list returns a list of vectors in the same order.
An empty list returns `[]` without calling the provider. `client.dimensions` reports the
vector width declared by the model spec.

## Semantic similarity

Related text lands close together in vector space; cosine similarity measures it:

```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

cat, kitten, taxes = client.embed(["a small cat", "a tiny kitten", "quarterly tax filing"])
cosine(cat, kitten)  # high
cosine(cat, taxes)   # low
```

## Providers

The `embed()` surface is identical across providers; pick one with a
`"provider:model_id"` string (or a provider `EmbeddingModel` enum member).

```python
aimu.embedding_client("openai:text-embedding-3-small")  # cloud, reads OPENAI_API_KEY
aimu.embedding_client("ollama:nomic-embed-text")        # local server (ollama pull first)
aimu.embedding_client("hf:BAAI/bge-small-en-v1.5")      # local sentence-transformers ([hf] extra)
```

- **OpenAI** (cloud) — reads `OPENAI_API_KEY`.
- **Ollama** (local server) — pull the model first, e.g. `ollama pull nomic-embed-text`.
- **HuggingFace** (local) — backed by `sentence-transformers` (the `[hf]` extra), so each
  model's own pooling/normalization config is honoured. Weights download on first use and
  are cached; free them with `aimu.clear_hf_cache()`. Retrieval-tuned models (BGE / E5)
  expect `"query: "` / `"passage: "` prefixes for asymmetric retrieval — pass already-prefixed
  strings when you need that; symmetric similarity does not.

## Available models

| Provider | Enum member | Model ID | Dims |
|---|---|---|---|
| OpenAI | `OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL` | `text-embedding-3-small` | 1536 |
| OpenAI | `OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE` | `text-embedding-3-large` | 3072 |
| Ollama | `OllamaEmbeddingModel.NOMIC_EMBED_TEXT` | `nomic-embed-text` | 768 |
| Ollama | `OllamaEmbeddingModel.MXBAI_EMBED_LARGE` | `mxbai-embed-large` | 1024 |
| HuggingFace | `HuggingFaceEmbeddingModel.BGE_SMALL_EN_V1_5` | `BAAI/bge-small-en-v1.5` | 384 |
| HuggingFace | `HuggingFaceEmbeddingModel.BGE_LARGE_EN_V1_5` | `BAAI/bge-large-en-v1.5` | 1024 |

`aimu.embedding_client(...).MODELS` (or each provider enum) lists the full catalog.

## Default model via env var

```bash
export AIMU_EMBEDDING_MODEL="openai:text-embedding-3-small"
```

Then `aimu.embedding_client()` and `aimu.embed()` resolve the model without an explicit
argument. No model is ever downloaded implicitly — if the var is unset and no model is
passed, a `ValueError` is raised.

## Pluggable embeddings in semantic memory

`SemanticMemoryStore` uses ChromaDB's built-in embedding model by default. Pass
`embedding_client=` to choose your own model instead:

```python
from aimu.memory import SemanticMemoryStore

store = SemanticMemoryStore(embedding_client=aimu.embedding_client("openai:text-embedding-3-small"))
store.store("Paul works at Google")
store.search("employment")
```

A custom embedding model is not persisted in the collection config, so reopen a persistent
store with the same `embedding_client=`.

## Async surface

Embedding clients have no chat lifecycle, so the async surface wraps a sync client and
routes `embed()` through `asyncio.to_thread` (the same pattern as the other modalities).
Construct a sync client first, then wrap it.

```python
from aimu import aio
import asyncio

async def main():
    sync_client = aimu.embedding_client("openai:text-embedding-3-small")
    async_client = aio.embedding_client(sync_client)
    vectors = await async_client.embed(["alpha", "beta"])
    print(len(vectors), "x", len(vectors[0]))

asyncio.run(main())
```

## See also

- [Use semantic memory](use-semantic-memory.md) — store and retrieve facts by meaning
- Notebook [20 - Embeddings](https://github.com/saxman/aimu/blob/main/notebooks/20%20-%20Embeddings.ipynb)
