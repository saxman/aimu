# Embed text

Turn text into fixed-length vectors with `embedding_client().embed()` or the one-shot
`aimu.embed()`. Embeddings power semantic similarity, clustering, retrieval, and
[semantic memory](use-semantic-memory.md).

## Basic usage

```python
import aimu

# One-shot: a single string returns one vector (list[float])
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

- **OpenAI** (cloud): reads `OPENAI_API_KEY`.
- **Ollama** (local server): pull the model first, e.g. `ollama pull nomic-embed-text`.
- **HuggingFace** (local): backed by `sentence-transformers` (the `[hf]` extra), so each
  model's own pooling/normalization config is honoured. Weights download on first use and
  are cached; free them with `aimu.clear_hf_cache()`.

## Queries and documents

Retrieval-tuned models embed a search differently from the text it searches: E5 wants
`"query: "` and `"passage: "` in front of each, EmbeddingGemma 2 and nomic have their own
pair, and BGE and mxbai prefix only the query. Say which side you are embedding and AIMU
prepends the prefix the model's card specifies:

```python
client = aimu.embedding_client("hf:intfloat/e5-large-v2")
doc_vectors = client.embed(["Paris is the capital of France."], input_type="document")
query_vector = client.embed("capital of france", input_type="query")
```

The prefixes are declared on the spec (`client.spec.query_prompt`,
`client.spec.document_prompt`), so you can read exactly what is sent. For a symmetric model
(OpenAI, MiniLM, GTE, BGE-M3) `input_type` changes nothing. `input_type=None`, the default,
sends the text as given. Embed a corpus and its queries the same way: a prompted query
compared against unprompted documents is a mismatch, not an improvement.

## Shorter vectors

Some models are trained so the front of each vector is a usable embedding on its own
(Matryoshka Representation Learning). Ask for a narrower width when you build the client and
every vector it returns has that width, stored at a fraction of the size:

```python
client = aimu.embedding_client("hf:google/embeddinggemma-2", dimensions=256)
client.dimensions                 # 256
len(client.embed("hello"))        # 256
```

The width is set once per client, not per call, so a corpus and its queries cannot end up
with different widths. Only widths the model was trained for are accepted; anything else
raises `ValueError` at construction, naming the ones that are. `client.spec.matryoshka_dimensions`
lists them: `(768, 512, 256, 128)` for EmbeddingGemma 2, and any width up to native for OpenAI's
text-embedding-3 models. A model whose spec declares none cannot be truncated through AIMU.

## Available models

| Provider | Enum member | Model ID | Dims |
|---|---|---|---|
| OpenAI | `OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL` | `text-embedding-3-small` | 1536 |
| OpenAI | `OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE` | `text-embedding-3-large` | 3072 |
| Ollama | `OllamaEmbeddingModel.NOMIC_EMBED_TEXT` | `nomic-embed-text` | 768 |
| Ollama | `OllamaEmbeddingModel.MXBAI_EMBED_LARGE` | `mxbai-embed-large` | 1024 |
| HuggingFace | `HuggingFaceEmbeddingModel.BGE_SMALL_EN_V1_5` | `BAAI/bge-small-en-v1.5` | 384 |
| HuggingFace | `HuggingFaceEmbeddingModel.BGE_LARGE_EN_V1_5` | `BAAI/bge-large-en-v1.5` | 1024 |
| HuggingFace | `HuggingFaceEmbeddingModel.EMBEDDING_GEMMA_2` | `google/embeddinggemma-2` | 768 |

`aimu.embedding_client(...).MODELS` (or each provider enum) lists the full catalog.

## Default model via env var

```bash
export AIMU_EMBEDDING_MODEL="openai:text-embedding-3-small"
```

Then `aimu.embedding_client()` and `aimu.embed()` resolve the model without an explicit
argument. No model is ever downloaded implicitly; if the var is unset and no model is
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

The store embeds stored facts with `input_type="document"` and searches with
`input_type="query"`, so a retrieval-tuned model gets its prefixes without extra wiring. A
persistent collection built before AIMU did this, with a model that declares prefixes, holds
unprefixed vectors: rebuild it.

EmbeddingGemma 2 loads only its 270M-parameter text encoder by default (the checkpoint also
carries vision and audio encoders that `embed()` cannot use), and refuses float16, which it
answers with NaN or degraded vectors rather than an error.

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

- [Use semantic memory](use-semantic-memory.md): store and retrieve facts by meaning
- Notebook [11 - Embeddings](https://github.com/saxman/aimu/blob/main/notebooks/11-embeddings.qmd)
