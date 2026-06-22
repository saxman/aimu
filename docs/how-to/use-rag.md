# Retrieval-augmented generation (RAG)

`aimu.rag` provides the primitives to ground a model's answers in your own documents:
chunk text, store it, retrieve the relevant pieces for a query, and fold them into the
prompt. They are **plain functions over the [`MemoryStore`](use-semantic-memory.md)
interface**. There is no retriever/splitter/loader class hierarchy to learn.

## The flow

```python
import aimu
from aimu.memory import SemanticMemoryStore
from aimu.rag import ingest, retrieve, format_context

# 1. Chunk documents into a store (any MemoryStore works)
store = SemanticMemoryStore(embedding_client=aimu.embedding_client("openai:text-embedding-3-small"))
ingest(store, my_documents, chunk_size=800, chunk_overlap=100)

# 2. Retrieve the chunks relevant to a question
question = "What is AIMU's design philosophy?"
chunks = retrieve(store, question, n_results=5)

# 3. Augment the prompt and answer
context = format_context(chunks)
answer = aimu.chat(f"Context:\n{context}\n\nQuestion: {question}")
```

`ingest` → `retrieve` → `format_context` → `chat` is just a workflow you compose; RAG is
not a separate object type.

## Splitting text

`split_text` chops a long string into overlapping chunks, breaking on the largest
separator that keeps each chunk under `chunk_size` (paragraphs → lines → sentences →
words → characters).

```python
from aimu.rag import split_text

chunks = split_text(long_text, chunk_size=1000, chunk_overlap=200)
```

- **`chunk_size` / `chunk_overlap`**: overlap repeats the tail of one chunk at the head
  of the next so context isn't lost at boundaries. `chunk_overlap` must be smaller than
  `chunk_size`.
- **`separators`**: override the hierarchy, e.g. `["\n## ", "\n", " ", ""]` for Markdown.
- **`length_function`**: defaults to `len` (characters). Pass a tokenizer's counter for
  **token-aware** chunking:

  ```python
  import tiktoken
  enc = tiktoken.get_encoding("cl100k_base")
  chunks = split_text(text, chunk_size=256, chunk_overlap=32,
                      length_function=lambda s: len(enc.encode(s)))
  ```

`ingest` forwards all of these, so `ingest(store, docs, chunk_size=..., length_function=...)`
chunks the same way.

## Retrieving

`retrieve` is a RAG-named pass-through to `store.search()`, returning the matching chunk
strings. Extra keyword arguments flow through to the store:

```python
chunks = retrieve(store, query, n_results=5)
chunks = retrieve(store, query, n_results=5, max_distance=0.5)  # SemanticMemoryStore cutoff
```

## Reranking (optional)

A vector search is fast but coarse. Fetch a wide candidate set, then rerank it with a
cross-encoder to keep the best few. Requires the `[hf]` extra (`sentence-transformers`).

```python
from aimu.rag import retrieve, rerank

candidates = retrieve(store, query, n_results=20)
top = rerank(query, candidates, top_n=5)
```

The cross-encoder is loaded once and cached across calls.

## As an agent tool

Let an agent fetch context on demand with `make_retrieval_tool`:

```python
from aimu.rag import ingest
from aimu.memory import SemanticMemoryStore
from aimu.tools.builtin import make_retrieval_tool
from aimu.agents import Agent

store = SemanticMemoryStore()
ingest(store, my_documents)

agent = Agent(client, tools=[make_retrieval_tool(store, n_results=5)])
agent.run("Using the knowledge base, explain X.")
# the agent can call retrieve_context(query="...") whenever it needs background
```

The tool returns numbered context (`[1] ...`, `[2] ...`) so the model can cite which
chunk it used, or `"No relevant context found."` when the store has nothing relevant.

## Choosing a store

Any `MemoryStore` works with `ingest` / `retrieve`:

- [`SemanticMemoryStore`](use-semantic-memory.md): ChromaDB vector search; the usual
  choice. Pair it with a chosen embedding model via `embedding_client=` (see
  [Embed text](use-embeddings.md)).
- `DocumentStore`: case-insensitive substring search; no embeddings, good for exact-term
  lookup or when you can't run an embedding model.

## Scope

`aimu.rag` ships the chunk/retrieve/rerank primitives only. Document **loaders** (PDF,
HTML, etc.) are intentionally not included. The built-in `read_file` and `get_webpage`
tools, or any library that returns text, feed `ingest` directly. Chunks are stored as
plain strings (the `MemoryStore` contract), so per-chunk metadata (source, page) is not
currently carried.

## See also

- [Embed text](use-embeddings.md): pick the embedding model behind your vector store
- [Use semantic memory](use-semantic-memory.md): the store interface RAG builds on
