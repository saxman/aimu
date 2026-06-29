# Use document memory

`DocumentStore` is a path-based document store that mirrors [Anthropic's Managed Agents Memory API](https://docs.claude.com/en/api/memory), with drop-in compatible naming for `write`, `read`, `edit`, `list`, `search`, `delete`.

## Basic usage

```python
from aimu.memory import DocumentStore

store = DocumentStore(persist_path="./doc_store")

store.write("/preferences.md", "Always use concise responses.")
store.write("/notes/2026/q1.md", "Focus on inference latency.")

store.read("/preferences.md")
# 'Always use concise responses.'

store.list_paths(prefix="/notes/")
# ['/notes/2026/q1.md']
```

Pass `persist_path=None` for ephemeral, in-memory use.

## Edit existing documents

```python
store.edit("/preferences.md", old_str="concise", new_str="detailed")
# Replaces the first occurrence of "concise" with "detailed"
# Raises ValueError if the old_str isn't found
```

## Search

`search_full_text(query, n_results=10)` returns a list of `{"path", "content"}` dicts, case-insensitive substring match:

```python
store.search_full_text("inference")
# [{"path": "/notes/2026/q1.md", "content": "Focus on inference latency."}]
```

## MemoryStore interface

`DocumentStore` also implements the abstract `MemoryStore` interface used by `SemanticMemoryStore`:

- `store(content)` auto-assigns `/note-{uuid}.md` and writes
- `search(query, n_results)` wraps `search_full_text()`
- `delete(identifier)` treats the identifier as a path

This lets you swap the two stores in code that only needs the common interface.

## Use as in-process agent tools

`make_memory_tools(store)` wraps the store as three `@tool`-decorated functions (`store_memory`, `search_memories`, and `list_memories`) that an agent can call directly without a separate MCP server process:

```python
from aimu.memory import DocumentStore
from aimu.tools.builtin import make_memory_tools
from aimu.agents import Agent
import aimu

store = DocumentStore(persist_path="./doc_store")
agent = Agent(
    aimu.client("anthropic:claude-sonnet-4-6"),
    system_message="Use store_memory to save notes and search_memories to find them.",
    tools=make_memory_tools(store),
)

agent.run("Save a note: the API deadline is June 15.")
agent.run("When is the API deadline?")
```

`search_memories` on a `DocumentStore` does case-insensitive full-text search (the same as `search_full_text()`), while on a `SemanticMemoryStore` it does vector similarity search. The agent interface is identical for both.

Pass the result to `make_tools(..., memory_store=store)` to combine memory with the full built-in tool set.

### Path-based document tools

`make_memory_tools` exposes the generic `MemoryStore` interface (store / search / list). To give an agent the document store's richer path-based API, use `make_document_tools(store)`, which returns `save_document(path, content)`, `read_document(path)`, `list_documents()`, and `search_documents(query)`:

```python
from aimu.tools.builtin import make_document_tools

agent.tools += make_document_tools(store)
```

Because these tool names are distinct from `make_memory_tools`' (`store_memory` etc.), one agent can carry both: a `SemanticMemoryStore` for facts and a `DocumentStore` for documents. These are the in-process counterpart to the `aimu.memory.document_mcp` server below.

## Expose as MCP tools

```bash
DOCUMENT_STORE_PATH=./doc_store python -m aimu.memory.document_mcp
```

Registers `memory_list`, `memory_search`, `memory_read`, `memory_write`, `memory_edit`, `memory_delete` (matching Anthropic's API naming).

## When to pick this vs semantic memory

- Use **`DocumentStore`** when you want named locations (`/preferences.md`, `/projects/x/spec.md`) and substring search. Best for agent scratchpads, configuration, and structured notes.
- Use **`SemanticMemoryStore`** when you want fuzzy retrieval of small natural-language facts ("Paul lives in San Francisco" found by searching for "where is Paul"). Best for episodic memory.

## See also

- [`aimu.memory.DocumentStore`](../reference/api/memory.md)
- [Use semantic memory](use-semantic-memory.md)
