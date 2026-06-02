# Use semantic memory

`SemanticMemoryStore` stores natural-language facts and retrieves them by semantic similarity. Backed by [ChromaDB](https://www.trychroma.com/) — cosine-similarity vector search, in-process, persisted to a directory.

## Basic usage

```python
from aimu.memory import SemanticMemoryStore

store = SemanticMemoryStore(persist_path="./memory_store")

store.store("Paul works at Google")
store.store("Paul lives in San Francisco")
store.store("Paul has two cats")

store.search("employment")     # → ["Paul works at Google"]
store.search("where is Paul")  # → ["Paul lives in San Francisco"]
store.search("Paul's pets")    # → ["Paul has two cats"]
```

Pass `persist_path=None` (or omit) for an ephemeral, in-memory store useful for tests.

## Distance cutoff

`max_distance` filters out matches below the cosine-similarity threshold. Distance ranges from `0` (identical) to `2` (maximally dissimilar); useful range is roughly 0.3–0.6:

```python
store.search("employment", max_distance=0.4)   # only close matches
store.search("employment", n_results=10)       # top 10 regardless of distance
```

`max_distance=None` (the default) returns the top `n_results` unconditionally.

## Manage entries

```python
store.list_all()              # every fact as a list of strings
len(store)                    # count
store.delete("Paul has two cats")   # exact-string match; no-op if not present
```

## Use as in-process agent tools

`make_memory_tools(store)` wraps the store as three `@tool`-decorated functions — `store_memory`, `search_memories`, and `list_memories` — that an agent can call directly without a separate MCP server process:

```python
from aimu.memory import SemanticMemoryStore
from aimu.tools.builtin import make_memory_tools
from aimu.agents import Agent
import aimu

store = SemanticMemoryStore(persist_path="./memory_store")
agent = Agent(
    aimu.client("anthropic:claude-sonnet-4-6"),
    system_message="Use store_memory to save facts the user shares, and search_memories before answering questions about them.",
    tools=make_memory_tools(store),
)

agent.run("Remember that my preferred language is Python.")
agent.run("What do you know about my programming preferences?")
```

Pass the result to `make_tools(..., memory_store=store)` to combine memory with the full built-in tool set:

```python
from aimu.tools.builtin import make_tools

tools = make_tools(base_client, memory_store=store)
agent = Agent(base_client, tools=tools)
```

The store is always explicit — there is no env-var singleton — because `persist_path` and collection name are meaningful choices. For cross-process or multi-agent memory, use the MCP server instead.

## Expose as MCP tools

Run the bundled memory MCP server:

```bash
MEMORY_STORE_PATH=./memory_store python -m aimu.memory.mcp
```

It registers `search_memories`, `add_memories`, `delete_memory`, and `list_memories` tools. Attach it to any agent:

```python
from aimu.tools import MCPClient
mem = MCPClient({"mcpServers": {"memory": {"command": "python", "args": ["-m", "aimu.memory.mcp"]}}})
client.mcp_client = mem
```

## See also

- [Use document memory](use-document-memory.md) — path-based store mirroring Anthropic's Memory API
- [`aimu.memory.SemanticMemoryStore`](../reference/api/memory.md)
- Notebook [07 - Memory](https://github.com/saxman/aimu/blob/main/notebooks/07%20-%20Memory.ipynb)
