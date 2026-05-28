# CLI

AIMU's CLI surface is a handful of `python -m` entry points that run the bundled FastMCP servers.

## Tools server

```bash
python -m aimu.tools.mcp
```

Runs a FastMCP server that registers every function in `aimu.tools.builtin.ALL_TOOLS` (weather, search, calculate, get_webpage, wikipedia, list_directory, read_file, echo, get_current_date_and_time).

Connect from another process:

```python
from aimu.tools import MCPClient
mcp = MCPClient({"mcpServers": {"aimu": {"command": "python", "args": ["-m", "aimu.tools.mcp"]}}})
```

## Semantic memory server

```bash
MEMORY_STORE_PATH=./.aimu/memory python -m aimu.memory.mcp
```

Wraps `SemanticMemoryStore`. Tools registered:

- `search_memories(search_request)`
- `add_memories(memories)`
- `delete_memory(memory)`
- `list_memories()`

Reads `MEMORY_STORE_PATH` (default: in-memory).

## Document memory server

```bash
DOCUMENT_STORE_PATH=./.aimu/docs python -m aimu.memory.document_mcp
```

Wraps `DocumentStore` with tools matching Anthropic's Managed Agents Memory API:

- `memory_list(path_prefix)`
- `memory_search(query)`
- `memory_read(path)`
- `memory_write(path, content)`
- `memory_edit(path, old_str, new_str)`
- `memory_delete(path)`

Reads `DOCUMENT_STORE_PATH` (default: in-memory).

## Prompt catalog server

```bash
PROMPT_CATALOG_PATH=./.aimu/prompts.db python -m aimu.prompts.mcp
```

Wraps `PromptCatalog`. Tools registered:

- `get_prompt(name, model_id)`
- `list_prompts()`
- `store_prompt_version(name, model_id, prompt, metrics)`

Reads `PROMPT_CATALOG_PATH` (default: `prompts.db` in cwd).

## Web chat apps

Not strictly CLI but launched the same way:

```bash
streamlit run web/streamlit_chatbot.py     # Streamlit UI
python web/gradio_chatbot_basic.py               # Gradio UI
```

Both demo a full-featured chat with streaming, tool calls, and persistent history.

## See also

- [Environment variables](env-vars.md) — every var these commands read
- [How-to: use MCP tools](../how-to/use-mcp-tools.md) — attaching MCP servers to a client
