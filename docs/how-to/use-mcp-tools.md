# Use MCP tools

For tools that live in a separate process — or to share a tool catalogue across many agents — wrap a [FastMCP 2.0](https://gofastmcp.com) server with `MCPClient` and attach it to your model client.

## Connect to a tool server

```python
from aimu.tools import MCPClient
import aimu

mcp_client = MCPClient({
    "mcpServers": {
        "mytools": {"command": "python", "args": ["tools.py"]},
    },
})
mcp_client.ping()  # raises MCPConnectionError if dead

client = aimu.client("ollama:qwen3.5:9b")
client.mcp_client = mcp_client
client.chat("Use the mytools to do something.")
```

`MCPClient` requires *exactly one* of:

- `config={...}` — FastMCP server config dict (the form above)
- `server=...` — an in-process `FastMCP` instance
- `file="path/to/server.py"` — a local server script

## Call a tool directly

You can use `MCPClient` standalone, without an agent:

```python
result = mcp_client.call_tool("mytool", {"input": "hello world!"})
# result.content[0].text  → tool's text response
```

## Use the built-in MCP server

AIMU ships a FastMCP server that exposes the entire `builtin` toolkit:

```bash
python -m aimu.tools.mcp
```

Connect to it from another process the same way:

```python
mcp_client = MCPClient({
    "mcpServers": {
        "aimu-builtin": {"command": "python", "args": ["-m", "aimu.tools.mcp"]},
    },
})
```

## Combine with in-process tools

Both routes work on the same client. Python `@tool` functions take precedence over MCP tools with the same name:

```python
client.mcp_client = mcp_client       # cross-process
client.tools = [my_local_tool]       # in-process; wins on name collision
```

## Loud failures

Connection problems raise `MCPConnectionError` with the original cause chained:

```python
try:
    mcp_client = MCPClient({"mcpServers": {"bad": {"command": "nonexistent"}}})
except MCPConnectionError as exc:
    print(exc)
# MCPConnectionError: failed to connect MCP transport ...: ...
```

A subsequent `.ping()` or `.call_tool()` failure also raises `MCPConnectionError`, so you can re-establish the connection without silently broken state.

## See also

- [Add a custom tool](add-custom-tool.md) — the in-process `@tool` route
- [Explanation: tool integration](../explanation/tool-integration.md) — when to pick which route
- [`aimu.tools.MCPClient`](../reference/api/tools.md#aimu.tools.MCPClient) — API reference
