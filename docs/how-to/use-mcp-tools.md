# Use MCP tools

For tools that live in a separate process (or to share a tool catalogue across many agents), wrap a [FastMCP 2.0](https://gofastmcp.com) server with `MCPClient`, then call `.as_tools()` to add its tools to your model client's `tools` list.

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
client.tools = mcp_client.as_tools()   # MCP tools become @tool-style callables
client.chat("Use the mytools to do something.")
```

`as_tools()` does one `list_tools()` round-trip and returns a callable per server tool, each carrying its `__tool_spec__`. They dispatch through the same path as in-process `@tool` functions, so MCP and Python tools are interchangeable from the client's point of view. Keep the `MCPClient` reference (or the callables, which hold one) alive for the connection's lifetime; call `as_tools()` again to refresh after the server's tool set changes.

`MCPClient` requires *exactly one* of:

- `config={...}`: FastMCP server config dict (the form above)
- `server=...`: an in-process `FastMCP` instance
- `file="path/to/server.py"`: a local server script
- `url="https://..."`: a remote HTTP/SSE server (see below)

## Connect to a remote server (URL)

Point `MCPClient` at a hosted MCP service with `url=`. The transport (streamable-HTTP vs SSE) is inferred from the URL; pass `auth=` (a bearer-token string or the literal `"oauth"`) and `headers=` for authenticated services:

```python
from aimu.tools import MCPClient

mcp_client = MCPClient(url="https://mcp.example.com/sse", auth="my-bearer-token")
client.tools = mcp_client.as_tools()
```

The async surface mirrors it:

```python
from aimu import aio

mcp = await aio.MCPClient.connect(url="https://mcp.example.com/mcp", auth="my-bearer-token")
agent = aio.Agent(aio.client(...), tools=await mcp.as_tools())
```

`auth=` and `headers=` apply only with `url=` (passing them with another source raises `MCPConnectionError`). Connect to a server you trust: its tools run with whatever access the service grants.

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

Both routes produce callables for the one `tools` list; concatenate them. On a name collision the **last** entry wins, so append a local override after the MCP tools to shadow one:

```python
client.tools = mcp_client.as_tools() + [my_local_tool]   # my_local_tool shadows a same-named MCP tool
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

- [Add a custom tool](add-custom-tool.md): the in-process `@tool` route
- [Explanation: tool integration](../explanation/tool-integration.md): when to pick which route
- [`aimu.tools.MCPClient`](../reference/api/tools.md#aimu.tools.MCPClient): API reference
