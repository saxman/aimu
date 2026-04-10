from typing import Optional

from anyio.from_thread import start_blocking_portal
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport


class _ToolResponse:
    """Wraps FastMCP 2.0 call_tool list result to preserve the .content API."""

    def __init__(self, content: list):
        self.content = content


class MCPClient:
    """Synchronous wrapper around an async FastMCP Client.

    Uses anyio's start_blocking_portal() to run the FastMCP Client in a
    background thread with a properly initialized anyio event loop.
    """

    def __init__(self, config: Optional[dict] = None, server: Optional[FastMCP] = None, file: Optional[str] = None):
        if config is not None:
            servers = config.get("mcpServers", {})
            first = next(iter(servers.values()))
            transport = StdioTransport(command=first["command"], args=first.get("args", []))
            self._transport = transport
        else:
            self._transport = server or file

        self._portal_cm = start_blocking_portal(backend="asyncio")
        self._portal = self._portal_cm.__enter__()
        self._portal.call(self._connect)

    async def _connect(self):
        self._client = Client(self._transport)
        await self._client.__aenter__()

    def __del__(self):
        try:
            self._portal.call(self._client.__aexit__, None, None, None)
        except Exception:
            pass
        try:
            self._portal_cm.__exit__(None, None, None)
        except Exception:
            pass

    def __deepcopy__(self, memo):
        # MCPClient holds a live async connection context; it cannot be duplicated.
        memo[id(self)] = self
        return self

    @property
    def client(self):
        return self._client

    def call_tool(self, tool_name: str, params: dict):
        result = self._portal.call(self._client.call_tool, tool_name, params)
        content = result.content if hasattr(result, "content") else result
        return _ToolResponse(content)

    def list_tools(self):
        return self._portal.call(self._client.list_tools)

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema},
            }
            for t in self.list_tools()
        ]
