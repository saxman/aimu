import asyncio
import threading
from typing import Optional

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport


class _ToolResponse:
    """Wraps FastMCP 2.0 call_tool list result to preserve the .content API."""

    def __init__(self, content: list):
        self.content = content


class MCPClient:
    """Synchronous wrapper around an async FastMCP Client.

    Runs an asyncio event loop in a background thread and submits coroutines
    via run_coroutine_threadsafe, avoiding anyio cancel-scope nesting issues.
    """

    def __init__(self, config: Optional[dict] = None, server: Optional[FastMCP] = None, file: Optional[str] = None):
        if config is not None:
            servers = config.get("mcpServers", {})
            first = next(iter(servers.values()))
            transport = StdioTransport(command=first["command"], args=first.get("args", []))
            self._transport = transport
        else:
            self._transport = server or file

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result()

    async def _connect(self):
        self._client = Client(self._transport)
        await self._client.__aenter__()

    def __del__(self):
        try:
            future = asyncio.run_coroutine_threadsafe(self._client.__aexit__(None, None, None), self._loop)
            future.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        try:
            self._loop.close()
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
        future = asyncio.run_coroutine_threadsafe(self._client.call_tool(tool_name, params), self._loop)
        return _ToolResponse(future.result())

    def list_tools(self):
        future = asyncio.run_coroutine_threadsafe(self._client.list_tools(), self._loop)
        return future.result()

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema},
            }
            for t in self.list_tools()
        ]
