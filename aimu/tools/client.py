from fastmcp import FastMCP
from fastmcp import Client
from anyio.from_thread import start_blocking_portal

from typing import Optional


class MCPClient:
    """Synchronous wrapper around an async FastMCP Client.

    Uses anyio's BlockingPortal to run async operations from synchronous contexts,
    making it safe to use from Jupyter notebooks and any other sync environment.
    """

    def __init__(self, config: Optional[dict] = None, server: Optional[FastMCP] = None, file: Optional[str] = None):
        self.mcp_config = server or config or file

        self._portal_cm = start_blocking_portal(backend="asyncio")
        self._portal = self._portal_cm.__enter__()

        async def connect():
            self._client = Client(self.mcp_config)
            await self._client.__aenter__()

        self._portal.call(connect)

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

    def call_tool(self, tool_name: str, params: dict):
        return self._portal.call(self._client.call_tool, tool_name, params)

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
