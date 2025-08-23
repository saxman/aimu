from fastmcp import FastMCP
from fastmcp import Client

import asyncio
from typing import Optional


class MCPClient:
    """A MCP client that wraps the FastMCP client for easier tool calls and management."""

    def __init__(self, config: Optional[dict] = None, server: Optional[FastMCP] = None, file: Optional[str] = None):
        self.loop = asyncio.new_event_loop()

        if config:
            self.mcp_config = config
        if file:
            self.mcp_config = file
        if server:
            self.mcp_config = server

        self.client = Client(self.mcp_config)
        self.loop.run_until_complete(self.client.__aenter__())

    def __del__(self):
        self.loop.run_until_complete(self.client.__aexit__(None, None, None))
        if self.loop.is_running():
            self.loop.stop()

    async def _call_tool(self, tool_name: str, params: dict):
        return await self.client.call_tool(tool_name, params)

    def call_tool(self, tool_name: str, params: dict):
        return self.loop.run_until_complete(self._call_tool(tool_name, params))

    async def _list_tools(self):
        return await self.client.list_tools()

    def list_tools(self):
        return self.loop.run_until_complete(self._list_tools())

    def get_tools(self) -> list[dict]:
        tools = []

        for tool in self.list_tools():
            tools.append(
                {
                    "type": "function",
                    "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
                }
            )

        return tools
