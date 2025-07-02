from fastmcp import FastMCP
from fastmcp import Client

import asyncio


class MCPClient:
    """A MCP client that wraps the FastMCP client for easier tool calls and management."""

    def __init__(self, config: dict = None, server: FastMCP = None, file: str = None):
        self.server = server
        self.file = file
        self.config = config

        self.loop = asyncio.new_event_loop()

        if self.config:
            self.client = Client(self.config)
        elif self.file:
            self.client = Client(self.file)
        else:
            self.client = Client(self.server)

    def __del__(self):
        if self.loop.is_running():
            self.loop.stop()

        self.loop.run_until_complete(self.client.close())

    async def _call_tool(self, tool_name: str, params: dict):
        async with self.client as client:
            return await client.call_tool(tool_name, params)

    def call_tool(self, tool_name: str, params: dict):
        return self.loop.run_until_complete(self._call_tool(tool_name, params))

    async def _list_tools(self):
        async with self.client as client:
            return await client.list_tools()

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
