from fastmcp import Client

import asyncio


class MCPClient:
    """A MCP client that wraps the FastMCP client for easier tool calls and management."""

    def __init__(self, config: dict = None):
        self.config = config
        self.loop = asyncio.new_event_loop()

    async def _call_tool(self, tool_name: str, params: dict):
        async with Client(self.config) as client:
            return await client.call_tool(tool_name, params)

    def call_tool(self, tool_name: str, params: dict):
        return self.loop.run_until_complete(self._call_tool(tool_name, params))

    async def _list_tools(self):
        async with Client(self.config) as client:
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
