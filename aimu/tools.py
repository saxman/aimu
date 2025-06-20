from fastmcp import FastMCP
from fastmcp import Client

import asyncio


mcp = FastMCP("AIMU Tools")

@mcp.tool()
def echo(echo_string: str) -> str:
    """Returns echo_string."""
    return echo_string

if __name__ == "__main__":
    mcp.run()


class MCPClient:
    """A MCP client that wraps the FastMCP client for easier tool calls and management."""

    def __init__(self, config: dict = None, client: FastMCP = None, file: str = None):
        self.client = None
        if not config and not file:
            self.client = client if client else mcp
        
        self.file = file
        self.config = config

        self.loop = asyncio.new_event_loop()

    async def _call_tool(self, tool_name: str, params: dict):
        if self.config:
            async with Client(self.config) as client:
                return await client.call_tool(tool_name, params)

        if self.file:
            async with Client(self.file) as client:
                return await client.call_tool(tool_name, params)

        async with Client(self.client) as client:
            return await client.call_tool(tool_name, params)

    def call_tool(self, tool_name: str, params: dict):
        return self.loop.run_until_complete(self._call_tool(tool_name, params))

    async def _list_tools(self):
        if self.config:
            async with Client(self.config) as client:
                return await client.list_tools()

        if self.file:
            async with Client(self.file) as client:
                return await client.list_tools()

        async with Client(self.client) as client:
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
