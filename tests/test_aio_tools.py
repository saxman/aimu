"""Async MCP integration tests.

Covers:
- ``mcp_tools_to_openai`` format conversion (shared helper)
- ``aio.MCPClient`` lifecycle (connect, list/call, close) against a FastMCP server
"""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from aimu.aio import MCPClient
from aimu.tools.mcp_format import mcp_tools_to_openai


def test_mcp_tools_to_openai_converts_spec():
    """Hand-built FastMCP tool spec converts to OpenAI function-calling format."""

    class FakeTool:
        def __init__(self, name, description, schema):
            self.name = name
            self.description = description
            self.inputSchema = schema

    fake = FakeTool(
        "add",
        "Adds two integers.",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
    )
    result = mcp_tools_to_openai([fake])
    assert result == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Adds two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]


def _build_test_server() -> FastMCP:
    server = FastMCP("test")

    @server.tool
    def add(a: int, b: int) -> int:
        """Adds two integers."""
        return a + b

    @server.tool
    def greet(name: str) -> str:
        """Greets the user."""
        return f"Hello, {name}!"

    return server


async def test_async_mcp_client_lifecycle():
    """Connect, list tools, call a tool, then close."""
    mcp = await MCPClient.connect(server=_build_test_server())
    try:
        tools = await mcp.get_tools()
        names = {t["function"]["name"] for t in tools}
        assert {"add", "greet"} <= names

        result = await mcp.call_tool("greet", {"name": "world"})
        # FastMCP returns a result with .content containing text blocks
        text = "".join(p.text for p in result.content if p.type == "text")
        assert "Hello, world!" in text
    finally:
        await mcp.aclose()


async def test_async_mcp_client_as_tools():
    """as_tools() returns async @tool-style callables that await cross-process dispatch."""
    mcp = await MCPClient.connect(server=_build_test_server())
    try:
        tools = await mcp.as_tools()
        by_name = {fn.__name__: fn for fn in tools}
        assert {"add", "greet"} <= set(by_name)
        greet = by_name["greet"]
        assert greet.__tool_is_async__ is True
        assert greet.__tool_is_streaming__ is False
        assert greet.__tool_spec__["function"]["name"] == "greet"
        assert "Hello, world!" in await greet(name="world")
    finally:
        await mcp.aclose()


async def test_async_mcp_client_rejects_two_sources():
    from aimu.tools.client import MCPConnectionError

    server = _build_test_server()
    with pytest.raises(MCPConnectionError, match="exactly one"):
        MCPClient(server=server, file="x")


async def test_async_mcp_client_call_before_connect_raises():
    from aimu.tools.client import MCPConnectionError

    mcp = MCPClient(server=_build_test_server())
    with pytest.raises(MCPConnectionError, match="not connected"):
        await mcp.call_tool("add", {"a": 1, "b": 2})
