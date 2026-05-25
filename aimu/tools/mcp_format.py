"""OpenAI-format conversion for FastMCP tool specs.

Shared by both ``aimu.tools.MCPClient.get_tools()`` (sync) and
``aimu.aio.MCPClient.get_tools()`` (async). One source of truth for the format
mapping.
"""

from __future__ import annotations


def mcp_tools_to_openai(tool_specs: list) -> list[dict]:
    """Convert FastMCP tool specs (from ``list_tools()``) to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema},
        }
        for t in tool_specs
    ]
