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


def mcp_content_to_text(tool_response) -> str:
    """Flatten a FastMCP ``call_tool`` result into a plain string.

    Accepts either a result object with a ``.content`` list or a bare list of content
    parts. Concatenates the text of every ``type == "text"`` part; non-text parts are
    skipped. Shared by the sync and async ``MCPClient.as_tools()`` wrappers so cross-process
    tool results look like any other ``@tool`` return value (a string) to the dispatch loop.
    """
    content = tool_response.content if hasattr(tool_response, "content") else tool_response
    parts = content if isinstance(content, list) else [content]
    return "".join(part.text for part in parts if getattr(part, "type", None) == "text")
