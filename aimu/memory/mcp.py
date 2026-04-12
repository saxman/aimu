"""
aimu.memory.mcp — MCP server exposing MemoryStore operations as tools.

The storage path is configured with the MEMORY_STORE_PATH environment
variable (defaults to ./memory_store).

Run as a standalone MCP server::

    python -m aimu.memory.mcp

Or connect programmatically::

    from aimu.memory.mcp import mcp
    client = MCPClient(server=mcp)
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from aimu.memory.store import MemoryStore

_DEFAULT_PERSIST_PATH = os.environ.get("MEMORY_STORE_PATH", "./memory_store")

mcp = FastMCP("AIMU Memory")
_store = MemoryStore(persist_path=_DEFAULT_PERSIST_PATH)


@mcp.tool()
def search_memories(search_request: str) -> str:
    """
    Search for memories about the user.

    Args:
        search_request: Information about the user that's relevant to the conversation.

    Returns:
        Information about the user that's relevant to the conversation.
    """
    facts = _store.retrieve_facts(search_request)
    return "\n".join(facts)


@mcp.tool()
def add_memories(memories: list[str]) -> None:
    """
    Add memories about the user.

    Args:
        memories: A list of memories, as strings, to save about the user.
    """
    for fact in memories:
        _store.store_fact(fact)


if __name__ == "__main__":
    mcp.run()
