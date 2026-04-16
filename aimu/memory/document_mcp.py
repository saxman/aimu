"""
aimu.memory.document_mcp — MCP server exposing DocumentStore as memory tools.

Tool names and semantics match Anthropic's Managed Agents Memory API so that
this server can be used as a drop-in replacement for Anthropic's hosted
memory stores in local or self-hosted deployments.

The storage path is configured with the DOCUMENT_STORE_PATH environment
variable (defaults to an ephemeral in-memory store).

Run as a standalone MCP server::

    python -m aimu.memory.document_mcp

Or connect programmatically::

    from aimu.memory.document_mcp import mcp
    client = MCPClient(server=mcp)
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from aimu.memory.document_store import DocumentStore

_DEFAULT_PERSIST_PATH = os.environ.get("DOCUMENT_STORE_PATH")  # None → ephemeral

mcp = FastMCP("AIMU Document Memory")
_store = DocumentStore(persist_path=_DEFAULT_PERSIST_PATH)


@mcp.tool()
def memory_list(path_prefix: str = "") -> list[dict]:
    """
    List memories in the store, optionally filtered by path prefix.

    Args:
        path_prefix: Only return memories whose path starts with this string.
                     Pass an empty string (default) to list all memories.

    Returns:
        List of memory objects with ``path`` and ``size`` fields.
    """
    prefix = path_prefix if path_prefix else None
    paths = _store.list_paths(prefix=prefix)
    return [{"path": p, "size": len(_store.read(p))} for p in paths]


@mcp.tool()
def memory_search(query: str) -> list[dict]:
    """
    Search memory contents for a query string (case-insensitive).

    Args:
        query: Search string to match against memory paths and content.

    Returns:
        List of memory objects with ``path`` and ``content`` fields.
    """
    return _store.search_full_text(query)


@mcp.tool()
def memory_read(path: str) -> str:
    """
    Read the content of a memory at the given path.

    Args:
        path: Memory path, e.g. ``"/preferences.md"``.

    Returns:
        The full text content of the memory.

    Raises:
        KeyError: If no memory exists at *path*.
    """
    return _store.read(path)


@mcp.tool()
def memory_write(path: str, content: str) -> dict:
    """
    Create or overwrite a memory at the given path.

    Args:
        path:    Memory path, e.g. ``"/project-context.md"``.
        content: Text content to store (≤ 100 KB recommended).

    Returns:
        Memory object with ``path`` and ``size`` fields.
    """
    _store.write(path, content)
    return {"path": path, "size": len(content)}


@mcp.tool()
def memory_edit(path: str, old_str: str, new_str: str) -> dict:
    """
    Edit an existing memory by replacing *old_str* with *new_str*.

    Args:
        path:    Memory path of the document to edit.
        old_str: Exact substring to find and replace.
        new_str: Replacement text.

    Returns:
        Updated memory object with ``path`` and ``size`` fields.

    Raises:
        KeyError:   If no memory exists at *path*.
        ValueError: If *old_str* is not found in the memory content.
    """
    _store.edit(path, old_str, new_str)
    updated = _store.read(path)
    return {"path": path, "size": len(updated)}


@mcp.tool()
def memory_delete(path: str) -> None:
    """
    Delete the memory at the given path.

    No-op if the path does not exist.

    Args:
        path: Memory path to remove.
    """
    _store.delete(path)


if __name__ == "__main__":
    mcp.run()
