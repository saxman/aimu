"""
aimu.memory.document_mcp: MCP server exposing DocumentStore as memory tools.

The tools are path-addressed and named after what they do (``memory_read``,
``memory_write``, ...), which is the same *shape* as Anthropic's memory tool but not the
same interface: that tool's commands are ``view`` / ``create`` / ``str_replace`` /
``insert`` / ``delete`` / ``rename``, and the Managed Agents memory-store API is different
again (``list`` / ``retrieve`` / ``create`` / ``update`` / ``delete``, where ``retrieve``
takes a ``mem_...`` id rather than a path, and a session reaches the store as a mounted
filesystem through ordinary file tools). This server is therefore **not** wire-compatible
with either, and nothing written against Anthropic's API can be pointed at it unchanged.
It is a local memory backend for AIMU agents that borrows the good idea, and its parameters
follow AIMU's own conventions -- ``memory_read`` windows with ``max_lines`` / ``offset``,
exactly like ``read_file`` and ``read_document``.

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

from aimu._window import past_end, window, window_complaint
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
def memory_read(path: str, max_lines: int = 2000, offset: int = 1) -> str:
    """
    Read a memory at the given path, up to max_lines lines starting at line offset.

    If the result says it was truncated, call again with the offset it names before drawing
    any conclusion from it: a partial document reads exactly like a complete one. The store
    is a directory a user can drop files into, so a memory can be paper-sized; paging is how
    one gets read whole without spending the context window the rest of the task needs.

    Args:
        path: Memory path, e.g. ``"/preferences.md"``.
        max_lines: Maximum number of lines to return (default 2000).
        offset: 1-indexed line to start reading from (default 1, the start).

    Returns:
        Up to *max_lines* lines of the memory from *offset*. When more remains, a trailing
        marker names the total and the offset that continues the read.

    Raises:
        KeyError: If no memory exists at *path*.
    """
    complaint = window_complaint(offset=offset, limit=max_lines, limit_name="max_lines", unit="line")
    if complaint:
        return complaint
    lines = _store.read(path).splitlines()
    # An empty memory read from the top is a legitimate whole-memory read, not a bad offset.
    if offset > len(lines) and offset > 1:
        return past_end(offset=offset, total=len(lines), unit="line", describe=path)
    return window(lines, offset=offset, limit=max_lines, unit="line", tool="memory_read", join="\n")


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
