"""
aimu.memory.document_store — Path-based document memory store.

Follows Anthropic's Managed Agents Memory API pattern: documents are
addressed by path (e.g. "/preferences.md"), stored as plain text, and
retrieved via full-text search or exact path lookup.

Persistence modes:
  - Ephemeral (persist_path=None): in-memory dict, lost on process exit.
  - Persistent (persist_path provided): each document is a file on disk
    under that directory; path segments map to subdirectories.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from aimu.memory.base import MemoryStore


class DocumentStore(MemoryStore):
    """
    Path-addressed document store backed by the filesystem or an in-memory dict.

    Implements the :class:`MemoryStore` abstract interface so it can be swapped
    with :class:`SemanticMemoryStore` in any application, and adds the richer
    path-based API that mirrors Anthropic's Managed Agents Memory tools.

    Args:
        persist_path: Root directory for persistent storage.  If *None* the
            store is ephemeral (in-memory only).

    Examples:
        >>> store = DocumentStore()
        >>> store.write("/preferences.md", "Use concise responses.")
        >>> store.read("/preferences.md")
        'Use concise responses.'
        >>> store.edit("/preferences.md", "concise", "detailed")
        >>> store.search_full_text("detailed")
        [{"path": "/preferences.md", "content": "Use detailed responses."}]
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = persist_path
        # Ephemeral backing store; ignored when persist_path is set.
        self._docs: dict[str, str] = {}
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _abs_path(self, path: str) -> str:
        """Resolve a memory path to an absolute filesystem path."""
        # Strip leading slash so os.path.join works correctly.
        return os.path.join(self._persist_path, path.lstrip("/"))

    def _load_from_disk(self) -> None:
        """Walk persist_path and populate the in-memory index."""
        for dirpath, _dirnames, filenames in os.walk(self._persist_path):
            for filename in filenames:
                abs_file = os.path.join(dirpath, filename)
                rel = os.path.relpath(abs_file, self._persist_path)
                # Normalise to forward-slash memory paths with leading slash.
                mem_path = "/" + rel.replace(os.sep, "/")
                with open(abs_file, encoding="utf-8") as f:
                    self._docs[mem_path] = f.read()

    def _write_to_disk(self, path: str, content: str) -> None:
        abs_file = self._abs_path(path)
        os.makedirs(os.path.dirname(abs_file), exist_ok=True)
        with open(abs_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _delete_from_disk(self, path: str) -> None:
        abs_file = self._abs_path(path)
        if os.path.exists(abs_file):
            os.remove(abs_file)

    # ------------------------------------------------------------------
    # Rich path-based API (mirrors Anthropic Managed Agents Memory tools)
    # ------------------------------------------------------------------

    def write(self, path: str, content: str) -> None:
        """
        Create or overwrite a document at *path*.

        Args:
            path:    Memory path, e.g. ``"/preferences.md"``.
            content: Text content to store (≤ 100 KB recommended).
        """
        self._docs[path] = content
        if self._persist_path:
            self._write_to_disk(path, content)

    def read(self, path: str) -> str:
        """
        Return the content of the document at *path*.

        Raises:
            KeyError: If no document exists at *path*.
        """
        if path not in self._docs:
            raise KeyError(path)
        return self._docs[path]

    def edit(self, path: str, old_str: str, new_str: str) -> None:
        """
        Replace the first occurrence of *old_str* with *new_str* in the
        document at *path*.

        Args:
            path:    Memory path of the document to edit.
            old_str: Exact substring to find.
            new_str: Replacement text.

        Raises:
            KeyError:   If no document exists at *path*.
            ValueError: If *old_str* is not found in the document.
        """
        content = self.read(path)
        if old_str not in content:
            raise ValueError(f"{old_str!r} not found in document at {path!r}")
        self.write(path, content.replace(old_str, new_str, 1))

    def list_paths(self, prefix: Optional[str] = None) -> list[str]:
        """
        Return all memory paths, optionally filtered by *prefix*.

        Args:
            prefix: Only return paths that start with this string.
                    Pass *None* (default) to return all paths.

        Returns:
            Sorted list of path strings.
        """
        paths = sorted(self._docs.keys())
        if prefix:
            paths = [p for p in paths if p.startswith(prefix)]
        return paths

    def search_full_text(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Case-insensitive substring search across all document contents.

        Args:
            query:     Search string.
            n_results: Maximum number of results to return.

        Returns:
            List of ``{"path": ..., "content": ...}`` dicts for matching
            documents, ordered by path.
        """
        query_lower = query.lower()
        matches = [
            {"path": path, "content": content}
            for path, content in sorted(self._docs.items())
            if query_lower in path.lower() or query_lower in content.lower()
        ]
        return matches[:n_results]

    # ------------------------------------------------------------------
    # MemoryStore abstract interface
    # ------------------------------------------------------------------

    def store(self, content: str) -> None:
        """
        Store *content* at an auto-assigned path (``/note-{uuid}.md``).

        Use :meth:`write` directly when you need control over the path.

        Args:
            content: Text content to store.
        """
        path = f"/note-{uuid.uuid4()}.md"
        self.write(path, content)

    def search(self, query: str, n_results: int = 10) -> list[str]:
        """
        Full-text search; returns a list of matching content strings.

        Args:
            query:     Search string.
            n_results: Maximum number of results.

        Returns:
            List of content strings from matching documents.
        """
        return [m["content"] for m in self.search_full_text(query, n_results)]

    def delete(self, identifier: str) -> None:
        """
        Delete the document at *identifier* (treated as a memory path).

        No-op if the path does not exist.

        Args:
            identifier: Memory path of the document to remove.
        """
        self._docs.pop(identifier, None)
        if self._persist_path:
            self._delete_from_disk(identifier)

    def list_all(self) -> list[str]:
        """Return all memory paths (sorted)."""
        return self.list_paths()
