"""
aimu.memory.document_store: Path-based document memory store.

Follows Anthropic's Managed Agents Memory API pattern: documents are
addressed by path (e.g. "/preferences.md"), stored as plain text, and
retrieved via full-text search or exact path lookup.

Persistence modes:
  - Ephemeral (persist_path=None): in-memory dict, lost on process exit.
  - Persistent (persist_path provided): the directory *is* the store.  Each
    document is a file on disk, path segments map to subdirectories, and every
    read/list/search goes to the filesystem.  A file copied into the directory
    out-of-band (by a user or another process) is therefore visible
    immediately, without reconstructing the store.
"""

from __future__ import annotations

import logging
import os
import posixpath
import threading
import uuid
from typing import Optional

from aimu.memory.base import MemoryStore, synchronized

logger = logging.getLogger(__name__)


class DocumentStore(MemoryStore):
    """
    Path-addressed document store backed by the filesystem or an in-memory dict.

    Implements the :class:`MemoryStore` abstract interface so it can be swapped
    with :class:`SemanticMemoryStore` in any application, and adds the richer
    path-based API that mirrors Anthropic's Managed Agents Memory tools.

    Thread-safe: every public method is serialized on a re-entrant per-instance
    lock, so the in-memory dict and on-disk files stay consistent when the store
    is shared across concurrent threads or turns (``edit`` -> ``read`` + ``write``
    stays atomic).

    With *persist_path* set, the directory is the single source of truth: reads,
    listings, and searches hit the filesystem on each call, so a document copied
    into the directory by hand shows up without restarting the process.  Files
    that are not UTF-8 text (a PDF, a binary artifact) are not documents; they
    are skipped, with a warning naming the file, and dot-files are ignored
    outright.

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
        # Ephemeral backing store; unused when persist_path is set (disk is authoritative then).
        self._docs: dict[str, str] = {}
        # Absolute paths already reported as unreadable, so a scan on every list/search call
        # warns once per file rather than on every turn.
        self._warned_unreadable: set[str] = set()
        # Serializes the public methods so the in-memory dict + on-disk files stay consistent when the
        # store is shared across concurrent turns (which dispatch sync tools from worker threads).
        # Re-entrant because public methods call each other (edit -> read + write, store -> write).
        self._lock = threading.RLock()
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(path: str) -> str:
        """Canonicalize a memory path to a single leading slash with forward slashes.

        Callers may address a document as ``"foo.md"`` or ``"/foo.md"`` interchangeably, and keys
        created by :meth:`write` match those rebuilt from disk by :meth:`_scan_disk` (which
        always prefixes a slash). ``posixpath.normpath`` also collapses redundant separators and
        resolves ``..`` segments, so a path can't escape the store's namespace.
        """
        cleaned = str(path).strip().replace("\\", "/")
        return posixpath.normpath("/" + cleaned.lstrip("/"))

    def _abs_path(self, path: str) -> str:
        """Resolve a memory path to an absolute filesystem path."""
        # Strip leading slash so os.path.join works correctly.
        return os.path.join(self._persist_path, path.lstrip("/"))

    @staticmethod
    def _is_hidden(path: str) -> bool:
        """Whether any segment of a memory path is a dot-file or dot-directory.

        ``.DS_Store``, ``.git/``, ``.gitkeep`` and friends share the directory but are not
        documents anyone placed there, so they are excluded silently rather than listed or
        reported as unreadable on every scan.
        """
        return any(segment.startswith(".") for segment in path.split("/") if segment)

    def _is_document(self, path: str) -> bool:
        """Whether *path* names a file eligible to be a document in this store."""
        return not self._is_hidden(path) and os.path.isfile(self._abs_path(path))

    def _read_file(self, abs_file: str) -> Optional[str]:
        """Return the text of *abs_file*, or *None* if it is not a readable UTF-8 document.

        Documents are text.  A file that is not (a PDF, an image, a partially written file) is
        skipped rather than aborting the scan, but never silently: it is logged once per path so
        a document that will never appear is visible to whoever put it there.
        """
        try:
            with open(abs_file, encoding="utf-8") as f:
                return f.read()
        except (UnicodeDecodeError, OSError) as exc:
            if abs_file not in self._warned_unreadable:
                self._warned_unreadable.add(abs_file)
                logger.warning("Skipping unreadable document %s: %s", abs_file, exc)
            return None

    def _scan_disk(self, collect_unreadable: Optional[list[str]] = None) -> dict[str, str]:
        """Read every document under persist_path into a fresh path -> content mapping.

        Pass *collect_unreadable* to also gather the memory paths of files that exist in the
        directory but are not readable UTF-8 text.
        """
        docs: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self._persist_path):
            # Prune dot-directories in place so os.walk never descends into them.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                abs_file = os.path.join(dirpath, filename)
                rel = os.path.relpath(abs_file, self._persist_path)
                # Normalise to forward-slash memory paths with leading slash.
                mem_path = self._normalize(rel.replace(os.sep, "/"))
                content = self._read_file(abs_file)
                if content is None:
                    if collect_unreadable is not None:
                        collect_unreadable.append(mem_path)
                    continue
                docs[mem_path] = content
        return docs

    def _documents(self) -> dict[str, str]:
        """Current contents of the store: the directory when persistent, the dict otherwise."""
        return self._scan_disk() if self._persist_path else dict(self._docs)

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

    @synchronized
    def write(self, path: str, content: str) -> None:
        """
        Create or overwrite a document at *path*.

        Args:
            path:    Memory path, e.g. ``"/preferences.md"``.
            content: Text content to store (≤ 100 KB recommended).
        """
        path = self._normalize(path)
        if self._persist_path:
            self._write_to_disk(path, content)
            # A path that was unreadable may now be a valid document; let it warn again if not.
            self._warned_unreadable.discard(self._abs_path(path))
        else:
            self._docs[path] = content

    @synchronized
    def read(self, path: str) -> str:
        """
        Return the content of the document at *path*.

        Raises:
            KeyError: If no document exists at *path*.
        """
        path = self._normalize(path)
        if self._persist_path:
            # Open the one file directly rather than scanning the whole directory.
            content = self._read_file(self._abs_path(path)) if self._is_document(path) else None
            if content is None:
                raise KeyError(path)
            return content
        if path not in self._docs:
            raise KeyError(path)
        return self._docs[path]

    @synchronized
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

    @synchronized
    def list_paths(self, prefix: Optional[str] = None) -> list[str]:
        """
        Return all memory paths, optionally filtered by *prefix*.

        Args:
            prefix: Only return paths that start with this string.
                    Pass *None* (default) to return all paths.

        Returns:
            Sorted list of path strings.
        """
        paths = sorted(self._documents().keys())
        if prefix:
            paths = [p for p in paths if p.startswith(self._normalize(prefix))]
        return paths

    @synchronized
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
            for path, content in sorted(self._documents().items())
            if query_lower in path.lower() or query_lower in content.lower()
        ]
        return matches[:n_results]

    @synchronized
    def unreadable_paths(self) -> list[str]:
        """Return the paths of files in the store's directory that are not readable as text.

        These are files someone put in the directory that can never become documents (a PDF, a
        Word export, a binary artifact).  They are excluded from :meth:`list_paths` and
        :meth:`search_full_text`, so this is what lets a caller tell the difference between "you
        gave me nothing" and "what you gave me is not text" -- the distinction an agent needs in
        order to say something useful back.  Dot-files are not reported: they are not documents
        anyone placed here.

        Rescans the directory; returns ``[]`` for an ephemeral store, which has no directory.

        Returns:
            Sorted list of memory paths.
        """
        if not self._persist_path:
            return []
        unreadable: list[str] = []
        self._scan_disk(collect_unreadable=unreadable)
        return sorted(unreadable)

    # ------------------------------------------------------------------
    # MemoryStore abstract interface
    # ------------------------------------------------------------------

    @synchronized
    def store(self, content: str) -> None:
        """
        Store *content* at an auto-assigned path (``/note-{uuid}.md``).

        Use :meth:`write` directly when you need control over the path.

        Args:
            content: Text content to store.
        """
        path = f"/note-{uuid.uuid4()}.md"
        self.write(path, content)

    @synchronized
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

    @synchronized
    def delete(self, identifier: str) -> None:
        """
        Delete the document at *identifier* (treated as a memory path).

        No-op if the path does not exist.

        Args:
            identifier: Memory path of the document to remove.
        """
        identifier = self._normalize(identifier)
        if self._persist_path:
            self._delete_from_disk(identifier)
            self._warned_unreadable.discard(self._abs_path(identifier))
        else:
            self._docs.pop(identifier, None)

    @synchronized
    def list_all(self) -> list[str]:
        """Return all memory paths (sorted)."""
        return self.list_paths()
