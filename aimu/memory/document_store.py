"""
aimu.memory.document_store: Path-based document memory store.

Borrows the shape of Anthropic's memory tool: documents are addressed by path
(e.g. "/preferences.md"), stored as plain text, and retrieved via full-text
search or exact path lookup.  The method names and signatures are AIMU's own,
not that tool's command set -- see ``aimu.memory.document_mcp`` for why this is
a local backend rather than a drop-in for anything of Anthropic's.

Persistence modes:
  - Ephemeral (persist_path=None): in-memory dict, lost on process exit.
  - Persistent (persist_path provided): the directory *is* the store.  Each
    document is a file on disk, path segments map to subdirectories, and every
    read/list/search goes to the filesystem.  A file copied into the directory
    out-of-band (by a user or another process) is therefore visible
    immediately, without reconstructing the store.

Passage search (:meth:`DocumentStore.search_passages`) adds an index beside the folder, never inside it:
a derived cache (see ``aimu.memory._index``) kept in step by a metadata walk before each search.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import posixpath
import shutil
import threading
import uuid
from typing import Any, Optional

from aimu.memory._index import IndexStatus, Passage, PassageIndex
from aimu.memory.base import MemoryStore, synchronized
from aimu.tools._documents import DocumentConversionError, pdf_to_markdown

logger = logging.getLogger(__name__)


class DocumentStore(MemoryStore):
    """
    Path-addressed document store backed by the filesystem or an in-memory dict.

    Implements the :class:`MemoryStore` abstract interface so it can be swapped
    with :class:`SemanticMemoryStore` in any application, and adds the richer
    path-based API in the spirit of Anthropic's memory tool (not its interface).

    Thread-safe: every public method is serialized on a re-entrant per-instance
    lock, so the in-memory dict and on-disk files stay consistent when the store
    is shared across concurrent threads or turns (``edit`` -> ``read`` + ``write``
    stays atomic).

    With *persist_path* set, the directory is the single source of truth: reads,
    listings, and searches hit the filesystem on each call, so a document copied
    into the directory by hand shows up without restarting the process.  Files
    that are neither UTF-8 text nor a PDF with a text layer (an image, a binary
    artifact), and files over *max_file_bytes*, are not documents; they are
    skipped, with a warning naming the file, and dot-files are ignored outright.
    So is anything whose real path lies outside the directory (a symlink out of
    it), which also cannot be written or deleted through the store.

    Args:
        persist_path: Root directory for persistent storage.  If *None* the
            store is ephemeral (in-memory only).
        index_path: Directory for the passage index behind :meth:`search_passages`.
            *None* keeps it in memory, rebuilt on the first search of each process.
            Must lie outside *persist_path*: the index is derived from the folder
            and never written into it.
        embedding_client: Reserved for vector search; must be *None* for now.
        max_file_bytes: Files larger than this are unreadable rather than read.

    Examples:
        >>> store = DocumentStore()
        >>> store.write("/preferences.md", "Use concise responses.")
        >>> store.read("/preferences.md")
        'Use concise responses.'
        >>> store.edit("/preferences.md", "concise", "detailed")
        >>> store.search_full_text("detailed")
        [{"path": "/preferences.md", "content": "Use detailed responses."}]
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        *,
        index_path: Optional[str] = None,
        embedding_client: Optional[Any] = None,
        max_file_bytes: int = 50_000_000,
    ):
        if embedding_client is not None:
            raise NotImplementedError("vector search is not available yet; pass embedding_client=None")
        self._persist_path = persist_path
        self._index_path = index_path
        self._max_file_bytes = max_file_bytes
        # Created on the first search_passages / index_status, so a store nobody searches costs nothing.
        self._index: Optional[PassageIndex] = None
        # Ephemeral backing store; unused when persist_path is set (disk is authoritative then).
        self._docs: dict[str, str] = {}
        # Absolute path -> why it could not be read. Doubles as the already-warned set, so a scan on
        # every list/search call warns once per file rather than on every turn.
        self._unreadable_reasons: dict[str, str] = {}
        # Absolute path -> ((st_mtime_ns, st_size), text or None, reason or None). Holds what is costly to
        # recompute on every scan: a PDF's extraction, and any file's failure to read. Plain text files are
        # cheap to reread and are not held, so the cache stays the size of the PDFs, not of the folder.
        self._extractions: dict[str, tuple[tuple[int, int], Optional[str], Optional[str]]] = {}
        # Serializes the public methods so the in-memory dict + on-disk files stay consistent when the
        # store is shared across concurrent turns (which dispatch sync tools from worker threads).
        # Re-entrant because public methods call each other (edit -> read + write, store -> write).
        self._lock = threading.RLock()
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            self._real_root = os.path.realpath(persist_path)
            if index_path and self._is_inside_root(index_path):
                raise ValueError(
                    f"index_path {index_path!r} must be outside the document folder {persist_path!r}: "
                    "the index is derived from the folder, and would otherwise be scanned as part of it."
                )

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

    def _is_inside_root(self, abs_path: str) -> bool:
        """Whether *abs_path*, symlinks resolved, lies inside the document folder."""
        real = os.path.realpath(abs_path)
        return real == self._real_root or real.startswith(self._real_root + os.sep)

    def _require_inside_root(self, path: str) -> str:
        """The real path *path* resolves to, refusing one that escapes the folder through a symlink."""
        abs_file = self._abs_path(path)
        if not self._is_inside_root(abs_file):
            raise ValueError(f"{path} resolves outside the document folder, so it is not a document in this store.")
        return os.path.realpath(abs_file)

    def _is_document(self, path: str) -> bool:
        """Whether *path* names a file eligible to be a document in this store."""
        abs_file = self._abs_path(path)
        return not self._is_hidden(path) and os.path.isfile(abs_file) and self._is_inside_root(abs_file)

    @staticmethod
    def _is_pdf(path: str) -> bool:
        return path.lower().endswith(".pdf")

    def _read_file(self, abs_file: str) -> Optional[str]:
        """Return the text of *abs_file*, or *None* if it cannot be read as a document.

        Documents are UTF-8 text, plus PDFs, which are converted to Markdown on read. Anything else
        (an image, a scanned or encrypted PDF, a file over *max_file_bytes*, one the process may not
        open) is skipped rather than aborting the scan, but never silently: the reason is recorded for
        :meth:`unreadable` and logged once per path, so a document that will never appear is visible
        to whoever put it there. A file that vanished since the walk found it is simply gone, not
        unreadable. An unchanged file (same ``st_mtime_ns`` and ``st_size``) is never extracted twice.
        """
        try:
            stat = os.stat(abs_file)
        except FileNotFoundError:
            self._extractions.pop(abs_file, None)
            self._unreadable_reasons.pop(abs_file, None)
            return None
        except OSError as exc:
            return self._unreadable(abs_file, str(exc))
        if stat.st_size > self._max_file_bytes:
            return self._unreadable(
                abs_file, f"too large to index ({stat.st_size} bytes; the limit is {self._max_file_bytes})"
            )
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._extractions.get(abs_file)
        if cached is not None and cached[0] == signature:
            _, text, reason = cached
            return text if reason is None else self._unreadable(abs_file, reason)
        try:
            if self._is_pdf(abs_file):
                with open(abs_file, "rb") as f:
                    text = pdf_to_markdown(f.read())
                self._extractions[abs_file] = (signature, text, None)
                return text
            with open(abs_file, encoding="utf-8") as f:
                return f.read()
        except (DocumentConversionError, UnicodeDecodeError) as exc:
            reason = str(exc) if isinstance(exc, DocumentConversionError) else "not UTF-8 text"
            self._extractions[abs_file] = (signature, None, reason)
            return self._unreadable(abs_file, reason)
        except OSError as exc:
            # Not cached: a permission change moves ctime, not the mtime the cache is keyed on.
            return self._unreadable(abs_file, str(exc))

    def _unreadable(self, abs_file: str, reason: str) -> None:
        """Record why *abs_file* is not a document, warning the first time, and return None for the caller."""
        if abs_file not in self._unreadable_reasons:
            logger.warning("Skipping unreadable document %s: %s", abs_file, reason)
        self._unreadable_reasons[abs_file] = reason
        return None

    def _walk(self) -> list[tuple[str, str]]:
        """Every candidate document under persist_path, as ``(memory path, absolute path)`` pairs.

        Dot-files and dot-directories are skipped, and so is a file symlink resolving outside the
        folder. Directory symlinks are never descended (``os.walk``'s default), so the file itself is
        the only place a walk can leave the folder. Drops extraction-cache entries for files that are gone.
        """
        found: list[tuple[str, str]] = []
        for dirpath, dirnames, filenames in os.walk(self._persist_path):
            # Prune dot-directories in place so os.walk never descends into them.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                abs_file = os.path.join(dirpath, filename)
                if os.path.islink(abs_file) and not self._is_inside_root(abs_file):
                    continue
                rel = os.path.relpath(abs_file, self._persist_path)
                # Normalise to forward-slash memory paths with leading slash.
                found.append((self._normalize(rel.replace(os.sep, "/")), abs_file))
        present = {abs_file for _, abs_file in found}
        for gone in self._extractions.keys() - present:
            del self._extractions[gone]
        return found

    def _scan_disk(self, collect_unreadable: Optional[dict[str, str]] = None) -> dict[str, str]:
        """Read every document under persist_path into a fresh path -> content mapping.

        Pass *collect_unreadable* to also gather the memory paths (with reasons) of files that exist
        in the directory but cannot be read as documents.
        """
        docs: dict[str, str] = {}
        for mem_path, abs_file in self._walk():
            content = self._read_file(abs_file)
            if content is not None:
                docs[mem_path] = content
            elif collect_unreadable is not None and abs_file in self._unreadable_reasons:
                collect_unreadable[mem_path] = self._unreadable_reasons[abs_file]
        return docs

    def _documents(self) -> dict[str, str]:
        """Current contents of the store: the directory when persistent, the dict otherwise."""
        return self._scan_disk() if self._persist_path else dict(self._docs)

    def _write_to_disk(self, path: str, content: str) -> None:
        """Write *content* atomically: a reader (an editor, a concurrent scan) sees the old file or the new one.

        The text goes to a dot-prefixed temp file in the same directory, which scans skip, then
        ``os.replace`` swaps it in. An in-folder symlink is written through to its target, not replaced.
        """
        target = self._require_inside_root(path)
        directory = os.path.dirname(target)
        os.makedirs(directory, exist_ok=True)
        temp = os.path.join(directory, f".{os.path.basename(target)}.{uuid.uuid4().hex}.tmp")
        try:
            # Mode "x" creates the file with the umask's permissions, as a plain open would have.
            with open(temp, "x", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(target):
                shutil.copymode(target, temp)
            os.replace(temp, target)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp)
            raise

    def _delete_from_disk(self, path: str) -> None:
        self._require_inside_root(path)
        abs_file = self._abs_path(path)
        if os.path.exists(abs_file):
            os.remove(abs_file)

    def _file_metadata(self) -> dict[str, tuple[float, int]]:
        """Every document's ``(mtime, size)``, without reading any of them: what the index refreshes from.

        An ephemeral store has no files, and nothing outside it can change its dict, so each document's
        "mtime" is a constant and its writes reach the index directly.
        """
        if not self._persist_path:
            return {path: (0.0, len(content)) for path, content in self._docs.items()}
        files: dict[str, tuple[float, int]] = {}
        for mem_path, abs_file in self._walk():
            try:
                stat = os.stat(abs_file)
            except OSError:
                continue
            files[mem_path] = (stat.st_mtime, stat.st_size)
        return files

    def _load(self, path: str) -> Optional[str]:
        if not self._persist_path:
            return self._docs.get(path)
        return self._read_file(self._abs_path(path))

    def _refreshed_index(self) -> tuple[PassageIndex, dict[str, str]]:
        """The passage index, created on first use and brought in step with the folder.

        Also returns the unreadable files the refresh met, with reasons. An unreadable file never enters
        the index's manifest, so every refresh tries it again (cheaply: the extraction cache holds its
        failure), which is what lets this report it without a second walk that reads every file.
        """
        if self._index is None:
            self._index = PassageIndex(self._index_path)
        unreadable: dict[str, str] = {}

        def load(path: str) -> Optional[str]:
            text = self._load(path)
            reason = self._unreadable_reasons.get(self._abs_path(path)) if self._persist_path else None
            if text is None and reason is not None:
                unreadable[path] = reason
            return text

        self._index.refresh(self._file_metadata(), load)
        return self._index, unreadable

    def _index_document(self, path: str, content: str) -> None:
        """Bring an existing index up to date with a write the store just made."""
        if self._index is None:
            return
        if not self._persist_path:
            self._index.update(path, content, 0.0, len(content))
            return
        abs_file = self._abs_path(path)
        # Indexed as read back, not as given: reading translates newlines, and the digest has to match
        # what a later refresh would compute from the file.
        text = self._read_file(abs_file)
        if text is None:
            self._index.remove(path)
        else:
            stat = os.stat(abs_file)
            self._index.update(path, text, stat.st_mtime, stat.st_size)

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
        if self._is_pdf(path):
            raise ValueError(f"PDFs are read-only: {path} can be read and searched, not written.")
        if self._persist_path:
            self._write_to_disk(path, content)
            # A path that was unreadable may now be a valid document; let it warn again if not.
            self._unreadable_reasons.pop(self._abs_path(path), None)
        else:
            self._docs[path] = content
        self._index_document(path, content)

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
        Replace the one occurrence of *old_str* with *new_str* in the document at *path*.

        *old_str* must appear exactly once.  Zero matches and two-or-more matches both
        raise, and nothing is written either way: replacing the *first* of several
        matches answers a question the caller did not ask, and it does so silently, in
        the file, where the mistake is discovered later by whoever reads it.  Include
        enough surrounding text to make the match unique.

        To genuinely intend the first of several, say so where it can be seen::

            store.write(path, store.read(path).replace(old_str, new_str, 1))

        Args:
            path:    Memory path of the document to edit.
            old_str: Exact substring to find, unique within the document.
            new_str: Replacement text.

        Raises:
            KeyError:   If no document exists at *path*.
            ValueError: If *old_str* is absent, or appears more than once.
        """
        if self._is_pdf(self._normalize(path)):
            raise ValueError(f"PDFs are read-only: {self._normalize(path)} can be read and searched, not written.")
        if self._persist_path:
            self._require_inside_root(self._normalize(path))
        content = self.read(path)
        occurrences = content.count(old_str)
        if occurrences == 0:
            raise ValueError(f"{old_str!r} not found in document at {path!r}")
        if occurrences > 1:
            raise ValueError(
                f"{old_str!r} appears {occurrences} times in document at {path!r}, so it is "
                "unclear which to change; nothing was written. Include surrounding text to "
                "make it unique."
            )
        self.write(path, content.replace(old_str, new_str))

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
    def unreadable(self) -> dict[str, str]:
        """Return the files in the store's directory that cannot be read as documents, with why.

        These are files someone put in the directory that never become documents: a scanned or
        password-protected PDF, a Word export, a binary artifact. They are excluded from
        :meth:`list_paths` and :meth:`search_full_text`, so this is what lets a caller tell the
        difference between "you gave me nothing" and "what you gave me cannot be read", and say why.
        Dot-files are not reported: they are not documents anyone placed here.

        Rescans the directory; returns ``{}`` for an ephemeral store, which has no directory.

        Returns:
            Mapping of normalized memory path to a human-readable reason.
        """
        if not self._persist_path:
            return {}
        unreadable: dict[str, str] = {}
        self._scan_disk(collect_unreadable=unreadable)
        return unreadable

    @synchronized
    def search_passages(self, query: str, n_results: int = 8) -> list[Passage]:
        """
        Rank passages of every document against *query*, best first.

        Where :meth:`search_full_text` returns whole documents containing a substring, this returns
        chunks of documents (about 1200 characters, each section or PDF page chunked on its own), each
        with its location, ranked by BM25 over the query's words. Any query is accepted: punctuation and
        search-syntax characters are dropped, never raised on, and common English words are ignored.

        The folder is the source of truth. A metadata walk before each search re-indexes only files
        whose size or modification time changed, so a file edited, added, or deleted out-of-band is
        reflected at once. The index is built on the first call.

        Args:
            query:     Natural-language or keyword query.
            n_results: Maximum number of passages to return.

        Returns:
            :class:`~aimu.memory.Passage` records, highest score first.
        """
        index, _ = self._refreshed_index()
        return index.keyword_search(query, n_results)

    @synchronized
    def index_status(self) -> IndexStatus:
        """Refresh the passage index and report how much of the store it covers, and what it cannot read."""
        index, unreadable = self._refreshed_index()
        return dataclasses.replace(index.status(), unreadable=unreadable)

    @synchronized
    def close(self) -> None:
        """Release the passage index's database connection, if one was opened. The store stays usable."""
        if self._index is not None:
            self._index.close()
            self._index = None

    @synchronized
    def unreadable_paths(self) -> list[str]:
        """Return the sorted memory paths of files that cannot be read as documents.

        See :meth:`unreadable` for the reasons.
        """
        return sorted(self.unreadable())

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
            self._unreadable_reasons.pop(self._abs_path(identifier), None)
        else:
            self._docs.pop(identifier, None)
        if self._index is not None:
            self._index.remove(identifier)

    @synchronized
    def list_all(self) -> list[str]:
        """Return all memory paths (sorted)."""
        return self.list_paths()
