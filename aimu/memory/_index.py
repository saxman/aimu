"""aimu.memory._index: the passage index behind :meth:`DocumentStore.search_passages`.

The index is a derived cache. The document folder stays the only source of truth: before each search the
store walks the folder's metadata and :meth:`PassageIndex.refresh` re-chunks only the files whose
``(mtime, size)`` changed, and drops the rows of files that are gone. Deleting the index loses nothing but
the time to rebuild it.

A document is indexed as *passages*, chunks small enough to quote, each carrying where in the document it
came from (a heading path, or a PDF page), so a search answers "which part of which file" instead of
returning whole documents. Ranking is SQLite FTS5's BM25, which needs no model and no dependency beyond the
standard library. Keyword search lives in its own method so a second ranking can be fused beside it.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import weakref
from dataclasses import dataclass
from typing import Callable, Mapping

from aimu.rag.bm25 import STOPWORDS
from aimu.rag.splitter import split_text

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# pdf_to_markdown's page marker. In a PDF it is the only heading: extracted text has no structure, and a
# line of it starting with "#" ("# of samples") is text, not a heading.
_PDF_PAGE = re.compile(r"^## Page (\d+)\s*$")
_FENCE = ("```", "~~~")
_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class Passage:
    """One chunk of a document, as a search returns it.

    Attributes:
        path: The document's store path, e.g. ``"/Notes/Methods.md"``.
        location: Where in the document the chunk sits: a heading path (``"Methods > Sampling"``), a PDF
            page (``"page 3"``), or ``""`` for text before any heading.
        text: The chunk itself.
        score: Relevance, higher is better. Comparable only within one search.
        digest: SHA-256 hex of the whole document's text when it was indexed. Two passages from copies of
            the same file share it, and a digest that no longer matches the file means the passage is stale.
    """

    path: str
    location: str
    text: str
    score: float
    digest: str


@dataclass(frozen=True)
class IndexStatus:
    """How much of a document folder the index covers.

    Attributes:
        files: Documents the index knows about (readable files in the folder).
        indexed: Documents whose passages are keyword-searchable.
        embedded: Documents whose passages have vectors (0 while vector search is not enabled).
        unreadable: Files in the folder that cannot be documents, mapped to why.
        error: Why part of the index could not be built, if it could not.
    """

    files: int
    indexed: int
    embedded: int
    unreadable: dict[str, str]
    error: str | None = None

    @property
    def complete(self) -> bool:
        """Whether every document is searchable, so a search that finds nothing found nothing."""
        return self.indexed == self.files


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block (Obsidian's ``---`` ... ``---``), which is metadata, not prose."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != "---":
        return text
    for number, line in enumerate(lines[1:], start=1):
        if line.rstrip() in ("---", "..."):
            return "".join(lines[number + 1 :])
    return text


def _sections(path: str, text: str) -> list[tuple[str, str]]:
    """Split *text* at its headings into ``(location, body)`` pairs, heading lines excluded from bodies."""
    is_pdf = path.lower().endswith(".pdf")
    sections: list[tuple[str, str]] = []
    headings: list[tuple[int, str]] = []
    location = ""
    body: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if not is_pdf and line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
        if is_pdf:
            page = _PDF_PAGE.match(line)
            new_location = f"page {page.group(1)}" if page else None
        else:
            heading = None if in_fence else _HEADING.match(line)
            new_location = None
            if heading:
                level = len(heading.group(1))
                headings = [h for h in headings if h[0] < level] + [(level, heading.group(2))]
                new_location = " > ".join(title for _, title in headings)
        if new_location is None:
            body.append(line)
            continue
        sections.append((location, "\n".join(body)))
        location, body = new_location, []
    sections.append((location, "\n".join(body)))
    return [(location, body) for location, body in sections if body.strip()]


def chunk_document(path: str, text: str, *, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[tuple[str, str]]:
    """Split a document into ``(location, chunk_text)`` passages, in document order.

    Each heading section (each page, for a PDF) is chunked on its own, so a passage never straddles two
    sections and its location is exact. Leading YAML frontmatter is dropped, and ``#`` lines inside fenced
    code are code, not headings.

    Args:
        path: The document's store path; a ``.pdf`` suffix selects page locations.
        text: The document's text (for a PDF, as ``pdf_to_markdown`` produced it).
        chunk_size: Maximum characters per passage.
        chunk_overlap: Characters repeated between consecutive passages of one section.
    """
    chunks: list[tuple[str, str]] = []
    for location, body in _sections(path, _strip_frontmatter(text)):
        for chunk in split_text(body, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            if chunk.strip():
                chunks.append((location, chunk.strip()))
    return chunks


def _match_expression(query: str) -> str | None:
    """An FTS5 MATCH expression for *query* that cannot be a syntax error, or None if nothing is left to match.

    A user's query is not FTS5 syntax: ``CRISPR-Cas13`` would parse as a column filter and a NOT, and a lone
    ``"`` as an unterminated string. Only word tokens survive, each double-quoted (``\\w`` never matches a
    quote, so a token cannot close its own string), joined by OR so a passage matching any word is a
    candidate and BM25 ranks the ones matching more of them higher.
    """
    tokens = dict.fromkeys(t for t in _TOKEN.findall(query.lower()) if t not in STOPWORDS)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


class PassageIndex:
    """SQLite FTS5 index of a document folder's passages. Private: reached through :class:`DocumentStore`.

    ``files`` is the manifest, one row per indexed document with the ``(mtime, size)`` it was indexed at and
    the digest of its text; ``passages`` holds its chunks. Both are written in one transaction per refresh,
    so a document is in the manifest exactly when its passages are searchable.

    Thread-safe: one re-entrant lock serializes all access to the connection, which is opened with
    ``check_same_thread=False`` because the store is called from worker threads.

    Args:
        index_path: Directory for ``index.sqlite3``, created if missing. *None* keeps the index in memory.
    """

    def __init__(self, index_path: str | None):
        self._lock = threading.RLock()
        if index_path is None:
            database = ":memory:"
        else:
            os.makedirs(index_path, exist_ok=True)
            database = os.path.join(index_path, "index.sqlite3")
        self._connection = sqlite3.connect(database, check_same_thread=False)
        # A store dropped without close() would otherwise leave the connection to the garbage collector,
        # which closes it with a ResourceWarning.
        self._closer = weakref.finalize(self, self._connection.close)
        with self._lock, self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, digest TEXT)"
            )
            self._connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5("
                "path UNINDEXED, location UNINDEXED, text, digest UNINDEXED, tokenize='unicode61')"
            )

    def refresh(self, files: Mapping[str, tuple[float, int]], load: Callable[[str], str | None]) -> None:
        """Bring the index in step with *files*, re-chunking only what changed.

        Args:
            files: Every document in the folder now, mapped to its ``(mtime, size)``.
            load: Returns a document's text, or None when it cannot be read or is gone. A None removes the
                document from the index rather than raising: a file deleted between the metadata walk and
                its read is simply not a document any more.
        """
        with self._lock, self._connection:
            indexed = {
                path: (mtime, size)
                for path, mtime, size in self._connection.execute("SELECT path, mtime, size FROM files")
            }
            for path, signature in files.items():
                if indexed.get(path) == signature:
                    continue
                text = load(path)
                if text is None:
                    self._delete(path)
                else:
                    self._write(path, text, *signature)
            for path in indexed.keys() - files.keys():
                self._delete(path)

    def update(self, path: str, text: str, mtime: float, size: int) -> None:
        """Index *text* as the current contents of *path*, replacing any earlier passages."""
        with self._lock, self._connection:
            self._write(path, text, mtime, size)

    def remove(self, path: str) -> None:
        """Drop *path* from the index; a no-op if it was never indexed."""
        with self._lock, self._connection:
            self._delete(path)

    def keyword_search(self, query: str, limit: int) -> list[Passage]:
        """Up to *limit* passages matching *query*'s words, best BM25 score first. Never raises on syntax."""
        expression = _match_expression(query)
        if expression is None or limit <= 0:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT path, location, text, -bm25(passages), digest FROM passages "
                "WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [Passage(*row) for row in rows]

    def status(self) -> IndexStatus:
        """Counts for the documents in the index. ``unreadable`` is the store's to fill in."""
        with self._lock:
            (files,) = self._connection.execute("SELECT COUNT(*) FROM files").fetchone()
        return IndexStatus(files=files, indexed=files, embedded=0, unreadable={})

    def close(self) -> None:
        with self._lock:
            self._closer()

    def _write(self, path: str, text: str, mtime: float, size: int) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._delete(path)
        self._connection.execute("INSERT INTO files VALUES (?, ?, ?, ?)", (path, mtime, size, digest))
        self._connection.executemany(
            "INSERT INTO passages VALUES (?, ?, ?, ?)",
            [(path, location, chunk, digest) for location, chunk in chunk_document(path, text)],
        )

    def _delete(self, path: str) -> None:
        # An UNINDEXED FTS5 column has no index, so deleting by path scans every passage. Asking the manifest
        # first (keyed by path) keeps a first build, where every write is of a path not yet indexed, linear.
        if self._connection.execute("DELETE FROM files WHERE path = ?", (path,)).rowcount:
            self._connection.execute("DELETE FROM passages WHERE path = ?", (path,))
