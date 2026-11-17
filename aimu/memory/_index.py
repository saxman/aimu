"""aimu.memory._index: the passage index behind :meth:`DocumentStore.search_passages`.

The index is a derived cache. The document folder stays the only source of truth: before each search the
store walks the folder's metadata and :meth:`PassageIndex.refresh` re-chunks only the files whose
``(mtime, size)`` changed, and drops the rows of files that are gone. Deleting the index loses nothing but
the time to rebuild it, which is also why an index written by a different version of this module is
dropped and rebuilt rather than migrated.

A document is indexed as *passages*, chunks small enough to quote, each carrying where in the document it
came from (a heading path, or a PDF page), so a search answers "which part of which file" instead of
returning whole documents. Two rankings cover them:

- **Keyword**: SQLite FTS5's BM25, which needs no model and no dependency beyond the standard library, and
  finds the exact terms a query names.
- **Vector**: with an embedding client, each passage is embedded into a Chroma collection, which finds a
  passage that says the same thing in other words. The two rankings are fused by reciprocal rank fusion,
  which uses only ranks, so BM25 scores and cosine distances never have to be put on one scale.

A first build can be long (extracting hundreds of PDFs, embedding tens of thousands of passages), and a
search must not wait for it. A refresh indexes changed files inline for :data:`INLINE_INDEX_SECONDS`, then
leaves the rest to one background worker thread, which also does all embedding. The worker takes the
index's lock (and, to read a file, the store's) one file at a time, so reads, writes, and searches
interleave with it, and a search meanwhile ranks what is indexed so far while :meth:`PassageIndex.status`
says how far along the build is. The worker exits once there is nothing left to do.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from aimu.rag.bm25 import STOPWORDS
from aimu.rag.fusion import reciprocal_rank_fusion
from aimu.rag.splitter import split_text

logger = logging.getLogger(__name__)

# The layout of index.sqlite3, stored as its PRAGMA user_version. An index of any other version is dropped and
# rebuilt on open. 0 (SQLite's default) is the keyword-only layout, which had no way to find a file's passages
# but scanning them all.
SCHEMA_VERSION = 1
# How long a refresh indexes changed files itself before leaving the rest to the background worker.
INLINE_INDEX_SECONDS = 2.0
# Passages per embedding call.
EMBEDDING_BATCH_SIZE = 32
WORKER_THREAD_NAME = "aimu-passage-index"
# How long close() waits for the worker to finish the file or batch it is on.
_CLOSE_TIMEOUT_SECONDS = 10.0
_COLLECTION = "passages"

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
        files: Documents in the folder: every file, less those found unreadable.
        indexed: Documents whose current text is keyword-searchable. Below *files* while a build is running.
        embedded: Indexed documents whose passages also have vectors (always 0 when *vectors* is False).
        unreadable: Files in the folder that cannot be documents, mapped to why.
        error: What is wrong, if anything: the embedding model failing to load or to embed (search is then
            keyword-only, and the next refresh retries it), or files the index failed to process (each
            left out of the index, and retried once it changes).
        vectors: Whether the index has an embedding client, so *embedded* counts toward *complete*.
    """

    files: int
    indexed: int
    embedded: int
    unreadable: dict[str, str]
    error: str | None = None
    vectors: bool = False

    @property
    def complete(self) -> bool:
        """Whether every document is fully searchable, so a search that finds nothing found nothing."""
        return self.indexed == self.files and (not self.vectors or self.embedded == self.indexed)


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


def _open_vectors(index_path: str | None) -> tuple[Any, str]:
    """A Chroma client for the index's vectors, and the name of its collection.

    Chroma is the ``[memory]`` extra, so it is imported only when vector search is asked for. In-memory
    Chroma clients share one store per process, so an in-memory index takes a collection name of its own.
    """
    try:
        import chromadb
    except ImportError as exc:
        if exc.name != "chromadb":
            raise
        raise ImportError(
            "DocumentStore vector search requires the [memory] extra (chromadb): pip install 'aimu[memory]'"
        ) from exc
    if index_path is None:
        return chromadb.EphemeralClient(), f"{_COLLECTION}-{uuid.uuid4().hex}"
    return chromadb.PersistentClient(path=os.path.join(index_path, "vectors")), _COLLECTION


def _release(connection: sqlite3.Connection, client: Any, private_collection: str | None) -> None:
    connection.close()
    if private_collection is not None:
        client.delete_collection(private_collection)


@dataclass(eq=False)
class _PendingFile:
    """A file a refresh found changed and has not indexed yet.

    Compared by identity: whoever indexes it checks the entry is still the one pending for its path, so a
    write, a removal, or a newer refresh that got there first wins over a slower read of older text.
    """

    signature: tuple[float, int]
    load: Callable[[str], str | None]


class PassageIndex:
    """Keyword (SQLite FTS5) and, optionally, vector (Chroma) index of a document folder's passages.

    Private: reached through :class:`DocumentStore`.

    ``files`` is the manifest, one row per indexed document: the ``(mtime, size)`` it was indexed at, the
    digest of its text, the rowids its passages occupy in ``passages`` (``first_rowid`` and ``chunks``), and
    whether they are ``embedded``. A document's manifest row and its passages are written in one transaction,
    so a document is in the manifest exactly when its passages are searchable.

    Vectors are keyed by content, not path: chunk ``n`` of a text with digest ``d`` is ``f"{d}:{n}"``. A
    copy of a document therefore needs no embedding of its own, and a vector hit is mapped back to passages
    through the manifest's digests, which drops any vector whose text no file holds any more.

    Thread-safe: one re-entrant lock guards the connection, the collection, and the pending work. It is
    held for one file's worth of work at a time and never while reading a file or calling the embedding
    model.

    Args:
        index_path: Directory for ``index.sqlite3`` (and ``vectors/``), created if missing. *None* keeps the
            index in memory.
        embedding_client: An AIMU embedding client (``embed(texts, input_type=...)`` and ``spec.id``), or
            *None* for keyword search alone. ``spec.id`` is recorded, and opening the index with a
            different model starts its vectors over, since vectors of two models are not comparable.

    Raises:
        ImportError: If *embedding_client* is given and chromadb (the ``[memory]`` extra) is not installed.
    """

    def __init__(self, index_path: str | None, embedding_client: Any | None = None):
        self._lock = threading.RLock()
        # Notified when the worker exits, which is when wait_until_embedded has something new to report.
        self._idle = threading.Condition(self._lock)
        self._embedder = embedding_client
        # The folder as of the last refresh, path -> (mtime, size); status() counts files from it.
        self._signatures: dict[str, tuple[float, int]] = {}
        self._pending: dict[str, _PendingFile] = {}
        # Paths whose load returned None, with the signature it did so at. Not retried until that changes.
        self._unloadable: dict[str, tuple[float, int]] = {}
        # Paths the worker failed to index, with the signature and why. Not retried until it changes.
        self._failed: dict[str, tuple[tuple[float, int], str]] = {}
        self._worker: threading.Thread | None = None
        self._closed = False
        # Set when embedding fails, so the worker stops trying until the next refresh or update.
        self._embedding_paused = False
        self._embedding_error: str | None = None
        # Why the worker stopped short of its work, outside any one file. Cleared when a refresh restarts it.
        self._worker_error: str | None = None
        client, collection_name = _open_vectors(index_path) if embedding_client is not None else (None, None)
        if index_path is None:
            database = ":memory:"
        else:
            os.makedirs(index_path, exist_ok=True)
            database = os.path.join(index_path, "index.sqlite3")
        self._connection = sqlite3.connect(database, check_same_thread=False)
        # A store dropped without close() would otherwise leave the connection to the garbage collector,
        # which closes it with a ResourceWarning.
        private_collection = collection_name if index_path is None else None
        self._closer = weakref.finalize(self, _release, self._connection, client, private_collection)
        with self._lock, self._connection:
            self._create_schema()
            self._collection = self._open_collection(client, collection_name)

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        (version,) = self._connection.execute("PRAGMA user_version").fetchone()
        if version != SCHEMA_VERSION:
            for table in ("files", "passages", "meta"):
                self._connection.execute(f"DROP TABLE IF EXISTS {table}")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, digest TEXT, "
            "first_rowid INTEGER, chunks INTEGER, embedded INTEGER NOT NULL DEFAULT 0)"
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS files_by_digest ON files (digest)")
        self._connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5("
            "path UNINDEXED, location UNINDEXED, text, digest UNINDEXED, tokenize='unicode61')"
        )
        self._connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _open_collection(self, client: Any, name: str | None) -> Any:
        """The vector collection, emptied first if the recorded embedding model is not this one."""
        row = self._connection.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        recorded = row[0] if row else None
        wanted = self._embedder.spec.id if self._embedder is not None else None
        if recorded != wanted:
            # Opened without a client, the vectors cannot be deleted (chromadb may not even be installed), so
            # the manifest stops vouching for them instead, and the next open with a client starts over.
            self._connection.execute("UPDATE files SET embedded = 0")
            if wanted is None:
                self._connection.execute("DELETE FROM meta WHERE key = 'embedding_model'")
            else:
                self._connection.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", (wanted,))
        if client is None:
            return None
        if recorded != wanted and name in {collection.name for collection in client.list_collections()}:
            client.delete_collection(name)
        # No embedding function: the index embeds passages and queries itself, with the input_type each needs.
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    # ------------------------------------------------------------------
    # Keeping in step with the folder
    # ------------------------------------------------------------------

    def refresh(self, files: Mapping[str, tuple[float, int]], load: Callable[[str], str | None]) -> None:
        """Bring the index in step with *files*, re-chunking only what changed.

        Changed files are indexed here until :data:`INLINE_INDEX_SECONDS` have passed, and by the worker
        after that; so is all embedding. While the worker is already running, changed files go straight to
        it, so a search during a long build pays for the metadata walk and not another inline budget. A
        refresh also retries embedding that failed.

        Args:
            files: Every document in the folder now, mapped to its ``(mtime, size)``.
            load: Returns a document's text, or None when it cannot be read or is gone. A None removes the
                document from the index rather than raising: a file deleted between the metadata walk and
                its read is simply not a document any more. Called from the worker thread too, so it must
                be safe to call from any thread.
        """
        deadline = time.monotonic() + INLINE_INDEX_SECONDS
        with self._lock, self._connection:
            indexed = {
                path: (mtime, size)
                for path, mtime, size in self._connection.execute("SELECT path, mtime, size FROM files")
            }
            for path in indexed.keys() - files.keys():
                self._delete(path)
            pending: dict[str, _PendingFile] = {}
            for path, signature in files.items():
                failed = self._failed.get(path)
                if indexed.get(path) == signature or self._unloadable.get(path) == signature:
                    continue
                if failed is not None and failed[0] == signature:
                    continue
                earlier = self._pending.get(path)
                keep = earlier is not None and earlier.signature == signature
                pending[path] = earlier if keep else _PendingFile(signature, load)
            self._signatures = dict(files)
            self._pending = pending
            self._unloadable = {path: sig for path, sig in self._unloadable.items() if files.get(path) == sig}
            self._failed = {path: failure for path, failure in self._failed.items() if files.get(path) == failure[0]}
            self._embedding_paused = False
            self._worker_error = None
            inline = self._worker is None
        for path, entry in list(pending.items()) if inline else []:
            if time.monotonic() >= deadline:
                break
            self._store_loaded(path, entry, entry.load(path))
        with self._lock:
            self._start_worker()

    def update(self, path: str, text: str, mtime: float, size: int) -> None:
        """Index *text* as the current contents of *path*, replacing any earlier passages."""
        with self._lock:
            with self._connection:
                self._write(path, text, mtime, size)
            self._pending.pop(path, None)
            self._unloadable.pop(path, None)
            self._failed.pop(path, None)
            self._signatures[path] = (mtime, size)
            self._embedding_paused = False
            self._start_worker()

    def remove(self, path: str) -> None:
        """Drop *path* from the index; a no-op if it was never indexed."""
        with self._lock:
            with self._connection:
                self._delete(path)
            self._pending.pop(path, None)
            self._unloadable.pop(path, None)
            self._failed.pop(path, None)
            self._signatures.pop(path, None)

    def _store_loaded(self, path: str, entry: _PendingFile, text: str | None) -> None:
        """Index what *entry*'s load returned, unless something newer has replaced it since."""
        with self._lock:
            if self._closed or self._pending.get(path) is not entry:
                return
            with self._connection:
                if text is None:
                    self._delete(path)
                    self._unloadable[path] = entry.signature
                else:
                    self._write(path, text, *entry.signature)
            # Only once written: if the write raises, the entry is still pending for whoever handles that.
            del self._pending[path]

    def _write(self, path: str, text: str, mtime: float, size: int) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self._connection.execute("SELECT digest FROM files WHERE path = ?", (path,)).fetchone()
        if row is not None and row[0] == digest:
            # Touched, not changed (a `touch`, a checkout of the same text): its passages and vectors stand.
            self._connection.execute("UPDATE files SET mtime = ?, size = ? WHERE path = ?", (mtime, size, path))
            return
        self._delete(path)
        chunks = chunk_document(path, text)
        last = self._connection.execute("SELECT rowid FROM passages ORDER BY rowid DESC LIMIT 1").fetchone()
        first_rowid = (last[0] if last else 0) + 1
        embedded = self._connection.execute(
            "SELECT 1 FROM files WHERE digest = ? AND embedded = 1 LIMIT 1", (digest,)
        ).fetchone()
        self._connection.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
            (path, mtime, size, digest, first_rowid, len(chunks), int(embedded is not None)),
        )
        self._connection.executemany(
            "INSERT INTO passages (rowid, path, location, text, digest) VALUES (?, ?, ?, ?, ?)",
            [(first_rowid + n, path, location, chunk, digest) for n, (location, chunk) in enumerate(chunks)],
        )

    def _delete(self, path: str) -> None:
        row = self._connection.execute(
            "SELECT digest, first_rowid, chunks FROM files WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return
        digest, first_rowid, chunks = row
        self._connection.execute("DELETE FROM files WHERE path = ?", (path,))
        self._connection.execute(
            "DELETE FROM passages WHERE rowid >= ? AND rowid < ?", (first_rowid, first_rowid + chunks)
        )
        still_held = self._connection.execute("SELECT 1 FROM files WHERE digest = ? LIMIT 1", (digest,)).fetchone()
        if self._collection is not None and chunks and still_held is None:
            self._collection.delete(ids=[f"{digest}:{n}" for n in range(chunks)])

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, limit: int) -> list[Passage]:
        """Up to *limit* passages matching *query*'s words, best BM25 score first. Never raises on syntax."""
        with self._lock:
            return [passage for _, passage in self._current(self._keyword_hits(query, limit))]

    def search(self, query: str, limit: int) -> list[Passage]:
        """Up to *limit* passages for *query*, best first: keyword and vector rankings fused, once vectors exist.

        Each half contributes its top ``max(4 * limit, 50)`` candidates. Scores are reciprocal-rank-fusion
        scores (keyword-only, they are BM25's). If the query cannot be embedded, the search is keyword-only
        and :meth:`status` says why; it never raises for that.
        """
        if limit <= 0 or not query.strip():
            return []
        candidates = max(4 * limit, 50)
        with self._lock:
            keyword = self._current(self._keyword_hits(query, candidates))
            use_vectors = self._collection is not None and self._collection.count() > 0
        if not use_vectors:
            return [passage for _, passage in keyword[:limit]]
        try:
            (query_vector,) = self._embedder.embed([query], input_type="query")
        except Exception as exc:
            message = f"Embedding the query failed, so search is keyword-only: {type(exc).__name__}: {exc}"
            logger.warning("%s", message)
            with self._lock:
                self._embedding_error = message
            return [passage for _, passage in keyword[:limit]]
        with self._lock:
            if not self._embedding_paused:
                self._embedding_error = None
            # Filtered again: a file may have become pending while the query was being embedded.
            keyword = self._current(keyword)
            vector = self._current(self._vector_hits(query_vector, candidates))
        found = dict(vector) | dict(keyword)
        fused = reciprocal_rank_fusion([[str(rowid) for rowid, _ in keyword], [str(rowid) for rowid, _ in vector]])
        return [dataclasses.replace(found[int(key)], score=score) for key, score in fused[:limit]]

    def _current(self, hits: list[tuple[int, Passage]]) -> list[tuple[int, Passage]]:
        """*hits* without passages of files awaiting re-indexing. Called with the lock held.

        A pending or failed file's manifest row still holds its previous text, which the folder no longer
        has. Until it is re-indexed the file counts as not indexed (``status().indexed`` already says so),
        so none of that text is returned as if the file held it.
        """
        return [(rowid, hit) for rowid, hit in hits if hit.path not in self._pending and hit.path not in self._failed]

    def _keyword_hits(self, query: str, limit: int) -> list[tuple[int, Passage]]:
        expression = _match_expression(query)
        if expression is None or limit <= 0:
            return []
        rows = self._connection.execute(
            "SELECT rowid, path, location, text, -bm25(passages), digest FROM passages "
            "WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(rowid, Passage(*passage)) for rowid, *passage in rows]

    def _vector_hits(self, query_vector: list[float], limit: int) -> list[tuple[int, Passage]]:
        """The *limit* nearest passages, resolved through the manifest so only current text comes back."""
        # Counted again: the last vectors may have been deleted while the query was being embedded.
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_vector], n_results=min(limit, count), include=["metadatas"]
        )
        hits: list[tuple[int, Passage]] = []
        for chunk_id, metadata in zip(result["ids"][0], result["metadatas"][0]):
            number = int(chunk_id.rsplit(":", 1)[1])
            holders = self._connection.execute(
                "SELECT first_rowid, chunks FROM files WHERE digest = ?", (metadata["digest"],)
            ).fetchall()
            for first_rowid, chunks in holders:
                if number >= chunks:
                    continue
                rowid = first_rowid + number
                row = self._connection.execute(
                    "SELECT path, location, text, digest FROM passages WHERE rowid = ?", (rowid,)
                ).fetchone()
                hits.append((rowid, Passage(row[0], row[1], row[2], 0.0, row[3])))
        return hits

    # ------------------------------------------------------------------
    # Status and lifetime
    # ------------------------------------------------------------------

    def status(self) -> IndexStatus:
        """Counts for the folder as of the last refresh. ``unreadable`` is the store's to fill in."""
        with self._lock:
            files = len(self._signatures) - len(self._unloadable)
            embedded = 0
            if self._embedder is not None:
                rows = self._connection.execute("SELECT path FROM files WHERE embedded = 1")
                embedded = sum(1 for (path,) in rows if path not in self._pending and path not in self._failed)
            problems = [problem for problem in (self._embedding_error, self._worker_error) if problem]
            if self._failed:
                path, (_, reason) = next(iter(self._failed.items()))
                others = f" and {len(self._failed) - 1} other file(s)" if len(self._failed) > 1 else ""
                problems.append(f"Could not index {path}{others}: {reason}")
            return IndexStatus(
                files=files,
                indexed=files - len(self._pending) - len(self._failed),
                embedded=embedded,
                unreadable={},
                error="; ".join(problems) or None,
                vectors=self._embedder is not None,
            )

    def wait_until_embedded(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for the background worker to finish; True if the index is complete.

        Returns early, with False, when the worker stops short of complete, as it does when embedding fails.
        """
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._worker is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return not self._closed and self.status().complete

    def close(self) -> None:
        """Stop the worker (after the file or batch it is on) and release the database. Idempotent."""
        with self._lock:
            self._closed = True
            worker = self._worker
        # Joined without the lock: the worker needs it to notice it should stop.
        if worker is not None:
            worker.join(timeout=_CLOSE_TIMEOUT_SECONDS)
        with self._lock:
            self._closer()

    # ------------------------------------------------------------------
    # The background worker
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Start the worker if there is work and none is running. Called with the lock held."""
        if self._worker is None and not self._closed and self._next_job() is not None:
            self._worker = threading.Thread(target=self._run, name=WORKER_THREAD_NAME, daemon=True)
            self._worker.start()

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    job = None if self._closed else self._next_job()
                    if job is None:
                        # Cleared under the same lock hold that found no work, so work added a moment
                        # later always finds no worker and starts one.
                        self._worker = None
                        self._idle.notify_all()
                        return
                job()
        except Exception as exc:
            # A failure outside any one file (the database, say). Raising would only reach threading.excepthook,
            # and carrying on could fail the same way forever, so the worker records it and stops; the next
            # refresh starts it again.
            message = f"Background indexing stopped: {type(exc).__name__}: {exc}"
            logger.exception("%s", message)
            with self._lock:
                self._worker_error = message
                self._worker = None
                self._idle.notify_all()
        except BaseException:
            with self._lock:
                self._worker = None
                self._idle.notify_all()
            raise

    def _next_job(self) -> Callable[[], None] | None:
        """The next file to index, else the next document to embed, else None. Called with the lock held."""
        if self._pending:
            path, entry = next(iter(self._pending.items()))
            return functools.partial(self._index_in_background, path, entry)
        if self._embedder is None or self._embedding_paused:
            return None
        rows = self._connection.execute("SELECT path, digest, first_rowid, chunks FROM files WHERE embedded = 0")
        for path, digest, first_rowid, chunks in rows:
            if path in self._pending:
                continue
            passages = self._connection.execute(
                "SELECT location, text FROM passages WHERE rowid >= ? AND rowid < ? ORDER BY rowid",
                (first_rowid, first_rowid + chunks),
            ).fetchall()
            return functools.partial(self._embed_document, digest, passages)
        return None

    def _index_in_background(self, path: str, entry: _PendingFile) -> None:
        try:
            self._store_loaded(path, entry, entry.load(path))
        except Exception as exc:
            # Inline, a failure raises to the caller. Here nobody would see it, and the file would be retried
            # forever, so it is recorded for status() and the file left out until it changes.
            logger.exception("Could not index %s", path)
            with self._lock:
                if self._pending.get(path) is entry:
                    del self._pending[path]
                    self._failed[path] = (entry.signature, f"{type(exc).__name__}: {exc}")

    def _embed_document(self, digest: str, passages: list[tuple[str, str]]) -> None:
        """Embed one document's passages, in batches, and store the vectors if its text is still current."""
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(passages), EMBEDDING_BATCH_SIZE):
                with self._lock:
                    if self._closed:
                        return
                batch = [text for _, text in passages[start : start + EMBEDDING_BATCH_SIZE]]
                vectors.extend(self._embedder.embed(batch, input_type="document"))
            with self._lock:
                if self._closed:
                    return
                held = self._connection.execute("SELECT 1 FROM files WHERE digest = ? LIMIT 1", (digest,)).fetchone()
                if held is None:
                    # Every file holding this text changed while it was embedded: these vectors are stale.
                    return
                if passages:
                    self._collection.upsert(
                        ids=[f"{digest}:{n}" for n in range(len(passages))],
                        embeddings=vectors,
                        metadatas=[{"digest": digest, "location": location} for location, _ in passages],
                    )
                with self._connection:
                    self._connection.execute("UPDATE files SET embedded = 1 WHERE digest = ?", (digest,))
                self._embedding_error = None
        except Exception as exc:
            message = f"Embedding failed, so search is keyword-only until it succeeds: {type(exc).__name__}: {exc}"
            with self._lock:
                if message != self._embedding_error:
                    logger.warning("%s", message)
                self._embedding_error = message
                self._embedding_paused = True
