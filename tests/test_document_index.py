"""Tests for DocumentStore's passage index and the store changes it relies on.

Mock-only: no model, no network. The keyword half is SQLite FTS5 and the vector half Chroma, each in memory
or in a temp directory, with :class:`FakeEmbedder` standing in for an embedding model. Background work is
awaited with ``wait_until_embedded`` and ``threading.Event``, never with a sleep.
"""

from __future__ import annotations

import builtins
import hashlib
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import aimu.memory._index as index_module
import aimu.memory.document_store as document_store_module
from aimu.memory import DocumentStore, IndexStatus, Passage, search_collections
from aimu.memory._index import PassageIndex, chunk_document
from aimu.models import EmbeddingSpec
from aimu.tools.builtin import make_document_tools
from tests.test_documents import minimal_pdf


def _all_files(root) -> set[str]:
    return {os.path.join(dirpath, name) for dirpath, _, names in os.walk(root) for name in names}


class FakeEmbedder:
    """A deterministic embedding client: bag-of-words vectors over a fixed vocabulary.

    Each group in ``VOCABULARY`` is one dimension, so words of a group are synonyms to it: "protein folding"
    and "structure prediction of polypeptides" embed to the same direction, while sharing no keyword. Words
    outside the vocabulary are ignored, and a text with none of its words lands on a last "other" dimension,
    so no vector is zero (cosine distance is undefined for one).

    Args:
        model_id: The ``spec.id`` an index records as its embedding model. ``spec.id`` stays settable.
        block: When given, a document embed waits for this event (bounded, so a failing test cannot hang).
            Queries are never blocked: a search embeds its query inline, and blocking it would block the
            search under test rather than the background build.
        error: When given, every embed raises it.
    """

    VOCABULARY = (
        ("protein", "folding", "structure", "prediction", "polypeptides"),
        ("glacier", "ice", "melt"),
        ("ribosome", "translation"),
        ("soil", "sampling"),
    )

    def __init__(self, model_id: str = "fake-1", *, block: threading.Event | None = None, error=None):
        self.spec = EmbeddingSpec(model_id)
        self.block = block
        self.error = error
        self.calls: list[tuple[str | None, int]] = []

    def embed(self, texts, *, input_type=None):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        self.calls.append((input_type, len(items)))
        if self.error is not None:
            raise self.error
        if self.block is not None and input_type == "document":
            self.block.wait(timeout=10)
        vectors = [self._vector(text) for text in items]
        return vectors[0] if single else vectors

    def _vector(self, text: str) -> list[float]:
        words = re.findall(r"\w+", text.lower())
        counts = [float(sum(word in group for word in words)) for group in self.VOCABULARY]
        if not any(counts):
            return [0.0] * len(self.VOCABULARY) + [1.0]
        norm = sum(count * count for count in counts) ** 0.5
        return [count / norm for count in counts] + [0.0]


def _index_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == index_module.WORKER_THREAD_NAME]


@pytest.fixture(autouse=True)
def no_leaked_index_workers():
    """Every test leaves no index worker behind: one exits once idle, and close() stops it."""
    yield
    for thread in _index_threads():
        thread.join(timeout=10)
    assert _index_threads() == []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunks_carry_heading_path():
    text = "---\ntags: [x]\n---\n# Methods\n## Sampling\nWe sampled soil.\n```\n# not a heading\n```\n"
    chunks = chunk_document("/n.md", text)
    assert chunks[0][0] == "Methods > Sampling"
    assert "tags:" not in "".join(c for _, c in chunks)
    assert all(location == "Methods > Sampling" for location, _ in chunks)
    assert "# not a heading" in "".join(c for _, c in chunks)


def test_text_before_any_heading_has_empty_location():
    chunks = chunk_document("/n.md", "Preamble.\n# Later\nBody.")
    assert chunks == [("", "Preamble."), ("Later", "Body.")]


def test_sibling_heading_replaces_its_level():
    chunks = chunk_document("/n.md", "# A\n## B\nb\n## C\nc\n# D\nd")
    assert [location for location, _ in chunks] == ["A > B", "A > C", "D"]


def test_long_section_is_split_with_the_same_location():
    text = "# Big\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    chunks = chunk_document("/n.md", text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(location == "Big" and len(chunk) <= 400 for location, chunk in chunks)


def test_pdf_page_headings_become_page_locations_and_hash_lines_stay_text():
    text = "## Page 1\nIntro\n## Page 3\n# of samples was 12\n"
    assert chunk_document("/p.pdf", text) == [("page 1", "Intro"), ("page 3", "# of samples was 12")]


def test_pdf_chunks_carry_page(tmp_path):
    (tmp_path / "p.pdf").write_bytes(minimal_pdf(["intro", "CRISPR-Cas13 knockdown"]))
    store = DocumentStore(persist_path=str(tmp_path))
    hits = store.search_passages("Cas13")
    assert hits[0].path == "/p.pdf" and hits[0].location == "page 2"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query", ["CRISPR-Cas13", '"quoted', "(a OR b)", "*", "key:value", "", "_", "NEAR(a b)", "a AND NOT b", "^x"]
)
def test_query_syntax_never_raises(tmp_path, query):
    (tmp_path / "a.md").write_text("CRISPR-Cas13 a b key value")
    DocumentStore(persist_path=str(tmp_path)).search_passages(query)


def test_hyphenated_query_matches_its_parts(tmp_path):
    (tmp_path / "a.md").write_text("Results of the CRISPR-Cas13 knockdown.")
    hits = DocumentStore(persist_path=str(tmp_path)).search_passages("CRISPR-Cas13")
    assert [hit.path for hit in hits] == ["/a.md"]


def test_stopword_only_query_returns_nothing(tmp_path):
    (tmp_path / "a.md").write_text("the cat is on the mat")
    assert DocumentStore(persist_path=str(tmp_path)).search_passages("the is on") == []


def test_passage_fields_and_ranking(tmp_path):
    (tmp_path / "a.md").write_text("# Topic\nribosome ribosome ribosome translation")
    (tmp_path / "b.md").write_text("one mention of ribosome among many other words here")
    (tmp_path / "c.md").write_text("nothing relevant")
    hits = DocumentStore(persist_path=str(tmp_path)).search_passages("ribosome")
    assert [hit.path for hit in hits] == ["/a.md", "/b.md"]
    assert isinstance(hits[0], Passage) and hits[0].location == "Topic"
    assert hits[0].score > hits[1].score
    assert hits[0].digest == hashlib.sha256((tmp_path / "a.md").read_text().encode()).hexdigest()


def test_n_results_limits_hits(tmp_path):
    for i in range(12):
        (tmp_path / f"n{i}.md").write_text(f"enzyme note {i}")
    store = DocumentStore(persist_path=str(tmp_path))
    assert len(store.search_passages("enzyme")) == 8
    assert len(store.search_passages("enzyme", n_results=3)) == 3


def test_non_ascii_query_matches_without_diacritics(tmp_path):
    (tmp_path / "a.md").write_text("Mon résumé est prêt.")
    assert DocumentStore(persist_path=str(tmp_path)).search_passages("resume")[0].path == "/a.md"


# ---------------------------------------------------------------------------
# What is indexed
# ---------------------------------------------------------------------------


def test_oversize_file_is_unreadable_not_indexed(tmp_path):
    (tmp_path / "huge.md").write_text("x" * 50)
    store = DocumentStore(persist_path=str(tmp_path), max_file_bytes=10)
    assert "too large" in store.unreadable()["/huge.md"] and store.search_passages("x") == []
    assert store.unreadable()["/huge.md"] == "too large to index (50 bytes; the limit is 10)"


def test_symlink_escaping_root_is_skipped_and_unwritable(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "s.md").write_text("secret")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "f.md").symlink_to(outside / "s.md")
    store = DocumentStore(persist_path=str(root))
    assert store.list_paths() == [] and store.search_passages("secret") == []
    with pytest.raises(ValueError, match="outside"):
        store.write("/link/new.md", "x")
    with pytest.raises(ValueError, match="outside"):
        store.edit("/f.md", "secret", "x")
    with pytest.raises(ValueError, match="outside"):
        store.delete("/link/s.md")
    with pytest.raises(KeyError):
        store.read("/link/s.md")
    assert (outside / "s.md").read_text() == "secret"
    assert not (outside / "new.md").exists()


def test_symlink_inside_root_is_written_through(tmp_path):
    (tmp_path / "real.md").write_text("one")
    (tmp_path / "alias.md").symlink_to(tmp_path / "real.md")
    store = DocumentStore(persist_path=str(tmp_path))
    store.write("/alias.md", "two")
    assert (tmp_path / "alias.md").is_symlink() and (tmp_path / "real.md").read_text() == "two"


def test_write_is_atomic(tmp_path, monkeypatch):
    store = DocumentStore(persist_path=str(tmp_path))
    store.write("/a.md", "original")

    def failing_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        store.write("/a.md", "replacement")
    assert (tmp_path / "a.md").read_text() == "original"
    assert os.listdir(tmp_path) == ["a.md"]


def test_write_keeps_existing_file_mode(tmp_path):
    (tmp_path / "a.md").write_text("x")
    os.chmod(tmp_path / "a.md", 0o640)
    DocumentStore(persist_path=str(tmp_path)).write("/a.md", "y")
    assert os.stat(tmp_path / "a.md").st_mode & 0o777 == 0o640


def test_index_path_inside_document_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outside the document folder"):
        DocumentStore(persist_path=str(tmp_path), index_path=str(tmp_path / ".index"))


# ---------------------------------------------------------------------------
# Keeping the index in step with the folder
# ---------------------------------------------------------------------------


def test_external_edit_and_delete_are_seen(tmp_path):
    note = tmp_path / "Notes" / "Résumé draft.md"
    note.parent.mkdir()
    note.write_text("old topic")
    store = DocumentStore(persist_path=str(tmp_path), index_path=str(tmp_path.parent / "idx"))
    assert store.search_passages("old")[0].path == "/Notes/Résumé draft.md"
    note.write_text("new topic")
    os.utime(note, (time.time() + 5, time.time() + 5))
    assert store.search_passages("old") == [] and store.search_passages("new")
    assert store.read(store.search_passages("new")[0].path) == "new topic"
    note.unlink()
    assert store.search_passages("new") == [] and store.index_status().files == 0
    store.close()


def test_index_persists_and_skips_unchanged_files(tmp_path, monkeypatch):
    folder, index_dir = tmp_path / "docs", tmp_path / "index"
    folder.mkdir()
    (folder / "a.md").write_text("photosynthesis notes")
    first = DocumentStore(persist_path=str(folder), index_path=str(index_dir))
    assert first.search_passages("photosynthesis")
    first.close()
    assert (index_dir / "index.sqlite3").exists()

    calls = []
    real_chunk_document = index_module.chunk_document

    def counting_chunk_document(*args, **kwargs):
        calls.append(args[0])
        return real_chunk_document(*args, **kwargs)

    monkeypatch.setattr(index_module, "chunk_document", counting_chunk_document)
    second = DocumentStore(persist_path=str(folder), index_path=str(index_dir))
    assert second.search_passages("photosynthesis")[0].path == "/a.md"
    assert calls == []

    (folder / "a.md").write_text("chlorophyll notes, longer")
    assert second.search_passages("chlorophyll")
    assert calls == ["/a.md"]
    second.close()


def test_writes_update_index(tmp_path):
    store = DocumentStore(persist_path=str(tmp_path / "d"), index_path=str(tmp_path / "i"))
    store.write("/x.md", "mitochondria")
    assert store.search_passages("mitochondria")
    store.delete("/x.md")
    assert store.search_passages("mitochondria") == []


def test_write_and_edit_after_index_exists_are_searchable_at_once(tmp_path):
    store = DocumentStore(persist_path=str(tmp_path / "d"), index_path=str(tmp_path / "i"))
    assert store.search_passages("anything") == []
    store.write("/x.md", "alpha")
    store.edit("/x.md", "alpha", "gamma")
    assert store._index.keyword_search("gamma", 8)[0].path == "/x.md"
    assert store._index.keyword_search("alpha", 8) == []


def test_ephemeral_store_searches_in_memory():
    store = DocumentStore()
    store.write("/a.md", "ribosome")
    assert store.search_passages("ribosome")[0].path == "/a.md"
    store.edit("/a.md", "ribosome", "lysosome")
    assert store.search_passages("ribosome") == [] and store.search_passages("lysosome")
    store.delete("/a.md")
    assert store.search_passages("lysosome") == [] and store.index_status().files == 0


def test_index_never_writes_under_the_document_folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("glacier melt")
    before = _all_files(folder)
    for index_path in (None, str(tmp_path / "index")):
        store = DocumentStore(persist_path=str(folder), index_path=index_path)
        assert store.search_passages("glacier")
        store.index_status()
        store.close()
    assert _all_files(folder) == before


def test_file_vanishing_between_walk_and_read_is_removed_not_raised():
    index = PassageIndex(None)
    index.refresh({"/a.md": (1.0, 5)}, lambda path: "zebra")
    assert index.keyword_search("zebra", 8)
    index.refresh({"/a.md": (2.0, 5)}, lambda path: None)
    assert index.keyword_search("zebra", 8) == [] and index.status().files == 0
    index.close()


def test_store_tolerates_a_file_gone_before_its_read(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("kept")
    store = DocumentStore(persist_path=str(tmp_path))
    real_metadata = store._file_metadata

    def metadata_with_a_ghost():
        files = real_metadata()
        files["/gone.md"] = (1.0, 1)
        return files

    monkeypatch.setattr(store, "_file_metadata", metadata_with_a_ghost)
    assert store.search_passages("kept")[0].path == "/a.md"
    status = store.index_status()
    assert status.files == 1 and "/gone.md" not in status.unreadable


def test_index_status_reports_counts_and_unreadable(tmp_path):
    (tmp_path / "a.md").write_text("one")
    (tmp_path / "b.md").write_text("two")
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00")
    status = DocumentStore(persist_path=str(tmp_path)).index_status()
    assert isinstance(status, IndexStatus)
    assert (status.files, status.indexed, status.embedded) == (2, 2, 0)
    assert status.unreadable == {"/bin.md": "not UTF-8 text"} and status.error is None
    assert status.complete


def test_plain_store_does_not_build_an_index(tmp_path):
    store = DocumentStore(persist_path=str(tmp_path))
    store.write("/a.md", "x")
    store.search_full_text("x")
    assert store._index is None


# ---------------------------------------------------------------------------
# Carried from PDF reading: the extraction cache and the OSError reason
# ---------------------------------------------------------------------------


def test_unchanged_pdf_is_extracted_once(tmp_path, monkeypatch):
    (tmp_path / "p.pdf").write_bytes(minimal_pdf(["alpha"]))
    calls = []
    real_pdf_to_markdown = document_store_module.pdf_to_markdown

    def counting_pdf_to_markdown(data):
        calls.append(len(data))
        return real_pdf_to_markdown(data)

    monkeypatch.setattr(document_store_module, "pdf_to_markdown", counting_pdf_to_markdown)
    store = DocumentStore(persist_path=str(tmp_path))
    assert store.list_paths() == ["/p.pdf"]
    assert store.list_paths() == ["/p.pdf"]
    assert len(calls) == 1

    (tmp_path / "p.pdf").write_bytes(minimal_pdf(["alpha", "beta"]))
    assert "beta" in store.read("/p.pdf")
    assert len(calls) == 2

    (tmp_path / "p.pdf").unlink()
    assert store.list_paths() == []
    assert store._extractions == {}


def test_os_error_has_its_own_reason_and_no_export_advice(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read a file with no permissions")
    locked = tmp_path / "locked.md"
    locked.write_text("x")
    os.chmod(locked, 0)
    try:
        store = DocumentStore(persist_path=str(tmp_path))
        reason = store.unreadable()["/locked.md"]
        assert reason != "not UTF-8 text" and "Permission denied" in reason
        list_documents = make_document_tools(store)[3]
        listing = list_documents()
        assert "/locked.md" in listing and "export" not in listing
    finally:
        os.chmod(locked, 0o600)


@pytest.mark.parametrize("embedder_type", [None, FakeEmbedder], ids=["keyword", "hybrid"])
def test_concurrent_writes_and_searches_from_threads(tmp_path, embedder_type):
    embedder = embedder_type() if embedder_type else None
    store = DocumentStore(persist_path=str(tmp_path / "d"), index_path=str(tmp_path / "i"), embedding_client=embedder)

    def work(number: int) -> list[Passage]:
        store.write(f"/n{number}.md", f"topic{number} shared")
        return store.search_passages(f"topic{number}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(24)))
    assert all(hits and hits[0].path == f"/n{number}.md" for number, hits in enumerate(results))
    assert store.index_status().files == 24
    assert store._index.wait_until_embedded(10)
    store.close()


# ---------------------------------------------------------------------------
# Rulings carried into the vector half: robustness of the store underneath it
# ---------------------------------------------------------------------------


def test_any_pdf_extraction_failure_is_unreadable_not_raised(tmp_path, monkeypatch):
    (tmp_path / "p.pdf").write_bytes(minimal_pdf(["alpha"]))
    (tmp_path / "a.md").write_text("alpha notes")

    def broken_extractor(data):
        raise KeyError("/Root")

    monkeypatch.setattr(document_store_module, "pdf_to_markdown", broken_extractor)
    store = DocumentStore(persist_path=str(tmp_path))
    assert [hit.path for hit in store.search_passages("alpha")] == ["/a.md"]
    assert "KeyError" in store.unreadable()["/p.pdf"]
    assert "KeyError" in store.index_status().unreadable["/p.pdf"]


def test_write_and_delete_drop_the_extraction_cache(tmp_path):
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "p.pdf").write_bytes(minimal_pdf(["alpha"]))
    store = DocumentStore(persist_path=str(tmp_path))
    store.unreadable()
    assert {str(tmp_path / "bin.md"), str(tmp_path / "p.pdf")} <= store._extractions.keys()
    store.write("/bin.md", "now text")
    store.delete("/p.pdf")
    assert store._extractions == {}
    assert store.read("/bin.md") == "now text" and store.unreadable() == {}


@pytest.mark.parametrize("persist", [True, False])
def test_write_over_max_file_bytes_is_refused(tmp_path, persist):
    store = DocumentStore(persist_path=str(tmp_path) if persist else None, max_file_bytes=10)
    store.write("/ok.md", "x" * 10)
    with pytest.raises(ValueError, match="max_file_bytes"):
        store.write("/big.md", "x" * 11)
    # Six characters, twelve bytes: the limit is on the encoded file, not the string.
    with pytest.raises(ValueError, match="12 bytes"):
        store.write("/accented.md", "é" * 6)
    assert store.list_paths() == ["/ok.md"]


def test_index_from_an_older_schema_is_rebuilt(tmp_path):
    folder, index_dir = tmp_path / "docs", tmp_path / "index"
    folder.mkdir()
    index_dir.mkdir()
    (folder / "a.md").write_text("photosynthesis notes")
    # The keyword-only index's schema, which set no user_version, holding a row for a file that is not there.
    old = sqlite3.connect(index_dir / "index.sqlite3")
    old.execute("CREATE TABLE files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, digest TEXT)")
    old.execute("CREATE VIRTUAL TABLE passages USING fts5(path UNINDEXED, location UNINDEXED, text, digest UNINDEXED)")
    old.execute("INSERT INTO files VALUES ('/ghost.md', 1.0, 5, 'x')")
    old.execute("INSERT INTO passages VALUES ('/ghost.md', '', 'photosynthesis ghost', 'x')")
    old.commit()
    old.close()

    store = DocumentStore(persist_path=str(folder), index_path=str(index_dir))
    assert [hit.path for hit in store.search_passages("photosynthesis")] == ["/a.md"]
    store.close()
    check = sqlite3.connect(index_dir / "index.sqlite3")
    try:
        assert check.execute("PRAGMA user_version").fetchone() == (index_module.SCHEMA_VERSION,)
        assert index_module.SCHEMA_VERSION == 1
    finally:
        check.close()


def test_complete_requires_vectors_only_when_enabled():
    keyword_only = IndexStatus(files=2, indexed=2, embedded=0, unreadable={})
    assert keyword_only.complete and keyword_only.vectors is False
    assert not IndexStatus(files=2, indexed=2, embedded=1, unreadable={}, vectors=True).complete
    assert IndexStatus(files=2, indexed=2, embedded=2, unreadable={}, vectors=True).complete
    assert not IndexStatus(files=3, indexed=2, embedded=2, unreadable={}, vectors=True).complete


def test_a_permanently_failed_file_leaves_the_status_complete():
    # A failed file is reported through problems and waits for a change, so it is not work still to come.
    assert IndexStatus(files=3, indexed=2, embedded=2, unreadable={}, failed=1, vectors=True).complete
    assert IndexStatus(files=3, indexed=2, embedded=0, unreadable={}, failed=1).complete
    assert not IndexStatus(files=3, indexed=1, embedded=1, unreadable={}, failed=1, vectors=True).complete
    assert not IndexStatus(files=3, indexed=2, embedded=1, unreadable={}, failed=1, vectors=True).complete


def test_first_build_over_the_inline_budget_finishes_in_the_background(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.05)
    for number in range(40):
        (tmp_path / f"n{number}.md").write_text(f"shared note {number}")
    store = DocumentStore(persist_path=str(tmp_path))
    real_load = store._load

    def slow_load(path):
        time.sleep(0.025)
        return real_load(path)

    monkeypatch.setattr(store, "_load", slow_load)
    # Loading every file takes a second; the search returns after about the inline budget.
    started = time.monotonic()
    hits = store.search_passages("shared")
    assert time.monotonic() - started < 0.5
    assert hits
    early = store._index.status()
    assert early.files == 40 and early.indexed < early.files and not early.complete

    # The worker takes the store's lock one file at a time, so a read never waits for the whole build.
    started = time.monotonic()
    assert store.read("/n39.md") == "shared note 39"
    assert time.monotonic() - started < 0.5

    assert store._index.wait_until_embedded(10)
    status = store.index_status()
    assert (status.files, status.indexed) == (40, 40) and status.complete
    assert len(store.search_passages("shared", n_results=50)) == 40
    store.close()


# ---------------------------------------------------------------------------
# Vector half and fusion
# ---------------------------------------------------------------------------


def _vector_store(tmp_path, embedder, **files):
    folder = tmp_path / "docs"
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / f"{name}.md").write_text(text)
    return DocumentStore(persist_path=str(folder), index_path=str(tmp_path / "index"), embedding_client=embedder)


def test_vector_search_finds_paraphrase(tmp_path):
    store = _vector_store(
        tmp_path, FakeEmbedder(), a="structure prediction of polypeptides", b="glacier melt rates", c="soil sampling"
    )
    store.index_status()
    assert store._index.wait_until_embedded(5)
    assert store._index.keyword_search("protein folding", 8) == []
    hits = store.search_passages("protein folding")
    assert hits[0].path == "/a.md" and hits[0].text == "structure prediction of polypeptides"
    assert hits[0].score > hits[1].score
    store.close()


def test_fusion_ranks_a_passage_both_halves_found_first(tmp_path):
    store = _vector_store(
        tmp_path,
        FakeEmbedder(),
        a="glacier ice",
        b="glacier glacier glacier soil sampling sampling sampling",
        d="glacier soil melt notes here",
    )
    store.index_status()
    assert store._index.wait_until_embedded(5)
    # Keyword ranks b, a, d (three "glacier"s first); vectors rank a, d, b (a is all ice words, b mostly soil).
    assert [hit.path for hit in store._index.keyword_search("glacier", 8)] == ["/b.md", "/a.md", "/d.md"]
    # a is 2nd and 1st, b 1st and 3rd: reciprocal ranks favour the passage both rankings placed high.
    assert [hit.path for hit in store.search_passages("glacier")] == ["/a.md", "/b.md", "/d.md"]
    store.close()


def test_first_build_embeds_in_background_and_reports_status(tmp_path):
    release = threading.Event()
    embedder = FakeEmbedder(block=release)
    store = _vector_store(tmp_path, embedder, a="ribosome notes", b="glacier notes")
    try:
        hits = store.search_passages("ribosome")
        assert [hit.path for hit in hits] == ["/a.md"]
        status = store.index_status()
        assert status.vectors and status.files == status.indexed == 2
        assert status.embedded < status.indexed and not status.complete and status.error is None
        assert not store._index.wait_until_embedded(0.05)
    finally:
        release.set()
    assert store._index.wait_until_embedded(5)
    status = store.index_status()
    assert status.embedded == status.indexed == 2 and status.complete
    assert all(
        input_type == "document" and size <= index_module.EMBEDDING_BATCH_SIZE for input_type, size in embedder.calls
    )
    store.close()


def test_embedding_batches_hold_at_most_the_batch_size(tmp_path):
    embedder = FakeEmbedder()
    text = "\n\n".join(f"# Section {number}\nglacier paragraph {number}" for number in range(70))
    store = _vector_store(tmp_path, embedder, big=text)
    store.index_status()
    assert store._index.wait_until_embedded(5)
    assert [size for _, size in embedder.calls] == [32, 32, 6]
    store.close()


def test_stale_vector_hits_are_dropped(tmp_path):
    embedder = FakeEmbedder()
    store = _vector_store(tmp_path, embedder, a="glacier melt", b="soil sampling")
    store.index_status()
    assert store._index.wait_until_embedded(5)
    index = store._index
    # A vector for text no file holds (left by a crash, or by a write this index never saw), naming a real
    # path and lying nearer the query than any current passage. Injected with no refresh after it, so
    # nothing but the hit's resolution through the manifest stands between it and the results.
    gone = hashlib.sha256(b"protein folding, since deleted").hexdigest()
    query_vector = embedder.embed("protein folding", input_type="query")
    index._collection.upsert(
        ids=[f"{gone}:0"], embeddings=[query_vector], metadatas=[{"digest": gone, "location": "", "path": "/a.md"}]
    )
    with index._lock:
        vector_hits = index._vector_hits(query_vector, 50)
    assert index._collection.get(ids=[f"{gone}:0"])["ids"] == [f"{gone}:0"]
    assert sorted(hit.path for _, hit in vector_hits) == ["/a.md", "/b.md"]
    hits = index.search("protein folding", 8)
    assert hits and gone not in {hit.digest for hit in hits}
    folder = tmp_path / "docs"
    for hit in hits:
        assert hit.digest == hashlib.sha256((folder / hit.path.lstrip("/")).read_text().encode()).hexdigest()
    store.close()


def test_a_pending_file_hides_its_superseded_text(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.0)
    store = _vector_store(tmp_path, FakeEmbedder(), a="protein folding notes", b="glacier melt")
    store.index_status()
    assert store._index.wait_until_embedded(5)

    # Hold the worker before it reads the rewritten file, outside the store's lock so the search can run.
    release = threading.Event()
    real_load = store._load

    def gated_load(path):
        release.wait(10)
        return real_load(path)

    monkeypatch.setattr(store, "_load", gated_load)
    note = tmp_path / "docs" / "a.md"
    note.write_text("soil sampling notes")
    os.utime(note, (time.time() + 5, time.time() + 5))
    try:
        # The manifest still holds a's old text, matched by both the keyword ("protein") and the vector half.
        hits = store.search_passages("protein folding")
        assert "/a.md" not in {hit.path for hit in hits}
        assert store._index.keyword_search("protein", 8) == []
        status = store._index.status()
        assert (status.files, status.indexed) == (2, 1)
    finally:
        release.set()
    assert store._index.wait_until_embedded(5)
    hits = store.search_passages("soil")
    assert hits[0].path == "/a.md" and hits[0].text == "soil sampling notes"
    store.close()


def test_a_search_during_a_background_build_does_not_index_inline_again(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.3)
    for number in range(40):
        (tmp_path / f"n{number}.md").write_text(f"shared note {number}")
    store = DocumentStore(persist_path=str(tmp_path))
    real_load = store._load
    loaded_by: list[str] = []

    def slow_load(path):
        loaded_by.append(threading.current_thread().name)
        time.sleep(0.025)
        return real_load(path)

    monkeypatch.setattr(store, "_load", slow_load)
    store.search_passages("shared")
    assert store._index._worker is not None
    # The worker has about 0.7 s of files left. Another search hands changed files to it and loads none on
    # the searching thread, rather than spending a second inline budget.
    caller = threading.current_thread().name
    inline_before = loaded_by.count(caller)
    store.search_passages("shared")
    assert loaded_by.count(caller) == inline_before
    assert store._index.wait_until_embedded(10)
    assert store.index_status().complete
    store.close()


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_worker_failure_on_one_file_is_recorded_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.0)
    (tmp_path / "bad.md").write_text("poison")
    (tmp_path / "good.md").write_text("fine words")
    chunked = []
    real_chunk_document = index_module.chunk_document

    def chunk_document(path, text, **kwargs):
        chunked.append(path)
        if "poison" in text:
            raise sqlite3.OperationalError("disk I/O error")
        return real_chunk_document(path, text, **kwargs)

    monkeypatch.setattr(index_module, "chunk_document", chunk_document)
    store = DocumentStore(persist_path=str(tmp_path))
    store.index_status()
    # Complete apart from the failure: bad.md waits for a change, not for the worker.
    assert store._index.wait_until_embedded(5)
    status = store.index_status()
    assert (status.files, status.indexed, status.failed) == (2, 1, 1) and status.complete
    assert "/bad.md" in status.error and "OperationalError: disk I/O error" in status.error
    assert store._index.wait_until_embedded(5)
    assert chunked.count("/bad.md") == 1
    assert [hit.path for hit in store.search_passages("fine")] == ["/good.md"]

    bad = tmp_path / "bad.md"
    bad.write_text("cured text")
    os.utime(bad, (time.time() + 5, time.time() + 5))
    store.index_status()
    assert store._index.wait_until_embedded(5)
    status = store.index_status()
    assert status.complete and status.error is None and status.failed == 0
    store.close()


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_worker_failure_outside_a_file_stops_the_worker_and_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.0)
    (tmp_path / "a.md").write_text("alpha")
    real_next_job = PassageIndex._next_job
    calls = []

    def next_job(self):
        calls.append(1)
        # The first call is refresh deciding to start the worker; the second is the worker's own.
        if len(calls) == 2:
            raise sqlite3.OperationalError("database is locked")
        return real_next_job(self)

    monkeypatch.setattr(PassageIndex, "_next_job", next_job)
    store = DocumentStore(persist_path=str(tmp_path))
    store.index_status()
    assert not store._index.wait_until_embedded(5)
    assert "OperationalError: database is locked" in store._index.status().error
    assert store._index._worker is None

    store.index_status()
    assert store._index.wait_until_embedded(5)
    assert store.index_status().error is None
    store.close()


def test_rewritten_file_loses_its_old_vectors(tmp_path):
    store = _vector_store(tmp_path, FakeEmbedder(), a="protein folding")
    store.index_status()
    assert store._index.wait_until_embedded(5)
    store.write("/a.md", "glacier melt")
    assert store._index.wait_until_embedded(5)
    digests = {metadata["digest"] for metadata in store._index._collection.get()["metadatas"]}
    assert digests == {store.search_passages("glacier")[0].digest}
    store.delete("/a.md")
    assert store._index._collection.count() == 0
    store.close()


def test_a_copy_reuses_the_vectors_of_its_original(tmp_path):
    embedder = FakeEmbedder()
    store = _vector_store(tmp_path, embedder, a="protein folding")
    store.index_status()
    assert store._index.wait_until_embedded(5)
    calls = len(embedder.calls)
    store.write("/copy.md", "protein folding")
    assert store._index.wait_until_embedded(5)
    assert len(embedder.calls) == calls
    hits = store.search_passages("polypeptides")
    assert {hit.path for hit in hits} == {"/a.md", "/copy.md"}
    store.delete("/a.md")
    assert [hit.path for hit in store.search_passages("polypeptides")] == ["/copy.md"]
    store.close()


def test_changing_embedding_model_rebuilds_vectors(tmp_path):
    first = _vector_store(tmp_path, FakeEmbedder("fake-1"), a="protein folding", b="glacier melt")
    first.index_status()
    assert first._index.wait_until_embedded(5)
    first.close()

    release = threading.Event()
    second = _vector_store(tmp_path, FakeEmbedder("fake-2", block=release))
    try:
        status = second.index_status()
        assert status.indexed == 2 and status.embedded == 0
        assert second._index._collection.count() == 0
    finally:
        release.set()
    assert second._index.wait_until_embedded(5)
    assert second.index_status().embedded == 2
    second.close()

    third = _vector_store(tmp_path, FakeEmbedder("fake-2"))
    assert third.index_status().embedded == 2
    third.close()


def test_reopening_without_an_embedder_forgets_the_vectors(tmp_path):
    first = _vector_store(tmp_path, FakeEmbedder("fake-1"), a="protein folding")
    first.index_status()
    assert first._index.wait_until_embedded(5)
    first.close()
    keyword_only = _vector_store(tmp_path, None)
    status = keyword_only.index_status()
    assert (status.vectors, status.embedded) == (False, 0) and status.complete
    keyword_only.close()
    # The vectors were never deleted from disk, but the manifest no longer vouches for them: rebuild.
    release = threading.Event()
    again = _vector_store(tmp_path, FakeEmbedder("fake-1", block=release))
    try:
        assert again.index_status().embedded == 0
    finally:
        release.set()
    assert again._index.wait_until_embedded(5)
    again.close()


def test_ephemeral_vector_indexes_do_not_share_vectors():
    one, two = DocumentStore(embedding_client=FakeEmbedder()), DocumentStore(embedding_client=FakeEmbedder())
    one.write("/a.md", "protein folding")
    two.write("/b.md", "glacier melt")
    for store in (one, two):
        store.index_status()
        assert store._index.wait_until_embedded(5)
    assert [hit.path for hit in one.search_passages("ice")] == ["/a.md"]
    assert [hit.path for hit in two.search_passages("polypeptides")] == ["/b.md"]
    one.close()
    two.close()


def test_missing_chromadb_names_the_extra(monkeypatch):
    real_import = builtins.__import__

    def without_chromadb(name, *args, **kwargs):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("No module named 'chromadb'", name="chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_chromadb)
    store = DocumentStore(embedding_client=FakeEmbedder())
    store.write("/a.md", "ribosome")
    with pytest.raises(ImportError, match=r"aimu\[memory\]"):
        store.search_passages("ribosome")
    keyword_only = DocumentStore()
    keyword_only.write("/a.md", "ribosome")
    assert keyword_only.search_passages("ribosome")[0].path == "/a.md"


def test_embedding_failure_degrades_to_keyword_with_error(tmp_path, caplog):
    embedder = FakeEmbedder(error=RuntimeError("model download failed"))
    store = _vector_store(tmp_path, embedder, a="ribosome notes")
    assert [hit.path for hit in store.search_passages("ribosome")] == ["/a.md"]
    assert not store._index.wait_until_embedded(5)
    status = store.index_status()
    assert "model download failed" in status.error and not status.complete
    assert "model download failed" in caplog.text

    # The next refresh tries again, and a success clears the error.
    embedder.error = None
    store.index_status()
    assert store._index.wait_until_embedded(5)
    status = store.index_status()
    assert status.error is None and status.complete
    store.close()


def test_query_embedding_failure_falls_back_to_keyword(tmp_path):
    embedder = FakeEmbedder()
    store = _vector_store(tmp_path, embedder, a="ribosome notes")
    store.index_status()
    assert store._index.wait_until_embedded(5)
    embedder.error = RuntimeError("embedding server went away")
    assert [hit.path for hit in store.search_passages("ribosome")] == ["/a.md"]
    assert "embedding server went away" in store._index.status().error
    store.close()


def test_close_stops_the_worker_mid_build(tmp_path):
    release = threading.Event()
    store = _vector_store(tmp_path, FakeEmbedder(block=release), a="ribosome", b="glacier")
    store.index_status()
    index = store._index
    assert _index_threads()
    closer = threading.Thread(target=store.close)
    closer.start()
    release.set()
    closer.join(timeout=10)
    assert not closer.is_alive() and _index_threads() == []
    assert index._worker is None


def _store_with(**documents: str) -> DocumentStore:
    store = DocumentStore()
    for path, text in documents.items():
        store.write(f"/{path}.md", text)
    return store


def test_cross_store_results_are_labeled_and_fused():
    stores = {
        "personal": _store_with(glaciers="# Glaciers\n\nGlacier ice melt is accelerating."),
        "work": _store_with(ribosomes="# Ribosomes\n\nRibosome translation of glacier proteins."),
    }
    results = search_collections(stores, "glacier")
    assert {result.collection for result in results} == {"personal", "work"}
    by_collection = {result.collection: result for result in results}
    assert by_collection["personal"].passage.path == "/glaciers.md"
    assert by_collection["work"].passage.path == "/ribosomes.md"
    assert all(result.also_in == () for result in results)
    assert len(search_collections(stores, "glacier", n_results=1)) == 1


def test_empty_stores_return_nothing():
    assert search_collections({}, "glacier") == []


def test_same_file_in_two_stores_appears_once():
    text = "# Paper\n\nProtein folding results."
    stores = {
        "personal": _store_with(paper=text),
        "work": _store_with(paper=text),
    }
    results = search_collections(stores, "protein folding")
    assert len(results) == 1
    only = results[0]
    other = "work" if only.collection == "personal" else "personal"
    assert only.also_in == ((other, "/paper.md"),)


def test_same_text_in_a_different_location_is_not_collapsed():
    stores = {
        "personal": _store_with(paper="# Intro\n\nProtein folding results."),
        "work": _store_with(paper="# Intro\n\nProtein folding results.", extra="# Other\n\nProtein folding too."),
    }
    results = search_collections(stores, "protein folding")
    assert {(r.collection, r.passage.path) for r in results} >= {("work", "/extra.md")}
    assert len([r for r in results if r.passage.path == "/paper.md"]) == 1


def test_rerank_reorders_with_stubbed_cross_encoder(monkeypatch):
    stores = {
        "a": _store_with(short="# Short\n\nglacier glacier glacier glacier"),
        "b": _store_with(long="# Long\n\nglacier " + "and so on " * 5),
    }
    plain = search_collections(stores, "glacier")
    assert plain[0].passage.path == "/short.md"

    calls = []

    def longest_first(query, documents, *, model=None, top_n=None):
        calls.append((query, list(documents)))
        return sorted(documents, key=len, reverse=True)

    monkeypatch.setattr(document_store_module, "rerank_texts", longest_first)
    reranked = search_collections(stores, "glacier", rerank=True)
    assert [r.passage.path for r in reranked] == ["/long.md", "/short.md"]
    assert calls[0][0] == "glacier"


def test_rerank_keeps_distinct_passages_with_identical_text(monkeypatch):
    stores = {
        "a": _store_with(one="# One\n\nglacier ice", two="# Two\n\nglacier ice"),
    }
    monkeypatch.setattr(document_store_module, "rerank_texts", lambda query, documents, **kw: list(documents))
    results = search_collections(stores, "glacier", rerank=True)
    assert sorted(r.passage.path for r in results) == ["/one.md", "/two.md"]


def test_two_chunks_of_one_section_in_one_store_both_appear():
    sentences = " ".join(f"Glacier ice melt observation number {n} was recorded." for n in range(60))
    stores = {"a": _store_with(long=f"# Long\n\n{sentences}")}
    results = search_collections(stores, "glacier", n_results=8)
    assert len(results) >= 2
    assert len({r.passage.text for r in results}) == len(results)
    assert len({r.passage.location for r in results}) == 1


# ---------------------------------------------------------------------------
# Final review: vault scale, lock discipline, and honest status
# ---------------------------------------------------------------------------


def test_an_inline_failure_on_one_file_is_recorded_not_raised(tmp_path, monkeypatch):
    # The default inline budget, so the failure happens in the search's own pass rather than the worker's.
    (tmp_path / "bad.md").write_text("poison")
    (tmp_path / "good.md").write_text("fine words")
    chunked = []
    real_chunk_document = index_module.chunk_document

    def chunk_document(path, text, **kwargs):
        chunked.append(path)
        if "poison" in text:
            raise sqlite3.OperationalError("disk I/O error")
        return real_chunk_document(path, text, **kwargs)

    monkeypatch.setattr(index_module, "chunk_document", chunk_document)
    store = DocumentStore(persist_path=str(tmp_path))
    assert [hit.path for hit in store.search_passages("fine")] == ["/good.md"]
    assert [hit.path for hit in store.search_passages("fine")] == ["/good.md"]
    assert chunked.count("/bad.md") == 1
    status = store.index_status()
    assert (status.files, status.indexed, status.failed) == (2, 1, 1)
    assert [problem.kind for problem in status.problems] == ["file"]
    assert "/bad.md" in status.problems[0].message and "disk I/O error" in status.error
    # Nothing is left to do until bad.md changes, so the index is complete apart from its reported problem.
    assert status.complete and store._index.wait_until_embedded(5)
    store.close()


def test_a_failed_file_is_not_counted_toward_embedding(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    real_chunk_document = index_module.chunk_document

    def chunk_document(path, text, **kwargs):
        if "poison" in text:
            raise RuntimeError("cannot chunk")
        return real_chunk_document(path, text, **kwargs)

    monkeypatch.setattr(index_module, "chunk_document", chunk_document)
    store = _vector_store(tmp_path, FakeEmbedder(), a="protein folding", bad="poison glacier")
    store.index_status()
    assert store._index.wait_until_embedded(5)
    status = store.index_status()
    assert (status.files, status.indexed, status.embedded, status.failed) == (2, 1, 1, 1)
    assert status.complete and status.has_problem("file")
    store.close()


@pytest.mark.parametrize("operation", ["read", "search_passages", "list_paths"])
def test_a_slow_pdf_extraction_does_not_block_another_file(tmp_path, monkeypatch, operation):
    (tmp_path / "slow.pdf").write_bytes(minimal_pdf(["alpha"]))
    (tmp_path / "a.md").write_text("quick")
    started, release = threading.Event(), threading.Event()
    real_pdf_to_markdown = document_store_module.pdf_to_markdown

    def slow_pdf_to_markdown(data):
        started.set()
        release.wait(10)
        return real_pdf_to_markdown(data)

    monkeypatch.setattr(document_store_module, "pdf_to_markdown", slow_pdf_to_markdown)
    store = DocumentStore(persist_path=str(tmp_path))
    calls = {
        "read": lambda: store.read("/slow.pdf"),
        "search_passages": lambda: store.search_passages("alpha"),
        "list_paths": store.list_paths,
    }
    extracting = threading.Thread(target=calls[operation])
    extracting.start()
    try:
        assert started.wait(5)
        done = []

        def read_and_write():
            done.append(store.read("/a.md"))
            store.write("/b.md", "written meanwhile")
            done.append(store.read("/b.md"))

        other = threading.Thread(target=read_and_write)
        other.start()
        other.join(timeout=2)
        assert done == ["quick", "written meanwhile"]
    finally:
        release.set()
        extracting.join(timeout=10)
        other.join(timeout=10)
    assert "alpha" in store.read("/slow.pdf")
    store.close()


def test_an_extraction_of_a_file_that_changed_meanwhile_is_not_cached(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(minimal_pdf(["alpha"]))
    real_pdf_to_markdown = document_store_module.pdf_to_markdown

    def rewriting_pdf_to_markdown(data):
        text = real_pdf_to_markdown(data)
        pdf.write_bytes(minimal_pdf(["alpha", "beta"]))
        os.utime(pdf, (time.time() + 5, time.time() + 5))
        return text

    monkeypatch.setattr(document_store_module, "pdf_to_markdown", rewriting_pdf_to_markdown)
    store = DocumentStore(persist_path=str(tmp_path))
    store.read("/p.pdf")
    assert str(pdf) not in store._extractions


def test_a_file_that_could_not_be_opened_is_retried_without_changing(tmp_path, monkeypatch):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read a file with no permissions")
    monkeypatch.setattr(index_module, "TRANSIENT_RETRY_SECONDS", 0.0, raising=False)
    locked = tmp_path / "locked.md"
    locked.write_text("ribosome notes")
    os.chmod(locked, 0)
    store = DocumentStore(persist_path=str(tmp_path))
    try:
        assert store.search_passages("ribosome") == []
        assert "Permission denied" in store.index_status().unreadable["/locked.md"]
    finally:
        # chmod moves the ctime, not the (mtime, size) the index keys on, so nothing about the file changed.
        os.chmod(locked, 0o600)
    assert [hit.path for hit in store.search_passages("ribosome")] == ["/locked.md"]
    status = store.index_status()
    assert status.unreadable == {} and status.complete
    store.close()


def test_a_transient_failure_waits_before_it_is_retried(tmp_path, monkeypatch):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read a file with no permissions")
    locked = tmp_path / "locked.md"
    locked.write_text("ribosome notes")
    os.chmod(locked, 0)
    store = DocumentStore(persist_path=str(tmp_path))
    loads = []
    real_load = store._load
    monkeypatch.setattr(store, "_load", lambda path: loads.append(path) or real_load(path))
    try:
        store.search_passages("ribosome")
        store.search_passages("ribosome")
        assert loads == ["/locked.md"]
    finally:
        os.chmod(locked, 0o600)
    store.close()


def test_a_file_that_is_not_text_is_not_read_again(tmp_path, monkeypatch):
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00")
    monkeypatch.setattr(index_module, "TRANSIENT_RETRY_SECONDS", 0.0, raising=False)
    store = DocumentStore(persist_path=str(tmp_path))
    loads = []
    real_load = store._load
    monkeypatch.setattr(store, "_load", lambda path: loads.append(path) or real_load(path))
    store.search_passages("x")
    store.search_passages("x")
    assert loads == ["/bin.md"]
    assert store.index_status().unreadable == {"/bin.md": "not UTF-8 text"}
    store.close()


def test_extensions_limit_which_files_are_documents(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "B.TXT").write_text("alpha two")
    (tmp_path / "c.json").write_text('{"alpha": 1}')
    (tmp_path / "d.docx").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "p.PDF").write_bytes(minimal_pdf(["alpha"]))
    store = DocumentStore(persist_path=str(tmp_path), extensions=["md", ".txt", ".PDF"])
    assert store.list_paths() == ["/B.TXT", "/a.md", "/p.PDF"]
    assert store.unreadable() == {}
    assert store.listing().paths == ["/B.TXT", "/a.md", "/p.PDF"]
    status = store.index_status()
    assert status.files == 3 and status.unreadable == {} and status.complete
    assert {hit.path for hit in store.search_passages("alpha")} == {"/a.md", "/B.TXT", "/p.PDF"}
    with pytest.raises(KeyError):
        store.read("/c.json")
    assert store.unreadable_reason("/c.json") is None
    with pytest.raises(ValueError, match=r"\.json"):
        store.write("/new.json", "{}")
    assert not (tmp_path / "new.json").exists()
    store.close()

    every = DocumentStore(persist_path=str(tmp_path))
    assert "/c.json" in every.list_paths() and "/d.docx" in every.unreadable()


def test_extensions_must_be_a_collection_of_suffixes(tmp_path):
    with pytest.raises(TypeError):
        DocumentStore(persist_path=str(tmp_path), extensions=".md")
    with pytest.raises(ValueError):
        DocumentStore(persist_path=str(tmp_path), extensions=[])


def test_paths_expand_the_home_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    store = DocumentStore("~/Vault", index_path="~/index")
    store.write("/a.md", "alpha")
    assert store.search_passages("alpha")[0].path == "/a.md"
    store.close()
    assert (home / "Vault" / "a.md").read_text() == "alpha"
    assert (home / "index" / "index.sqlite3").exists()
    assert not (tmp_path / "~").exists()


def test_listing_reads_no_file_contents(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "p.pdf").write_bytes(minimal_pdf(["alpha"]))
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "big.md").write_text("x" * 20_000)
    store = DocumentStore(persist_path=str(tmp_path), max_file_bytes=10_000)
    # Reading records why bin.md is no document; the listing reports that reason without reading again.
    assert "/bin.md" in store.unreadable()

    def no_reading(*args, **kwargs):
        raise AssertionError("listing read a file")

    monkeypatch.setattr(document_store_module, "open", no_reading, raising=False)
    monkeypatch.setattr(document_store_module, "pdf_to_markdown", no_reading)
    listing = store.listing()
    assert listing.paths == ["/a.md", "/p.pdf"]
    assert listing.unreadable["/bin.md"] == "not UTF-8 text"
    assert "too large" in listing.unreadable["/big.md"]
    assert store.listing(prefix="/b").paths == []
    assert sorted(store.listing(prefix="/b").unreadable) == ["/big.md", "/bin.md"]
    assert DocumentStore().listing() == ([], {})


def test_a_fresh_store_over_a_persisted_index_lists_without_extracting(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "p.pdf").write_bytes(minimal_pdf(["alpha"]))
    first = DocumentStore(persist_path=str(folder), index_path=str(tmp_path / "index"))
    first.index_status()
    first.close()
    calls = []
    monkeypatch.setattr(document_store_module, "pdf_to_markdown", lambda data: calls.append(1) or "x")
    second = DocumentStore(persist_path=str(folder), index_path=str(tmp_path / "index"))
    assert second.listing().paths == ["/p.pdf"]
    assert second.index_status().complete
    assert calls == []
    second.close()


def test_unreadable_reason_names_why_an_existing_file_is_no_document(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00")
    store = DocumentStore(persist_path=str(tmp_path))
    assert store.unreadable_reason("/a.md") is None
    assert store.unreadable_reason("/missing.md") is None
    assert store.unreadable_reason("bin.md") == "not UTF-8 text"
    assert DocumentStore().unreadable_reason("/a.md") is None


def test_an_embedding_failure_is_a_structured_problem(tmp_path):
    store = _vector_store(tmp_path, FakeEmbedder(error=RuntimeError("model download failed")), a="ribosome notes")
    store.search_passages("ribosome")
    assert not store._index.wait_until_embedded(5)
    status = store.index_status()
    assert [problem.kind for problem in status.problems] == ["embedding"]
    assert "model download failed" in status.problems[0].message
    assert status.error == status.problems[0].message
    store.close()


def test_a_worker_error_survives_the_next_status(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.0)
    (tmp_path / "a.md").write_text("alpha")
    real_next_job = PassageIndex._next_job
    worker_calls = []

    def next_job(self):
        if threading.current_thread().name == index_module.WORKER_THREAD_NAME:
            worker_calls.append(1)
            if len(worker_calls) == 1:
                raise sqlite3.OperationalError("database is locked")
        return real_next_job(self)

    monkeypatch.setattr(PassageIndex, "_next_job", next_job)
    store = DocumentStore(persist_path=str(tmp_path))
    release = threading.Event()
    real_load = store._load

    def gated_load(path):
        release.wait(10)
        return real_load(path)

    monkeypatch.setattr(store, "_load", gated_load)
    try:
        store.index_status()
        assert not store._index.wait_until_embedded(5)
        # This refresh restarts the worker, which is now held reading a.md: nothing has succeeded since the
        # failure, so the failure is still the news.
        for _ in range(2):
            status = store.index_status()
            assert [problem.kind for problem in status.problems] == ["worker"]
            assert "database is locked" in status.error
    finally:
        release.set()
    assert store._index.wait_until_embedded(5)
    assert store.index_status().problems == ()
    store.close()


def test_status_without_refresh_reports_the_last_refresh(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    store = DocumentStore(persist_path=str(tmp_path))
    store.search_passages("alpha")
    (tmp_path / "b.md").write_text("beta")
    assert store.index_status(refresh=False).files == 1
    assert store.index_status().files == 2
    store.close()


def test_a_corrupt_keyword_index_is_rebuilt(tmp_path, caplog):
    folder, index_dir = tmp_path / "docs", tmp_path / "index"
    folder.mkdir()
    index_dir.mkdir()
    (folder / "a.md").write_text("photosynthesis notes")
    (index_dir / "index.sqlite3").write_bytes(b"not a database " * 100)
    store = DocumentStore(persist_path=str(folder), index_path=str(index_dir))
    assert [hit.path for hit in store.search_passages("photosynthesis")] == ["/a.md"]
    assert "rebuilding" in caplog.text
    store.close()


def test_a_corrupt_vector_index_is_rebuilt(tmp_path, caplog):
    pytest.importorskip("chromadb")
    folder, index_dir = tmp_path / "docs", tmp_path / "index"
    folder.mkdir()
    (index_dir / "vectors").mkdir(parents=True)
    (folder / "a.md").write_text("structure prediction of polypeptides")
    (index_dir / "vectors" / "chroma.sqlite3").write_bytes(b"not a database " * 100)
    store = DocumentStore(persist_path=str(folder), index_path=str(index_dir), embedding_client=FakeEmbedder())
    store.index_status()
    assert store._index.wait_until_embedded(5)
    assert store.search_passages("protein folding")[0].path == "/a.md"
    assert "rebuilding" in caplog.text
    store.close()


@pytest.mark.parametrize(
    "error",
    [
        OSError(28, "No space left on device"),
        sqlite3.OperationalError("database is locked"),
        "error returned from database: (code: 5) database is locked",
        "error returned from database: (code: 14) unable to open database file",
    ],
)
def test_vectors_that_fail_to_open_without_damage_are_kept(tmp_path, monkeypatch, error):
    pytest.importorskip("chromadb")
    from chromadb.errors import InternalError

    if isinstance(error, str):
        error = InternalError(error)
    first = _vector_store(tmp_path, FakeEmbedder(), a="structure prediction of polypeptides", b="glacier melt")
    first.index_status()
    assert first._index.wait_until_embedded(5)
    first.close()
    vectors_before = _all_files(tmp_path / "index" / "vectors")

    real_open_vectors = index_module._open_vectors

    def failing_open_vectors(index_path):
        raise error

    monkeypatch.setattr(index_module, "_open_vectors", failing_open_vectors)
    embedder = FakeEmbedder()
    second = _vector_store(tmp_path, embedder)
    status = second.index_status()
    assert second._index.wait_until_embedded(5) is False
    assert [problem.kind for problem in status.problems] == ["embedding"]
    assert "keyword-only" in status.error and str(error) in status.error
    assert status.embedded == 0 and status.indexed == 2
    # Keyword search still works, and nothing was embedded against a collection that is not open.
    assert [hit.path for hit in second.search_passages("glacier")] == ["/b.md"]
    assert embedder.calls == []
    second.close()
    assert _all_files(tmp_path / "index" / "vectors") == vectors_before
    check = sqlite3.connect(tmp_path / "index" / "index.sqlite3")
    try:
        assert check.execute("SELECT path, embedded FROM files ORDER BY path").fetchall() == [
            ("/a.md", 1),
            ("/b.md", 1),
        ]
    finally:
        check.close()

    monkeypatch.setattr(index_module, "_open_vectors", real_open_vectors)
    embedder = FakeEmbedder()
    third = _vector_store(tmp_path, embedder)
    status = third.index_status()
    assert (status.embedded, status.problems) == (2, ())
    # The existing vectors answer a paraphrase: only the query was embedded, no document.
    assert third.search_passages("protein folding")[0].path == "/a.md"
    assert [input_type for input_type, _ in embedder.calls] == ["query"]
    third.close()


def test_vectors_that_failed_to_open_are_retried_once_due(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from chromadb.errors import InternalError

    first = _vector_store(tmp_path, FakeEmbedder(), a="structure prediction of polypeptides", b="glacier melt")
    first.index_status()
    assert first._index.wait_until_embedded(5)
    first.close()

    real_open_vectors = index_module._open_vectors
    opens = []

    def failing_open_vectors(index_path):
        opens.append(index_path)
        raise InternalError("error returned from database: (code: 5) database is locked")

    monkeypatch.setattr(index_module, "_open_vectors", failing_open_vectors)
    embedder = FakeEmbedder()
    store = _vector_store(tmp_path, embedder)
    status = store.index_status()
    # The note says where the vectors are and how to recover if this was damage after all.
    vectors = str(tmp_path / "index" / "vectors")
    assert vectors in status.error and "deleting it rebuilds the vectors" in status.error
    # Not due yet: later searches stay keyword-only without opening Chroma again.
    store.search_passages("glacier")
    store.index_status()
    assert len(opens) == 1

    monkeypatch.setattr(index_module, "TRANSIENT_RETRY_SECONDS", 0.0)
    store._index._vectors_retry_at = time.monotonic()
    store.search_passages("glacier")
    assert len(opens) == 2 and store.index_status().has_problem("embedding")

    monkeypatch.setattr(index_module, "_open_vectors", real_open_vectors)
    assert store.search_passages("protein folding")[0].path == "/a.md"
    status = store.index_status()
    assert (status.embedded, status.problems) == (2, ())
    assert store._index.wait_until_embedded(5)
    # The existing vectors were reused: queries were embedded, no document was.
    assert {input_type for input_type, _ in embedder.calls} == {"query"}
    store.close()
    assert store._index is None


def test_only_damage_counts_as_damaged_vectors():
    pytest.importorskip("chromadb")
    from chromadb.errors import InternalError

    damaged = [
        InternalError("error returned from database: (code: 26) file is not a database"),
        InternalError("error returned from database: (code: 11) database disk image is malformed"),
        # An extended result code (SQLITE_CORRUPT_INDEX) is damage too: its low byte is SQLITE_CORRUPT.
        InternalError("error returned from database: (code: 779) database disk image is malformed"),
        InternalError("Inconsistent hash: db=bogus, source=38352d725ad1c16074fac420b22b4633"),
    ]
    sound = [
        InternalError("error returned from database: (code: 5) database is locked"),
        InternalError("error returned from database: (code: 1544) attempt to write a readonly database"),
        InternalError("File exists (os error 17)"),
        OSError(5, "Input/output error"),
        RuntimeError("file is not a database"),
    ]
    assert [index_module._vectors_damaged(exc) for exc in damaged] == [True] * len(damaged)
    assert [index_module._vectors_damaged(exc) for exc in sound] == [False] * len(sound)


def test_vectors_lost_while_the_manifest_says_embedded_are_rebuilt(tmp_path, caplog):
    import shutil

    first = _vector_store(tmp_path, FakeEmbedder(), a="protein folding", b="glacier melt")
    first.index_status()
    assert first._index.wait_until_embedded(5)
    first.close()
    shutil.rmtree(tmp_path / "index" / "vectors")

    release = threading.Event()
    second = _vector_store(tmp_path, FakeEmbedder(block=release))
    try:
        assert second.index_status().embedded == 0
    finally:
        release.set()
    assert second._index.wait_until_embedded(5)
    assert second.search_passages("polypeptides")[0].path == "/a.md"
    assert "vectors" in caplog.text
    second.close()


def test_changing_embedding_dimensions_rebuilds_vectors(tmp_path):
    wide = FakeEmbedder("fake-1")
    wide.dimensions = 8
    first = _vector_store(tmp_path, wide, a="protein folding")
    first.index_status()
    assert first._index.wait_until_embedded(5)
    first.close()

    release = threading.Event()
    narrow = FakeEmbedder("fake-1", block=release)
    narrow.dimensions = 4
    second = _vector_store(tmp_path, narrow)
    try:
        assert second.index_status().embedded == 0
        assert second._index._collection.count() == 0
    finally:
        release.set()
    assert second._index.wait_until_embedded(5)
    second.close()


def test_vector_clients_send_no_telemetry(tmp_path):
    pytest.importorskip("chromadb")
    from aimu.memory import SemanticMemoryStore

    for client, _ in (index_module._open_vectors(None), index_module._open_vectors(str(tmp_path))):
        assert client.get_settings().anonymized_telemetry is False
    assert SemanticMemoryStore()._client.get_settings().anonymized_telemetry is False


def test_a_failing_store_is_skipped_and_reported():
    stores = {"good": _store_with(a="glacier ice"), "bad": _store_with(b="glacier melt")}

    def broken(query, n_results=8):
        raise RuntimeError("index database is locked")

    stores["bad"].search_passages = broken
    with pytest.raises(RuntimeError):
        search_collections(stores, "glacier")
    failures = {}
    results = search_collections(stores, "glacier", on_error=lambda name, exc: failures.setdefault(name, exc))
    assert [result.collection for result in results] == ["good"]
    assert isinstance(failures["bad"], RuntimeError)
