"""Tests for DocumentStore's passage index (keyword half) and the store changes it relies on.

Mock-only: no model, no network. The index is SQLite FTS5, in memory or in a temp directory.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import aimu.memory._index as index_module
import aimu.memory.document_store as document_store_module
from aimu.memory import DocumentStore, IndexStatus, Passage
from aimu.memory._index import PassageIndex, chunk_document
from aimu.tools.builtin import make_document_tools
from tests.test_documents import minimal_pdf


def _all_files(root) -> set[str]:
    return {os.path.join(dirpath, name) for dirpath, _, names in os.walk(root) for name in names}


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


def test_concurrent_writes_and_searches_from_threads(tmp_path):
    store = DocumentStore(persist_path=str(tmp_path / "d"), index_path=str(tmp_path / "i"))

    def work(number: int) -> list[Passage]:
        store.write(f"/n{number}.md", f"topic{number} shared")
        return store.search_passages(f"topic{number}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(24)))
    assert all(hits and hits[0].path == f"/n{number}.md" for number, hits in enumerate(results))
    assert store.index_status().files == 24
    store.close()
