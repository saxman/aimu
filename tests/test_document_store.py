"""
Tests for DocumentStore and the document_mcp server tools.
"""

import pytest

import aimu.memory.document_mcp as document_mcp
from aimu.memory import DocumentStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    """Ephemeral in-memory DocumentStore for each test."""
    return DocumentStore()


@pytest.fixture
def mcp_store(monkeypatch):
    """Replace the module-level _store with a fresh ephemeral DocumentStore."""
    fresh = DocumentStore()
    monkeypatch.setattr(document_mcp, "_store", fresh)
    return fresh


# ---------------------------------------------------------------------------
# DocumentStore: rich path-based API
# ---------------------------------------------------------------------------


def test_initial_store_is_empty(store):
    assert len(store) == 0
    assert store.list_paths() == []


def test_write_increases_count(store):
    store.write("/note.md", "Hello")
    assert len(store) == 1


def test_write_and_read_roundtrip(store):
    store.write("/prefs.md", "Use concise responses.")
    assert store.read("/prefs.md") == "Use concise responses."


def test_write_overwrites_existing(store):
    store.write("/note.md", "first")
    store.write("/note.md", "second")
    assert store.read("/note.md") == "second"
    assert len(store) == 1


def test_read_missing_path_raises(store):
    with pytest.raises(KeyError):
        store.read("/does-not-exist.md")


def test_edit_replaces_first_occurrence(store):
    store.write("/doc.md", "foo foo bar")
    store.edit("/doc.md", "foo", "baz")
    assert store.read("/doc.md") == "baz foo bar"


def test_edit_missing_path_raises(store):
    with pytest.raises(KeyError):
        store.edit("/missing.md", "old", "new")


def test_edit_missing_string_raises(store):
    store.write("/doc.md", "hello world")
    with pytest.raises(ValueError):
        store.edit("/doc.md", "not-here", "replacement")


def test_list_paths_returns_sorted(store):
    store.write("/b.md", "B")
    store.write("/a.md", "A")
    store.write("/c.md", "C")
    assert store.list_paths() == ["/a.md", "/b.md", "/c.md"]


def test_list_paths_with_prefix(store):
    store.write("/notes/todo.md", "task")
    store.write("/notes/done.md", "finished")
    store.write("/prefs.md", "settings")
    filtered = store.list_paths(prefix="/notes/")
    assert set(filtered) == {"/notes/todo.md", "/notes/done.md"}
    assert "/prefs.md" not in filtered


def test_search_full_text_matches_content(store):
    store.write("/a.md", "Paul works at Google")
    store.write("/b.md", "Sarah is the sister of Emma")
    results = store.search_full_text("Google")
    assert len(results) == 1
    assert results[0]["path"] == "/a.md"
    assert results[0]["content"] == "Paul works at Google"


def test_search_full_text_matches_path(store):
    store.write("/project-notes.md", "some content")
    results = store.search_full_text("project")
    assert any(r["path"] == "/project-notes.md" for r in results)


def test_search_full_text_case_insensitive(store):
    store.write("/doc.md", "UPPERCASE content")
    results = store.search_full_text("uppercase")
    assert len(results) == 1


def test_search_full_text_respects_n_results(store):
    for i in range(10):
        store.write(f"/doc{i}.md", "keyword")
    results = store.search_full_text("keyword", n_results=3)
    assert len(results) == 3


def test_search_full_text_empty_store(store):
    assert store.search_full_text("anything") == []


def test_delete_removes_document(store):
    store.write("/a.md", "content a")
    store.write("/b.md", "content b")
    store.delete("/a.md")
    assert len(store) == 1
    with pytest.raises(KeyError):
        store.read("/a.md")


def test_delete_nonexistent_is_noop(store):
    store.write("/a.md", "content")
    store.delete("/does-not-exist.md")
    assert len(store) == 1


def test_persistent_store_survives_reinstantiation(tmp_path):
    s1 = DocumentStore(persist_path=str(tmp_path))
    s1.write("/prefs.md", "Use bullet points.")
    s1.write("/context.md", "Project: AIMU")

    s2 = DocumentStore(persist_path=str(tmp_path))
    assert len(s2) == 2
    assert s2.read("/prefs.md") == "Use bullet points."
    assert s2.read("/context.md") == "Project: AIMU"


def test_persistent_edit_survives_reinstantiation(tmp_path):
    s1 = DocumentStore(persist_path=str(tmp_path))
    s1.write("/doc.md", "version one")
    s1.edit("/doc.md", "one", "two")

    s2 = DocumentStore(persist_path=str(tmp_path))
    assert s2.read("/doc.md") == "version two"


# ---------------------------------------------------------------------------
# MemoryStore abstract interface on DocumentStore
# ---------------------------------------------------------------------------


def test_store_auto_assigns_path(store):
    store.store("some content")
    assert len(store) == 1
    path = store.list_all()[0]
    assert path.startswith("/note-")
    assert path.endswith(".md")


def test_store_multiple_items(store):
    store.store("first")
    store.store("second")
    assert len(store) == 2


def test_search_abstract_returns_content_strings(store):
    store.write("/doc.md", "needle in a haystack")
    results = store.search("needle")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    assert any("needle" in r for r in results)


def test_search_abstract_empty_store(store):
    assert store.search("anything") == []


def test_delete_abstract_removes_by_path(store):
    store.write("/to-remove.md", "content")
    store.delete("/to-remove.md")
    assert len(store) == 0


def test_list_all_returns_paths(store):
    store.write("/x.md", "X")
    store.write("/y.md", "Y")
    paths = store.list_all()
    assert set(paths) == {"/x.md", "/y.md"}


def test_len_matches_list_all(store):
    store.write("/a.md", "A")
    store.write("/b.md", "B")
    assert len(store) == len(store.list_all())


# ---------------------------------------------------------------------------
# document_mcp server tool tests
# ---------------------------------------------------------------------------


def test_mcp_memory_write_and_read(mcp_store):
    document_mcp.memory_write("/prefs.md", "Use concise responses.")
    content = document_mcp.memory_read("/prefs.md")
    assert content == "Use concise responses."


def test_mcp_memory_write_returns_metadata(mcp_store):
    result = document_mcp.memory_write("/doc.md", "hello")
    assert result["path"] == "/doc.md"
    assert result["size"] == len("hello")


def test_mcp_memory_list_empty(mcp_store):
    assert document_mcp.memory_list() == []


def test_mcp_memory_list_returns_all(mcp_store):
    document_mcp.memory_write("/a.md", "A")
    document_mcp.memory_write("/b.md", "BB")
    items = document_mcp.memory_list()
    paths = {i["path"] for i in items}
    assert paths == {"/a.md", "/b.md"}


def test_mcp_memory_list_with_prefix(mcp_store):
    document_mcp.memory_write("/notes/todo.md", "task")
    document_mcp.memory_write("/prefs.md", "settings")
    items = document_mcp.memory_list(path_prefix="/notes/")
    assert len(items) == 1
    assert items[0]["path"] == "/notes/todo.md"


def test_mcp_memory_search(mcp_store):
    document_mcp.memory_write("/doc.md", "Paul works at Google")
    document_mcp.memory_write("/other.md", "Sarah lives in Paris")
    results = document_mcp.memory_search("Google")
    assert len(results) == 1
    assert results[0]["path"] == "/doc.md"


def test_mcp_memory_search_empty(mcp_store):
    assert document_mcp.memory_search("anything") == []


def test_mcp_memory_edit(mcp_store):
    document_mcp.memory_write("/doc.md", "old content")
    result = document_mcp.memory_edit("/doc.md", "old", "new")
    assert document_mcp.memory_read("/doc.md") == "new content"
    assert result["path"] == "/doc.md"


def test_mcp_memory_delete(mcp_store):
    document_mcp.memory_write("/a.md", "content")
    document_mcp.memory_write("/b.md", "content")
    document_mcp.memory_delete("/a.md")
    items = document_mcp.memory_list()
    assert len(items) == 1
    assert items[0]["path"] == "/b.md"


def test_mcp_memory_delete_nonexistent_is_noop(mcp_store):
    document_mcp.memory_write("/a.md", "content")
    document_mcp.memory_delete("/does-not-exist.md")
    assert len(document_mcp.memory_list()) == 1
