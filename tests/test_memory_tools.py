"""Mock-only unit tests for make_memory_tools and its make_tools integration.

No ChromaDB or heavy deps required; uses an in-memory MemoryStore stub.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aimu.tools import builtin
from aimu.tools.builtin import make_memory_tools


# ---------------------------------------------------------------------------
# Minimal in-memory store stub (no ChromaDB needed)
# ---------------------------------------------------------------------------


class _SimpleStore:
    def __init__(self):
        self._items: list[str] = []

    def store(self, content: str) -> None:
        self._items.append(content)

    def search(self, query: str, n_results: int = 10) -> list[str]:
        return [i for i in self._items if query.lower() in i.lower()][:n_results]

    def delete(self, identifier: str) -> None:
        self._items = [i for i in self._items if i != identifier]

    def list_all(self) -> list[str]:
        return list(self._items)


@pytest.fixture
def store():
    return _SimpleStore()


@pytest.fixture
def tools(store):
    return make_memory_tools(store)


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_returns_three_tools(tools):
    assert len(tools) == 3


def test_tool_names(tools):
    names = {t.__tool_spec__["function"]["name"] for t in tools}
    assert names == {"store_memory", "search_memories", "list_memories"}


def test_store_memory_spec(tools):
    store_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "store_memory")
    params = store_tool.__tool_spec__["function"]["parameters"]
    assert "content" in params["properties"]
    assert params["required"] == ["content"]


def test_search_memories_spec(tools):
    search_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "search_memories")
    params = search_tool.__tool_spec__["function"]["parameters"]
    assert "query" in params["properties"]
    assert "n_results" in params["properties"]
    assert params["required"] == ["query"]


def test_list_memories_spec(tools):
    list_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "list_memories")
    params = list_tool.__tool_spec__["function"]["parameters"]
    assert params.get("required", []) == []


def test_tools_are_sync_and_not_streaming(tools):
    for t in tools:
        assert t.__tool_is_async__ is False
        assert t.__tool_is_streaming__ is False


# ---------------------------------------------------------------------------
# Functional behaviour
# ---------------------------------------------------------------------------


def test_store_memory_stores_and_returns_confirmation(store, tools):
    store_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "store_memory")
    result = store_tool("the sky is blue")
    assert result == "Stored."
    assert "the sky is blue" in store.list_all()


def test_search_memories_returns_matches(store, tools):
    store.store("the sky is blue")
    store.store("the grass is green")
    search_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "search_memories")
    result = search_tool("sky")
    assert "the sky is blue" in result
    assert "grass" not in result


def test_search_memories_no_results_message(store, tools):
    search_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "search_memories")
    result = search_tool("nothing matches this")
    assert "No relevant memories found" in result


def test_list_memories_returns_all(store, tools):
    store.store("fact one")
    store.store("fact two")
    list_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "list_memories")
    result = list_tool()
    assert "fact one" in result
    assert "fact two" in result


def test_list_memories_empty_message(store, tools):
    list_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "list_memories")
    result = list_tool()
    assert "empty" in result.lower()


def test_each_call_creates_independent_tools():
    store_a = _SimpleStore()
    store_b = _SimpleStore()
    tools_a = make_memory_tools(store_a)
    tools_b = make_memory_tools(store_b)

    store_fn_a = next(t for t in tools_a if t.__tool_spec__["function"]["name"] == "store_memory")
    store_fn_b = next(t for t in tools_b if t.__tool_spec__["function"]["name"] == "store_memory")

    store_fn_a("only in a")
    assert store_a.list_all() == ["only in a"]
    assert store_b.list_all() == []

    store_fn_b("only in b")
    assert store_a.list_all() == ["only in a"]
    assert store_b.list_all() == ["only in b"]


# ---------------------------------------------------------------------------
# make_tools integration
# ---------------------------------------------------------------------------


def _fake_base_client(supports_vision=False):
    client = MagicMock()
    client.model = MagicMock()
    client.model.supports_vision = supports_vision
    return client


def test_make_tools_without_memory_store_excludes_memory_tools():
    tools = builtin.make_tools(_fake_base_client())
    names = {t.__tool_spec__["function"]["name"] for t in tools}
    assert "store_memory" not in names
    assert "search_memories" not in names
    assert "list_memories" not in names


def test_make_tools_with_memory_store_appends_memory_tools():
    store = _SimpleStore()
    tools = builtin.make_tools(_fake_base_client(), memory_store=store)
    names = {t.__tool_spec__["function"]["name"] for t in tools}
    assert {"store_memory", "search_memories", "list_memories"}.issubset(names)


def test_make_tools_memory_tools_bound_to_provided_store():
    store = _SimpleStore()
    tools = builtin.make_tools(_fake_base_client(), memory_store=store)
    store_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "store_memory")
    store_tool("bound correctly")
    assert "bound correctly" in store.list_all()


# ---------------------------------------------------------------------------
# make_document_tools (real ephemeral DocumentStore: pure-Python, no deps)
# ---------------------------------------------------------------------------


from aimu.memory import DocumentStore  # noqa: E402
from aimu.tools.builtin import make_document_tools  # noqa: E402


@pytest.fixture
def doc_store():
    return DocumentStore()  # ephemeral, in-memory


@pytest.fixture
def doc_tools(doc_store):
    return make_document_tools(doc_store)


def _by_name(tools, name):
    return next(t for t in tools if t.__tool_spec__["function"]["name"] == name)


def test_document_tools_names_and_specs(doc_tools):
    names = {t.__tool_spec__["function"]["name"] for t in doc_tools}
    assert names == {"save_document", "read_document", "edit_document", "list_documents", "search_documents"}
    for t in doc_tools:
        assert t.__tool_is_async__ is False
        assert t.__tool_is_streaming__ is False
    save = _by_name(doc_tools, "save_document")
    assert save.__tool_spec__["function"]["parameters"]["required"] == ["path", "content"]


def test_document_tools_distinct_from_memory_tools(doc_tools):
    """Both tool sets coexist on one agent without name collisions."""
    doc_names = {t.__tool_spec__["function"]["name"] for t in doc_tools}
    mem_names = {"store_memory", "search_memories", "list_memories"}
    assert doc_names.isdisjoint(mem_names)


def test_save_then_read_round_trip(doc_tools):
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    assert save("/notes/standup.md", "Yesterday, Today, Blockers") == "Saved /notes/standup.md."
    assert read("/notes/standup.md") == "Yesterday, Today, Blockers"


def test_read_missing_document_returns_message_not_raise(doc_tools):
    read = _by_name(doc_tools, "read_document")
    result = read("/nope.md")
    assert "No document found" in result and "/nope.md" in result


def test_edit_document_changes_part_without_touching_the_rest(doc_tools):
    # The reason this tool exists: without it, the only way to change a line of a long
    # document is read_document + save_document, and read_document returns a *window*, so
    # saving it back deletes everything outside the window. See the test below.
    save = _by_name(doc_tools, "save_document")
    edit = _by_name(doc_tools, "edit_document")
    read = _by_name(doc_tools, "read_document")
    save("/paper.md", "\n".join(f"line {i}" for i in range(3000)))

    result = edit("/paper.md", "line 7\n", "line seven\n")

    assert "replaced 1 occurrence" in result
    whole = read("/paper.md", max_lines=5000)
    assert len(whole.splitlines()) == 3000
    assert "line seven" in whole
    assert "line 2999" in whole


def test_saving_back_a_windowed_read_would_truncate_the_document(doc_tools):
    """The hazard edit_document exists to remove, pinned so it cannot return unnoticed.

    read_document returns one window of a long document. save_document replaces the whole
    thing. So the read-modify-save round-trip an agent would otherwise have to use destroys
    everything past the window, silently, and reports success. This test documents that the
    combination is destructive; save_document's docstring is what steers the model away from
    it, and edit_document is what it should reach for instead.
    """
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/paper.md", "\n".join(f"line {i}" for i in range(3000)))

    window = read("/paper.md", max_lines=50)
    save("/paper.md", window)  # what an agent without edit_document has to do

    assert len(read("/paper.md", max_lines=5000).splitlines()) == 51  # 50 + the marker
    # The tool's own docstring must carry the warning, since neither the tool nor the store
    # can tell a full rewrite from a window.
    assert "truncated" in save.__doc__
    assert "edit_document" in save.__doc__


def test_edit_document_refuses_an_ambiguous_match(doc_tools):
    save = _by_name(doc_tools, "save_document")
    edit = _by_name(doc_tools, "edit_document")
    original = "timeout = 30\nretries = 3\ntimeout = 30"
    save("/conf.md", original)

    result = edit("/conf.md", "timeout = 30", "timeout = 60")

    # The store's own message, surfaced bare rather than behind a prefix that repeated it.
    assert "appears 2 times" in result
    assert "nothing was written" in result.lower()
    assert _by_name(doc_tools, "read_document")("/conf.md") == original


def test_edit_document_reports_a_missing_document_rather_than_raising(doc_tools):
    # This group's convention: a miss is a message the model can act on, unlike
    # document_mcp's memory_edit, which raises.
    result = _by_name(doc_tools, "edit_document")("/nope.md", "a", "b")
    assert "No document found" in result and "/nope.md" in result


def test_edit_document_reports_a_missing_match(doc_tools):
    save = _by_name(doc_tools, "save_document")
    edit = _by_name(doc_tools, "edit_document")
    save("/a.md", "hello world")

    result = edit("/a.md", "not-here", "x")

    assert "not found" in result
    assert _by_name(doc_tools, "read_document")("/a.md") == "hello world"


def test_read_document_windows_a_long_document_and_names_the_next_offset(doc_tools):
    # The store holds documents the user drops in, which can be paper-sized. Returning one
    # whole is how a single read spends the context window the rest of the task needs.
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/papers/long.md", "\n".join(f"line {i}" for i in range(3000)))

    out = read("/papers/long.md", max_lines=10)

    assert out.startswith("line 0")
    assert "truncated: showing lines 1-10 of 3000" in out
    assert "call read_document with offset=11 to continue" in out


def test_read_document_pages_a_document_larger_than_one_window(doc_tools):
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    body = [f"line {i}" for i in range(2500)]
    save("/papers/long.md", "\n".join(body))

    seen, offset = [], 1
    while True:
        lines = read("/papers/long.md", max_lines=1000, offset=offset).splitlines()
        truncated = bool(lines) and lines[-1].startswith("... (truncated")
        seen.extend(lines[:-1] if truncated else lines)
        if not truncated:
            break
        offset += 1000

    assert seen == body


def test_read_document_offset_past_the_end_says_how_long_it_is(doc_tools):
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/notes/a.md", "one\ntwo\nthree")

    out = read("/notes/a.md", offset=99)

    assert "past the end" in out and "3 lines" in out


def test_read_document_rejects_a_non_positive_offset(doc_tools):
    read = _by_name(doc_tools, "read_document")
    assert "1-indexed" in read("/notes/a.md", offset=0)


def test_read_document_advertises_its_window_to_the_model(doc_tools):
    read = _by_name(doc_tools, "read_document")
    params = read.__tool_spec__["function"]["parameters"]
    assert {"max_lines", "offset"} <= set(params["properties"])
    assert params["required"] == ["path"]


def test_search_documents_finds_substring(doc_tools):
    save = _by_name(doc_tools, "save_document")
    search = _by_name(doc_tools, "search_documents")
    save("/notes/a.md", "the gate code is 4242")
    save("/notes/b.md", "buy milk")
    result = search("gate code")
    assert "/notes/a.md" in result and "4242" in result
    assert "buy milk" not in result


def test_search_documents_no_match_message(doc_tools):
    search = _by_name(doc_tools, "search_documents")
    assert "No matching documents" in search("nothing here")


def test_list_documents(doc_tools):
    save = _by_name(doc_tools, "save_document")
    list_docs = _by_name(doc_tools, "list_documents")
    assert "No documents stored" in list_docs()
    save("/a.md", "one")
    save("/b.md", "two")
    result = list_docs()
    assert "/a.md" in result and "/b.md" in result


# ---------------------------------------------------------------------------
# Document tools: a file the store cannot read must be reported to the model,
# and a search must not dump whole documents into the context window.
# ---------------------------------------------------------------------------


def test_list_documents_reports_unreadable_files(tmp_path):
    from aimu.memory import DocumentStore
    from aimu.tools.builtin import make_document_tools

    store = DocumentStore(persist_path=str(tmp_path))
    store.write("/notes.md", "keep me")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4\n\xe9\xff\x00binary")

    result = _by_name(make_document_tools(store), "list_documents")()

    assert "/notes.md" in result
    assert "paper.pdf" in result
    assert "could not be read" in result


def test_list_documents_says_nothing_about_unreadable_when_there_are_none(doc_tools):
    save = _by_name(doc_tools, "save_document")
    save("/a.md", "one")
    assert "could not be read" not in _by_name(doc_tools, "list_documents")()


def test_search_documents_truncates_a_long_document(doc_tools):
    save = _by_name(doc_tools, "save_document")
    search = _by_name(doc_tools, "search_documents")
    save("/paper.md", "findings " + ("x" * 20000))

    result = search("findings")

    assert len(result) < 5000
    assert "read_document" in result
    assert "/paper.md" in result


def test_search_documents_does_not_truncate_a_short_document(doc_tools):
    save = _by_name(doc_tools, "save_document")
    search = _by_name(doc_tools, "search_documents")
    save("/note.md", "the gate code is 4242")

    result = search("gate")

    assert "the gate code is 4242" in result
    assert "read_document" not in result
