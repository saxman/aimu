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
    assert names == {"save_document", "read_document", "list_documents", "search_documents"}
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
