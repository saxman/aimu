"""
Tests for MemoryStore class and the memory MCP server tools.
"""

import uuid

import pytest

import aimu.memory.mcp as memory_mcp
from aimu.memory import SemanticMemoryStore


@pytest.fixture
def store():
    """Ephemeral in-memory SemanticMemoryStore for each test."""
    return SemanticMemoryStore(collection_name=str(uuid.uuid4()))


def test_initial_store_is_empty(store):
    assert len(store) == 0


def test_store_increases_count(store):
    store.store("Paul works at Google")
    assert len(store) == 1


def test_store_multiple(store):
    store.store("Paul works at Google")
    store.store("Sarah is the sister of Emma")
    store.store("Paul is married to Sarah")
    assert len(store) == 3


def test_search_empty_store(store):
    results = store.search("work")
    assert results == []


def test_search_returns_list_of_strings(store):
    store.store("Paul works at Google")
    results = store.search("work")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_search_semantic_match(store):
    store.store("Paul works at Google")
    store.store("Sarah is the sister of Emma")
    store.store("Paul is married to Sarah")

    results = store.search("employment and career")
    assert "Paul works at Google" in results


def test_search_respects_n_results(store):
    for i in range(10):
        store.store(f"Person{i} works at Company{i}")

    results = store.search("work", n_results=3)
    assert len(results) <= 3


def test_search_n_results_larger_than_store(store):
    store.store("Paul works at Google")
    store.store("Sarah is the sister of Emma")

    # Requesting more results than stored facts should not raise
    results = store.search("work", n_results=100)
    assert len(results) <= 2


def test_delete_removes_exact_match(store):
    store.store("Paul works at Google")
    store.store("Sarah is the sister of Emma")

    store.delete("Paul works at Google")

    assert len(store) == 1
    results = store.search("Paul works at Google")
    assert "Paul works at Google" not in results


def test_delete_nonexistent_does_not_raise(store):
    store.store("Paul works at Google")
    store.delete("This fact does not exist")
    assert len(store) == 1


def test_delete_only_removes_exact_match(store):
    store.store("Paul works at Google")
    store.store("Paul works at Google")  # duplicate
    store.store("Sarah is the sister of Emma")

    store.delete("Paul works at Google")

    # Both duplicates should be removed; the unrelated fact remains
    assert len(store) == 1


def test_store_and_retrieve_roundtrip(store):
    facts = [
        "Paul works at Google",
        "Sarah is the sister of Emma",
        "Paul is married to Sarah",
        "Emma lives in Paris",
    ]
    for fact in facts:
        store.store(fact)

    assert len(store) == len(facts)

    work_results = store.search("job and employment")
    assert any("works" in r for r in work_results)

    family_results = store.search("family relationships")
    assert any("sister" in r or "married" in r for r in family_results)


def test_persistent_store_survives_reinstantiation(tmp_path):
    store = SemanticMemoryStore(persist_path=str(tmp_path))
    store.store("Paul works at Google")
    store.store("Sarah is the sister of Emma")

    # Reopen the same path; facts must still be there
    store2 = SemanticMemoryStore(persist_path=str(tmp_path))
    assert len(store2) == 2
    results = store2.search("Paul works at Google")
    assert "Paul works at Google" in results


def test_list_all_empty(store):
    assert store.list_all() == []


def test_list_all_returns_all(store):
    facts = ["Paul works at Google", "Sarah is the sister of Emma", "Paul is married to Sarah"]
    for f in facts:
        store.store(f)
    result = store.list_all()
    assert len(result) == 3
    assert set(result) == set(facts)


# ---------------------------------------------------------------------------
# Memory MCP server tool tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_store(monkeypatch):
    """Replace the module-level _store with a fresh ephemeral SemanticMemoryStore."""
    fresh = SemanticMemoryStore(collection_name=str(uuid.uuid4()))
    monkeypatch.setattr(memory_mcp, "_store", fresh)
    return fresh


def test_mcp_add_and_search(mcp_store):
    memory_mcp.add_memories(["Paul works at Google", "Sarah is the sister of Emma"])
    result = memory_mcp.search_memories("Paul works at Google")
    assert "Paul works at Google" in result


def test_mcp_search_empty_store(mcp_store):
    result = memory_mcp.search_memories("anything")
    assert result == ""


def test_mcp_delete_memory(mcp_store):
    memory_mcp.add_memories(["Paul works at Google", "Sarah is the sister of Emma"])
    memory_mcp.delete_memory("Paul works at Google")
    assert len(mcp_store) == 1
    result = memory_mcp.search_memories("Paul works at Google")
    assert "Paul works at Google" not in result


def test_mcp_list_memories_empty(mcp_store):
    assert memory_mcp.list_memories() == []


def test_mcp_list_memories_returns_all(mcp_store):
    facts = ["Paul works at Google", "Sarah is the sister of Emma"]
    memory_mcp.add_memories(facts)
    result = memory_mcp.list_memories()
    assert len(result) == 2
    assert set(result) == set(facts)
