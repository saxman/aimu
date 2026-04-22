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


def test_store_fact_increases_count(store):
    store.store_fact("Paul works at Google")
    assert len(store) == 1


def test_store_multiple_facts(store):
    store.store_fact("Paul works at Google")
    store.store_fact("Sarah is the sister of Emma")
    store.store_fact("Paul is married to Sarah")
    assert len(store) == 3


def test_retrieve_facts_empty_store(store):
    results = store.retrieve_facts("work")
    assert results == []


def test_retrieve_facts_returns_list_of_strings(store):
    store.store_fact("Paul works at Google")
    results = store.retrieve_facts("work")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_retrieve_facts_semantic_match(store):
    store.store_fact("Paul works at Google")
    store.store_fact("Sarah is the sister of Emma")
    store.store_fact("Paul is married to Sarah")

    results = store.retrieve_facts("employment and career")
    assert "Paul works at Google" in results


def test_retrieve_facts_respects_n_results(store):
    for i in range(10):
        store.store_fact(f"Person{i} works at Company{i}")

    results = store.retrieve_facts("work", n_results=3)
    assert len(results) <= 3


def test_retrieve_facts_n_results_larger_than_store(store):
    store.store_fact("Paul works at Google")
    store.store_fact("Sarah is the sister of Emma")

    # Requesting more results than stored facts should not raise
    results = store.retrieve_facts("work", n_results=100)
    assert len(results) <= 2


def test_delete_fact_removes_exact_match(store):
    store.store_fact("Paul works at Google")
    store.store_fact("Sarah is the sister of Emma")

    store.delete_fact("Paul works at Google")

    assert len(store) == 1
    results = store.retrieve_facts("Paul works at Google")
    assert "Paul works at Google" not in results


def test_delete_fact_nonexistent_does_not_raise(store):
    store.store_fact("Paul works at Google")
    store.delete_fact("This fact does not exist")
    assert len(store) == 1


def test_delete_fact_only_removes_exact_match(store):
    store.store_fact("Paul works at Google")
    store.store_fact("Paul works at Google")  # duplicate
    store.store_fact("Sarah is the sister of Emma")

    store.delete_fact("Paul works at Google")

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
        store.store_fact(fact)

    assert len(store) == len(facts)

    work_results = store.retrieve_facts("job and employment")
    assert any("works" in r for r in work_results)

    family_results = store.retrieve_facts("family relationships")
    assert any("sister" in r or "married" in r for r in family_results)


def test_persistent_store_survives_reinstantiation(tmp_path):
    store = SemanticMemoryStore(persist_path=str(tmp_path))
    store.store_fact("Paul works at Google")
    store.store_fact("Sarah is the sister of Emma")

    # Reopen the same path; facts must still be there
    store2 = SemanticMemoryStore(persist_path=str(tmp_path))
    assert len(store2) == 2
    results = store2.retrieve_facts("Paul works at Google")
    assert "Paul works at Google" in results


def test_list_facts_empty(store):
    assert store.list_facts() == []


def test_list_facts_returns_all(store):
    facts = ["Paul works at Google", "Sarah is the sister of Emma", "Paul is married to Sarah"]
    for f in facts:
        store.store_fact(f)
    result = store.list_facts()
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
