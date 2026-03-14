"""
Tests for MemoryStore class.
"""

import pytest

from aimu.memory import MemoryStore


@pytest.fixture
def store():
    """Ephemeral in-memory MemoryStore for each test."""
    return MemoryStore()


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
