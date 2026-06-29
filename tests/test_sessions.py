"""Tests for aimu.sessions (mock-only: in-memory + TinyDB stores, key derivation, locks)."""

from __future__ import annotations

import asyncio

from aimu.sessions import (
    InMemorySessionStore,
    Session,
    SessionLocks,
    TinyDBSessionStore,
    session_key,
)


def test_session_key_single_user_defaults():
    assert session_key(None, None) == "default:default"
    assert session_key("telegram", "12345") == "telegram:12345"
    assert session_key("web", None) == "web:default"


def test_in_memory_round_trip_and_detached_snapshot():
    store = InMemorySessionStore()
    # Absent key -> fresh empty session, not yet persisted.
    s = store.get("telegram:1")
    assert s.key == "telegram:1" and s.messages == []
    assert store.list_keys() == []

    s.messages.append({"role": "user", "content": "hi"})
    # Mutating the returned session without save does not persist.
    assert store.get("telegram:1").messages == []

    store.save(s)
    assert store.list_keys() == ["telegram:1"]
    assert store.get("telegram:1").messages == [{"role": "user", "content": "hi"}]

    # The stored copy is detached: mutating the saved-from object doesn't change the store.
    s.messages.append({"role": "assistant", "content": "yo"})
    assert len(store.get("telegram:1").messages) == 1


def test_in_memory_sessions_are_isolated_by_key():
    store = InMemorySessionStore()
    store.save(Session(key="a", messages=[{"role": "user", "content": "A"}]))
    store.save(Session(key="b", messages=[{"role": "user", "content": "B"}]))
    assert store.get("a").messages[0]["content"] == "A"
    assert store.get("b").messages[0]["content"] == "B"
    assert set(store.list_keys()) == {"a", "b"}


def test_tinydb_persists_across_restart(tmp_path):
    path = str(tmp_path / "sessions.json")
    store = TinyDBSessionStore(path)
    store.save(Session(key="web:default", messages=[{"role": "user", "content": "remember"}], metadata={"name": "Ada"}))
    store.close()

    # A fresh store over the same file (a "restart") sees the saved session.
    reopened = TinyDBSessionStore(path)
    s = reopened.get("web:default")
    assert s.messages == [{"role": "user", "content": "remember"}]
    assert s.metadata == {"name": "Ada"}
    assert reopened.list_keys() == ["web:default"]
    # Absent key -> fresh empty, not persisted.
    assert reopened.get("nope").messages == []
    reopened.close()


async def test_session_locks_same_key_serializes_diff_keys_concurrent():
    locks = SessionLocks()
    assert locks("a") is locks("a")  # same key -> same lock
    assert locks("a") is not locks("b")

    order: list[str] = []

    async def turn(key: str, label: str, hold: float):
        async with locks(key):
            order.append(f"{label}-start")
            await asyncio.sleep(hold)
            order.append(f"{label}-end")

    # Same key: the second turn must wait for the first to finish (serialized).
    await asyncio.gather(turn("a", "a1", 0.05), turn("a", "a2", 0.0))
    assert order in (["a1-start", "a1-end", "a2-start", "a2-end"], ["a2-start", "a2-end", "a1-start", "a1-end"])

    # Different keys: turns overlap (both start before either ends).
    order.clear()
    await asyncio.gather(turn("x", "x", 0.05), turn("y", "y", 0.05))
    assert order[:2] == ["x-start", "y-start"] or order[:2] == ["y-start", "x-start"]
