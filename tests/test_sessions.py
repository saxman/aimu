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


def test_delete_removes_session_and_is_noop_when_absent(tmp_path):
    for store in (InMemorySessionStore(), TinyDBSessionStore(str(tmp_path / "sessions.json"))):
        store.save(Session(key="a", messages=[{"role": "user", "content": "A"}]))
        store.save(Session(key="b", messages=[{"role": "user", "content": "B"}]))
        store.delete("a")
        assert store.list_keys() == ["b"]
        assert store.get("a").messages == []  # gone -> fresh empty
        store.delete("missing")  # no-op, does not raise
        assert store.list_keys() == ["b"]


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


async def test_routing_isolates_and_accumulates_session_history():
    """The documented restore->run->snapshot routing keeps each session's history isolated and
    accumulating across turns. Guards the gotcha: the system prompt must be on the client (not the
    agent), else `_prepare_run` resets the client every run and wipes the restored history."""
    from aimu import aio
    from helpers_aio import MockAsyncModelClient

    client = MockAsyncModelClient(["a1", "b1", "a2"])
    client.system_message = "You are helpful."  # on the client; agent.system_message stays None
    agent = aio.Agent(client)
    store = InMemorySessionStore()
    locks = SessionLocks()

    async def handle(channel, sender, text):
        async with locks(session_key(channel, sender)):
            s = store.get(session_key(channel, sender))
            agent.restore(s.messages)
            await agent.run(text)
            s.messages = list(agent.model_client.messages)
            store.save(s)

    await handle("web", "ada", "I am Ada")
    await handle("web", "bob", "I am Bob")
    await handle("web", "ada", "again")  # second Ada turn must still see her first

    ada_users = [m["content"] for m in store.get("web:ada").messages if m.get("role") == "user"]
    bob_users = [m["content"] for m in store.get("web:bob").messages if m.get("role") == "user"]
    assert ada_users == ["I am Ada", "again"]  # restore survived run (history accumulated)
    assert bob_users == ["I am Bob"]  # no cross-session leakage
    assert set(store.list_keys()) == {"web:ada", "web:bob"}
