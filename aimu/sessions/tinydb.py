"""TinyDB-backed session store (durable across restarts).

Reuses the same JSON/TinyDB mechanics as :class:`~aimu.history.ConversationManager`, one row per
session keyed by ``key``. No new dependency (TinyDB is already a core dependency).
"""

from __future__ import annotations

from pathlib import Path

from tinydb import Query, TinyDB

from aimu.sessions.base import Session, SessionStore


class TinyDBSessionStore(SessionStore):
    """A durable :class:`SessionStore` persisting one document per session to a TinyDB file."""

    def __init__(self, db_path: str = "sessions.json"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = TinyDB(db_path)
        self._table = self._db.table("sessions")
        self._query = Query()

    def get(self, key: str) -> Session:
        row = self._table.get(self._query.key == key)
        if row is None:
            return Session(key=key)
        return Session(
            key=key,
            messages=row.get("messages", []),
            memory_namespace=row.get("memory_namespace"),
            metadata=row.get("metadata", {}),
        )

    def save(self, session: Session) -> None:
        self._table.upsert(
            {
                "key": session.key,
                "messages": session.messages,
                "memory_namespace": session.memory_namespace,
                "metadata": session.metadata,
            },
            self._query.key == session.key,
        )

    def list_keys(self) -> list[str]:
        return [row["key"] for row in self._table.all()]

    def delete(self, key: str) -> None:
        self._table.remove(self._query.key == key)

    def close(self) -> None:
        self._db.close()
