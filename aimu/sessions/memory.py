"""In-memory session store (the trivial default; not durable)."""

from __future__ import annotations

from aimu.sessions.base import Session, SessionStore


class InMemorySessionStore(SessionStore):
    """A non-durable :class:`SessionStore` backed by a dict. Sessions are lost on process exit.

    ``get`` and ``save`` copy, so the stored state is detached from the returned/passed object
    (same contract as the persisted stores).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, key: str) -> Session:
        stored = self._sessions.get(key)
        if stored is None:
            return Session(key=key)
        return Session(
            key=key,
            messages=list(stored.messages),
            memory_namespace=stored.memory_namespace,
            metadata=dict(stored.metadata),
        )

    def save(self, session: Session) -> None:
        self._sessions[session.key] = Session(
            key=session.key,
            messages=list(session.messages),
            memory_namespace=session.memory_namespace,
            metadata=dict(session.metadata),
        )

    def list_keys(self) -> list[str]:
        return list(self._sessions.keys())

    def delete(self, key: str) -> None:
        self._sessions.pop(key, None)
