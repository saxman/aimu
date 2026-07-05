"""Multi-user session state: per-conversation history keyed by ``channel:sender``.

A :class:`Session` holds one conversation's message history (plain ``list[dict]`` in OpenAI
format, the same data an agent uses) plus an optional per-session memory namespace and free-form
metadata. A :class:`SessionStore` persists sessions by key so one process can serve many users /
chats. Sync, matching the :class:`~aimu.memory.MemoryStore` / :class:`~aimu.history.ConversationManager`
family; the async assistant loop calls these directly (in-memory and small TinyDB ops are cheap),
exactly as it already uses ``ConversationManager``.

Routing pattern (no new agent primitive needed): per inbound message, under that session's lock,
``model_client.reset(system_message=...)`` then ``agent.restore(session.messages)``, run the turn,
snapshot ``agent.model_client.messages`` back onto the session, and ``store.save(session)``. Agents
never share a live ``messages`` list across sessions.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


def session_key(channel: Optional[str], sender: Optional[str]) -> str:
    """Canonical session key from a message's ``channel`` + ``sender``.

    Single-user transports (no channel/sender) collapse to ``"default:default"``, so single-user
    usage needs no ceremony.
    """
    return f"{channel or 'default'}:{sender or 'default'}"


@dataclass
class Session:
    """One conversation's persisted state.

    Attributes:
        key: The session key (see :func:`session_key`).
        messages: Conversation history as ``list[dict]`` in OpenAI format (plain data).
        memory_namespace: Optional scope a caller can pass to a ``MemoryStore`` to isolate this
            session's memories (e.g. a collection name / prefix). The store ABC is unchanged.
        metadata: Free-form, opaque to the library (display name, locale, ...).
    """

    key: str
    messages: list[dict] = field(default_factory=list)
    memory_namespace: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStore(ABC):
    """A store of :class:`Session` state keyed by ``session_key(channel, sender)``.

    ``get`` returns a detached snapshot (mutating it does nothing until ``save``); ``save`` persists.
    """

    @abstractmethod
    def get(self, key: str) -> Session:
        """Return the session for ``key``, or a fresh empty :class:`Session` if none is stored."""
        ...

    @abstractmethod
    def save(self, session: Session) -> None:
        """Persist ``session`` (create or replace by ``session.key``)."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """Return the keys of all saved sessions."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the session for ``key``. A no-op if no such session exists."""
        ...

    def close(self) -> None:
        """Release resources. Default no-op."""
        return None


class SessionLocks:
    """Lazily-created per-key ``asyncio.Lock``.

    Serializes a single session's turns (shared message state, ordering) while letting different
    sessions run concurrently::

        locks = SessionLocks()
        async with locks(key):
            ...  # this session's turn

    Created locks are retained, so the same key always returns the same lock. Bound to the running
    event loop via ``asyncio.Lock``; use one ``SessionLocks`` per assistant process.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def __call__(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock
