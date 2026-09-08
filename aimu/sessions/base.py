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
from copy import deepcopy
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


@dataclass(frozen=True)
class SessionSummary:
    """One stored session, without its messages.

    Everything a :class:`Session` carries except ``messages``, plus how many of them there are. That
    definition is the point: a caller listing conversations wants titles and timestamps, and reading
    every transcript to get them is the cost this type exists to avoid.

    ``message_count`` is the one field that is not a projection, since it is computed over the very
    thing being excluded. It is here because "how many messages" is the most universal summary fact
    about a conversation, and recovering it otherwise means the full read again.

    Attributes:
        key: The session key (see :func:`session_key`).
        message_count: How many messages the stored session holds.
        memory_namespace: The stored session's ``memory_namespace``, unchanged.
        metadata: A detached copy of the stored session's metadata. Mutating it does nothing until
            the corresponding :class:`Session` is fetched, changed, and saved.
    """

    key: str
    message_count: int
    memory_namespace: Optional[str]
    metadata: dict[str, Any]


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

    def list_summaries(self) -> list[SessionSummary]:
        """Every stored session without its messages, in unspecified order.

        Concrete rather than abstract, so an existing implementation keeps working: this default is
        correct everywhere and slow anywhere a full read is expensive, and a store that can do better
        overrides it. The same reasoning as :meth:`close`'s default no-op.

        Deep-copies each session's metadata before handing it back, rather than trusting whatever
        detachment ``get()`` already did. ``SessionSummary.metadata`` promises a caller can mutate it
        freely, and this default has no way to know how deep an arbitrary subclass's ``get()`` copies:
        this library's own ``InMemorySessionStore.get`` copies only one level, which is enough for a
        ``Session`` a caller is expected to fetch, change, and save again, but not enough for a
        summary nobody saves. Overriding ``list_summaries`` (as ``TinyDBSessionStore`` does) can skip
        this copy when the override's own read path already returns freshly-built data.

        No ordering parameter and no paging, because both need a sort key and every candidate lives
        in ``metadata``, which this library treats as opaque. A caller sorts the result by whichever
        of its own keys it means.
        """
        return [
            SessionSummary(
                key=session.key,
                message_count=len(session.messages),
                memory_namespace=session.memory_namespace,
                metadata=deepcopy(session.metadata),
            )
            for session in (self.get(key) for key in self.list_keys())
        ]


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
