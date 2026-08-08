"""
aimu.memory.base: Abstract base class for memory store implementations.

Defines the minimal common interface so that SemanticMemoryStore,
DocumentStore, and any future implementations are interchangeable
in applications.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable)


def synchronized(method: _F) -> _F:
    """Serialize a store method on the instance's ``self._lock`` (a ``threading.RLock``).

    Store methods run in worker threads (e.g. an async agent dispatches sync tools via
    ``asyncio.to_thread``), so a store shared across concurrent turns can be entered from several
    threads at once. Wrapping the public methods with this makes each store safe under that
    concurrency; the lock is re-entrant so a method may call another (``edit`` -> ``read`` + ``write``).
    Concrete stores must set ``self._lock = threading.RLock()`` in ``__init__``.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class MemoryStore(ABC):
    """
    Abstract interface for memory store implementations.

    All implementations must support storing, searching, deleting, and
    listing content.  Concrete subclasses add implementation-specific
    methods on top of this baseline.
    """

    @abstractmethod
    def store(self, content: str) -> None:
        """Store a string in the memory store."""

    @abstractmethod
    def search(self, query: str, n_results: int = 10) -> list[str]:
        """Return up to *n_results* strings relevant to *query*.

        Concrete stores may accept additional store-specific keyword arguments (e.g.
        ``SemanticMemoryStore.search(..., max_distance=...)``); those are not part of this
        interface, so code typed against ``MemoryStore`` should not rely on them.
        """

    @abstractmethod
    def delete(self, identifier: str) -> None:
        """Remove content by identifier (path, exact string, or UUID)."""

    @abstractmethod
    def list_all(self) -> list[str]:
        """Return all stored identifiers or content strings."""

    def __len__(self) -> int:
        """Return the number of stored items."""
        return len(self.list_all())
