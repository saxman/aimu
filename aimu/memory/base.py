"""
aimu.memory.base — Abstract base class for memory store implementations.

Defines the minimal common interface so that SemanticMemoryStore,
DocumentStore, and any future implementations are interchangeable
in applications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
        """Return up to *n_results* strings relevant to *query*."""

    @abstractmethod
    def delete(self, identifier: str) -> None:
        """Remove content by identifier (path, exact string, or UUID)."""

    @abstractmethod
    def list_all(self) -> list[str]:
        """Return all stored identifiers or content strings."""

    def __len__(self) -> int:
        """Return the number of stored items."""
        return len(self.list_all())
