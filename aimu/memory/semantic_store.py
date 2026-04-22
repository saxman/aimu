"""
aimu.memory.store: Semantic fact storage using subject-predicate-object triples.

Facts like "Paul works at Google" are stored in ChromaDB and can be retrieved
by semantic topic (e.g. "work", "family life") or by exact subject.
"""

from __future__ import annotations

import uuid
from typing import Optional

import chromadb

from aimu.memory.base import MemoryStore


class SemanticMemoryStore(MemoryStore):
    """
    Persistent store for factual memories backed by ChromaDB.

    Memories are natural-language strings in subject-predicate-object form,
    e.g. "Paul works at Google" or "Sarah is the sister of Emma". ChromaDB
    embeds each fact so that topic-based retrieval ("work", "family life")
    finds semantically relevant facts without exact-string matching.

    Args:
        collection_name: ChromaDB collection name (default: "memories").
        persist_path: Directory path for persistent storage. If None, uses an
            in-memory ephemeral client.

    Examples:
        >>> store = SemanticMemoryStore()
        >>> store.store("Paul works at Google")
        >>> store.store("Paul is married to Sarah")
        >>> store.store("Sarah is the sister of Emma")
        >>> store.search("work")        # ["Paul works at Google"]
        >>> store.search("family life") # marriage / sibling facts
    """

    def __init__(self, collection_name: str = "memories", persist_path: Optional[str] = None):
        if persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # MemoryStore abstract interface
    # ------------------------------------------------------------------

    def store(self, content: str) -> None:
        """
        Store a fact string in the memory store.

        The fact should be a natural-language sentence in subject-predicate-object
        form, e.g. "Paul works at Google" or "Sarah is the sister of Emma".

        Args:
            content: The fact string to store.
        """
        self._collection.add(
            ids=[str(uuid.uuid4())],
            documents=[content],
        )

    def search(self, query: str, n_results: int = 10, max_distance: float | None = None) -> list[str]:
        """
        Retrieve facts semantically related to a query.

        Uses ChromaDB's vector similarity search so broad topics like "work"
        or "family life" match relevant facts without exact-string matching.

        Args:
            query:        Natural-language query (e.g. "work", "family").
            n_results:    Maximum number of facts to return.
            max_distance: Optional cosine-distance cutoff (0 = identical,
                          2 = maximally dissimilar). Facts with a distance
                          above this threshold are excluded. Typical useful
                          range: 0.3 (tight) – 0.6 (loose). Defaults to
                          None (no cutoff).

        Returns:
            List of fact strings ordered by relevance.
        """
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        documents = results["documents"][0]
        if max_distance is not None:
            distances = results["distances"][0]
            documents = [doc for doc, dist in zip(documents, distances) if dist <= max_distance]
        return documents

    def delete(self, identifier: str) -> None:
        """
        Remove all stored facts that exactly match the given string.

        Args:
            identifier: The exact fact string to remove.
        """
        results = self._collection.get(where_document={"$contains": identifier})
        matching_ids = [
            doc_id for doc_id, document in zip(results["ids"], results["documents"]) if document == identifier
        ]
        if matching_ids:
            self._collection.delete(ids=matching_ids)

    def list_all(self) -> list[str]:
        """Return all stored fact strings."""
        result = self._collection.get()
        return result["documents"] or []

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    def store_fact(self, fact: str) -> None:
        """Alias for :meth:`store` (backward compatibility)."""
        self.store(fact)

    def retrieve_facts(self, topic: str, n_results: int = 10, max_distance: float | None = None) -> list[str]:
        """Alias for :meth:`search` (backward compatibility)."""
        return self.search(topic, n_results, max_distance)

    def delete_fact(self, fact: str) -> None:
        """Alias for :meth:`delete` (backward compatibility)."""
        self.delete(fact)

    def list_facts(self) -> list[str]:
        """Alias for :meth:`list_all` (backward compatibility)."""
        return self.list_all()
