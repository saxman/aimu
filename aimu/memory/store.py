"""
aimu.memory.store — Semantic fact storage using subject-predicate-object triples.

Facts like "Paul works at Google" are stored in ChromaDB and can be retrieved
by semantic topic (e.g. "work", "family life") or by exact subject.
"""

from __future__ import annotations

import uuid
from typing import Optional

import chromadb


class MemoryStore:
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
        >>> store = MemoryStore()
        >>> store.store_fact("Paul works at Google")
        >>> store.store_fact("Paul is married to Sarah")
        >>> store.store_fact("Sarah is the sister of Emma")
        >>> store.retrieve_facts("work")        # ["Paul works at Google"]
        >>> store.retrieve_facts("family life") # marriage / sibling facts
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

    def store_fact(self, fact: str) -> None:
        """
        Store a fact string in the memory store.

        The fact should be a natural-language sentence in subject-predicate-object
        form, e.g. "Paul works at Google" or "Sarah is the sister of Emma".

        Args:
            fact: The fact string to store.
        """
        self._collection.add(
            ids=[str(uuid.uuid4())],
            documents=[fact],
        )

    def retrieve_facts(self, topic: str, n_results: int = 10) -> list[str]:
        """
        Retrieve facts semantically related to a topic.

        Uses ChromaDB's vector similarity search so broad topics like "work"
        or "family life" match relevant facts without exact-string matching.

        Args:
            topic:     Natural-language topic to search for (e.g. "work", "family").
            n_results: Maximum number of facts to return.

        Returns:
            List of fact strings ordered by relevance.
        """
        count = len(self)
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[topic],
            n_results=min(n_results, count),
        )
        return results["documents"][0]

    def delete_fact(self, fact: str) -> None:
        """
        Remove all stored facts that exactly match the given string.

        Args:
            fact: The exact fact string to remove.
        """
        results = self._collection.get(where_document={"$contains": fact})
        matching_ids = [doc_id for doc_id, document in zip(results["ids"], results["documents"]) if document == fact]
        if matching_ids:
            self._collection.delete(ids=matching_ids)

    def list_facts(self) -> list[str]:
        """Return all stored fact strings."""
        result = self._collection.get()
        return result["documents"] or []

    def __len__(self) -> int:
        """Return the total number of stored facts."""
        return self._collection.count()
