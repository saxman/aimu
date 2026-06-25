"""
aimu.memory.store: Semantic fact storage using subject-predicate-object triples.

Facts like "Paul works at Google" are stored in ChromaDB and can be retrieved
by semantic topic (e.g. "work", "family life") or by exact subject.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import chromadb

from aimu.memory.base import MemoryStore


class _EmbeddingClientFunction(chromadb.EmbeddingFunction):
    """Adapt an AIMU embedding client to ChromaDB's ``EmbeddingFunction`` protocol.

    ChromaDB calls this with a list of documents and expects one vector per document;
    it delegates to the client's ``embed()`` (which returns ``list[list[float]]`` for a
    list input). Lets a caller pick the embedding model instead of ChromaDB's built-in
    default.
    """

    def __init__(self, client: Any):
        self._client = client

    def __call__(self, input: Any) -> Any:  # noqa: A002 - ChromaDB requires the param named `input`
        return self._client.embed(list(input))

    @staticmethod
    def name() -> str:
        return "aimu_embedding_client"

    def get_config(self) -> Any:
        # The wrapped live client isn't serialisable. Returning the NotImplemented sentinel
        # is ChromaDB's signal to skip embedding-function config persistence (avoids a
        # per-call deprecation warning). Reopen a persistent store with the same
        # embedding_client= (documented on SemanticMemoryStore).
        return NotImplemented


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
        embedding_client: Optional AIMU embedding client (e.g. from
            ``aimu.embedding_client("openai:text-embedding-3-small")``) to embed facts.
            When ``None`` (default), ChromaDB's built-in default embedding model is used,
            preserving the original behaviour exactly. A custom embedding model is not
            persisted in the collection config, so reopen a persistent store with the same
            ``embedding_client=``.

    Examples:
        >>> store = SemanticMemoryStore()
        >>> store.store("Paul works at Google")
        >>> store.store("Paul is married to Sarah")
        >>> store.store("Sarah is the sister of Emma")
        >>> store.search("work")        # ["Paul works at Google"]
        >>> store.search("family life") # marriage / sibling facts

        >>> client = aimu.embedding_client("openai:text-embedding-3-small")
        >>> store = SemanticMemoryStore(embedding_client=client)
    """

    def __init__(
        self,
        collection_name: str = "memories",
        persist_path: Optional[str] = None,
        embedding_client: Optional[Any] = None,
    ):
        if persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()

        collection_kwargs: dict[str, Any] = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if embedding_client is not None:
            collection_kwargs["embedding_function"] = _EmbeddingClientFunction(embedding_client)

        self._collection = self._client.get_or_create_collection(**collection_kwargs)

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

        Note: ``max_distance`` is specific to this store and is **not** part of the
        :class:`~aimu.memory.base.MemoryStore` interface; code typed against the base
        ``MemoryStore`` (or a ``DocumentStore``, which has no distance notion) cannot pass
        it. Type the variable as ``SemanticMemoryStore`` to use it.

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
