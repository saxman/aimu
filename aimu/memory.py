"""
aimu.memory — Semantic fact storage using subject-predicate-object triples.

Facts like "Paul works at Google" are stored in ChromaDB and can be retrieved
by semantic topic (e.g. "work", "family life") or by exact subject.

MCP server
----------
This module also exposes MemoryStore operations as MCP tools via a FastMCP
server. The storage path is configured with the MEMORY_STORE_PATH environment
variable (defaults to ./memory_store).

Run as a standalone MCP server::

    python -m aimu.memory

Or connect programmatically::

    from aimu.tools.client import MCPClient
    from aimu.memory import mcp
    client = MCPClient(server=mcp)
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import chromadb
from fastmcp import FastMCP


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

    def __len__(self) -> int:
        """Return the total number of stored facts."""
        return self._collection.count()


_DEFAULT_PERSIST_PATH = os.environ.get("MEMORY_STORE_PATH", "./memory_store")

mcp = FastMCP("AIMU Memory")
_store = MemoryStore(persist_path=_DEFAULT_PERSIST_PATH)


@mcp.tool()
def search_memories(search_request: str) -> str:
    """
    Search for memories about the user.

    Args:
        search_request: Information about the user that's relevant to the conversation.

    Returns:
        Information about the user that's relevant to the conversation.
    """
    facts = _store.retrieve_facts(search_request)
    return "\n".join(facts)


@mcp.tool()
def add_memories(memories: list[str]) -> None:
    """
    Add memories about the user.

    Args:
        memories: A list of memories, as strings, to save about the user.
    """
    for fact in memories:
        _store.store_fact(fact)


if __name__ == "__main__":
    mcp.run()
