"""aimu.memory._chroma: how AIMU's stores open a Chroma client.

Both :class:`~aimu.memory.SemanticMemoryStore` and the passage index behind
:meth:`~aimu.memory.DocumentStore.search_passages` keep their vectors in Chroma, and both open it here so
they open it the same way: with Chroma's anonymized usage telemetry turned off, since a local memory store
has no reason to send anything off the machine.

The caller imports chromadb (the ``[memory]`` extra) and passes the module in, so this module never imports
it and stays importable without the extra.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def open_client(chromadb: Any, path: str | None) -> Any:
    """A persistent Chroma client at *path*, or an in-memory one when *path* is None, with telemetry off.

    Chroma shares one underlying system per path (and one for every in-memory client) across a process, and
    refuses to open a second client on it with different settings. A host that already opened one with
    Chroma's defaults would make the telemetry setting an error, so in that case the client joins the
    existing system with its settings, and telemetry is whatever that host chose.
    """
    from chromadb.config import Settings

    def construct(**kwargs: Any) -> Any:
        if path is None:
            return chromadb.EphemeralClient(**kwargs)
        return chromadb.PersistentClient(path=path, **kwargs)

    try:
        return construct(settings=Settings(anonymized_telemetry=False))
    except ValueError as exc:
        if "different settings" not in str(exc):
            raise
        logger.info("Joining an existing Chroma client at %s with its own settings: %s", path or "memory", exc)
        return construct()


def close_client(client: Any) -> None:
    """Release *client*'s share of its Chroma system, where the installed chromadb supports that."""
    close = getattr(client, "close", None)
    if close is not None:
        close()


def forget_system(path: str) -> None:
    """Drop the process's cached Chroma system for *path*, so the next client there starts a fresh one.

    A workaround: when opening a client fails (a corrupt ``chroma.sqlite3``, say), chromadb 1.x leaves the
    half-started system in its process-wide cache, and every later client for that path fails against it,
    even once the files are gone. The cache is private to chromadb, so this does nothing where it is absent.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except ImportError:
        return
    systems = getattr(SharedSystemClient, "_identifier_to_system", None)
    if isinstance(systems, dict):
        systems.pop(path, None)
    refcounts = getattr(SharedSystemClient, "_identifier_to_refcount", None)
    if isinstance(refcounts, dict):
        refcounts.pop(path, None)
