"""RAG pipeline helpers over the :class:`~aimu.memory.MemoryStore` interface.

Plain functions, not a retriever class tree: `ingest` chunks documents into a store,
`retrieve` pulls relevant chunks back out, and `format_context` joins them for prompt
augmentation. Any `MemoryStore` works (``SemanticMemoryStore`` for vector search,
``DocumentStore`` for substring search, or a custom one) -- they share `store()` / `search()`.
"""

from __future__ import annotations

from typing import Any, Callable, Union

from .splitter import split_text


def ingest(
    store: Any,
    documents: Union[str, list[str]],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
    length_function: Callable[[str], int] = len,
) -> int:
    """Split *documents* into chunks and store each chunk in *store*.

    ``chunk_size``, ``chunk_overlap``, ``separators``, and ``length_function`` are forwarded
    to :func:`~aimu.rag.split_text`.

    Args:
        store: Any object with a ``store(content: str)`` method (a
            :class:`~aimu.memory.MemoryStore`).
        documents: A single document string or a list of them.

    Returns:
        The number of chunks stored.
    """
    if isinstance(documents, str):
        documents = [documents]

    count = 0
    for document in documents:
        for chunk in split_text(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=length_function,
        ):
            store.store(chunk)
            count += 1
    return count


def retrieve(store: Any, query: str, *, n_results: int = 5, **search_kwargs: Any) -> list[str]:
    """Return up to *n_results* stored chunks relevant to *query*.

    A thin, RAG-named pass-through to ``store.search()``. Extra keyword arguments are
    forwarded to the store (e.g. ``max_distance=`` for
    :class:`~aimu.memory.SemanticMemoryStore`). To rerank, fetch a wider set here and pass
    it through :func:`~aimu.rag.rerank`.
    """
    return store.search(query, n_results=n_results, **search_kwargs)


def format_context(chunks: list[str], *, separator: str = "\n\n", numbered: bool = False) -> str:
    """Join retrieved chunks into a single context string for prompt augmentation.

    With ``numbered=True`` each chunk is prefixed ``[1] ``, ``[2] `` ... (useful when you
    want the model to cite which chunk it used).
    """
    if numbered:
        return separator.join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
    return separator.join(chunks)
