"""Retrieval-augmented generation primitives.

Plain functions over the :class:`~aimu.memory.MemoryStore` interface -- no retriever /
splitter / loader class hierarchy. The typical flow:

    from aimu.rag import ingest, retrieve, format_context

    ingest(store, documents)                       # chunk + store
    chunks = retrieve(store, "what is X?", n_results=5)
    context = format_context(chunks)               # join for the prompt
    answer = client.chat(f"Context:\\n{context}\\n\\nQuestion: what is X?")

`rerank` (optional, needs the `[hf]` extra) re-scores a wide candidate set with a
cross-encoder. `make_retrieval_tool(store)` (in :mod:`aimu.tools.builtin`) wraps `retrieve`
as an agent tool.
"""

from .pipeline import format_context, ingest, retrieve
from .rerank import rerank
from .splitter import split_text

__all__ = [
    "split_text",
    "ingest",
    "retrieve",
    "format_context",
    "rerank",
]
