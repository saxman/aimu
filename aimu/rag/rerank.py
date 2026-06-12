"""Optional cross-encoder reranking for retrieved chunks.

A retriever (vector or keyword search) is fast but coarse. Reranking re-scores each
candidate against the query with a cross-encoder -- slower but more accurate -- so you
fetch a wide set, then keep the best few. Backed by ``sentence-transformers`` (the `[hf]`
extra). Plain function, not a class.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cross-encoders are reused across calls (weights load once), keyed by model id.
_encoder_registry: dict[str, Any] = {}
_registry_lock = threading.Lock()


def _get_encoder(model: str) -> Any:
    with _registry_lock:
        if model in _encoder_registry:
            return _encoder_registry[model]
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError("rerank() requires sentence-transformers (the [hf] extra): pip install -e '.[hf]'") from exc
    encoder = CrossEncoder(model)
    with _registry_lock:
        _encoder_registry[model] = encoder
    return encoder


def rerank(
    query: str,
    documents: list[str],
    *,
    model: str = DEFAULT_RERANK_MODEL,
    top_n: Optional[int] = None,
) -> list[str]:
    """Reorder *documents* by cross-encoder relevance to *query*, most relevant first.

    Args:
        query: The search query.
        documents: Candidate chunks (e.g. the output of :func:`~aimu.rag.retrieve` with a
            generous ``n_results``).
        model: A sentence-transformers ``CrossEncoder`` model id.
        top_n: Keep only the top N after reranking (default: keep all, reordered).

    Returns:
        The documents reordered by descending relevance, truncated to ``top_n`` if set.
        An empty input returns ``[]`` without loading the model.
    """
    documents = list(documents)
    if not documents:
        return []
    encoder = _get_encoder(model)
    scores = encoder.predict([(query, document) for document in documents])
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    ordered = [document for document, _ in ranked]
    return ordered[:top_n] if top_n is not None else ordered
