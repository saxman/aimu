"""aimu.rag.fusion: combine several rankings of the same items into one.

Reciprocal rank fusion (Cormack, Clarke, and Buettcher, 2009) scores each item by the ranks it reached, not by
the scores that put it there: ``sum(1 / (k + rank))`` over every ranking that holds it. That is what makes it
the usual way to merge a keyword ranking with a vector one. BM25 scores and cosine distances are on scales
with nothing in common, and normalizing either is a guess, while a rank means the same thing in both.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse *rankings* (each a sequence of keys, best first) into one ranking.

    Args:
        rankings: The rankings to fuse. A key may appear in any number of them; within one ranking only its
            first (best) position counts.
        k: Damps the advantage of the very top ranks. 60 is the value the original paper found robust.

    Returns:
        ``(key, fused score)`` pairs, highest score first. Ties keep the order in which their keys first
        appear, reading the rankings in the order given.

    Raises:
        ValueError: If *k* is negative.
    """
    if k < 0:
        raise ValueError(f"k must be zero or more. Got: {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    # sorted() is stable and the dict keeps first-appearance order, so equal scores stay in that order.
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
