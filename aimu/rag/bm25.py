"""BM25 relevance scores: ranking passages with no model and no dependency.

Used where text has to be ranked against a question and nothing heavier can be assumed, such as
choosing which passages of a fetched page enter an agent's context
(:func:`~aimu.tools.builtin.make_page_tools`). Lexical matching misses paraphrase, and an embedding
model ranks markedly better when one is available (BrowseComp-Plus, arXiv 2508.06600, measured a
large gap between BM25 and a dense retriever). Any ``(query, documents) -> scores`` callable can stand
in for this one, e.g. ``make_page_tools(ranker=...)`` over an embedding client.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Collection

_TOKEN = re.compile(r"\w+")

#: English function words dropped from a query before scoring. BM25's IDF stays positive, so without
#: this every passage sharing "the" or "is" with a natural-language question scores above zero, and
#: a caller that treats 0 as "no match" (``make_page_tools``) would never see one.
STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in into is it its "
    "me my of on or our should so than that the their them then there these they this those to us was "
    "we were what when where which who whom why will with would you your".split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    stopwords: Collection[str] = STOPWORDS,
) -> list[float]:
    """Score each document's relevance to *query* with Okapi BM25.

    Args:
        query: The text to rank against.
        documents: The candidates, scored as a collection (term rarity is measured across them).
        k1: Term-frequency saturation.
        b: Length normalization, from 0 (none) to 1 (full).
        stopwords: Query words to ignore. Defaults to :data:`STOPWORDS`, which is English; pass
            ``frozenset()`` to score every word.

    Returns:
        One score per document, in input order; 0.0 for a document sharing no term with the
        query. An empty ``documents`` returns ``[]``.
    """
    tokenized = [_tokens(document) for document in documents]
    if not tokenized:
        return []
    query_terms = set(_tokens(query)) - set(stopwords)
    count = len(tokenized)
    average_length = sum(len(tokens) for tokens in tokenized) / count or 1.0
    frequency = Counter(term for tokens in tokenized for term in set(tokens) if term in query_terms)
    # The Lucene form of IDF, which stays positive for a term in more than half the documents:
    # the classic form goes negative there and would rank a document lower for matching.
    idf = {term: math.log(1 + (count - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    scores = []
    for tokens in tokenized:
        counts = Counter(tokens)
        normalizer = k1 * (1 - b + b * len(tokens) / average_length)
        scores.append(
            sum(
                idf[term] * counts[term] * (k1 + 1) / (counts[term] + normalizer)
                for term in query_terms
                if counts[term]
            )
        )
    return scores
