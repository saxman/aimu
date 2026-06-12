"""Mock-only tests for the RAG primitives in ``aimu.rag``.

Splitter behaviour, ingest/retrieve over a deterministic in-memory store,
``format_context``, ``rerank`` (with a stubbed cross-encoder), and the
``make_retrieval_tool`` agent-tool factory.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from aimu.rag import format_context, ingest, retrieve, split_text


# ---------------------------------------------------------------------------
# A deterministic store (substring search) so retrieval tests don't depend on embeddings.
# ---------------------------------------------------------------------------


class ListStore:
    def __init__(self):
        self.items: list[str] = []

    def store(self, content: str) -> None:
        self.items.append(content)

    def search(self, query: str, n_results: int = 10, **kwargs) -> list[str]:
        self.last_search_kwargs = kwargs
        hits = [item for item in self.items if query.lower() in item.lower()]
        return hits[:n_results]


# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------


def test_split_short_text_single_chunk():
    assert split_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_split_empty_or_whitespace_returns_empty():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_split_respects_chunk_size():
    text = "\n\n".join(f"Paragraph {i} has several words here." for i in range(12))
    chunks = split_text(text, chunk_size=90, chunk_overlap=20)
    assert len(chunks) > 1
    # Every chunk fits (paragraphs are well under 90, so packing never overflows).
    assert all(len(c) <= 90 for c in chunks)


def test_split_hard_cut_when_no_separator_fits():
    chunks = split_text("x" * 250, chunk_size=100, chunk_overlap=10)
    assert [len(c) for c in chunks] == [100, 100, 70]


def test_split_overlap_carries_content():
    words = " ".join(f"w{i}" for i in range(40))
    word_count = lambda s: len(s.split())  # noqa: E731
    chunks = split_text(words, chunk_size=30, chunk_overlap=10, length_function=word_count)
    # Overlap means total emitted words exceed the original (boundary repetition).
    assert sum(len(c.split()) for c in chunks) > 40


def test_split_token_aware_length_function():
    sentence = " ".join(f"word{i}" for i in range(50))
    word_count = lambda s: len(s.split())  # noqa: E731
    chunks = split_text(sentence, chunk_size=10, chunk_overlap=2, length_function=word_count)
    assert all(len(c.split()) <= 10 for c in chunks)


def test_split_custom_separators():
    chunks = split_text("a|b|c|d|e", chunk_size=3, chunk_overlap=0, separators=["|", ""])
    assert chunks  # splits on the custom pipe separator without error


def test_split_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        split_text("abc", chunk_size=10, chunk_overlap=10)


# ---------------------------------------------------------------------------
# ingest / retrieve / format_context
# ---------------------------------------------------------------------------


def test_ingest_single_document_returns_chunk_count():
    store = ListStore()
    text = "\n\n".join(f"Paragraph {i} with words." for i in range(10))
    n = ingest(store, text, chunk_size=60, chunk_overlap=10)
    assert n == len(store.items) > 1


def test_ingest_accepts_list_of_documents():
    store = ListStore()
    n = ingest(store, ["short one", "short two"], chunk_size=100, chunk_overlap=10)
    assert n == 2
    assert store.items == ["short one", "short two"]


def test_retrieve_passes_through_to_store_search():
    store = ListStore()
    ingest(store, ["the cat sat", "the dog ran", "fiscal policy report"], chunk_size=100, chunk_overlap=0)
    hits = retrieve(store, "cat", n_results=5)
    assert hits == ["the cat sat"]


def test_retrieve_forwards_extra_kwargs():
    store = ListStore()
    store.store("anything")
    retrieve(store, "anything", n_results=3, max_distance=0.5)
    assert store.last_search_kwargs == {"max_distance": 0.5}


def test_format_context_plain_and_numbered():
    chunks = ["alpha", "beta"]
    assert format_context(chunks) == "alpha\n\nbeta"
    assert format_context(chunks, numbered=True) == "[1] alpha\n\n[2] beta"


# ---------------------------------------------------------------------------
# rerank (stubbed cross-encoder)
# ---------------------------------------------------------------------------


def test_rerank_empty_returns_empty_without_loading_model():
    from aimu.rag import rerank

    # No sentence_transformers import happens for empty input.
    assert rerank("q", []) == []


def test_rerank_orders_by_score_and_truncates(monkeypatch):
    import importlib

    # The package binds the `rerank` function as an attribute, shadowing the submodule;
    # import_module returns the real module object so we can patch its encoder registry.
    rerank_mod = importlib.import_module("aimu.rag.rerank")

    class _FakeCrossEncoder:
        def __init__(self, model):
            self.model = model

        def predict(self, pairs):
            # Score = length of the document (second element); longer ranks higher.
            return [len(doc) for _, doc in pairs]

    st_stub = ModuleType("sentence_transformers")
    st_stub.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)
    monkeypatch.setattr(rerank_mod, "_encoder_registry", {})

    docs = ["tiny", "medium length", "the longest document here"]
    ranked = rerank_mod.rerank("query", docs)
    assert ranked == ["the longest document here", "medium length", "tiny"]
    assert rerank_mod.rerank("query", docs, top_n=1) == ["the longest document here"]


# ---------------------------------------------------------------------------
# make_retrieval_tool
# ---------------------------------------------------------------------------


def test_make_retrieval_tool_spec_and_output():
    from aimu.tools.builtin import make_retrieval_tool

    store = ListStore()
    ingest(store, ["the cat sat on the mat", "unrelated finance note"], chunk_size=100, chunk_overlap=0)

    tool = make_retrieval_tool(store, n_results=3)
    assert tool.__tool_spec__["function"]["name"] == "retrieve_context"

    out = tool("cat")
    assert "the cat sat on the mat" in out
    assert out.startswith("[1] ")  # numbered context

    assert tool("no-such-topic") == "No relevant context found."
