# Search document folders

`DocumentStore.search_passages` answers "which part of which file?" for a folder of notes, papers, or
reference material. Where [`search_full_text`](use-document-memory.md) returns whole documents that contain a
substring, this returns ranked **passages**: chunks of about 1200 characters, each tagged with the file and the
heading (or PDF page) it came from, so an agent can cite its source.

## Index a folder

```python
from aimu.memory import DocumentStore

store = DocumentStore("~/Vault", index_path="~/.cache/aimu/vault-index")

for passage in store.search_passages("sampling temperature", n_results=5):
    print(passage.path, passage.location, round(passage.score, 3))
    print(passage.text)
```

Each `Passage` has `path`, `location` (a heading path such as `"Methods > Sampling"`, `"page 3"` for a PDF, or
`""` before any heading), `text`, `score`, and `digest` (a hash of the whole file's text). `score` is higher for a
better match and comparable only within one search.

`index_path` is where the index lives, and it must be **outside** the folder: a `ValueError` is raised if it lies
inside, so the index is never mistaken for a document. Leave it as `None` and the index is held in memory and
rebuilt each run. Nothing is built until the first `search_passages` or `index_status` call, so a store nobody
searches pays nothing.

Any query is accepted. Only its words reach the index, so punctuation or search syntax in a user's question never
raises, and common English words are ignored.

## Keyword only by default

Without an embedding client the index is a keyword (SQLite FTS5, BM25) index. That needs no extra install and
works well when the question shares words with the answer.

## Add vectors

Pass an embedding client and a search fuses keyword and vector rankings (reciprocal rank fusion), which finds
passages that mean the same thing in other words:

```python
from aimu.models import EmbeddingClient

store = DocumentStore(
    "~/Vault",
    index_path="~/.cache/aimu/vault-index",
    embedding_client=EmbeddingClient("hf:BAAI/bge-base-en-v1.5"),
)
```

Vectors need the `[memory]` extra (`pip install "aimu[memory]"`) and, for an `hf:` model, the `[hf]` extra. They
are imported on the first search, not at import, so a missing extra raises `ImportError` naming it there. Changing
the embedding model re-embeds the folder on its own.

## The first build runs in the background

A search never waits for a large folder to be indexed. The first search indexes what it can in about two seconds,
and a background thread finishes the rest and does all the embedding, so early results may come from part of the
folder. Check progress with `index_status()`:

```python
status = store.index_status()
print(f"{status.indexed}/{status.files} indexed, {status.embedded} embedded")
if status.error:
    print(status.error)
```

| Field | Meaning |
| --- | --- |
| `files` | Documents in the folder, less those found unreadable |
| `indexed` | Documents whose current text is keyword-searchable |
| `unreadable` | Files that cannot be documents, as a dict of path to reason |
| `embedded` | Indexed documents that also have vectors (0 without an embedding client) |
| `vectors` | Whether the store has an embedding client, so `embedded` counts toward `complete` |
| `error` | What is wrong, if anything: the embedding model failing, or files the index could not process |
| `complete` | `True` when every document is indexed (and embedded, if `vectors`) |

An embedding failure never raises into a search: it falls back to keyword-only results and `error` says why. A
file that changed since it was last indexed is never returned with its old text. Call `store.close()` when you are
done to stop the background thread; it also exits on its own when idle.

## PDFs

PDFs are read as Markdown with one `## Page N` heading per page, so passages from a paper cite `"page 3"`. They
are read-only, and a scanned or password-protected PDF is reported as unreadable rather than indexed. See
[Use document memory](use-document-memory.md).

## Search several folders

`search_collections` ranks passages across a `{name: store}` mapping:

```python
from aimu.memory import DocumentStore, search_collections

stores = {
    "personal": DocumentStore("~/Vault", index_path="~/.cache/aimu/personal"),
    "work": DocumentStore("~/Work", index_path="~/.cache/aimu/work"),
}

for hit in search_collections(stores, "quarterly goals", n_results=5):
    print(hit.collection, hit.passage.path, hit.passage.location, hit.also_in)
```

A passage that appears verbatim in several folders is returned once, with the other copies listed in `also_in`.
Pass `rerank=True` to rerank the top results with a cross-encoder (needs the `[hf]` extra).

To give an agent the same folders, use `make_collection_tools`, the multi-folder counterpart to
`make_document_tools`:

```python
from aimu.tools.builtin import make_collection_tools

agent.tools += make_collection_tools(stores, writable=["work"])
```

It returns `list_documents`, `read_document`, and `search_documents`, plus `save_document` and `edit_document`
only for the collections named in `writable`, so an agent given read-only folders holds no tool that writes. Its
search ends with a note when an index is still building or has an error, so an incomplete search does not read as
one that found nothing. An agent carries either this set or `make_document_tools`', not both.

## The folder stays the source of truth

The index is a derived cache. Everything in it is rebuilt from the files, so deleting `index_path` costs only
rebuild time, and editing, adding, or removing a file in the folder is picked up on the next search. Do not
edit the index by hand.
