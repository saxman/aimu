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
searches pays nothing. Both paths expand `~`. Give each index directory to **one process**: two processes sharing
one would each rebuild over the other's rows.

## Choose which files are documents

By default every file in the folder is a candidate, and one that is not text (an image, a Word export) is reported
as unreadable. A notes vault holds plenty of files that are not notes, so name the kinds that are:

```python
store = DocumentStore(
    "~/Vault",
    index_path="~/.cache/aimu/vault-index",
    extensions=(".md", ".markdown", ".txt", ".pdf"),
)
```

Suffixes match case-insensitively, with or without the leading dot. Any other file is not a candidate at all: it
is not listed, not reported unreadable, not counted, and `write` refuses its path.

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
for problem in status.problems:
    print(problem.kind, problem.message)
```

| Field | Meaning |
| --- | --- |
| `files` | Documents in the folder, less those found unreadable |
| `indexed` | Documents whose current text is keyword-searchable |
| `unreadable` | Files that cannot be documents, as a dict of path to reason |
| `embedded` | Indexed documents that also have vectors (0 without an embedding client) |
| `vectors` | Whether the store has an embedding client, so `embedded` counts toward `complete` |
| `problems` | What is wrong, as `IndexProblem(kind, message)` records (see below); `()` when nothing is |
| `error` | Every problem's message joined into one string, or `None` |
| `complete` | `True` when every document is indexed (and embedded, if `vectors`) |

A problem's `kind` says what it means for a search:

- `"embedding"`: the embedding model failed to load or to embed. Searches are keyword-only until it works, and
  each search retries it. Nothing raises.
- `"file"`: files the index failed to process. Each is left out until it changes; the message names the first.
- `"worker"`: the background thread stopped on a failure outside any one file. The next search restarts it, and
  the problem stays reported until the thread gets work done again.

A file that changed since it was last indexed is never returned with its old text. A file that could not be opened
(a permission change, or a cloud-synced file not downloaded yet) is tried again a minute later, changed or not.
`index_status(refresh=False)` reports on the last search's walk of the folder instead of walking it again. Call
`store.close()` when you are done to stop the background thread; it also exits on its own when idle.

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

agent.tools += make_collection_tools(stores, writable=["work"], descriptions={"work": "Quarterly plans and reviews."})
```

It returns `list_documents`, `read_document`, and `search_documents`, plus `save_document` and `edit_document`
only for the collections named in `writable`, so an agent given read-only folders holds no tool that writes.
`descriptions` puts what each collection holds beside its name in `list_documents`' overview, which is where the
model chooses a collection. Its
search ends with a note when an index is still building or has a problem, so an incomplete search does not read as
one that found nothing; a collection whose search fails is skipped and named in a note while the others' results
come back (`search_collections(..., on_error=...)` does the same for your own code). `list_documents` lists from the
folder's metadata (`DocumentStore.listing`), opening no file, and shows at most 200 paths per collection, then the
folders below the prefix with their sizes so the model can narrow. A file nothing has read yet is listed as a
document; once a search has read it, an unreadable one is listed with its reason. `save_document` refuses to replace
a file that exists but cannot be read. An agent carries either this set or `make_document_tools`', not both.

## The folder stays the source of truth

The index is a derived cache. Everything in it is rebuilt from the files, so deleting `index_path` costs only
rebuild time, and editing, adding, or removing a file in the folder is picked up on the next search. An index that
is damaged when it is opened (a database that is not one or is malformed, or vectors missing while the index says
they exist) is logged and rebuilt once rather than failing every search. Vectors that fail to open for any other
reason (a locked database, an I/O error, a full disk) are kept, since re-embedding a large folder can take hours:
search runs keyword-only, `index_status()` reports it as an `"embedding"` problem, and the next time the store
opens its index it uses the vectors where they were. Do not edit the index by hand. Nothing leaves the machine: the vectors are kept in
Chroma, opened with its anonymized telemetry off, and the embedding model is whichever client you pass.
