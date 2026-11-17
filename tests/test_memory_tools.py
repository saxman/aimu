"""Mock-only unit tests for make_memory_tools and its make_tools integration.

No ChromaDB or heavy deps required; uses an in-memory MemoryStore stub.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aimu.tools import builtin
from aimu.tools.builtin import make_memory_tools


# ---------------------------------------------------------------------------
# Minimal in-memory store stub (no ChromaDB needed)
# ---------------------------------------------------------------------------


class _SimpleStore:
    def __init__(self):
        self._items: list[str] = []

    def store(self, content: str) -> None:
        self._items.append(content)

    def search(self, query: str, n_results: int = 10) -> list[str]:
        return [i for i in self._items if query.lower() in i.lower()][:n_results]

    def delete(self, identifier: str) -> None:
        self._items = [i for i in self._items if i != identifier]

    def list_all(self) -> list[str]:
        return list(self._items)


@pytest.fixture
def store():
    return _SimpleStore()


@pytest.fixture
def tools(store):
    return make_memory_tools(store)


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_returns_three_tools(tools):
    assert len(tools) == 3


def test_tool_names(tools):
    names = {t.__tool_spec__["function"]["name"] for t in tools}
    assert names == {"store_memory", "search_memories", "list_memories"}


def test_store_memory_spec(tools):
    store_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "store_memory")
    params = store_tool.__tool_spec__["function"]["parameters"]
    assert "content" in params["properties"]
    assert params["required"] == ["content"]


def test_search_memories_spec(tools):
    search_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "search_memories")
    params = search_tool.__tool_spec__["function"]["parameters"]
    assert "query" in params["properties"]
    assert "n_results" in params["properties"]
    assert params["required"] == ["query"]


def test_list_memories_spec(tools):
    list_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "list_memories")
    params = list_tool.__tool_spec__["function"]["parameters"]
    assert params.get("required", []) == []


def test_tools_are_sync_and_not_streaming(tools):
    for t in tools:
        assert t.__tool_is_async__ is False
        assert t.__tool_is_streaming__ is False


# ---------------------------------------------------------------------------
# Functional behaviour
# ---------------------------------------------------------------------------


def test_store_memory_stores_and_returns_confirmation(store, tools):
    store_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "store_memory")
    result = store_tool("the sky is blue")
    assert result == "Stored."
    assert "the sky is blue" in store.list_all()


def test_search_memories_returns_matches(store, tools):
    store.store("the sky is blue")
    store.store("the grass is green")
    search_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "search_memories")
    result = search_tool("sky")
    assert "the sky is blue" in result
    assert "grass" not in result


def test_search_memories_no_results_message(store, tools):
    search_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "search_memories")
    result = search_tool("nothing matches this")
    assert "No relevant memories found" in result


def test_list_memories_returns_all(store, tools):
    store.store("fact one")
    store.store("fact two")
    list_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "list_memories")
    result = list_tool()
    assert "fact one" in result
    assert "fact two" in result


def test_list_memories_empty_message(store, tools):
    list_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "list_memories")
    result = list_tool()
    assert "empty" in result.lower()


def test_each_call_creates_independent_tools():
    store_a = _SimpleStore()
    store_b = _SimpleStore()
    tools_a = make_memory_tools(store_a)
    tools_b = make_memory_tools(store_b)

    store_fn_a = next(t for t in tools_a if t.__tool_spec__["function"]["name"] == "store_memory")
    store_fn_b = next(t for t in tools_b if t.__tool_spec__["function"]["name"] == "store_memory")

    store_fn_a("only in a")
    assert store_a.list_all() == ["only in a"]
    assert store_b.list_all() == []

    store_fn_b("only in b")
    assert store_a.list_all() == ["only in a"]
    assert store_b.list_all() == ["only in b"]


# ---------------------------------------------------------------------------
# make_tools integration
# ---------------------------------------------------------------------------


def _fake_base_client(supports_vision=False):
    client = MagicMock()
    client.model = MagicMock()
    client.model.supports_vision = supports_vision
    return client


def test_make_tools_without_memory_store_excludes_memory_tools():
    tools = builtin.make_tools(_fake_base_client())
    names = {t.__tool_spec__["function"]["name"] for t in tools}
    assert "store_memory" not in names
    assert "search_memories" not in names
    assert "list_memories" not in names


def test_make_tools_with_memory_store_appends_memory_tools():
    store = _SimpleStore()
    tools = builtin.make_tools(_fake_base_client(), memory_store=store)
    names = {t.__tool_spec__["function"]["name"] for t in tools}
    assert {"store_memory", "search_memories", "list_memories"}.issubset(names)


def test_make_tools_memory_tools_bound_to_provided_store():
    store = _SimpleStore()
    tools = builtin.make_tools(_fake_base_client(), memory_store=store)
    store_tool = next(t for t in tools if t.__tool_spec__["function"]["name"] == "store_memory")
    store_tool("bound correctly")
    assert "bound correctly" in store.list_all()


# ---------------------------------------------------------------------------
# make_document_tools (real ephemeral DocumentStore: pure-Python, no deps)
# ---------------------------------------------------------------------------


from aimu.memory import DocumentStore  # noqa: E402
from aimu.tools.builtin import make_document_tools  # noqa: E402


@pytest.fixture
def doc_store():
    return DocumentStore()  # ephemeral, in-memory


@pytest.fixture
def doc_tools(doc_store):
    return make_document_tools(doc_store)


def _by_name(tools, name):
    return next(t for t in tools if t.__tool_spec__["function"]["name"] == name)


def test_document_tools_names_and_specs(doc_tools):
    names = {t.__tool_spec__["function"]["name"] for t in doc_tools}
    assert names == {"save_document", "read_document", "edit_document", "list_documents", "search_documents"}
    for t in doc_tools:
        assert t.__tool_is_async__ is False
        assert t.__tool_is_streaming__ is False
    save = _by_name(doc_tools, "save_document")
    assert save.__tool_spec__["function"]["parameters"]["required"] == ["path", "content"]


def test_document_tools_distinct_from_memory_tools(doc_tools):
    """Both tool sets coexist on one agent without name collisions."""
    doc_names = {t.__tool_spec__["function"]["name"] for t in doc_tools}
    mem_names = {"store_memory", "search_memories", "list_memories"}
    assert doc_names.isdisjoint(mem_names)


def test_save_then_read_round_trip(doc_tools):
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    assert save("/notes/standup.md", "Yesterday, Today, Blockers") == "Saved /notes/standup.md."
    assert read("/notes/standup.md") == "Yesterday, Today, Blockers"


def test_read_missing_document_returns_message_not_raise(doc_tools):
    read = _by_name(doc_tools, "read_document")
    result = read("/nope.md")
    assert "No document found" in result and "/nope.md" in result


def test_edit_document_changes_part_without_touching_the_rest(doc_tools):
    # The reason this tool exists: without it, the only way to change a line of a long
    # document is read_document + save_document, and read_document returns a *window*, so
    # saving it back deletes everything outside the window. See the test below.
    save = _by_name(doc_tools, "save_document")
    edit = _by_name(doc_tools, "edit_document")
    read = _by_name(doc_tools, "read_document")
    save("/paper.md", "\n".join(f"line {i}" for i in range(3000)))

    result = edit("/paper.md", "line 7\n", "line seven\n")

    assert "replaced 1 occurrence" in result
    whole = read("/paper.md", max_lines=5000)
    assert len(whole.splitlines()) == 3000
    assert "line seven" in whole
    assert "line 2999" in whole


def test_saving_back_a_windowed_read_is_refused(doc_tools):
    """The hazard, now blocked in code rather than warned about in a docstring.

    read_document returns one window of a long document and save_document replaces the whole
    thing, so the read-modify-save round-trip used to delete everything past the window --
    silently, reporting success. save_document now refuses to replace a document whose full
    text it has not shown the model.

    The document is created through the tool here on purpose: authoring it grants the
    permission, and the windowed read has to take that permission back. A model reading a
    window of a document it wrote no longer holds the text, which is why it is reading.
    """
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    body = "\n".join(f"line {i}" for i in range(3000))
    save("/paper.md", body)

    window = read("/paper.md", max_lines=50)
    result = save("/paper.md", window)

    assert "Refusing to replace" in result
    assert "3000" in result  # the size it would have destroyed, so a full read can be sized
    assert "edit_document" in result  # and the tool that does the job safely
    assert read("/paper.md", max_lines=5000).splitlines() == body.splitlines()


def test_reading_a_document_in_full_permits_replacing_it(doc_tools):
    # The guard must not block the legitimate rewrite: a model that has seen the whole
    # document knows what it is discarding.
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/short.md", "one\ntwo\nthree")

    read("/short.md")  # fits in one window, so this is a complete read
    assert save("/short.md", "replaced") == "Saved /short.md."
    assert read("/short.md") == "replaced"


def test_replacing_an_unread_document_is_refused(doc_tools):
    # A fresh tool set has shown the model nothing, so it cannot knowingly replace what is
    # already there -- the same read-before-write rule Claude Code's own Write tool enforces.
    store = DocumentStore()
    store.write("/prior.md", "written by someone else")
    tools = {t.__name__: t for t in make_document_tools(store)}

    result = tools["save_document"]("/prior.md", "clobbered")

    assert "Refusing to replace" in result
    assert store.read("/prior.md") == "written by someone else"


def test_creating_a_new_document_needs_no_read(doc_tools):
    save = _by_name(doc_tools, "save_document")
    assert save("/brand-new.md", "hello") == "Saved /brand-new.md."


def test_an_out_of_band_change_revokes_the_permission_to_replace(doc_tools, doc_store):
    # The digest key expires on its own: whoever changed the document after the read holds
    # content the model has never seen, so the whole-document write must stop.
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/shared.md", "original")
    read("/shared.md")

    doc_store.write("/shared.md", "changed by a user in the directory")
    result = save("/shared.md", "replaced")

    assert "Refusing to replace" in result
    assert doc_store.read("/shared.md") == "changed by a user in the directory"


def test_an_edit_after_a_full_read_still_permits_replacing(doc_tools):
    # A model that saw the document whole and then made a change it chose still knows the
    # result, so its own edit must not revoke the permission.
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    edit = _by_name(doc_tools, "edit_document")
    save("/notes.md", "alpha\nbeta")

    read("/notes.md")
    edit("/notes.md", "beta", "gamma")

    assert save("/notes.md", "rewritten") == "Saved /notes.md."


def test_save_document_still_warns_in_the_text_the_model_reads(doc_tools):
    # The guard refuses after the fact; the description is what stops the model trying.
    description = _by_name(doc_tools, "save_document").__tool_spec__["function"]["description"]
    assert "deleted" in description
    assert "edit_document" in description


def test_edit_document_refuses_an_ambiguous_match(doc_tools):
    save = _by_name(doc_tools, "save_document")
    edit = _by_name(doc_tools, "edit_document")
    original = "timeout = 30\nretries = 3\ntimeout = 30"
    save("/conf.md", original)

    result = edit("/conf.md", "timeout = 30", "timeout = 60")

    # The store's own message, surfaced bare rather than behind a prefix that repeated it.
    assert "appears 2 times" in result
    assert "nothing was written" in result.lower()
    assert _by_name(doc_tools, "read_document")("/conf.md") == original


def test_edit_document_reports_a_missing_document_rather_than_raising(doc_tools):
    # This group's convention: a miss is a message the model can act on, unlike
    # document_mcp's memory_edit, which raises.
    result = _by_name(doc_tools, "edit_document")("/nope.md", "a", "b")
    assert "No document found" in result and "/nope.md" in result


def test_edit_document_reports_a_missing_match(doc_tools):
    save = _by_name(doc_tools, "save_document")
    edit = _by_name(doc_tools, "edit_document")
    save("/a.md", "hello world")

    result = edit("/a.md", "not-here", "x")

    assert "not found" in result
    assert _by_name(doc_tools, "read_document")("/a.md") == "hello world"


def test_read_document_windows_a_long_document_and_names_the_next_offset(doc_tools):
    # The store holds documents the user drops in, which can be paper-sized. Returning one
    # whole is how a single read spends the context window the rest of the task needs.
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/papers/long.md", "\n".join(f"line {i}" for i in range(3000)))

    out = read("/papers/long.md", max_lines=10)

    assert out.startswith("line 0")
    assert "truncated: showing lines 1-10 of 3000" in out
    assert "call read_document with offset=11 to continue" in out


def test_read_document_pages_a_document_larger_than_one_window(doc_tools):
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    body = [f"line {i}" for i in range(2500)]
    save("/papers/long.md", "\n".join(body))

    seen, offset = [], 1
    while True:
        lines = read("/papers/long.md", max_lines=1000, offset=offset).splitlines()
        truncated = bool(lines) and lines[-1].startswith("... (truncated")
        seen.extend(lines[:-1] if truncated else lines)
        if not truncated:
            break
        offset += 1000

    assert seen == body


def test_read_document_offset_past_the_end_says_how_long_it_is(doc_tools):
    save = _by_name(doc_tools, "save_document")
    read = _by_name(doc_tools, "read_document")
    save("/notes/a.md", "one\ntwo\nthree")

    out = read("/notes/a.md", offset=99)

    assert "past the end" in out and "3 lines" in out


def test_read_document_rejects_a_non_positive_offset(doc_tools):
    read = _by_name(doc_tools, "read_document")
    assert "1-indexed" in read("/notes/a.md", offset=0)


def test_read_document_advertises_its_window_to_the_model(doc_tools):
    read = _by_name(doc_tools, "read_document")
    params = read.__tool_spec__["function"]["parameters"]
    assert {"max_lines", "offset"} <= set(params["properties"])
    assert params["required"] == ["path"]


def test_search_documents_finds_substring(doc_tools):
    save = _by_name(doc_tools, "save_document")
    search = _by_name(doc_tools, "search_documents")
    save("/notes/a.md", "the gate code is 4242")
    save("/notes/b.md", "buy milk")
    result = search("gate code")
    assert "/notes/a.md" in result and "4242" in result
    assert "buy milk" not in result


def test_search_documents_no_match_message(doc_tools):
    search = _by_name(doc_tools, "search_documents")
    assert "No matching documents" in search("nothing here")


def test_list_documents(doc_tools):
    save = _by_name(doc_tools, "save_document")
    list_docs = _by_name(doc_tools, "list_documents")
    assert "No documents stored" in list_docs()
    save("/a.md", "one")
    save("/b.md", "two")
    result = list_docs()
    assert "/a.md" in result and "/b.md" in result


# ---------------------------------------------------------------------------
# Document tools: a file the store cannot read must be reported to the model,
# and a search must not dump whole documents into the context window.
# ---------------------------------------------------------------------------


def test_list_documents_reports_unreadable_files(tmp_path):
    from aimu.memory import DocumentStore
    from aimu.tools.builtin import make_document_tools

    store = DocumentStore(persist_path=str(tmp_path))
    store.write("/notes.md", "keep me")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4\n\xe9\xff\x00binary")

    result = _by_name(make_document_tools(store), "list_documents")()

    assert "/notes.md" in result
    assert "paper.pdf" in result
    assert "could not be read" in result


def test_list_documents_names_a_scanned_pdf_with_its_reason(tmp_path):
    from tests.test_documents import _blank_pdf

    store = DocumentStore(persist_path=str(tmp_path))
    (tmp_path / "scan.pdf").write_bytes(_blank_pdf())

    result = _by_name(make_document_tools(store), "list_documents")()

    assert "/scan.pdf" in result
    assert "OCR" in result
    assert "export" not in result.lower()


def test_list_documents_says_nothing_about_unreadable_when_there_are_none(doc_tools):
    save = _by_name(doc_tools, "save_document")
    save("/a.md", "one")
    assert "could not be read" not in _by_name(doc_tools, "list_documents")()


def test_search_documents_truncates_a_long_document(doc_tools):
    save = _by_name(doc_tools, "save_document")
    search = _by_name(doc_tools, "search_documents")
    save("/paper.md", "findings " + ("x" * 20000))

    result = search("findings")

    assert len(result) < 5000
    assert "read_document" in result
    assert "/paper.md" in result


def test_search_documents_does_not_truncate_a_short_document(doc_tools):
    save = _by_name(doc_tools, "save_document")
    search = _by_name(doc_tools, "search_documents")
    save("/note.md", "the gate code is 4242")

    result = search("gate")

    assert "the gate code is 4242" in result
    assert "read_document" not in result


# ---------------------------------------------------------------------------
# make_collection_tools: several named DocumentStores behind one tool set
# ---------------------------------------------------------------------------


from aimu.memory import IndexProblem, IndexStatus  # noqa: E402
from aimu.tools.builtin import make_collection_tools  # noqa: E402


def _collection_tools(stores, **kwargs):
    return {t.__tool_spec__["function"]["name"]: t for t in make_collection_tools(stores, **kwargs)}


def _two_collections():
    personal, work = DocumentStore(), DocumentStore()
    personal.write("/a.md", "personal a")
    work.write("/a.md", "work a")
    return {"personal": personal, "work": work}


def test_write_tools_only_when_something_is_writable():
    stores = _two_collections()
    read_only = make_collection_tools(stores)
    assert [t.__name__ for t in read_only] == ["list_documents", "read_document", "search_documents"]

    writable = make_collection_tools(stores, writable=["work"])
    assert [t.__name__ for t in writable] == [
        "list_documents",
        "read_document",
        "search_documents",
        "save_document",
        "edit_document",
    ]
    for t in writable:
        assert t.__tool_is_async__ is False

    with pytest.raises(ValueError, match="nowhere"):
        make_collection_tools(stores, writable=["nowhere"])


def test_list_documents_describes_each_collection_it_was_given_a_description_for():
    """What a collection is for is the host's to say, and the overview is where the model decides which
    collection a question belongs to, so a description rides on that collection's line."""
    tools = _collection_tools(_two_collections(), writable=["work"], descriptions={"personal": "Research notes."})

    overview = tools["list_documents"]().split("\n\n")[0]
    assert overview.splitlines() == [
        "Collections:",
        "- personal (read-only): Research notes.",
        "- work (writable)",
    ]
    # A listing narrowed to one collection has no overview to describe.
    assert "Research notes." not in tools["list_documents"](collection="personal")


def test_descriptions_must_name_held_collections():
    with pytest.raises(ValueError, match="nowhere"):
        make_collection_tools(_two_collections(), descriptions={"nowhere": "Not held."})


def test_omitted_collection_resolves_when_only_one_is_held():
    store = DocumentStore()
    store.write("/a.md", "only one")
    tools = _collection_tools({"kokua": store}, writable=["kokua"])

    assert tools["read_document"]("/a.md") == "only one"
    assert tools["edit_document"]("/a.md", "one", "two").startswith("Edited /a.md")
    assert store.read("/a.md") == "only two"
    assert tools["save_document"]("/b.md", "new").startswith("Saved /b.md")
    assert store.read("/b.md") == "new"


def test_omitted_collection_with_several_names_them():
    tools = _collection_tools(_two_collections(), writable=["work"])

    assert "personal, work" in tools["read_document"]("/a.md")
    assert "personal, work" in tools["save_document"]("/b.md", "x")
    assert "personal, work" in tools["edit_document"]("/a.md", "a", "b")
    # The collection that was named is the one read.
    assert tools["read_document"]("/a.md", collection="work") == "work a"
    assert tools["read_document"]("/a.md", collection="personal") == "personal a"


def test_unknown_collection_names_the_held_ones():
    tools = _collection_tools(_two_collections(), writable=["work"])

    for result in (
        tools["read_document"]("/a.md", collection="vault"),
        tools["list_documents"](collection="vault"),
        tools["search_documents"]("a", collections=["vault"]),
        tools["save_document"]("/a.md", "x", collection="vault"),
    ):
        assert "'vault'" in result
        assert "personal, work" in result


def test_write_to_read_only_collection_is_refused_by_name():
    stores = _two_collections()
    tools = _collection_tools(stores, writable=["personal"])

    edit_document = tools["edit_document"]
    assert edit_document("/a.md", "x", "y", collection="work").startswith(
        "Refusing: the 'work' collection is read-only"
    )
    assert tools["save_document"]("/new.md", "x", collection="work").startswith(
        "Refusing: the 'work' collection is read-only"
    )
    assert stores["work"].list_paths() == ["/a.md"]
    assert stores["work"].read("/a.md") == "work a"


def test_pdf_write_refused_as_text(tmp_path):
    from tests.test_documents import minimal_pdf

    (tmp_path / "paper.pdf").write_bytes(minimal_pdf(["glacier melt"]))
    store = DocumentStore(persist_path=str(tmp_path))
    tools = _collection_tools({"vault": store}, writable=["vault"])

    assert "glacier melt" in tools["read_document"]("/paper.pdf")  # read in full, so the guard is satisfied
    for result in (
        tools["save_document"]("/paper.pdf", "replaced"),
        tools["edit_document"]("/paper.pdf", "glacier", "ice"),
        tools["save_document"]("/new.pdf", "fresh"),
    ):
        assert result.startswith("PDFs are read-only")
    assert (tmp_path / "paper.pdf").read_bytes() == minimal_pdf(["glacier melt"])
    assert not (tmp_path / "new.pdf").exists()


def test_pdf_write_is_refused_as_read_only_even_unread(tmp_path):
    # The read-before-replace refusal would be the wrong lesson here: reading the PDF would not help.
    from tests.test_documents import minimal_pdf

    (tmp_path / "paper.pdf").write_bytes(minimal_pdf(["glacier melt"]))
    tools = _collection_tools({"vault": DocumentStore(persist_path=str(tmp_path))}, writable=["vault"])

    assert tools["save_document"]("/paper.pdf", "replaced").startswith("PDFs are read-only")


def test_search_result_format():
    personal, work, kokua = DocumentStore(), DocumentStore(), DocumentStore()
    paper = "# Methods\nglacier melt rates were measured weekly\n"
    personal.write("/paper.md", paper)
    work.write("/paper.md", paper)
    kokua.write("/notes.md", "no heading here, glacier talk")
    tools = _collection_tools({"personal": personal, "work": work, "kokua": kokua})

    result = tools["search_documents"]("glacier melt")

    blocks = result.split("\n\n")
    assert blocks[0].splitlines()[0] == "[personal] /paper.md (Methods)"
    assert "glacier melt rates were measured weekly" in blocks[0]
    assert "also in: work:/paper.md" in blocks[0]
    assert any(block.splitlines()[0] == "[kokua] /notes.md" for block in blocks)
    assert "Note:" not in result  # ephemeral keyword-only stores are complete once searched

    # Narrowed to one collection, the other is neither searched nor cited.
    narrowed = tools["search_documents"]("glacier melt", collections=["work"])
    assert narrowed.splitlines()[0] == "[work] /paper.md (Methods)"
    assert "personal" not in narrowed and "kokua" not in narrowed

    # Incomplete indexes are summed across the searched stores that are incomplete.
    personal.index_status = lambda **_: IndexStatus(files=10, indexed=10, embedded=3, unreadable={}, vectors=True)
    work.index_status = lambda **_: IndexStatus(files=4, indexed=4, embedded=1, unreadable={}, vectors=True)
    result = tools["search_documents"]("glacier melt")
    assert result.endswith("Note: indexing in progress (4 of 14 files embedded); results may be incomplete.")

    # Without vectors, the count is of indexed files.
    personal.index_status = lambda **_: IndexStatus(files=10, indexed=6, embedded=0, unreadable={})
    work.index_status = lambda **_: IndexStatus(files=4, indexed=4, embedded=0, unreadable={})
    result = tools["search_documents"]("glacier melt")
    assert "Note: indexing in progress (6 of 10 files indexed); results may be incomplete." in result

    # An embedding failure on a store with vectors is the semantic-search note, its error verbatim.
    embedding_error = "Embedding failed, so search is keyword-only until it succeeds: RuntimeError: model down"
    personal.index_status = lambda **_: IndexStatus(
        files=1,
        indexed=1,
        embedded=1,
        unreadable={},
        problems=(IndexProblem("embedding", embedding_error),),
        vectors=True,
    )
    result = tools["search_documents"]("glacier melt")
    assert f"Note: semantic search is unavailable ({embedding_error}); these are keyword matches only." in result
    assert "indexing problem" not in result
    assert "indexing in progress" not in result


def test_search_names_a_non_embedding_index_problem_by_collection():
    personal, work = DocumentStore(), DocumentStore()
    personal.write("/a.md", "glacier")
    work.write("/b.md", "glacier")
    tools = _collection_tools({"personal": personal, "work": work})

    stopped = "Background indexing stopped: OperationalError: database is locked"
    work.index_status = lambda **_: IndexStatus(
        files=1, indexed=1, embedded=0, unreadable={}, problems=(IndexProblem("worker", stopped),)
    )
    result = tools["search_documents"]("glacier")
    assert f"Note: indexing problem in 'work': {stopped}" in result
    assert "semantic search" not in result

    # One status can carry both kinds; each is reported as what it is.
    embedding = "Embedding the query failed, so search is keyword-only: RuntimeError: gone; really gone"
    failed = "Could not index /c.md: OperationalError: disk I/O error"
    work.index_status = lambda **_: IndexStatus(
        files=1,
        indexed=1,
        embedded=1,
        unreadable={},
        problems=(IndexProblem("embedding", embedding), IndexProblem("file", failed)),
        vectors=True,
    )
    result = tools["search_documents"]("glacier")
    assert f"Note: semantic search is unavailable ({embedding}); these are keyword matches only." in result
    assert f"Note: indexing problem in 'work': {failed}" in result


def test_index_problem_notes_match_the_real_index_wording(tmp_path, monkeypatch):
    # The notes follow each problem's kind; this pins the kinds the real index assigns, and their wording.
    pytest.importorskip("chromadb")
    import aimu.memory._index as index_module
    from tests.test_document_index import FakeEmbedder

    (tmp_path / "vectors").mkdir()
    (tmp_path / "vectors" / "a.md").write_text("ribosome notes")
    embedded = DocumentStore(
        persist_path=str(tmp_path / "vectors"),
        index_path=str(tmp_path / "vectors-index"),
        embedding_client=FakeEmbedder(error=RuntimeError("model download failed")),
    )
    embedded.search_passages("ribosome")
    embedded._index.wait_until_embedded(5)

    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.0)
    real_chunk_document = index_module.chunk_document

    def chunk_document(path, text, **kwargs):
        if "poison" in text:
            raise RuntimeError("cannot chunk")
        return real_chunk_document(path, text, **kwargs)

    monkeypatch.setattr(index_module, "chunk_document", chunk_document)
    (tmp_path / "keyword").mkdir()
    (tmp_path / "keyword" / "bad.md").write_text("poison ribosome")
    keyword = DocumentStore(persist_path=str(tmp_path / "keyword"))
    keyword.index_status()
    keyword._index.wait_until_embedded(5)

    result = _collection_tools({"vectors": embedded, "keyword": keyword})["search_documents"]("ribosome")
    embedded.close()
    keyword.close()

    assert "Note: semantic search is unavailable (Embedding failed" in result
    assert "model download failed); these are keyword matches only." in result
    assert "Note: indexing problem in 'keyword': Could not index /bad.md" in result
    assert "indexing problem in 'vectors'" not in result


def test_pointers_name_the_collection_when_several_are_held():
    from aimu.memory import Passage

    personal, work = DocumentStore(), DocumentStore()
    # The chunker keeps real passages under the excerpt cap, so a longer one is stubbed in to reach the pointer.
    long_passage = Passage(path="/long.md", location="", text="glacier " * 400, score=1.0, digest="d")
    personal.search_passages = lambda query, n_results=8: [long_passage]
    work.write("/lines.md", "\n".join(f"line {i}" for i in range(50)))
    tools = _collection_tools({"personal": personal, "work": work})

    search = tools["search_documents"]("glacier", collections=["personal"])
    assert "call read_document('/long.md', collection='personal')" in search
    window = tools["read_document"]("/lines.md", collection="work", max_lines=10)
    assert "call read_document('/lines.md', collection='work') with offset=11" in window

    # With one collection, collection may be omitted, so the pointers stay as make_document_tools writes them.
    single = _collection_tools({"personal": personal})
    assert "call read_document('/long.md')" in single["search_documents"]("glacier")
    single_window = _collection_tools({"work": work})["read_document"]("/lines.md", max_lines=10)
    assert "call read_document with offset=11" in single_window


def test_also_in_names_a_second_copy_in_the_same_collection():
    store = DocumentStore()
    store.write("/paper.md", "glacier melt rates")
    store.write("/copy/paper.md", "glacier melt rates")
    tools = _collection_tools({"personal": store})

    result = tools["search_documents"]("glacier")

    assert result.count("[personal]") == 1
    assert "also in: personal:/" in result


def test_search_with_no_match_still_carries_its_notes():
    store = DocumentStore()
    store.write("/a.md", "alpha")
    store.index_status = lambda **_: IndexStatus(files=5, indexed=2, embedded=0, unreadable={})
    result = _collection_tools({"personal": store})["search_documents"]("zebra")

    assert result.startswith("No matching passages found")
    assert "Note: indexing in progress (2 of 5 files indexed)" in result


def test_a_failing_collection_is_skipped_and_named():
    stores = _two_collections()

    def broken(query, n_results=8):
        raise RuntimeError("index database is locked")

    stores["work"].search_passages = broken
    result = _collection_tools(stores)["search_documents"]("personal")

    assert result.startswith("[personal] /a.md")
    assert "Note: could not search 'work' (RuntimeError: index database is locked)" in result


def test_a_failing_status_keeps_the_results():
    stores = _two_collections()

    def broken(**_):
        raise RuntimeError("status unavailable")

    stores["work"].index_status = broken
    result = _collection_tools(stores)["search_documents"]("personal")

    assert result.startswith("[personal] /a.md")
    assert "'work'" in result and "status unavailable" in result


def test_a_failure_of_the_whole_search_is_returned_not_raised(monkeypatch):
    import aimu.memory as memory

    def broken(*args, **kwargs):
        raise ImportError("rerank needs the [hf] extra")

    monkeypatch.setattr(memory, "search_collections", broken)
    result = _collection_tools(_two_collections())["search_documents"]("a")

    assert result.startswith("Search failed") and "[hf]" in result


def test_search_rejects_a_bad_n_results():
    result = _collection_tools(_two_collections())["search_documents"]("a", n_results=0)
    assert "n_results" in result


def test_guard_is_per_collection():
    personal, kokua = DocumentStore(), DocumentStore()
    personal.write("/a.md", "same text")
    kokua.write("/a.md", "same text")
    tools = _collection_tools({"personal": personal, "kokua": kokua}, writable=["personal", "kokua"])

    assert tools["read_document"]("/a.md", collection="personal") == "same text"

    refused = tools["save_document"]("/a.md", "replaced", collection="kokua")
    assert refused.startswith("Refusing to replace /a.md")
    assert kokua.read("/a.md") == "same text"
    assert tools["save_document"]("/a.md", "replaced", collection="personal").startswith("Saved")
    assert personal.read("/a.md") == "replaced"


def test_windowed_read_revokes_and_edit_carries_the_guard():
    store = DocumentStore()
    store.write("/long.md", "\n".join(f"line {i}" for i in range(100)))
    tools = _collection_tools({"kokua": store}, writable=["kokua"])

    tools["read_document"]("/long.md", max_lines=500)
    tools["edit_document"]("/long.md", "line 7\n", "line seven\n")
    assert tools["save_document"]("/long.md", "short").startswith("Saved")

    store.write("/long.md", "\n".join(f"line {i}" for i in range(100)))
    tools["read_document"]("/long.md", max_lines=500)
    tools["read_document"]("/long.md", max_lines=10)
    assert tools["save_document"]("/long.md", "short").startswith("Refusing to replace")


def test_edit_misses_are_returned():
    store = DocumentStore()
    store.write("/a.md", "x x")
    tools = _collection_tools({"kokua": store}, writable=["kokua"])

    assert "No document found at /missing.md" in tools["edit_document"]("/missing.md", "a", "b")
    assert "appears 2 times" in tools["edit_document"]("/a.md", "x", "y")
    assert "not found" in tools["edit_document"]("/a.md", "zebra", "y")
    assert "No document found at /missing.md" in tools["read_document"]("/missing.md")
    assert "past the end" in tools["read_document"]("/a.md", offset=5)


def test_list_documents_reports_unreadable_with_reasons(tmp_path):
    from tests.test_documents import _blank_pdf

    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "notes.md").write_text("keep me")
    (tmp_path / "vault" / "scan.pdf").write_bytes(_blank_pdf())
    (tmp_path / "vault" / "export.docx").write_bytes(b"\xff\xfe\x00binary")
    vault = DocumentStore(persist_path=str(tmp_path / "vault"))
    kokua = DocumentStore()
    kokua.write("/plan.md", "the plan")
    tools = _collection_tools({"vault": vault, "kokua": kokua}, writable=["kokua"])

    # Listing reads no file, so it reports the reasons a read has found: here, a search's index build.
    assert "could not be read" not in tools["list_documents"]()
    tools["search_documents"]("keep")
    listing = tools["list_documents"]()

    # The held collections come first, with what each permits.
    head = listing.split("\n\n")[0]
    assert "vault (read-only)" in head and "kokua (writable)" in head
    assert head.index("vault") < head.index("kokua")
    assert listing.index("/notes.md") < listing.index("/plan.md")
    assert "/scan.pdf" in listing and "OCR" in listing
    assert "/export.docx: not UTF-8 text" in listing
    assert "export them to Markdown" in listing

    one = tools["list_documents"](collection="kokua")
    assert "/plan.md" in one and "/notes.md" not in one and "could not be read" not in one


def test_list_documents_keeps_export_advice_for_text_reasons_only(tmp_path):
    from tests.test_documents import _blank_pdf

    (tmp_path / "scan.pdf").write_bytes(_blank_pdf())
    tools = _collection_tools({"vault": DocumentStore(persist_path=str(tmp_path))})

    tools["search_documents"]("anything")
    listing = tools["list_documents"]()
    assert "/scan.pdf" in listing and "could not be read" in listing
    assert "export" not in listing.lower()


def test_list_documents_filters_by_prefix():
    store = DocumentStore()
    store.write("/papers/a.md", "a")
    store.write("/notes/b.md", "b")
    tools = _collection_tools({"vault": store})

    listing = tools["list_documents"](collection="vault", prefix="/papers")
    assert "/papers/a.md" in listing and "/notes/b.md" not in listing
    assert "No documents" in tools["list_documents"](prefix="/nowhere")


def test_collection_tools_are_re_exported_for_async():
    from aimu.aio.tools import builtin as aio_builtin

    assert aio_builtin.make_collection_tools is make_collection_tools


# ---------------------------------------------------------------------------
# Final review: vault-scale listing, honest notes, safe saves
# ---------------------------------------------------------------------------


def test_list_documents_caps_each_collection_and_says_how_to_narrow():
    store = DocumentStore()
    for number in range(150):
        store.write(f"/Papers/p{number:03}.md", "x")
        store.write(f"/Daily/d{number:03}.md", "x")
    small = DocumentStore()
    small.write("/a.md", "x")
    tools = _collection_tools({"vault": store, "kokua": small})

    listing = tools["list_documents"]()
    vault = listing.split("\n\nkokua:")[0]
    assert vault.count("\n- /") == 200
    assert "100 more documents not listed" in vault
    assert "prefix" in vault and "/Daily/ (150)" in vault and "/Papers/ (150)" in vault
    assert "- /a.md" in listing

    narrowed = tools["list_documents"](collection="vault", prefix="/Papers/")
    assert narrowed.count("\n- /") == 150 and "not listed" not in narrowed


def test_list_documents_reads_no_file(tmp_path, monkeypatch):
    import aimu.memory.document_store as document_store_module
    from tests.test_documents import minimal_pdf

    for number in range(3):
        (tmp_path / f"p{number}.pdf").write_bytes(minimal_pdf(["glacier"]))
    calls = []
    monkeypatch.setattr(document_store_module, "pdf_to_markdown", lambda data: calls.append(1) or "x")
    listing = _collection_tools({"vault": DocumentStore(persist_path=str(tmp_path))})["list_documents"]()

    assert "/p0.pdf" in listing and "/p2.pdf" in listing
    assert calls == []


def test_in_progress_note_leaves_out_a_store_whose_embedding_fails():
    personal, work = DocumentStore(), DocumentStore()
    personal.write("/a.md", "glacier")
    work.write("/b.md", "glacier")
    tools = _collection_tools({"personal": personal, "work": work})
    failing = "Embedding failed, so search is keyword-only until it succeeds: RuntimeError: offline"
    problems = (IndexProblem("embedding", failing),)

    # Keyword indexing complete, embedding failing: nothing is in progress, so there is nothing to wait for.
    personal.index_status = lambda **_: IndexStatus(
        files=5, indexed=5, embedded=0, unreadable={}, problems=problems, vectors=True
    )
    result = tools["search_documents"]("glacier")
    assert "indexing in progress" not in result
    assert "semantic search is unavailable" in result

    # Keyword indexing still running: its progress is the note, not embedding progress that is not happening.
    personal.index_status = lambda **_: IndexStatus(
        files=5, indexed=2, embedded=0, unreadable={}, problems=problems, vectors=True
    )
    work.index_status = lambda **_: IndexStatus(files=4, indexed=4, embedded=1, unreadable={}, vectors=True)
    result = tools["search_documents"]("glacier")
    assert "Note: indexing in progress (1 of 4 files embedded, 2 of 5 files indexed)" in result


def test_in_progress_note_leaves_out_permanently_failed_files():
    personal, work = DocumentStore(), DocumentStore()
    personal.write("/a.md", "glacier")
    work.write("/b.md", "glacier")
    tools = _collection_tools({"personal": personal, "work": work})
    failed = "Could not index /c.md: RuntimeError: cannot chunk"
    problems = (IndexProblem("file", failed),)

    # The only gap is a file that failed: nothing is in progress, and the problem note says why.
    personal.index_status = lambda **_: IndexStatus(
        files=3, indexed=2, embedded=2, unreadable={}, problems=problems, failed=1, vectors=True
    )
    work.index_status = lambda **_: IndexStatus(files=2, indexed=1, embedded=0, unreadable={}, failed=1)
    result = tools["search_documents"]("glacier")
    assert "indexing in progress" not in result
    assert f"Note: indexing problem in 'personal': {failed}" in result

    # Work still to do is counted, without the failed files in either number.
    personal.index_status = lambda **_: IndexStatus(
        files=4, indexed=3, embedded=1, unreadable={}, problems=problems, failed=1, vectors=True
    )
    work.index_status = lambda **_: IndexStatus(files=5, indexed=2, embedded=0, unreadable={}, failed=1)
    result = tools["search_documents"]("glacier")
    assert "Note: indexing in progress (1 of 3 files embedded, 2 of 4 files indexed)" in result

    # A stopped worker's files are still pending and retried on the next search, so they stay in progress.
    stopped = (IndexProblem("worker", "Background indexing stopped: OperationalError: database is locked"),)
    personal.index_status = lambda **_: IndexStatus(files=3, indexed=1, embedded=0, unreadable={}, problems=stopped)
    work.index_status = lambda **_: IndexStatus(files=1, indexed=1, embedded=0, unreadable={})
    result = tools["search_documents"]("glacier")
    assert "Note: indexing in progress (1 of 3 files indexed)" in result


@pytest.mark.parametrize("vectors", [False, True])
def test_a_permanently_failed_file_is_a_problem_not_progress(tmp_path, monkeypatch, vectors):
    import aimu.memory._index as index_module
    from tests.test_document_index import FakeEmbedder

    if vectors:
        pytest.importorskip("chromadb")
    real_chunk_document = index_module.chunk_document

    def chunk_document(path, text, **kwargs):
        if "poison" in text:
            raise RuntimeError("cannot chunk")
        return real_chunk_document(path, text, **kwargs)

    monkeypatch.setattr(index_module, "chunk_document", chunk_document)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "good.md").write_text("glacier melt")
    (tmp_path / "docs" / "bad.md").write_text("poison glacier")
    store = DocumentStore(
        persist_path=str(tmp_path / "docs"),
        index_path=str(tmp_path / "index"),
        embedding_client=FakeEmbedder() if vectors else None,
    )
    store.index_status()
    store._index.wait_until_embedded(5)
    tools = _collection_tools({"vault": store})
    for _ in range(2):
        result = tools["search_documents"]("glacier")
        assert "indexing in progress" not in result
        assert "Note: indexing problem in 'vault': Could not index /bad.md: RuntimeError: cannot chunk" in result
    store.close()


def test_an_offline_first_build_says_semantic_search_is_unavailable_not_in_progress(tmp_path):
    pytest.importorskip("chromadb")
    from tests.test_document_index import FakeEmbedder

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("ribosome notes")
    store = DocumentStore(
        persist_path=str(tmp_path / "docs"),
        index_path=str(tmp_path / "index"),
        embedding_client=FakeEmbedder(error=RuntimeError("model download failed")),
    )
    tools = _collection_tools({"vault": store})
    tools["search_documents"]("ribosome")
    store._index.wait_until_embedded(5)

    result = tools["search_documents"]("ribosome")
    store.close()
    assert result.startswith("[vault] /a.md")
    assert "semantic search is unavailable" in result and "model download failed" in result
    assert "embedded" not in result and "in progress" not in result


def test_a_worker_error_reaches_the_search_note(tmp_path, monkeypatch):
    import threading

    import aimu.memory._index as index_module
    from aimu.memory._index import PassageIndex

    monkeypatch.setattr(index_module, "INLINE_INDEX_SECONDS", 0.0)
    (tmp_path / "a.md").write_text("glacier")
    real_next_job = PassageIndex._next_job
    worker_calls = []

    def next_job(self):
        if threading.current_thread().name == index_module.WORKER_THREAD_NAME:
            worker_calls.append(1)
            if len(worker_calls) == 1:
                raise RuntimeError("database is locked")
        return real_next_job(self)

    monkeypatch.setattr(PassageIndex, "_next_job", next_job)
    store = DocumentStore(persist_path=str(tmp_path))
    release = threading.Event()
    real_load = store._load
    monkeypatch.setattr(store, "_load", lambda path: release.wait(10) and real_load(path))
    tools = _collection_tools({"vault": store})
    try:
        store.index_status()
        store._index.wait_until_embedded(5)
        result = tools["search_documents"]("glacier")
    finally:
        release.set()
    store._index.wait_until_embedded(5)
    store.close()
    assert "Note: indexing problem in 'vault': Background indexing stopped: RuntimeError: database is locked" in result


@pytest.mark.parametrize("factory", ["document", "collection"])
def test_save_document_refuses_to_replace_a_file_it_cannot_read(tmp_path, factory):
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00binary")
    store = DocumentStore(persist_path=str(tmp_path))
    if factory == "document":
        save = _by_name(make_document_tools(store), "save_document")
    else:
        save = _collection_tools({"vault": store}, writable=["vault"])["save_document"]

    result = save("/bin.md", "replacement")

    assert result.startswith("Refusing to replace /bin.md")
    assert "not UTF-8 text" in result and "nothing was written" in result.lower()
    assert (tmp_path / "bin.md").read_bytes() == b"\xff\xfe\x00binary"
    assert save("/new.md", "fresh").startswith("Saved")


@pytest.mark.parametrize("factory", ["document", "collection"])
def test_read_document_says_why_a_file_cannot_be_read(tmp_path, factory):
    (tmp_path / "bin.md").write_bytes(b"\xff\xfe\x00binary")
    store = DocumentStore(persist_path=str(tmp_path))
    if factory == "document":
        read = _by_name(make_document_tools(store), "read_document")
    else:
        read = _collection_tools({"vault": store})["read_document"]

    result = read("/bin.md")
    assert "/bin.md" in result and "cannot be read" in result and "not UTF-8 text" in result
    assert "No document found" in read("/missing.md")


def test_pointers_quote_a_path_with_an_apostrophe():
    from aimu.memory import Passage

    personal, work = DocumentStore(), DocumentStore()
    long_passage = Passage(path="/Paul's notes.md", location="", text="glacier " * 400, score=1.0, digest="d")
    personal.search_passages = lambda query, n_results=8: [long_passage]
    tools = _collection_tools({"personal": personal, "work": work})

    search = tools["search_documents"]("glacier", collections=["personal"])
    assert """call read_document("/Paul's notes.md", collection='personal')""" in search


def test_search_documents_tells_the_model_to_pass_the_cited_collection():
    tools = _collection_tools(_two_collections())
    description = tools["search_documents"].__tool_spec__["function"]["description"]
    assert "read_document" in description and "[collection]" in description
