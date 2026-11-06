"""Tests for the page store, outline, and offset-exact chunks behind ``make_page_tools``."""

from __future__ import annotations

from aimu.tools._pages import PageStore, chunks, outline

DOC = "# Title\n\nIntro para.\n\n## Methods\n\nWe measured.\n\n```\n# not a heading\n```\n\n## Results\n\nIt worked."


def test_put_returns_content_addressed_handle_and_is_idempotent():
    store = PageStore()
    first = store.put("https://a.example", "hello")
    again = store.put("https://a.example", "hello")
    assert first.handle.startswith("page-") and len(first.handle) == 17
    assert again == first


def test_same_url_changed_text_gets_new_handle():
    store = PageStore()
    old = store.put("https://a.example", "v1")
    new = store.put("https://a.example", "v2")
    assert old.handle != new.handle
    assert store.get(old.handle).text == "v1" and store.get(new.handle).text == "v2"


def test_directory_store_round_trips_and_survives_a_new_instance(tmp_path):
    page = PageStore(tmp_path).put("https://a.example", "body\ntext")
    assert (tmp_path / f"{page.handle}.md").read_text() == "body\ntext"
    reread = PageStore(tmp_path).get(page.handle)
    assert reread == page


def test_store_rejects_malformed_handles(tmp_path):
    (tmp_path.parent / "secret.md").write_text("nope")
    store = PageStore(tmp_path)
    for bad in ["page-../secret", "../secret", "https://a.example", "page-XYZ", ""]:
        assert store.get(bad) is None


def test_outline_finds_headings_with_offsets_and_skips_code_fences():
    heads = outline(DOC)
    assert [h.title for h in heads] == ["Title", "Methods", "Results"]
    assert [h.level for h in heads] == [1, 2, 2]
    assert DOC[heads[1].offset - 1 :].startswith("## Methods")
    assert heads[0].size == heads[1].offset - heads[0].offset
    assert heads[2].offset - 1 + heads[2].size == len(DOC)


def test_outline_of_text_without_headings_is_empty():
    assert outline("just text\n\nmore text") == []


def test_chunks_offsets_are_exact_and_carry_their_heading():
    for chunk in chunks(DOC, size=30):
        assert DOC[chunk.offset - 1 : chunk.offset - 1 + len(chunk.text)] == chunk.text
    assert any(c.heading == "Results" and "worked" in c.text for c in chunks(DOC, size=30))


def test_chunks_never_cross_a_heading():
    parts = chunks(DOC, size=10_000)
    assert [c.heading for c in parts] == ["Title", "Methods", "Results"]
    assert all("\n## " not in c.text for c in parts)


def test_chunks_hard_split_an_oversized_paragraph():
    text = "x" * 2500
    parts = chunks(text, size=1000)
    assert [len(c.text) for c in parts] == [1000, 1000, 500]
    assert [c.offset for c in parts] == [1, 1001, 2001]
