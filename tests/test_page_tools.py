"""Mock-only tests for ``make_page_tools``: fetch once, admit only ranked passages, read the rest by handle."""

from __future__ import annotations

import re
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import requests

from aimu.tools._pages import OUTLINE_LIMIT
from aimu.tools.builtin import make_page_tools

LONG = (
    "# Guide\n\n"
    + "Filler paragraph about nothing.\n\n" * 200
    + "## Pricing\n\nThe enterprise plan costs 40 dollars per seat.\n\n"
    + "Filler paragraph about nothing.\n\n" * 200
)


@contextmanager
def _serving(text: str):
    """Serve *text* as text/plain, which the fetch keeps verbatim, so offsets are exact against it."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.headers = {"content-type": "text/plain"}
    response.encoding = "utf-8"
    response.iter_content = lambda chunk_size=1: iter([text.encode("utf-8")])
    with patch("aimu.tools.builtin.requests.request", return_value=response):
        yield


def _handle(text: str) -> str:
    return re.search(r"page-[0-9a-f]{12}", text).group(0)


def test_fetch_page_returns_outline_and_ranked_verbatim_passage():
    fetch_page, _, _ = make_page_tools()
    with _serving(LONG):
        out = fetch_page("https://example.com", question="enterprise plan price per seat")
    assert out.startswith("Page page-") and f"{len(LONG)} characters" in out
    assert "- ## Pricing (offset" in out
    assert "The enterprise plan costs 40 dollars per seat." in out
    assert len(out) < 4000  # the point: far less than the page


def test_fetch_page_passage_offsets_point_at_the_text():
    fetch_page, read_page, _ = make_page_tools()
    with _serving(LONG):
        out = fetch_page("https://example.com", question="enterprise plan")
    offset = int(re.search(r'\[offset (\d+), under "Pricing"\]', out).group(1))
    assert "enterprise" in read_page(_handle(out), offset=offset, max_chars=200)


def test_fetch_page_reports_no_match_and_falls_back_to_opening():
    fetch_page, _, _ = make_page_tools(opening_chars=100)
    with _serving(LONG):
        out = fetch_page("https://example.com", question="zebra quantum")
    assert "No passage matched" in out and "# Guide" in out and "call read_page with offset=101" in out


def test_fetch_page_without_question_returns_opening_window():
    fetch_page, _, _ = make_page_tools(opening_chars=100)
    with _serving(LONG):
        out = fetch_page("https://example.com")
    assert "Passages" not in out and "call read_page with offset=101" in out


def test_fetch_page_shows_published_line_in_header():
    fetch_page, _, _ = make_page_tools()
    with _serving("Published: 2026-01-02\n\n# A\n\nbody"):
        out = fetch_page("https://example.com", question="body")
    assert out.splitlines()[1] == "Published: 2026-01-02"


def test_outline_is_capped_with_a_count_of_the_rest():
    many = "".join(f"## Page {n}\n\ntext {n}\n\n" for n in range(1, 101))
    fetch_page, _, _ = make_page_tools()
    with _serving(many):
        out = fetch_page("https://example.com")
    assert out.count("- ## Page") == OUTLINE_LIMIT
    assert f"- ... and {100 - OUTLINE_LIMIT} more headings" in out


def test_outline_says_when_page_has_no_headings():
    fetch_page, _, _ = make_page_tools()
    with _serving("plain text only"):
        out = fetch_page("https://example.com")
    assert "Outline: none (the page has no headings)." in out


def test_find_in_page_lists_matching_lines_with_offsets():
    fetch_page, _, find_in_page = make_page_tools()
    with _serving(LONG):
        handle = _handle(fetch_page("https://example.com"))
    out = find_in_page(handle, "ENTERPRISE")
    assert out.startswith('1 lines match "ENTERPRISE"')
    offset = int(re.search(r"offset (\d+)", out).group(1))
    assert LONG[offset - 1 :].startswith("The enterprise plan")


def test_find_in_page_caps_matches_and_reports_none():
    fetch_page, _, find_in_page = make_page_tools()
    with _serving(LONG):
        handle = _handle(fetch_page("https://example.com"))
    capped = find_in_page(handle, "filler", max_matches=3)
    assert capped.startswith('400 lines match "filler"; showing the first 3')
    assert capped.count("\n- offset ") == 3
    assert find_in_page(handle, "zebra") == f'No line in {handle} contains "zebra".'


def test_find_in_page_clips_long_lines_around_the_match():
    line = "a" * 1000 + " needle " + "b" * 1000
    fetch_page, _, find_in_page = make_page_tools()
    with _serving(line):
        handle = _handle(fetch_page("https://example.com"))
    out = find_in_page(handle, "needle")
    match_line = out.splitlines()[1]
    assert "needle" in match_line and "..." in match_line and len(match_line) < 400


def test_read_page_and_find_in_page_refuse_unknown_handles():
    _, read_page, find_in_page = make_page_tools()
    assert "No page with handle" in read_page("page-000000000000")
    assert "No page with handle" in find_in_page("../etc/passwd", "root")


def test_read_page_complains_about_bad_windows():
    fetch_page, read_page, _ = make_page_tools()
    with _serving("short"):
        handle = _handle(fetch_page("https://example.com"))
    assert "offset must be 1 or greater" in read_page(handle, offset=0)
    assert "past the end" in read_page(handle, offset=50)


def test_fetch_page_passes_fetch_failures_through():
    fetch_page, _, _ = make_page_tools()
    with patch("aimu.tools.builtin.requests.request", side_effect=requests.RequestException("refused")):
        assert fetch_page("https://example.com").startswith("Error fetching page")


def test_custom_ranker_is_used():
    calls = []

    def ranker(query, passages):
        calls.append(query)
        return [1.0 if "enterprise" in passage else 0.0 for passage in passages]

    fetch_page, _, _ = make_page_tools(ranker=ranker)
    with _serving(LONG):
        out = fetch_page("https://example.com", question="anything")
    assert calls == ["anything"] and "enterprise plan" in out


def test_directory_store_persists_pages(tmp_path):
    fetch_page, _, _ = make_page_tools(directory=tmp_path)
    with _serving(LONG):
        handle = _handle(fetch_page("https://example.com"))
    _, read_page, _ = make_page_tools(directory=tmp_path)
    assert read_page(handle, max_chars=7).startswith("# Guide")


def test_factory_rejects_bad_budgets():
    for bad in ({"top_k": 0}, {"chunk_chars": 0}, {"opening_chars": 0}):
        with pytest.raises(ValueError):
            make_page_tools(**bad)


def test_page_tools_are_reexported_on_aio():
    from aimu.aio.tools.builtin import make_page_tools as aio_make_page_tools

    assert aio_make_page_tools is make_page_tools
