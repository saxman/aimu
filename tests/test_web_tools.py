"""Mock-only unit tests for the web-browsing tools.

Covers the stateless ``get_webpage_html`` and ``get_web_content`` tools, the stateful
``make_web_tools`` factory (``find_forms`` / ``submit_form``), and the classifier,
capped read, and ``_truncate`` helpers ``get_web_content`` is built on. No network
access; ``requests`` is stubbed.
"""

from __future__ import annotations

import pytest
import requests

from aimu.tools import builtin
from aimu.tools.builtin import (
    _BodyTooLarge,
    _classify,
    _declared_size,
    _read_capped_body,
    _truncate,
    _window,
    _window_complaint,
    _WEB_CONTENT_LIMIT_BYTES,
    get_web_content,
    get_webpage_html,
    make_web_tools,
)

FORM_HTML = """
<html><body>
  <form action="/login" method="post">
    <input type="hidden" name="csrf" value="tok123">
    <input type="text" name="username">
    <input type="password" name="password">
    <textarea name="note"></textarea>
    <select name="role"></select>
  </form>
</body></html>
"""

TWO_FORMS_HTML = """
<form action="https://other.example/a" method="GET"><input name="q"></form>
<form action="/b"><input type="hidden" name="t" value="v"></form>
"""


class FakeResponse:
    def __init__(self, text="", status_code=200, url="http://site.example/", headers=None, body=None, encoding="utf-8"):
        self.text = text
        self.status_code = status_code
        self.url = url
        self.encoding = encoding
        # A real requests.Response has case-insensitive headers; the tools only ever read
        # Content-Type and Content-Length, so a plain lowercase dict plus .get is enough.
        self.headers = headers if headers is not None else {"content-type": "text/html"}
        self._body = body if body is not None else text.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


class FakeSession:
    """Records requests and hands back queued responses; shares one cookie jar."""

    def __init__(self, responses):
        self.headers: dict = {}
        self.cookies: dict = {}
        self.calls: list[tuple] = []
        self._responses = list(responses)

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# get_webpage_html (stateless)
# ---------------------------------------------------------------------------


def test_get_webpage_html_returns_raw_markup(monkeypatch):
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text=FORM_HTML))
    out = get_webpage_html("http://site.example/")
    assert "<form" in out and 'name="csrf"' in out  # tags preserved, not stripped


def test_get_webpage_html_truncates_long_pages(monkeypatch):
    big = "x" * 25000
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text=big))
    out = get_webpage_html("http://site.example/")
    assert "truncated: showing characters 1-20000 of 25000" in out
    assert "call get_webpage_html with offset=20001 to continue" in out
    assert len(out) < len(big)


def test_get_webpage_html_offset_reaches_markup_past_the_first_window(monkeypatch):
    # The form or link a caller is after is routinely past a real page's head and nav, and
    # this tool's window is fixed, so offset is the only way to it.
    page = "x" * 20000 + '<form action="/login" method="post"></form>'
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text=page))

    out = get_webpage_html("http://site.example/", offset=20001)

    assert '<form action="/login"' in out
    assert "truncated" not in out


def test_get_webpage_html_rejects_a_non_positive_offset(monkeypatch):
    calls = []
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: calls.append(1) or FakeResponse(text="x"))

    out = get_webpage_html("http://site.example/", offset=0)

    assert "1-indexed" in out
    assert calls == []  # and it did not spend a request finding that out


def test_get_webpage_html_reports_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(builtin.requests, "request", boom)
    out = get_webpage_html("http://site.example/")
    assert out.startswith("Error fetching page:")


def test_get_webpage_html_is_stateless(monkeypatch):
    """Uses the module-level requests, never a Session."""
    used = {}

    def record(method, url, **k):
        used["hit"] = True
        return FakeResponse()

    monkeypatch.setattr(builtin.requests, "request", record)
    get_webpage_html("http://site.example/")
    assert used["hit"] is True


# ---------------------------------------------------------------------------
# find_forms
# ---------------------------------------------------------------------------


def test_find_forms_parses_fields_and_hidden_csrf():
    session = FakeSession([FakeResponse(text=FORM_HTML, url="http://site.example/page")])
    find_forms, _ = make_web_tools(session=session)
    out = find_forms("http://site.example/page")
    assert "POST http://site.example/login" in out  # relative action resolved to absolute
    assert "csrf [hidden] = 'tok123'" in out  # hidden field surfaced with value
    assert "username [text]" in out
    assert "note [textarea]" in out
    assert "role [select]" in out


def test_find_forms_no_forms():
    session = FakeSession([FakeResponse(text="<html><body>nothing</body></html>")])
    find_forms, _ = make_web_tools(session=session)
    assert find_forms("http://site.example/") == "No forms found on the page."


def test_find_forms_multiple_and_absolute_action():
    session = FakeSession([FakeResponse(text=TWO_FORMS_HTML, url="http://site.example/")])
    find_forms, _ = make_web_tools(session=session)
    out = find_forms("http://site.example/")
    assert "Form 0: GET https://other.example/a" in out  # already-absolute action preserved
    assert "Form 1: GET http://site.example/b" in out  # relative resolved; default method GET


def test_find_forms_reports_errors():
    class Boom(FakeSession):
        def request(self, *a, **k):
            raise requests.Timeout("slow")

    find_forms, _ = make_web_tools(session=Boom([]))
    assert find_forms("http://site.example/").startswith("Error fetching page:")


# ---------------------------------------------------------------------------
# submit_form
# ---------------------------------------------------------------------------


def test_submit_form_post_routes_data_to_body():
    session = FakeSession([FakeResponse(text="ok", status_code=201, url="http://site.example/login")])
    _, submit_form = make_web_tools(session=session)
    out = submit_form("http://site.example/login", method="POST", data={"user": "a"})
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert kwargs == {"data": {"user": "a"}}
    assert "Status: 201" in out and "ok" in out


def test_submit_form_get_routes_data_to_params():
    session = FakeSession([FakeResponse(text="<html>results</html>")])
    _, submit_form = make_web_tools(session=session)
    submit_form("http://site.example/search", method="GET", data={"q": "cats"})
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert kwargs == {"params": {"q": "cats"}}


def test_submit_form_rejects_unknown_method():
    _, submit_form = make_web_tools(session=FakeSession([]))
    assert "Unsupported method" in submit_form("http://x/", method="PUT")


def test_submit_form_reports_errors():
    class Boom(FakeSession):
        def request(self, *a, **k):
            raise requests.ConnectionError("down")

    _, submit_form = make_web_tools(session=Boom([]))
    assert submit_form("http://x/", data={}).startswith("Error submitting form:")


# ---------------------------------------------------------------------------
# session sharing / cookie persistence
# ---------------------------------------------------------------------------


def test_find_forms_and_submit_form_share_one_session():
    session = FakeSession([FakeResponse(text=FORM_HTML), FakeResponse(text="ok")])
    find_forms, submit_form = make_web_tools(session=session)
    find_forms("http://site.example/login")
    # A cookie set during the first exchange persists into the second call (same jar).
    session.cookies["sid"] = "abc"
    submit_form("http://site.example/login", data={"user": "a"})
    assert len(session.calls) == 2  # both tools drove the same session
    assert session.cookies["sid"] == "abc"


def test_make_web_tools_creates_session_when_none_given():
    find_forms, submit_form = make_web_tools()
    assert find_forms.__tool_spec__["function"]["name"] == "find_forms"
    assert submit_form.__tool_spec__["function"]["name"] == "submit_form"


# ---------------------------------------------------------------------------
# tool specs + async re-exports
# ---------------------------------------------------------------------------


def test_tool_specs():
    assert get_webpage_html.__tool_spec__["function"]["name"] == "get_webpage_html"
    find_forms, submit_form = make_web_tools()
    params = submit_form.__tool_spec__["function"]["parameters"]["properties"]
    assert set(params) == {"url", "method", "data"}
    # only url is required (method + data have defaults)
    assert submit_form.__tool_spec__["function"]["parameters"]["required"] == ["url"]


def test_async_reexports_importable():
    from aimu.aio.tools.builtin import get_webpage_html as aio_html
    from aimu.aio.tools.builtin import make_web_tools as aio_web

    assert aio_html is get_webpage_html
    assert aio_web is make_web_tools


def test_get_webpage_html_in_web_subgroup_and_all_tools():
    assert get_webpage_html in builtin.web
    assert get_webpage_html in builtin.ALL_TOOLS


# ---------------------------------------------------------------------------
# fetch cap and classification
# ---------------------------------------------------------------------------


def test_read_capped_body_returns_the_whole_small_body():
    response = FakeResponse(body=b"hello there")
    assert _read_capped_body(response) == b"hello there"


def test_read_capped_body_refuses_an_oversized_body():
    oversized = b"x" * (_WEB_CONTENT_LIMIT_BYTES + 1)
    response = FakeResponse(body=oversized, headers={"content-type": "application/pdf"})
    with pytest.raises(_BodyTooLarge):
        _read_capped_body(response)


def test_read_capped_body_allows_exactly_the_cap():
    """The comparison is strictly greater-than; a body of exactly the cap must not raise."""
    exact = b"x" * _WEB_CONTENT_LIMIT_BYTES
    response = FakeResponse(body=exact, headers={"content-type": "application/pdf"})
    assert len(_read_capped_body(response)) == _WEB_CONTENT_LIMIT_BYTES


def test_read_capped_body_reports_the_read_count_not_a_smaller_declared_length():
    """Content-Length describes the encoded body while iter_content yields decoded bytes,
    so a gzipped response can declare a small length and still stream past the cap. The
    refusal must report what was actually read, not the header, or the message
    contradicts itself (e.g. "2000000 bytes, and the limit is 10485760 bytes")."""
    oversized = b"x" * (_WEB_CONTENT_LIMIT_BYTES + 1)
    response = FakeResponse(body=oversized, headers={"content-type": "application/pdf", "content-length": "2000000"})
    with pytest.raises(_BodyTooLarge) as excinfo:
        _read_capped_body(response)
    assert "2000000" not in excinfo.value.size
    assert "bytes read" in excinfo.value.size


def test_read_capped_body_stops_reading_at_the_cap():
    """The point of the cap is not allocating the rest, so it must stop, not read then check."""
    read = {"bytes": 0}

    class CountingResponse(FakeResponse):
        def iter_content(self, chunk_size=1):
            while True:
                read["bytes"] += chunk_size
                yield b"x" * chunk_size

    with pytest.raises(_BodyTooLarge):
        _read_capped_body(CountingResponse())
    assert read["bytes"] < _WEB_CONTENT_LIMIT_BYTES * 2


def test_classify_reads_content_type_first():
    assert _classify(FakeResponse(headers={"content-type": "text/html; charset=utf-8"}), b"<p>x</p>") == "html"
    assert _classify(FakeResponse(headers={"content-type": "application/pdf"}), b"%PDF-1.7 ...") == "pdf"
    assert _classify(FakeResponse(headers={"content-type": "text/plain"}), b"plain") == "text"
    assert _classify(FakeResponse(headers={"content-type": "image/png"}), b"\x89PNG") == "unsupported"


def test_classify_falls_back_to_magic_bytes_for_a_lying_content_type():
    """A PDF served as application/octet-stream is common; the header is the half that lies.
    application/octet-stream is not one of the HTML content types, so it falls through the
    (now-first) HTML check unaffected and still reaches the magic-byte check."""
    response = FakeResponse(headers={"content-type": "application/octet-stream"})
    assert _classify(response, b"%PDF-1.7 rest of the file") == "pdf"


def test_classify_does_not_let_a_lying_html_content_type_override_a_pdf_body():
    """A server that stamps text/html on every response is a real misconfiguration, not a
    hypothetical, and the HTML content-type check running first must not let that header
    override a body that is demonstrably a PDF; that would launder the PDF's bytes into the
    Markdown path as though they were HTML."""
    response = FakeResponse(headers={"content-type": "text/html; charset=utf-8"})
    assert _classify(response, b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>") == "pdf"


def test_classify_treats_xhtml_as_html_rather_than_the_plus_xml_text_branch():
    """application/xhtml+xml is one of the HTML content types and must be caught there,
    before the +xml suffix check that would otherwise classify it as plain text."""
    assert _classify(FakeResponse(headers={"content-type": "application/xhtml+xml"}), b"<html/>") == "html"


def test_classify_accepts_json_and_xml_and_their_vendor_variants():
    """application/json and application/xml are structured text, not the binary-laundered-
    as-text failure this tool exists to prevent; RSS and Atom feeds are the +xml case
    examples/news-summarizer depends on."""
    assert _classify(FakeResponse(headers={"content-type": "application/json"}), b"{}") == "text"
    assert _classify(FakeResponse(headers={"content-type": "application/xml"}), b"<a/>") == "text"
    assert _classify(FakeResponse(headers={"content-type": "application/rss+xml"}), b"<rss/>") == "text"
    assert _classify(FakeResponse(headers={"content-type": "application/atom+xml"}), b"<feed/>") == "text"


def test_classify_refuses_a_response_with_no_content_type():
    """Guessing "text" for an undeclared body is the same failure mode this tool exists to
    remove, in a different shape, so an absent Content-Type stays refused deliberately."""
    assert _classify(FakeResponse(headers={}), b"anything") == "unsupported"


def test_declared_size_prefers_content_length():
    response = FakeResponse(headers={"content-type": "application/pdf", "content-length": "4096"})
    assert _declared_size(response, b"xx") == "4096 bytes"


def test_declared_size_says_what_it_read_when_no_length_is_declared():
    response = FakeResponse(headers={"content-type": "application/pdf"})
    assert _declared_size(response, b"xxxx") == "4 bytes read"


# ---------------------------------------------------------------------------
# _truncate / _window
# ---------------------------------------------------------------------------


def test_truncate_marker_is_unchanged():
    """submit_form's output must not move: it is the one caller left that cannot page."""
    assert _truncate("x" * 12, 10) == "x" * 10 + "\n[... truncated 2 chars]"


def test_truncate_leaves_short_text_alone():
    assert _truncate("short", 10) == "short"


def test_window_marker_names_the_next_offset():
    out = _window("abcdefghij", offset=1, limit=4, unit="character", tool="get_web_content")
    assert out.startswith("abcd")
    assert "characters 1-4 of 10" in out
    assert "call get_web_content with offset=5 to continue" in out


def test_window_serves_lines_and_characters_from_one_implementation():
    """The unit differs; the marker, the arithmetic, and the last-window silence do not."""
    chars = _window("abcdef", offset=3, limit=2, unit="character", tool="t")
    lines = _window(["a", "b", "c", "d", "e", "f"], offset=3, limit=2, unit="line", tool="t", join="\n")

    assert chars.startswith("cd") and lines.startswith("c\nd")
    assert "characters 3-4 of 6" in chars
    assert "lines 3-4 of 6" in lines


def test_window_does_not_mark_a_final_window():
    assert _window("abcdef", offset=5, limit=100, unit="character", tool="t") == "ef"


def test_window_complaint_catches_an_unusable_request():
    assert "1-indexed" in _window_complaint(offset=0, limit=10, limit_name="max_chars", unit="character")
    assert "max_chars must be 1 or greater" in _window_complaint(
        offset=1, limit=0, limit_name="max_chars", unit="character"
    )
    assert _window_complaint(offset=1, limit=10, limit_name="max_chars", unit="character") is None


# ---------------------------------------------------------------------------
# get_web_content
# ---------------------------------------------------------------------------

ARTICLE_HTML = """
<html><head><meta property="article:published_time" content="2026-01-02T00:00:00Z"></head>
<body><h1>Headline</h1><p>Body <em>text</em>.</p></body></html>
"""


def test_get_web_content_returns_markdown_for_html(monkeypatch):
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(text=ARTICLE_HTML, headers={"content-type": "text/html"}),
    )
    out = get_web_content("http://site.example/")
    assert "# Headline" in out
    assert "*text*" in out


def test_get_web_content_keeps_the_published_line(monkeypatch):
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(text=ARTICLE_HTML, headers={"content-type": "text/html"}),
    )
    out = get_web_content("http://site.example/")
    assert out.startswith("Published: 2026-01-02T00:00:00Z")


def test_get_web_content_extracts_a_pdf(monkeypatch):
    from tests.test_documents import minimal_pdf

    pdf = minimal_pdf(["Quarterly results follow"])
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=pdf, headers={"content-type": "application/pdf"}),
    )
    out = get_web_content("http://site.example/report.pdf")
    assert "## Page 1" in out
    assert "Quarterly results follow" in out


def test_get_web_content_extracts_a_pdf_served_as_octet_stream(monkeypatch):
    from tests.test_documents import minimal_pdf

    pdf = minimal_pdf(["Served with the wrong type"])
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=pdf, headers={"content-type": "application/octet-stream"}),
    )
    assert "Served with the wrong type" in get_web_content("http://site.example/x")


def test_get_web_content_refuses_an_unsupported_type(monkeypatch):
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=b"\x89PNG\r\n", headers={"content-type": "image/png", "content-length": "6"}),
    )
    out = get_web_content("http://site.example/x.png")
    assert "image/png" in out
    assert "6 bytes" in out
    assert "PNG" not in out  # the bytes themselves never reach the caller


def test_get_web_content_returns_json_as_is(monkeypatch):
    body = b'{"result": "ok"}'
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=body, headers={"content-type": "application/json"}),
    )
    out = get_web_content("http://site.example/api")
    assert out == '{"result": "ok"}'


def test_get_web_content_returns_an_atom_feed_as_is(monkeypatch):
    body = b"<feed><entry><title>Item</title></entry></feed>"
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=body, headers={"content-type": "application/atom+xml"}),
    )
    out = get_web_content("http://site.example/feed")
    assert "<title>Item</title>" in out


def test_get_web_content_refuses_a_response_with_no_content_type(monkeypatch):
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=b"mystery bytes", headers={}),
    )
    out = get_web_content("http://site.example/x")
    assert "declared no content type" in out
    assert "13 bytes" in out  # len(b"mystery bytes")


def test_get_web_content_reports_a_password_protected_pdf(monkeypatch):
    from tests.test_documents import _encrypted_pdf, minimal_pdf

    locked = _encrypted_pdf(minimal_pdf(["Secret"]), user_password="letmein", owner_password="s")
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=locked, headers={"content-type": "application/pdf"}),
    )
    assert "password" in get_web_content("http://site.example/locked.pdf")


def test_get_web_content_caps_its_return_and_names_the_continuation(monkeypatch):
    html = "<p>" + ("word " * 20000) + "</p>"
    monkeypatch.setattr(
        builtin.requests, "request", lambda *a, **k: FakeResponse(text=html, headers={"content-type": "text/html"})
    )
    out = get_web_content("http://site.example/")
    # The remedy named is the one that does not re-read what this call already returned;
    # raising max_chars to swallow the whole document is what the cap exists to prevent.
    assert "truncated: showing characters 1-20000 of" in out
    assert "call get_web_content with offset=20001 to continue" in out


def test_get_web_content_honors_a_raised_max_chars(monkeypatch):
    html = "<p>" + ("word " * 20000) + "</p>"
    monkeypatch.setattr(
        builtin.requests, "request", lambda *a, **k: FakeResponse(text=html, headers={"content-type": "text/html"})
    )
    assert "[... truncated" not in get_web_content("http://site.example/", max_chars=200000)


def test_get_web_content_refuses_an_oversized_download(monkeypatch):
    oversized = b"x" * (_WEB_CONTENT_LIMIT_BYTES + 1)
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=oversized, headers={"content-type": "application/pdf"}),
    )
    out = get_web_content("http://site.example/huge.pdf")
    assert "too large" in out
    assert str(_WEB_CONTENT_LIMIT_BYTES) in out


def test_get_web_content_reports_transport_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(builtin.requests, "request", boom)
    assert get_web_content("http://site.example/").startswith("Error fetching page:")


def test_get_web_content_never_touches_response_text(monkeypatch):
    """Pins the streaming contract: requests raises on .text once iter_content ran."""

    class TextRaisingResponse(FakeResponse):
        @property
        def text(self):
            raise RuntimeError("The content for this response was already consumed")

        @text.setter
        def text(self, value):
            self._text = value

    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: TextRaisingResponse(
            body=b"<html><body><h1>Streamed</h1></body></html>", headers={"content-type": "text/html"}
        ),
    )
    assert "# Streamed" in get_web_content("http://site.example/")


def test_get_web_content_falls_back_when_the_declared_charset_is_unknown(monkeypatch):
    """A bogus charset label (utf8mb4, unicode, none, ...) is common in the wild; the old
    get_webpage tolerated it via response.text's own LookupError/TypeError retry, and this
    tool must not regress to raising out of a call whose whole contract is a string back."""
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(
            text="<html><body><h1>Hello</h1></body></html>",
            headers={"content-type": "text/html"},
            encoding="utf8mb4",
        ),
    )
    out = get_web_content("http://site.example/")
    assert "# Hello" in out


def test_get_web_content_falls_back_when_the_declared_charset_is_not_a_string(monkeypatch):
    """bytes.decode raises TypeError, not LookupError, when the encoding argument itself is
    not a usable codec name (e.g. requests handed back something other than a str). The
    decode guard's own rationale cites response.text's LookupError/TypeError retry, so both
    must be caught."""
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(
            text="<html><body><h1>Hello</h1></body></html>",
            headers={"content-type": "text/html"},
            encoding=12345,
        ),
    )
    out = get_web_content("http://site.example/")
    assert "# Hello" in out


def test_get_web_content_falls_back_to_utf8_when_no_charset_is_declared(monkeypatch):
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(
            text="<html><body><h1>Hello</h1></body></html>",
            headers={"content-type": "text/html"},
            encoding=None,
        ),
    )
    out = get_web_content("http://site.example/")
    assert "# Hello" in out


def test_get_web_content_honors_a_declared_non_utf8_charset(monkeypatch):
    """caf\xe9 (latin-1 "café") decodes to different text under utf-8 with errors="replace":
    the trailing byte is not a valid utf-8 continuation on its own, so the wrong codec would
    replace it rather than reproduce "café". This pins that the declared charset is honored."""
    body = "café".encode("iso-8859-1")
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(body=body, headers={"content-type": "text/plain"}, encoding="iso-8859-1"),
    )
    out = get_web_content("http://site.example/")
    assert "café" in out


def test_get_web_content_reports_a_javascript_shell_page_instead_of_returning_empty(monkeypatch):
    """html_to_markdown on a page with no prose (a JS-rendered shell) returns "", which would
    otherwise flow out of the tool as an empty tool result. pdf_to_markdown already refuses
    this for PDFs; the HTML path must say something too, rather than reading as an empty
    document."""
    shell_html = "<html><head><script>var x = 1;</script></head><body></body></html>"
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(text=shell_html, headers={"content-type": "text/html"}),
    )
    out = get_web_content("http://site.example/app")
    assert "no readable text" in out
    assert "get_webpage_html" in out


def test_get_webpage_is_gone():
    assert not hasattr(builtin, "get_webpage")


# ---------------------------------------------------------------------------
# get_web_content windowing
# ---------------------------------------------------------------------------


def test_get_web_content_truncation_marker_names_the_next_offset(monkeypatch):
    body = "<html><body><p>" + ("word " * 6000) + "</p></body></html>"
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text=body))

    out = get_web_content("http://site.example/", max_chars=500)

    assert "truncated: showing characters 1-500 of" in out
    assert "call get_web_content with offset=501 to continue" in out


def test_get_web_content_offset_reads_a_later_window(monkeypatch):
    body = "<html><body><p>" + ("A" * 400) + "TARGET" + "</p></body></html>"
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text=body))

    first = get_web_content("http://site.example/", max_chars=100)
    assert "TARGET" not in first

    later = get_web_content("http://site.example/", max_chars=400, offset=101)
    assert "TARGET" in later


def test_get_web_content_pages_a_document_larger_than_one_window(monkeypatch):
    body = "<html><body><p>" + ("word " * 2000) + "</p></body></html>"
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text=body))

    seen, offset = "", 1
    while True:
        lines = get_web_content("http://site.example/", max_chars=1000, offset=offset).splitlines()
        truncated = bool(lines) and lines[-1].startswith("... (truncated")
        seen += "\n".join(lines[:-1] if truncated else lines)
        if not truncated:
            break
        offset += 1000

    whole = get_web_content("http://site.example/", max_chars=10**6)
    assert seen == whole
    assert "truncated" not in whole


def test_get_web_content_offset_past_the_end_says_how_long_it_is(monkeypatch):
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: FakeResponse(text="<p>short</p>"))

    out = get_web_content("http://site.example/", offset=99999)

    assert "past the end" in out
    assert "characters" in out


def test_get_web_content_rejects_a_non_positive_offset_without_fetching(monkeypatch):
    calls = []
    monkeypatch.setattr(builtin.requests, "request", lambda *a, **k: calls.append(1) or FakeResponse(text="<p>x</p>"))

    out = get_web_content("http://site.example/", offset=-5)

    assert "1-indexed" in out
    assert calls == []


def test_get_web_content_windows_a_pdf_too(monkeypatch):
    # A PDF is the body most likely to exceed one window, and the path that converts it is
    # separate from the HTML one, so it needs its own proof that the window applies there.
    from tests.test_documents import minimal_pdf

    pdf = minimal_pdf(["alpha " * 200, "beta " * 200])
    monkeypatch.setattr(
        builtin.requests,
        "request",
        lambda *a, **k: FakeResponse(text="", headers={"content-type": "application/pdf"}, body=pdf),
    )

    first = get_web_content("http://site.example/doc.pdf", max_chars=300)
    assert "truncated: showing characters 1-300 of" in first
    assert "call get_web_content with offset=301 to continue" in first

    later = get_web_content("http://site.example/doc.pdf", max_chars=300, offset=301)
    assert "alpha" in later or "beta" in later
