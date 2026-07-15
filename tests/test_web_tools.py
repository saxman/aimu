"""Mock-only unit tests for the web-browsing tools.

Covers the stateless ``get_webpage_html`` tool and the stateful ``make_web_tools``
factory (``find_forms`` / ``submit_form``). No network access; ``requests`` is stubbed.
"""

from __future__ import annotations

import requests

from aimu.tools import builtin
from aimu.tools.builtin import get_webpage_html, make_web_tools

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
    def __init__(self, text="", status_code=200, url="http://site.example/"):
        self.text = text
        self.status_code = status_code
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


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
    assert "[... truncated" in out
    assert len(out) < len(big)


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
