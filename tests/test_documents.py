"""Unit tests for the private document converters behind get_web_content.

No network and no checked-in binaries: the PDF fixtures are built literally by
``minimal_pdf`` below, so a reader can see exactly what each one contains.
"""

from __future__ import annotations

import pytest

from aimu.tools._documents import DocumentConversionError, html_to_markdown


def test_html_to_markdown_converts_structure():
    html = """
    <html><body>
      <h1>Title</h1>
      <p>Hello <strong>world</strong> and <a href="/l">link</a>.</p>
      <ul><li>one</li><li>two</li></ul>
    </body></html>
    """
    out = html_to_markdown(html)
    assert "# Title" in out
    assert "**world**" in out
    assert "[link](/l)" in out
    assert "* one" in out


def test_html_to_markdown_drops_script_and_style_content():
    html = "<html><head><title>T</title></head><body><script>alert('x')</script><style>.a{}</style><p>Visible</p></body></html>"
    out = html_to_markdown(html)
    assert "Visible" in out
    assert "alert" not in out
    assert ".a{}" not in out


def test_html_to_markdown_on_empty_input_returns_empty():
    assert html_to_markdown("") == ""


def test_document_conversion_error_is_an_exception():
    with pytest.raises(DocumentConversionError):
        raise DocumentConversionError("boom")
