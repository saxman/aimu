"""Unit tests for the private document converters behind get_web_content.

No network and no checked-in binaries: the PDF fixtures are built literally by
``minimal_pdf`` below, so a reader can see exactly what each one contains.
"""

from __future__ import annotations

import pytest

from aimu.tools._documents import DocumentConversionError, html_to_markdown, pdf_to_markdown


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


def minimal_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one line of Helvetica text per page, built literally.

    pypdf can read PDFs but cannot draw text (that needs reportlab), and a checked-in
    binary fixture is a thing no reader can inspect. Building the bytes here costs
    twenty lines and leaves the fixture readable: object numbers, the xref table, and
    the text of every page are all visible above.
    """
    objects: list[bytes] = []
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for text in pages:
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents " + str(len(objects) + 2).encode() + b" 0 R >>"
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _encrypted_pdf(data: bytes, user_password: str, owner_password: str) -> bytes:
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter(clone_from=BytesIO(data))
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _blank_pdf() -> bytes:
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_to_markdown_marks_each_page():
    out = pdf_to_markdown(minimal_pdf(["Hello page one", "Second page here"]))
    assert "## Page 1" in out
    assert "Hello page one" in out
    assert "## Page 2" in out
    assert "Second page here" in out
    assert out.index("## Page 1") < out.index("## Page 2")


def test_pdf_to_markdown_opens_an_owner_password_only_pdf():
    """The common case for a published report: encrypted, but the user password is empty."""
    data = _encrypted_pdf(minimal_pdf(["Restricted but readable"]), user_password="", owner_password="secret")
    assert "Restricted but readable" in pdf_to_markdown(data)


def test_pdf_to_markdown_reports_a_password_protected_pdf():
    data = _encrypted_pdf(minimal_pdf(["Secret"]), user_password="letmein", owner_password="secret")
    with pytest.raises(DocumentConversionError, match="password"):
        pdf_to_markdown(data)


def test_pdf_to_markdown_reports_a_pdf_with_no_text_layer():
    with pytest.raises(DocumentConversionError, match="no extractable text"):
        pdf_to_markdown(_blank_pdf())


def test_pdf_to_markdown_reports_unparseable_bytes():
    with pytest.raises(DocumentConversionError, match="could not be read"):
        pdf_to_markdown(b"%PDF-1.4 this is not really a pdf")


def test_pdf_to_markdown_omits_a_page_with_no_text_but_keeps_later_numbering():
    """Blank pages are skipped to avoid inflating context with empty sections."""
    out = pdf_to_markdown(minimal_pdf(["Page one text", "", "Page three text"]))
    assert "## Page 1" in out
    assert "Page one text" in out
    assert "## Page 2" not in out
    assert "## Page 3" in out
    assert "Page three text" in out
