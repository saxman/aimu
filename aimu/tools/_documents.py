"""Converters turning fetched document bytes into Markdown, for ``get_web_content``.

Private for now, and deliberately: each function has exactly one caller, and the public
shape of a document-conversion API should be decided by the on-disk file handling that
will actually consume it (a path-taking, type-dispatching tool) rather than guessed at
here. When that work lands, this module is what gets promoted.

Nothing here knows about HTTP, tools, or ``requests``. A converter takes what was
fetched and returns Markdown, or raises :class:`DocumentConversionError` naming what it
could not do. That split is what lets the tool own every model-facing string while these
stay testable on literals.
"""

from __future__ import annotations

from io import BytesIO

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from pypdf import PasswordType, PdfReader
from pypdf.errors import DependencyError, PdfReadError

# Tags whose *content* is not page content. markdownify's own ``strip`` option removes a
# tag while keeping its text, which is the opposite of what these need: a <script> body
# is code, not prose, and it would otherwise be converted right into the output.
_NON_CONTENT_TAGS = ("script", "style", "head", "noscript", "template")


class DocumentConversionError(Exception):
    """A document could not be converted, for a reason worth telling the caller.

    Raised rather than returned so a converter has one success type. The tool catches it
    and hands the message to the model, which is why every message here names the
    condition in terms a reader can act on.
    """


def html_to_markdown(html: str) -> str:
    """*html* as Markdown, with non-content tags removed first.

    Two passes rather than one call: BeautifulSoup drops the tags whose content is not
    prose, and markdownify converts what is left. Passing the parsed soup on directly
    (``convert_soup``) avoids re-parsing the serialized tree.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(_NON_CONTENT_TAGS)):
        tag.decompose()
    return MarkdownConverter(heading_style="ATX").convert_soup(soup).strip()


def pdf_to_markdown(data: bytes) -> str:
    """*data* as Markdown, one ``## Page N`` heading per page with extractable text.

    Pages with no extractable text are omitted rather than emitting empty headings.
    A document whose figures are full-page images would otherwise contribute a run of
    empty sections, inflating the model's context for zero content. The numbering of
    the pages that do appear is unaffected, so both reasons for page markers still hold:
    a caller citing a report refers to a page number, and a document later cut by a
    character cap still reports how far it got in terms the source itself has.
    pypdf's extraction exposes no heading structure that could be promoted honestly, so
    nothing else is invented.

    An encrypted PDF is opened with an empty password before anything else is tried.
    Published reports are routinely encrypted with an owner password alone, which
    restricts printing and editing while leaving the text readable, and refusing those
    would refuse the common case.
    """
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise DocumentConversionError(
                "This PDF is encrypted and needs a password to open. Ask the user for it, "
                "or find a copy that is not password protected."
            )
        pages = [(number, page.extract_text()) for number, page in enumerate(reader.pages, start=1)]
    except DocumentConversionError:
        raise
    except (PdfReadError, DependencyError, NotImplementedError, OSError, ValueError) as exc:
        # pypdf raises PdfReadError for a malformed file, DependencyError for missing
        # decompression dependencies, NotImplementedError for unsupported encryption
        # filters or /V versions, and ValueError or OSError on inputs that are not PDFs
        # at all despite the header.
        raise DocumentConversionError(f"This PDF could not be read: {exc}") from exc

    if not any(text.strip() for _, text in pages):
        raise DocumentConversionError(
            f"This PDF has no extractable text across its {len(pages)} page(s). "
            "It is most likely a scan, which needs OCR rather than text extraction."
        )

    sections = [f"## Page {number}\n\n{text.strip()}" for number, text in pages if text.strip()]
    return "\n\n".join(sections)
