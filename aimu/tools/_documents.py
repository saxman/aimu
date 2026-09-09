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

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

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
