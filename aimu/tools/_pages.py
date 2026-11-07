"""The store and text helpers behind :func:`~aimu.tools.builtin.make_page_tools`.

A fetched page is kept whole here so that what enters an agent's context can be a small, ranked
part of it, with the rest still reachable. Two choices shape the store:

- **Pages are addressed by handle, never by path.** A handle that does not match
  ``HANDLE_PATTERN`` is refused before anything touches the filesystem, so a read tool given a
  model-chosen string cannot be aimed at an arbitrary file.
- **The handle is content-addressed** (a digest of the URL and the text). Re-fetching an unchanged
  page returns the same handle; a page that changed gets a new one, so the copy an earlier answer
  was drawn from is never overwritten by a later fetch.

Offsets everywhere are 1-indexed characters, the unit ``get_web_content(offset=)`` uses, so a model
can carry an offset from any of these tools to any other.
"""

from __future__ import annotations

import bisect
import datetime
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HANDLE_PATTERN = re.compile(r"^page-[0-9a-f]{12}$")

#: The most outline entries a tool shows; a long PDF has a ``## Page N`` heading per page.
OUTLINE_LIMIT = 30

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")
_FENCE = ("```", "~~~")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


@dataclass(frozen=True)
class Page:
    handle: str
    url: str
    text: str
    fetched_at: str


class PageStore:
    """Fetched pages by handle, in memory or as plain files in a directory.

    A directory holds ``<handle>.md`` (the text exactly) and ``<handle>.json`` (its URL and fetch
    time), so the store is readable without AIMU. Each file is written to a temporary name and then
    renamed, so a concurrent reader never sees half a page.
    """

    def __init__(self, directory: Optional[Path] = None):
        self._directory = Path(directory) if directory is not None else None
        self._pages: dict[str, Page] = {}

    def put(self, url: str, text: str) -> Page:
        handle = "page-" + hashlib.sha256(f"{url}\n{text}".encode("utf-8")).hexdigest()[:12]
        existing = self.get(handle)
        if existing is not None:
            return existing
        fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        page = Page(handle=handle, url=url, text=text, fetched_at=fetched_at)
        if self._directory is None:
            self._pages[handle] = page
        else:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._write(f"{handle}.md", text)
            self._write(f"{handle}.json", json.dumps({"url": url, "fetched_at": fetched_at}))
        return page

    def get(self, handle: str) -> Optional[Page]:
        if not HANDLE_PATTERN.match(handle):
            return None
        if self._directory is None:
            return self._pages.get(handle)
        try:
            # newline="" keeps "\r\n" as written, so offsets match the text the page was ranked on.
            text = (self._directory / f"{handle}.md").read_text(encoding="utf-8", newline="")
            meta = json.loads((self._directory / f"{handle}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        return Page(handle=handle, url=meta["url"], text=text, fetched_at=meta["fetched_at"])

    def _write(self, name: str, content: str) -> None:
        descriptor, temporary = tempfile.mkstemp(dir=self._directory, prefix=".", suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(content)
        os.replace(temporary, self._directory / name)


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    offset: int
    size: int


def outline(text: str) -> list[Heading]:
    """The page's Markdown headings, each with its offset and the size of its section.

    A section runs to the next heading of any level. Lines inside fenced code blocks are skipped, so
    a shell comment in a code sample is not mistaken for a heading.
    """
    found: list[tuple[int, str, int]] = []
    in_fence = False
    position = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(_FENCE):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING.match(line.rstrip("\r\n"))
            if match:
                found.append((len(match.group(1)), match.group(2), position + 1))
        position += len(line)
    ends = [offset for _, _, offset in found[1:]] + [len(text) + 1]
    return [
        Heading(level=level, title=title, offset=offset, size=end - offset)
        for (level, title, offset), end in zip(found, ends)
    ]


@dataclass(frozen=True)
class Chunk:
    offset: int
    text: str
    heading: str


def chunks(text: str, *, size: int = 1200) -> list[Chunk]:
    """Split *text* into passages of at most *size* characters, each with its exact offset.

    Paragraphs are merged up to *size*, but never across a heading, so a passage stays inside one
    section and its ``heading`` label is true of all of it. A paragraph longer than *size* is cut
    every *size* characters. Not ``aimu.rag.split_text``: that returns strings without positions,
    and a passage is only useful to an agent if it can be found again with ``read_page``.
    """
    headings = outline(text)
    heading_starts = [heading.offset - 1 for heading in headings]

    def heading_at(index: int) -> str:
        position = bisect.bisect_right(heading_starts, index) - 1
        return headings[position].title if position >= 0 else ""

    # Paragraph spans as [start, end) indices, cut additionally at every heading line so a heading
    # that follows text without a blank line still starts its own passage.
    cuts: list[tuple[int, int]] = []
    start = 0
    for separator in _PARAGRAPH_BREAK.finditer(text):
        cuts.append((start, separator.start()))
        start = separator.end()
    cuts.append((start, len(text)))
    spans: list[tuple[int, int]] = []
    for span_start, span_end in cuts:
        inner = [index for index in heading_starts if span_start < index < span_end]
        for begin, finish in zip([span_start, *inner], [*inner, span_end]):
            if text[begin:finish].strip():
                spans.append((begin, finish))

    result: list[Chunk] = []

    def emit(begin: int, finish: int) -> None:
        result.append(Chunk(offset=begin + 1, text=text[begin:finish], heading=heading_at(begin)))

    group: Optional[list[int]] = None
    starts = set(heading_starts)
    for span_start, span_end in spans:
        if group is not None and (span_start in starts or span_end - group[0] > size):
            emit(*group)
            group = None
        if span_end - span_start > size:
            for piece in range(span_start, span_end, size):
                emit(piece, min(piece + size, span_end))
        elif group is None:
            group = [span_start, span_end]
        else:
            group[1] = span_end
    if group is not None:
        emit(*group)
    return result
