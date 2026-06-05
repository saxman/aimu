"""Stream-chunk utilities shared by sync and async model surfaces.

Pure stateless helpers — no I/O, no provider knowledge. Lifted out of
``BaseModelClient`` so the async surface can reuse them.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterable, Iterator, Union

from .._base.shared import StreamChunk, StreamingContentType


def resolve_include(include: Iterable[Union[str, StreamingContentType]]) -> set[StreamingContentType]:
    """Normalise an ``include=`` argument to a set of :class:`StreamingContentType`."""
    return {StreamingContentType(p) if not isinstance(p, StreamingContentType) else p for p in include}


def filter_chunks(chunks: Iterator[StreamChunk], include: set[StreamingContentType]) -> Iterator[StreamChunk]:
    """Drop chunks whose phase isn't in the include set."""
    for chunk in chunks:
        if chunk.phase in include:
            yield chunk


async def afilter_chunks(
    chunks: AsyncIterator[StreamChunk], include: set[StreamingContentType]
) -> AsyncIterator[StreamChunk]:
    """Async equivalent of :func:`filter_chunks`."""
    async for chunk in chunks:
        if chunk.phase in include:
            yield chunk
