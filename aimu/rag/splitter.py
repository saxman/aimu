"""Recursive text splitting -- a plain function, not a class hierarchy.

`split_text` chops a long string into overlapping chunks small enough to embed or
fit in a prompt. It tries a hierarchy of separators (paragraphs, then lines, then
sentences, then words, then characters), preferring to break on the largest separator
that keeps chunks under ``chunk_size``. Size is measured by ``length_function`` (default
character count); pass a tokenizer's token-counter for token-aware chunking.
"""

from __future__ import annotations

from typing import Callable

# Tried in order: split on the largest separator that keeps pieces under chunk_size,
# recursing into still-too-big pieces with the next separator. "" is the base case
# (split into individual characters) so no piece is ever larger than chunk_size.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
    length_function: Callable[[str], int] = len,
) -> list[str]:
    """Split *text* into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum size of each chunk, measured by ``length_function``.
        chunk_overlap: How much trailing content of one chunk repeats at the start of the
            next (continuity across boundaries). Must be smaller than ``chunk_size``.
        separators: Separator hierarchy, largest semantic unit first. Defaults to
            paragraphs -> lines -> sentences -> words -> characters.
        length_function: Measures chunk size. Defaults to ``len`` (characters); pass e.g.
            a tokenizer's encode-and-count for token-aware chunking.

    Returns:
        A list of chunk strings in document order. Empty / whitespace-only input returns
        ``[]``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

    seps = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)
    if not text.strip():
        return []
    return _split(text, seps, chunk_size, chunk_overlap, length_function)


def _split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
) -> list[str]:
    # Pick the first separator that occurs in the text; fall back to the last one.
    separator = separators[-1]
    remaining: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            remaining = []
            break
        if sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    pieces = list(text) if separator == "" else text.split(separator)

    chunks: list[str] = []
    accumulated: list[str] = []
    for piece in pieces:
        if length_function(piece) < chunk_size:
            accumulated.append(piece)
            continue
        # Flush what we've accumulated, then handle the oversized piece.
        if accumulated:
            chunks.extend(_merge(accumulated, separator, chunk_size, chunk_overlap, length_function))
            accumulated = []
        if remaining:
            chunks.extend(_split(piece, remaining, chunk_size, chunk_overlap, length_function))
        else:
            # No finer separator left (separator == ""): the piece is a single
            # character or otherwise unsplittable; keep it as its own chunk.
            chunks.append(piece)
    if accumulated:
        chunks.extend(_merge(accumulated, separator, chunk_size, chunk_overlap, length_function))
    return chunks


def _merge(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
) -> list[str]:
    """Greedily pack splits into chunks under ``chunk_size``, carrying ``chunk_overlap``."""
    sep_len = length_function(separator)
    chunks: list[str] = []
    window: list[str] = []
    total = 0

    for split in splits:
        split_len = length_function(split)
        added = split_len + (sep_len if window else 0)
        if total + added > chunk_size and window:
            chunks.append(separator.join(window))
            # Drop from the front until the carried overlap fits under chunk_overlap
            # (and the chunk would otherwise overflow).
            while window and (total > chunk_overlap or total + added > chunk_size):
                removed = length_function(window[0]) + (sep_len if len(window) > 1 else 0)
                total -= removed
                window.pop(0)
        window.append(split)
        total += split_len + (sep_len if len(window) > 1 else 0)

    if window:
        chunks.append(separator.join(window))
    return [c for c in chunks if c]
