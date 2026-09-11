"""Windowed reads of output too large to return whole.

Every tool that returns part of something larger returns the same shape: a window, plus a
marker naming the call that continues from where it stopped. A cap with no offset is not a
smaller read, it is an unreachable tail, and the only remedy such a tool can offer is a
bigger cap: on exactly the documents where the cap bites, that overflows the context window
the cap was protecting.

This lives at the package root rather than in ``aimu.tools`` because both ``aimu.tools`` and
``aimu.memory`` window their model-facing reads, and neither package should depend on the
other (``aimu.tools`` eagerly imports ``builtin``, so an import from ``aimu.memory`` would
pull ``requests`` and the whole built-in tool set into an MCP server that wants neither).
Keeping one implementation is the point: four markers written separately become four
dialects, and a model that learns one of them is then wrong about the rest.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

Units = Union[str, Sequence[str]]


def window_complaint(*, offset: int, limit: int, limit_name: str, unit: str) -> Optional[str]:
    """Return a model-facing complaint about an unusable window request, else None.

    Returned rather than raised, like the rest of this codebase's model-facing argument
    complaints, so the model corrects its own next call instead of the run failing on a
    mistake it could have fixed.
    """
    if offset < 1:
        return f"offset must be 1 or greater (it is 1-indexed; the first {unit} is offset=1), got {offset}"
    if limit < 1:
        return f"{limit_name} must be 1 or greater, got {limit}"
    return None


def past_end(*, offset: int, total: int, unit: str, describe: str) -> str:
    """Report an offset past the end of *describe*, naming its real size to correct against."""
    return f"offset {offset} is past the end of {describe}: it has {total} {unit}s"


def window(units: Units, *, offset: int, limit: int, unit: str, tool: str, join: str = "") -> str:
    """Return *limit* units from 1-indexed *offset*, marked when more remains.

    *units* is a ``str`` (character windows) or a list of lines; slicing and ``join.join``
    behave identically on both, so line-windowed and character-windowed tools share this one
    implementation.
    """
    start = offset - 1
    selected = units[start : start + limit]
    last = start + len(selected)
    content = join.join(selected)
    if last < len(units):
        # Name the window and the total, so the model sees the size of the gap rather than
        # only that something was cut, and name the exact call that continues from here.
        content += (
            f"\n... (truncated: showing {unit}s {offset}-{last} of {len(units)}; "
            f"call {tool} with offset={last + 1} to continue)"
        )
    return content
