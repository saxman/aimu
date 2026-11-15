"""Async twin of :mod:`aimu.context.compaction` -- only :func:`summarize_messages` needs one.

Masking and trimming are pure, synchronous functions over ``list[dict]``, and so are the
ready-made compactions built from them; there is nothing for an async surface to add, so callers
use :mod:`aimu.context.compaction` directly from either surface (an ``aio.Agent`` takes the same
``compaction=``). Only summarization makes a model call, so only it gets an async twin here --
same signature, ``await``ing the client instead of calling it. The partitioning logic is imported
from the sync module, not duplicated, so the two cannot disagree about what gets summarized.
"""

from __future__ import annotations

from typing import Any, Optional

from aimu.context.compaction import DEFAULT_SUMMARIZE_PROMPT, _partition_for_summary, _summary_message, _summary_prompt

__all__ = ["summarize_messages"]


async def summarize_messages(
    client: Any,
    messages: list[dict],
    *,
    keep_first: int = 1,
    keep_last: int = 2,
    prompt: Optional[str] = None,
) -> list[dict]:
    """Async twin of :func:`aimu.context.compaction.summarize_messages`; awaits ``client.generate()``.

    ``client`` is any object exposing ``async generate(prompt: str) -> str`` (an
    :class:`~aimu.aio.AsyncModelClient`, or ``agent.as_model_client()`` on an async
    :class:`~aimu.aio.Agent`). See the sync version's docstring for the full behaviour:
    system messages are always preserved, the first ``keep_first`` and last ``keep_last``
    messages (not exchanges) are kept verbatim (extended outward to whole tool-call groups),
    and everything between them is replaced by one summary message.
    """
    kept_system, head, prefix, tail = _partition_for_summary(messages, keep_first, keep_last)
    if not prefix:
        return kept_system + head + tail

    summary_text = await client.generate(_summary_prompt(prompt or DEFAULT_SUMMARIZE_PROMPT, head, prefix))
    return kept_system + [_summary_message(summary_text)] + head + tail
