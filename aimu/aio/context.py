"""Async twin of :mod:`aimu.context` -- only :func:`summarize_messages` needs one.

``count_tokens`` and ``trim_messages`` are pure, synchronous functions over ``list[dict]``;
there is nothing for an async surface to add, so callers use :func:`aimu.context.count_tokens`
/ :func:`aimu.context.trim_messages` directly, from either surface. Only summarization makes
a model call, so only it gets an async twin here -- same signature, ``await``ing the client
instead of calling it. The grouping/partitioning logic is imported from :mod:`aimu.context`,
not duplicated.
"""

from __future__ import annotations

from typing import Any, Optional

# Direct import of the sibling module's underscore-prefixed helpers, not routed through an
# aimu._internal package: aimu.context is a single small module, not a multi-file package,
# and this mirrors the existing precedent of aimu.aio.providers.openai_compat importing
# _ThinkingParser / _split_thinking straight from aimu.models.providers._thinking -- a
# private helper shared by exactly one sync/async pair doesn't warrant a new shared package.
from aimu.context import DEFAULT_SUMMARIZE_PROMPT, _partition_for_summary, _summary_message, _summary_prompt

__all__ = ["summarize_messages"]


async def summarize_messages(
    client: Any,
    messages: list[dict],
    *,
    keep_first: int = 1,
    keep_last: int = 2,
    prompt: Optional[str] = None,
) -> list[dict]:
    """Async twin of :func:`aimu.context.summarize_messages`; awaits ``client.generate()``.

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
