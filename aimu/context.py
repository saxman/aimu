"""Context management: plain functions over the conversation.

Mirrors :mod:`aimu.rag`'s shape: conversation state in AIMU is a ``list[dict]`` (OpenAI
message format) you can print and edit, so the tools for reshaping it are plain functions
that take a list and return a new one -- not a ``ContextPolicy`` class applied inside a
client. Rewriting a caller's conversation from inside the request path would be invisible
by construction, and this library exists to make a model's behaviour visible, not to hide
another layer of it.

**The invariant that justifies these existing at all**: trimming must never orphan a
``tool`` message from the ``assistant`` message carrying its ``tool_calls``. Every provider
rejects that shape, and it is exactly what a naive ``messages[-n:]`` slice produces. So
:func:`trim_messages` drops messages oldest-first over the *non-system* messages, and an
``assistant`` message carrying ``tool_calls`` is dropped together with every ``tool``
message answering it -- the pair (or group, for multiple tool calls in one turn) is treated
as one indivisible unit, never split.

**Token counting is an estimate, not a measurement.** :func:`count_tokens` defaults to
``len(text) // 4`` (the common English-text rule of thumb), and this is stated here plainly
rather than dressed up: an exact count depends on the provider's own tokenizer, which is
not available before a request is sent. The only exact count AIMU can report is post-hoc,
via ``client.last_usage`` after a real call. Pass ``counter=`` with a real tokenizer's
``encode``-and-``len`` (e.g. ``lambda text: len(enc.encode(text))``) when accuracy matters
more than a zero-dependency default.

**What counts as "text" for counting.** A message is JSON-serialized (``json.dumps``) and
that whole string is counted -- not just ``content``. The structural overhead (``role``,
tool-call ``id``/``type``/``arguments``, JSON punctuation) genuinely rides along in the
request payload a provider receives, so it belongs in the estimate. AIMU also attaches
non-standard, inert bookkeeping keys to some messages (``timestamp`` on every message,
``thinking`` and ``provenance`` on some; see :mod:`aimu.models._internal.message_meta`),
and those are never sent to a provider, so they are stripped before serializing: a token
count that silently included a ``thinking`` blob the provider never sees would overcount
and mislead. The three inert key names are duplicated here as literals rather than
importing that module's list, since this module intentionally imports nothing but stdlib
and ``typing`` (it joins the ``import aimu`` chain).

**``keep_last`` counts messages, not exchanges.** A "turn" is ambiguous -- one exchange is
usually a user message plus an assistant reply, i.e. 2 messages, but an agentic turn can
carry a user message, an assistant tool-call message, one-or-more tool results, and a final
assistant answer, i.e. 5+ messages. ``keep_last=2`` keeps the last 2 *messages*, not the
last 2 *exchanges*; a caller who wants "the last exchange" for a plain (non-tool-using)
conversation should pass ``keep_last=2`` knowing that, and more for a tool-using one. When
the boundary would fall inside a protected tool-call group, the boundary is pushed outward
to the group's edge instead of splitting it, so the tail always satisfies the same
no-orphaned-tool-result invariant as the dropped prefix.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

DEFAULT_SUMMARIZE_PROMPT = (
    "Summarize the conversation below in a few sentences. Preserve concrete facts, "
    "decisions, and any open questions a continuation of this conversation would still "
    "need to know."
)

# Duplicated literals, not an import: mirrors aimu.models._internal.message_meta.INERT_MESSAGE_KEYS,
# the non-standard bookkeeping keys AIMU attaches to message dicts that a provider never sees. Kept
# as local literals (rather than importing that module) so this module stays stdlib+typing only.
_INERT_KEYS = frozenset({"thinking", "provenance", "timestamp"})


def _estimate_tokens(text: str) -> int:
    """Default token counter: ~4 characters per token. An estimate, not a measurement."""
    return len(text) // 4


def _countable_text(message: dict) -> str:
    """Serialize the parts of *message* that a provider actually receives, for token counting.

    JSON-serializes *message* with :data:`_INERT_KEYS` stripped first. The structural
    overhead (``role``, tool-call ``id``/``type``/``arguments``, JSON punctuation) rides
    along in the real request payload, so it belongs in the estimate; AIMU's own inert
    bookkeeping keys never reach a provider, so they are removed before serializing.
    """
    filtered = {k: v for k, v in message.items() if k not in _INERT_KEYS}
    return json.dumps(filtered, sort_keys=True, default=str)


def _message_text(message: dict) -> str:
    """Extract human-readable text from one message, for rendering a summarization transcript.

    Only ``content`` (string, or the text blocks of a content-block list; image/audio
    blocks contribute nothing meaningful as text) and ``tool_calls`` (function name +
    arguments) are considered -- this is for a human/LLM reader, not token counting (see
    :func:`_countable_text` for that).
    """
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments", "")
        text = f"{name} {arguments}".strip()
        if text:
            parts.append(text)

    return " ".join(p for p in parts if p)


def count_tokens(messages: list[dict], *, counter: Optional[Callable[[str], int]] = None) -> int:
    """Estimate the token count of *messages*.

    The default counter is ``len(text) // 4``, a rule-of-thumb **estimate** -- it is not a
    real tokenizer and will be wrong (usually within 20-30%) for any specific model. Exact
    counts exist only after the fact, via ``client.last_usage`` on a real request. Pass
    ``counter=`` (any ``Callable[[str], int]``, e.g. a real tokenizer's ``encode``+``len``)
    for accuracy.

    ``text`` is each message JSON-serialized (structural overhead included, since it rides
    along in the real request) with AIMU's own inert bookkeeping keys (``timestamp``,
    ``thinking``, ``provenance``) stripped first, since a provider never sees those. See the
    module docstring for the full rationale.
    """
    counter = counter or _estimate_tokens
    text = "\n".join(_countable_text(m) for m in messages)
    return counter(text)


def _group_messages(messages: list[dict]) -> list[list[dict]]:
    """Partition *messages* into indivisible drop/keep units.

    An ``assistant`` message carrying ``tool_calls`` is grouped together with every
    immediately-following ``tool`` message (its results); every other message is its own
    singleton group. Groups preserve order and, concatenated, reproduce *messages* exactly.
    """
    groups: list[list[dict]] = []
    i, n = 0, len(messages)
    while i < n:
        message = messages[i]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            group = [message]
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                group.append(messages[j])
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([message])
            i += 1
    return groups


def _flatten(groups: list[list[dict]]) -> list[dict]:
    return [message for group in groups for message in group]


def _protect_tail(groups: list[list[dict]], keep_last: int) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split *groups* into ``(droppable, protected)`` from the end.

    Walks backward from the end of *groups*, accumulating message counts (not group counts,
    since ``keep_last`` counts messages), until at least *keep_last* messages are covered.
    The walk only ever takes whole groups, so a protected tail can never split a tool-call
    group in two -- the boundary is pushed outward to the group edge instead.
    """
    if keep_last <= 0:
        return list(groups), []
    protected: list[list[dict]] = []
    covered = 0
    index = len(groups)
    while index > 0 and covered < keep_last:
        index -= 1
        group = groups[index]
        protected.insert(0, group)
        covered += len(group)
    return groups[:index], protected


def trim_messages(
    messages: list[dict],
    max_tokens: int,
    *,
    keep_system: bool = True,
    keep_last: int = 2,
    counter: Optional[Callable[[str], int]] = None,
) -> list[dict]:
    """Return a new, trimmed message list that fits within *max_tokens* (estimated).

    Always returns a new list; *messages* (and its dicts) are never mutated. A conversation
    already at or under *max_tokens* is returned unchanged (a copy) -- this is a no-op, not
    a forced rewrite.

    Dropping proceeds oldest-first over the non-system messages, one indivisible group at a
    time (see :func:`_group_messages`): an ``assistant`` message carrying ``tool_calls`` is
    always dropped together with every ``tool`` message answering it, never separately. This
    is the invariant that justifies this function over a hand-written ``messages[-n:]``
    slice, which produces exactly that illegal shape.

    Args:
        messages: The conversation, OpenAI message-dict format.
        max_tokens: The token budget to trim down to, per ``counter`` (or the default
            estimate; see :func:`count_tokens`).
        keep_system: When ``True`` (default), every ``{"role": "system"}`` message is kept
            regardless of budget. When ``False``, system messages are ordinary droppable
            messages like any other.
        keep_last: The number of trailing **messages** (not exchanges -- see the module
            docstring) to always protect from dropping, extended outward to whole tool-call
            groups when the raw count would otherwise split one.
        counter: Optional real tokenizer; see :func:`count_tokens`.
    """
    messages = list(messages)
    if count_tokens(messages, counter=counter) <= max_tokens:
        return messages

    if keep_system:
        kept_system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
    else:
        kept_system = []
        rest = list(messages)

    droppable, protected = _protect_tail(_group_messages(rest), keep_last)

    while (
        droppable
        and count_tokens(kept_system + _flatten(droppable) + _flatten(protected), counter=counter) > max_tokens
    ):
        droppable.pop(0)

    return kept_system + _flatten(droppable) + _flatten(protected)


def _render_transcript(messages: list[dict]) -> str:
    """Render *messages* as a simple ``role: text`` transcript for a summarization prompt."""
    return "\n".join(f"{m.get('role', '?')}: {_message_text(m)}" for m in messages)


def summarize_messages(
    client: Any,
    messages: list[dict],
    *,
    keep_last: int = 2,
    prompt: Optional[str] = None,
) -> list[dict]:
    """Replace the older part of a conversation with a one-message LLM summary.

    Any ``{"role": "system"}`` messages are always preserved unchanged. Of the rest, the
    last ``keep_last`` **messages** (see the module docstring on why messages, not
    exchanges; extended outward to whole tool-call groups per :func:`_group_messages`, the
    same invariant :func:`trim_messages` enforces) are kept verbatim as the tail. Everything
    older than that (the "prefix") is rendered to a plain transcript, summarized by one
    ``client.generate()`` call, and replaced with a single ``{"role": "system", ...}``
    message carrying the summary text -- system, rather than user or assistant, because it
    is out-of-band context neither party actually said, not a real conversational turn.

    ``client`` is any object exposing ``generate(prompt: str) -> str`` (a plain
    :class:`~aimu.models.BaseModelClient`, or a :class:`~aimu.agents.Agent` via
    ``agent.as_model_client()``); this function takes it as an argument rather than
    constructing one itself, so it stays free of any provider dependency.

    When there is nothing older than the protected tail (the whole conversation already
    fits in ``keep_last``), no summarization call is made and the conversation is returned
    unchanged (a new list; *messages* is never mutated).

    Args:
        client: Anything with a ``generate(prompt: str) -> str`` method.
        messages: The conversation, OpenAI message-dict format.
        keep_last: Trailing messages (not exchanges) to keep verbatim; see above.
        prompt: Optional instruction prepended to the rendered transcript, replacing
            :data:`DEFAULT_SUMMARIZE_PROMPT`.
    """
    messages = list(messages)
    kept_system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    prefix_groups, tail_groups = _protect_tail(_group_messages(rest), keep_last)
    prefix = _flatten(prefix_groups)
    tail = _flatten(tail_groups)

    if not prefix:
        return kept_system + tail

    instruction = prompt or DEFAULT_SUMMARIZE_PROMPT
    summary_text = client.generate(f"{instruction}\n\n{_render_transcript(prefix)}")
    summary_message = {"role": "system", "content": f"Summary of earlier conversation:\n{summary_text}"}
    return kept_system + [summary_message] + tail


__all__ = ["count_tokens", "trim_messages", "summarize_messages", "DEFAULT_SUMMARIZE_PROMPT"]
