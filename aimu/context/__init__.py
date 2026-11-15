"""Context management: what a request to a model holds, as plain functions over the conversation.

Conversation state in AIMU is a ``list[dict]`` (OpenAI message format) you can print and edit, so
the tools that decide what a request carries are plain functions that take a list and return a
new one, the same shape as :mod:`aimu.rag`, rather than a ``ContextPolicy`` class. This package
holds two things:

- :func:`count_tokens`, the estimate every budget here is measured in.
- :mod:`aimu.context.compaction`, the strategies (trim, mask, summarize) and the ready-made
  compactions an ``Agent(compaction=...)`` takes, bundled the way :mod:`aimu.tools.builtin`
  bundles tools.

:data:`Compaction` is the contract between the two halves of a compaction: the strategy (a
function from the conversation's full history to what one request sends) and the agent loop that
applies it to every request and announces what it leaves out (see
``aimu.agents._tool_loop._BaseToolLoop._compacted_request``). The history itself is never
rewritten.

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
importing that module's list, since this package intentionally imports nothing but stdlib
and ``typing`` (it joins the ``import aimu`` chain).
"""

from __future__ import annotations

import json
from typing import Callable, Optional

Compaction = Callable[[list[dict]], list[dict]]
"""What ``Agent(compaction=...)`` takes: the conversation's full history in, what one request sends out.

It runs on every request of a run, on a copy of the history, so it must not mutate its argument,
and it should depend only on its argument (and its own settings): the next request calls it again
on a longer history, and a strategy that remembered earlier calls could not be swapped for another
without losing track. :mod:`aimu.context.compaction` holds the ones AIMU ships.
"""

# Duplicated literals, not an import: mirrors aimu.models._internal.message_meta.INERT_MESSAGE_KEYS,
# the non-standard bookkeeping keys AIMU attaches to message dicts that a provider never sees. Kept
# as local literals (rather than importing that module) so this package stays stdlib+typing only.
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


# Last, so `compaction` can import count_tokens from this package while it initializes; the same
# arrangement as aimu.tools importing its builtin submodule.
from . import compaction  # noqa: E402

__all__ = ["Compaction", "compaction", "count_tokens"]
