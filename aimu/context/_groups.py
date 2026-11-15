"""The one invariant every compaction strategy shares: a tool call and its results stay together.

Every provider rejects a ``tool`` message whose ``assistant`` turn carrying the ``tool_calls`` is
gone, and a call whose result is gone, and that orphaned shape is exactly what a naive
``messages[-n:]`` slice produces. So the strategies in :mod:`aimu.context.compaction` work on
*groups*: an ``assistant`` message carrying ``tool_calls`` together with every ``tool`` message
answering it is one indivisible unit, and every other message is a unit of its own.

**``keep_first`` / ``keep_last`` count messages, not exchanges.** A "turn" is ambiguous -- one
exchange is usually a user message plus an assistant reply, i.e. 2 messages, but an agentic turn
can carry a user message, an assistant tool-call message, one-or-more tool results, and a final
assistant answer, i.e. 5+ messages. ``keep_last=2`` keeps the last 2 *messages*, not the last 2
*exchanges*. When a boundary would fall inside a tool-call group it is pushed outward to the
group's edge instead of splitting it.
"""

from __future__ import annotations


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


def _protect_head(groups: list[list[dict]], keep_first: int) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split *groups* into ``(protected, rest)`` from the start.

    The mirror of :func:`_protect_tail`: takes whole groups until at least *keep_first*
    messages are covered, so a protected head can never split a tool-call group either.
    """
    protected: list[list[dict]] = []
    covered = 0
    index = 0
    while index < len(groups) and covered < keep_first:
        protected.append(groups[index])
        covered += len(groups[index])
        index += 1
    return protected, groups[index:]
