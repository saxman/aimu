"""An agent's inbox: messages handed to a loop that is already running.

A host application that keeps reading its input while an agent runs needs somewhere to put a
message that arrives mid-run. ``Inbox`` is that seam: the loop asks for pending messages at
each round boundary and sends whatever it gets as that round's user message, on a model call it
was going to make anyway.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Inbox(Protocol):
    """A source of messages for runs that are already in progress.

    ``reader()`` is called **once per run**, at its start, and returns that run's own drain.
    A reader per run rather than one shared drain, because nothing about the calling context
    identifies a reader: with sequential tool dispatch a spawned sub-agent's loop executes in
    the *same* asyncio task as the agent that spawned it, so two runs keyed by task would share
    one cursor and consume each other's messages. Two runs of the same agent type would collide
    on a name for the same reason.

    The drain returns every message that arrived since it was last called, oldest first, and an
    empty list when there is nothing. It is called from inside the loop and must not block.

    Called ``Inbox`` because each run opens a reader of its own: that is inbox semantics from
    the run's side, even though the one object underneath serves every run.
    """

    def reader(self) -> Callable[[], list[str]]: ...
