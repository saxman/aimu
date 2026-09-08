"""Multi-user session storage: per-conversation state keyed by ``channel:sender``.

See [how-to: use sessions](https://saxman.github.io/aimu/how-to/use-sessions/).
"""

from aimu.sessions.base import Session, SessionLocks, SessionStore, SessionSummary, session_key
from aimu.sessions.memory import InMemorySessionStore
from aimu.sessions.tinydb import TinyDBSessionStore

__all__ = [
    "InMemorySessionStore",
    "Session",
    "SessionLocks",
    "SessionStore",
    "SessionSummary",
    "TinyDBSessionStore",
    "session_key",
]
