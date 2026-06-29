# `aimu.sessions`

Multi-user session storage: per-conversation state keyed by `channel:sender`, so one process serves many users/chats. See [how-to: use sessions](../../how-to/use-sessions.md).

Sync, matching the [`MemoryStore`](memory.md) / [`ConversationManager`](history.md) family; the async assistant loop calls it directly.

::: aimu.sessions.Session

::: aimu.sessions.SessionStore

::: aimu.sessions.InMemorySessionStore

::: aimu.sessions.TinyDBSessionStore

::: aimu.sessions.SessionLocks

::: aimu.sessions.session_key
