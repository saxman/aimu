# Use sessions (multi-user)

[`ConversationManager`](persist-conversations.md) holds **one** conversation. When a single process
serves many users or chats (a multi-channel assistant), you need **per-conversation** state keyed by
who is talking. `aimu.sessions` provides that: a `Session` (one conversation's history + optional
memory scope + metadata) and a `SessionStore` that persists sessions by key.

It's a thin, sync layer (same family as `MemoryStore` / `ConversationManager`); the async assistant
loop calls it directly.

## The pieces

```python
from aimu.sessions import (
    Session, SessionStore, InMemorySessionStore, TinyDBSessionStore, SessionLocks, session_key,
)

store = InMemorySessionStore()          # non-durable; or TinyDBSessionStore("sessions.json")
key = session_key("telegram", "12345")  # "telegram:12345"  (single-user -> "default:default")

session = store.get(key)                # a detached snapshot; empty Session if none stored yet
session.messages.append({"role": "user", "content": "hi"})
store.save(session)                     # persist (create or replace by key)

store.list_keys()                       # ["telegram:12345"]
```

`Session` carries `messages` (plain `list[dict]` in OpenAI format, the same data an agent uses),
an optional `memory_namespace` (a scope you can pass to a `MemoryStore` to isolate a session's
memories), and free-form `metadata`. `get` returns a *detached* snapshot, so mutating it does nothing
until you `save`.

## Routing a conversation to its session (the per-turn pattern)

Agents never share a live `messages` list across sessions. Each turn loads that session's history
into the agent via the existing [`restore()`](using-llms-inside-tools.md) seam, runs, then snapshots
back. Serialize a single session's turns with `SessionLocks` while letting different sessions run
concurrently:

```python
from aimu import aio
from aimu.sessions import InMemorySessionStore, SessionLocks, session_key

agent = aio.Agent(aio.client("ollama:qwen3:8b"), system_message="You are helpful.")
store = InMemorySessionStore()
locks = SessionLocks()

async def handle(channel_name, sender, text):
    key = session_key(channel_name, sender)
    async with locks(key):                       # this session's turns are serialized
        session = store.get(key)
        agent.model_client.reset(system_message=agent.system_message)
        agent.restore(session.messages)           # load THIS conversation
        reply = await agent.run(text)
        session.messages = list(agent.model_client.messages)   # snapshot back
        store.save(session)
        return reply
```

Different users (different keys) hold different locks, so `handle("telegram", "A", ...)` and
`handle("telegram", "B", ...)` run concurrently while a single user's turns never interleave.

## Single user

Single-user needs no ceremony: `session_key(None, None)` is `"default:default"`, so one fixed key
holds the only conversation. (If that's all you need, plain `ConversationManager` is also fine.)

## Durable sessions

Swap the store; the rest is unchanged. `TinyDBSessionStore` writes one row per session and survives
restarts:

```python
store = TinyDBSessionStore("sessions.json")
# ... same handle() loop; sessions reload on the next run.
store.close()
```

## See also

- [Persist conversations](persist-conversations.md): `ConversationManager` (single conversation)
- [Build a personal assistant](build-personal-assistant.md): channels, scheduler, skills
- [`aimu.sessions` API reference](../reference/api/sessions.md)
