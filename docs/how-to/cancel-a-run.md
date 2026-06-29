# Cancel a run

A long agent run, many tool-calling rounds, a slow model, sometimes needs to be stopped: a user
hits "stop", a timeout fires, a supervisor moves on. On the async surface (`aimu.aio`), cancellation
rides on asyncio task cancellation, and `RunHandle` is the thin wrapper that exposes it.

## RunHandle

`RunHandle.start(coro)` schedules a run as a task and hands you a handle to cancel or await it:

```python
from aimu import aio

agent = aio.Agent(aio.client("ollama:qwen3:8b"), tools=[...])
handle = aio.RunHandle.start(agent.run("a big task"))

# ...later, from a /stop command, a timeout, another task...
handle.cancel()

try:
    reply = await handle.result()
except asyncio.CancelledError:
    ...  # the run was stopped
```

`cancel()` returns `False` if the run had already finished; `done` / `cancelled` report status; you
can also `await handle` directly. It wraps any run coroutine, so it works for a workflow run too.

Cancellation takes effect at the run's natural `await` points (each `model_client.chat(...)` call in
the loop), so an in-flight model request or tool round is interrupted cleanly. `CancelledError` is a
`BaseException`, so AIMU's `except Exception` handlers never swallow it.

## Resume after a stop

A cancelled run still records its partial turn: the agent snapshots `messages` in a `finally`, so
`agent.model_client.messages` holds what happened so far. Resume with
[`restore()`](using-llms-inside-tools.md) just as you would after a failure:

```python
handle = aio.RunHandle.start(agent.run(task))
handle.cancel()
try:
    await handle.result()
except asyncio.CancelledError:
    partial = list(agent.model_client.messages)   # capture the partial turn

# later, on a fresh agent (or the same one):
agent.restore(partial)
await agent.run("continue where we left off")
```

## Wiring `/stop` into an app

To cancel a turn *while it is running*, the app must keep reading input concurrently with the turn,
so run each turn as a background `RunHandle` rather than awaiting it inline. The
[personal-assistant example](build-personal-assistant.md) does exactly this: its serve loop starts
each turn as a `RunHandle`, and a `/stop` message cancels the current one (which then reports
"(stopped)" and persists its partial state, while the daemon keeps serving).

## See also

- [Build a personal assistant](build-personal-assistant.md): the `/stop` demo in context
- [Use async (`aio`)](use-async.md): the async surface this lives on
