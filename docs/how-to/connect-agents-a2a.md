# Connect agents (A2A)

To call an agent that lives in another process — or to publish one of your agents for others to call — use AIMU's [Agent2Agent](https://a2a-protocol.org/) interop. It is the agent-level analog of [MCP tools](use-mcp-tools.md): MCP shares *tools* across a process boundary, A2A shares whole *agents*.

Requires the optional extra:

```bash
pip install 'aimu[a2a]'
```

`aimu.agents.HAS_A2A` reports whether it is installed. See [A2A vs MCP](../explanation/a2a-vs-mcp.md) for when to reach for which.

## Consume a remote agent

`RemoteAgent.connect(url)` resolves the remote agent card and returns a local `Runner`:

```python
from aimu.agents import RemoteAgent

remote = RemoteAgent.connect("http://localhost:9000")
print(remote.name)                       # from the remote agent card
print(remote.run("Summarise the news"))  # round-trips to the remote agent
remote.close()                            # tear down the connection (or let it be GC'd)
```

Because `RemoteAgent` *is a* `Runner`, the remote agent composes exactly like a local one — no A2A-specific wiring:

```python
from aimu.agents import Agent, Chain, OrchestratorAgent
import aimu

# As a tool an autonomous agent can call (description comes from the remote card):
local = Agent(aimu.client(), tools=[remote.as_tool()])

# As a workflow step:
pipeline = Chain(agents=[local_planner, remote])

# As an orchestrator worker:
orch = OrchestratorAgent.assemble(aimu.client(), "Delegate to the worker.", workers=[remote])
```

## Expose an agent

Wrap any `Runner` (an `Agent`, a workflow, a custom `Runner`) as an A2A server. `serve_a2a` blocks:

```python
from aimu.agents import Agent, serve_a2a
import aimu

agent = Agent(aimu.client(), system_message="You are a research assistant.", name="researcher")
serve_a2a(agent, host="127.0.0.1", port=9000)
# Agent card served at http://127.0.0.1:9000/.well-known/agent-card.json
```

To get the ASGI app without running it (to embed in an existing server, or for tests), use `build_a2a_app`, which returns a Starlette app:

```python
from aimu.agents import build_a2a_app

app = build_a2a_app(agent, url="https://my-host/")  # url is the externally reachable base URL
```

From the command line, serve a single `Agent` with no code:

```bash
python -m aimu.agents.a2a --model anthropic:claude-sonnet-4-6 \
    --system "You are a research assistant." --port 9000
```

## Async

`aimu.aio.a2a` mirrors the sync surface. The async `RemoteAgent` uses the `a2a-sdk` client natively (no thread portal) and supports incremental streaming via `message/stream`:

```python
from aimu.aio.a2a import RemoteAgent

remote = await RemoteAgent.connect("http://localhost:9000")
print(await remote.run("Summarise the news"))

async for chunk in await remote.run("Summarise the news", stream=True):
    print(chunk.phase.value, chunk.content)

await remote.aclose()
```

## Loud failures

Connection and call problems raise `A2AConnectionError` with the original cause chained:

```python
from aimu.agents import A2AConnectionError, RemoteAgent

try:
    remote = RemoteAgent.connect("http://localhost:1/")
except A2AConnectionError as exc:
    print(exc)
# A2AConnectionError: failed to connect to A2A agent at 'http://localhost:1/': ...
```

## Notes and limitations

- **Streaming is approximate.** The map from A2A SSE events to `StreamChunk` phases is lossy; the bundled server emits a single final message rather than incremental task-status updates. The sync `RemoteAgent` returns the full response as one `GENERATING` chunk; real incremental streaming is on the async surface.
- **`RemoteAgent.messages`** records the local `(user, assistant)` exchange, not the remote agent's internal transcript (A2A keeps that opaque by design).
- **SDK pin.** AIMU pins `a2a-sdk` to the `0.3.x` line (pydantic-native API); the protobuf `1.x` line is a tracked future migration.

## See also

- [Use MCP tools](use-mcp-tools.md) — the tool-level counterpart
- [Build an orchestrator](build-orchestrator.md) — `assemble(workers=...)` accepts any `Runner`, local or remote
- [Explanation: A2A vs MCP](../explanation/a2a-vs-mcp.md) — agents over the wire vs tools over the wire
- [`aimu.agents.RemoteAgent`](../reference/api/agents.md#aimu.agents.RemoteAgent) — API reference
