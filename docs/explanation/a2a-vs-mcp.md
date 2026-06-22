# A2A vs MCP: agents over the wire, tools over the wire

AIMU speaks two cross-process protocols, and they are complementary, not competing:

| | **MCP** (Model Context Protocol) | **A2A** (Agent2Agent) |
|---|---|---|
| Exposes | **Tools**: typed function calls | **Agents**: opaque, autonomous units of work |
| Unit of work | One stateless call → one result | A task with a lifecycle (submit, work, complete) |
| Discovery | `list_tools()` | An *agent card* at `/.well-known/agent-card.json` |
| AIMU surface | `aimu.tools.MCPClient` / `python -m aimu.tools.mcp` | `aimu.agents.a2a.RemoteAgent` / `serve_a2a()` |
| The other side knows | Your function names + JSON schemas | Only that an agent exists and what it claims to do |

Use **MCP** when you want a remote process to contribute *capabilities* (a search tool, a database query) that *your* model decides how to orchestrate. Use **A2A** when the remote process *is itself an agent* (it has its own model, its own loop, its own judgment) and you want to delegate a whole task to it and get an answer back.

## Both are optional adapters, not core abstractions

A2A lives behind the `a2a` extra (`pip install 'aimu[a2a]'`) and adapts entirely at the boundary. Nothing about A2A leaks into `Runner`, `Agent`, or the workflow classes:

- **Plain data.** A2A `Message` / `Task` / `AgentCard` types exist only inside `aimu/agents/a2a/`. `runner.run()` still takes a `str` and returns a `str`. The adapters in `aimu/agents/a2a/_card.py` convert at the edge, the same role `aimu.tools.mcp_format` plays for MCP.
- **Composability through uniform interfaces.** This is the strongest fit. A remote A2A agent is wrapped as `RemoteAgent`, which *is a* `Runner`. So it drops into `Chain` / `Router` / `Parallel` / `OrchestratorAgent.assemble(workers=[...])` with no special-casing, and, via the `Runner.as_tool()` method, into any local `Agent`'s tool list:

  ```python
  from aimu.agents import Agent, Chain, RemoteAgent

  remote = RemoteAgent.connect("http://localhost:9000")

  # As a workflow step:
  pipeline = Chain(agents=[local_planner, remote])

  # As a tool an autonomous agent can call:
  orchestrator = Agent(client, tools=[remote.as_tool()])
  ```

  This is the exact analog of `MCPClient.as_tools()`, lifted from the tool level to the agent level.
- **Progressive disclosure / direct paths.** One obvious entry point per direction: `RemoteAgent.connect(url)` to consume, `serve_a2a(runner)` to expose. They sit beside the MCP equivalents on the same rung of the ladder.
- **Failures are apparent.** `A2AConnectionError` is raised at connection time (and on a failed call), with the underlying cause chained via `raise ... from exc`, mirroring `MCPConnectionError`.

## Sync and async, mirroring the MCP split

`aimu.agents.a2a.RemoteAgent` (sync) drives the async `a2a-sdk` client through an anyio blocking portal, the same mechanism `aimu.tools.MCPClient` uses, so synchronous code never has to touch `await`. `aimu.aio.a2a.RemoteAgent` (async) uses the SDK client natively (no portal), exactly as `aimu.aio.MCPClient` uses FastMCP's async `Client` directly. Real incremental streaming (`message/stream` → `StreamChunk`s) is available on the async surface; the sync surface returns the full response as a single `GENERATING` chunk.

## Known limitations / notes

- **Streaming is approximate.** The map from A2A SSE status/artifact events to AIMU `StreamChunk` phases is lossy; the bundled server emits a single final message rather than incremental task-status updates.
- **`RemoteAgent.messages`** is best-effort: it records the local `(user, assistant)` exchange, not the remote agent's internal transcript, which A2A keeps opaque by design.
- **SDK version pin.** AIMU pins `a2a-sdk` to the `0.3.x` line, whose pydantic-native API matches the broader A2A ecosystem and reference samples. The `1.x` line is a protobuf/gRPC transport rework with a still-churning surface; migrating to it is a tracked follow-up, not a blocker for interop today.
