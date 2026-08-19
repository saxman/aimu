# Spawn sub-agents

`make_subagent_tool` gives an agent a `spawn_subagent` tool: at runtime the LLM decides to delegate an independent subtask to a **fresh sub-agent with its own isolated context**, and can spawn several to run **in parallel**. This is AIMU's answer to the dynamic sub-agent pattern (as in Claude Code's `Task` tool).

It is the *dynamic complement* to [`OrchestratorAgent`](build-orchestrator.md): an orchestrator dispatches to a **fixed roster** of workers you wire up in Python; `spawn_subagent` hands the LLM one tool and lets it decide **how many** sub-agents to spawn and with what tasks. Both reduce to `subagent.run(task)` under the hood — the difference is who decides the roster.

## Generic sub-agents

Pass a model (enum, `"provider:model_id"` string, or a client to clone from). Each spawn builds a fresh `ModelClient` — its own message history, so concurrent spawns share no state.

```python
import aimu
from aimu.agents import Agent
from aimu.tools.builtin import make_subagent_tool, web

spawn = make_subagent_tool("anthropic:claude-sonnet-4-6", tools=web)

agent = Agent(
    aimu.client("anthropic:claude-sonnet-4-6"),
    "Break the request into independent subtasks and spawn a sub-agent for each, then synthesize.",
    tools=[spawn],
    concurrent_tool_calls=True,   # see "Run them in parallel" below
)
print(agent.run("Compare GDP growth of France, Japan, and Brazil since 2019."))
```

The tool the model sees is `spawn_subagent(task: str) -> str`. Give each sub-agent a complete, self-contained task — it has no view of the parent conversation.

## Typed sub-agents (a registry of specialists)

Pass `agent_types` to switch the tool to `spawn_subagent(agent_type, task)`, mirroring Claude Code's `subagent_type` + `prompt`. Each entry is a dict with `system_message` and optional `tools` / `model` / `thinking` (see [Control thinking effort](control-thinking.md)) — those four keys and no others. The available type names are listed in the tool description, so the model knows the menu.

```python
spawn = make_subagent_tool(
    "anthropic:claude-sonnet-4-6",
    agent_types={
        "researcher": {"system_message": "Research the topic and cite sources.", "tools": web},
        "writer":     {"system_message": "Write clear, concise prose.", "tools": []},
    },
)
agent = Agent(client, "Delegate to researcher then writer.", tools=[spawn], concurrent_tool_calls=True)
```

An unknown `agent_type` is returned to the model as a tool result (so it self-corrects), not raised. Programmer errors — `max_depth < 1`, an empty `agent_types`, a type missing `system_message`, or a spec carrying a key outside the four above — raise `ValueError` at `make_subagent_tool(...)` call time.

The spec key set is closed on purpose, because an ignored key reads exactly like an applied one: a misspelled `thinkng` or a hopeful `temperature` would leave the spawned agent at its default with nothing raised anywhere. Note the split in who is expected to recover: a bad *spec key* is the programmer's mistake and raises, while a bad *`agent_type`* is the model's and comes back as a tool result it can retry from.

## Run them in parallel

Parallelism is free and rides the existing `concurrent_tool_calls` machinery: give the **parent** `Agent` `concurrent_tool_calls=True`, and when the model emits several `spawn_subagent` calls in one turn they run concurrently (`ThreadPoolExecutor` on the sync surface, `asyncio.TaskGroup` on `aio`).

Caveats:

- Genuine overlap is for **cloud** models (each sub-agent makes independent network requests). A single **local** model serializes on the GIL/CUDA, and concurrent in-process client construction touches the shared weight cache — warm it once or spawn serially for HuggingFace/LlamaCpp.
- Provider rate limits are the practical ceiling (same as `Parallel` / `ResearchReportAgent`).
- A `deps` object shared across concurrent spawns must be thread-safe/async-safe.

## Bound recursion with `max_depth`

`max_depth` (default `1`) counts the caller's agent as level 1, so by default spawned sub-agents get **no** spawn tool of their own — a safe anti-fan-out default. Raise it to let sub-agents spawn their own:

```python
spawn = make_subagent_tool(model, max_depth=2)   # sub-agents may spawn one more level, no deeper
```

The nested tool is rebuilt with a decremented depth, so recursion always terminates.

## Gate a sub-agent's tools

`tool_approval` forwards an approval gate to every spawned sub-agent (and, with `max_depth > 1`, to the sub-agents they spawn). It is the same hook `Agent` accepts — a callable `(name, arguments) -> bool` (may be a coroutine) run before each tool call; returning `False` skips the call and feeds the model a refusal. Use it so a delegated sub-agent does not run a sensitive tool unattended:

```python
from aimu.tools.builtin import make_subagent_tool, web, compute

def approve(name, arguments):
    return name != "execute_python"   # let sub-agents look things up, but not run code unattended

spawn = make_subagent_tool("anthropic:claude-sonnet-4-6", tools=[*web, *compute], tool_approval=approve)
```

Without `tool_approval` a sub-agent runs its tools unattended (the default `approve_all`). See [Gate tool calls](gate-tool-calls.md) for the gate contract.

## Async

`aimu.aio.tools.builtin.make_async_subagent_tool` has the identical surface and produces an `async def` tool that builds an `aio.Agent` per spawn:

```python
from aimu import aio
from aimu.aio.tools.builtin import make_async_subagent_tool

spawn = make_async_subagent_tool("anthropic:claude-sonnet-4-6")
agent = aio.Agent(aio.client("anthropic:claude-sonnet-4-6"), tools=[spawn], concurrent_tool_calls=True)
print(await agent.run("Fan out three independent lookups and combine them."))
```

In-process providers (HuggingFace, LlamaCpp) can't be constructed from an enum on the async surface, so each spawn wraps a **fresh** sync client (fresh preserves isolation; the process weight cache prevents reloading weights).

## Watching a sub-agent work

`make_async_subagent_tool` accepts an `observer` that gets a live report of what each spawn is doing, without turning `spawn_subagent` itself into a streaming tool (so `concurrent_tool_calls` still lets several spawns overlap). Implement the three methods of `SubagentObserver`:

```python
from aimu.aio.tools.builtin import make_async_subagent_tool


class PrintingObserver:  # structurally satisfies SubagentObserver; no inheritance required
    async def spawned(self, spawn_id, agent_type, task):
        print(f"[{spawn_id}] started ({agent_type}): {task}")

    async def chunk(self, spawn_id, chunk):
        print(f"[{spawn_id}] {chunk.phase}: {chunk.content}")

    async def finished(self, spawn_id, result, error):
        print(f"[{spawn_id}] done: {error or result}")


spawn = make_async_subagent_tool("anthropic:claude-sonnet-4-6", observer=PrintingObserver())
```

`spawn_id` is stable across all three callbacks for one spawn and unique per spawn, so a front end can key a display panel per sub-agent even when several run concurrently. `finished` always fires, including on cancellation, with whatever text had accumulated and the exception (if any) that ended the run. An observer is display-only: if a callback raises, the failure is logged and swallowed rather than aborting the spawn it was reporting on, and that holds for a callback whose signature no longer matches as much as for one that raises in its body. Nested spawns (`max_depth > 1`) inherit the same observer automatically.

Attaching an observer is not purely additive, though: an observed spawn runs the child through the provider's **streaming** request path, where an unobserved spawn uses the non-streaming one. The returned answer is the same either way (the accumulator resets on each loop iteration, so it is the final answer rather than every intermediate response concatenated), but any provider behavior that differs between its two request paths applies to observed spawns.

## When to use this vs `OrchestratorAgent`

- **`OrchestratorAgent`** — you know the specialist roster up front and want named, hand-wired workers (possibly workflows or remote agents). See [Build an orchestrator](build-orchestrator.md).
- **`make_subagent_tool`** — the model should decide the fan-out at runtime: how many sub-agents, what each one does. Typed mode still lets you constrain *which kinds* of specialist it may spawn.

Streaming a spawned sub-agent's tokens through the parent stream is intentionally **not** supported: it would disable the concurrent-dispatch path and interleave concurrent sub-agents' output. `spawn_subagent` returns the sub-agent's final answer; use `observer=` (async only, see above) to watch a spawn's progress without that interleaving.

## See also

- [Build an orchestrator](build-orchestrator.md): the static-roster counterpart
- The `--method spawn` variant in [`examples/news-summarizer`](https://github.com/saxman/aimu/blob/main/examples/news-summarizer/news_summarizer.py)
- [`aimu.tools.builtin.make_subagent_tool`](../reference/api/tools.md)
