# Agents vs workflows

AIMU follows the taxonomy from Anthropic's *[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)*. The split — autonomous agents vs code-controlled workflows — is the most important conceptual divide in the library, and worth understanding before you build anything non-trivial.

## The distinction

**Agents** are autonomous: the LLM directs the process. You give it tools and a goal; the LLM decides which tools to call, in what order, and when to stop. The control flow lives in the model's outputs.

**Workflows** are code-controlled: you direct the process. The LLM is invoked at fixed points; routing, sequencing, and aggregation happen in Python. The control flow lives in your code.

Both have a place. Picking the wrong one is the most common source of frustration when building LLM apps.

## When to pick which

| You want… | Pick |
|---|---|
| An open-ended task with unknown number of steps | **`Agent`** |
| A fixed pipeline: extract → analyse → summarise | **`Chain`** (workflow) |
| Classify input, dispatch to a specialist | **`Router`** (workflow) |
| Run independent perspectives in parallel | **`Parallel`** (workflow) |
| Iterate generate → critique → revise | **`EvaluatorOptimizer`** (workflow) |
| Coordinate sub-agents via tool calls | **`OrchestratorAgent`** (agent dispatching to other agents) |

Rules of thumb:

- **If you can write the steps in code with `if/else`, use a workflow.** Code is more debuggable, predictable, cheaper to run, and easier to test.
- **If the steps depend on the model's understanding of the task, use an agent.** Open-ended research, multi-step problem solving, tasks where the next step depends on what was just learned.
- **You can mix.** A `Router` can dispatch to an `Agent`. A `Chain` can have an `OrchestratorAgent` as a step. The interfaces are the same.

## Why the split matters

The taxonomy isn't just nomenclature. It's a question about *who controls the loop*.

Autonomous agents are powerful but expensive: every step the model decides is a round-trip, with full message history. They're also harder to reason about — when something goes wrong, you have to trace through what the model decided, not what your code did.

Workflows are constrained but predictable. Each step has a known purpose. Failures are localised. Latency is bounded by the number of steps in the code, not by however many tool rounds the model wants.

For most production systems, workflows do more of the work than people expect. Autonomous agents are best at the *leaves* — at points where the task genuinely is "use these tools to figure out what to do next".

## In AIMU

Both branches share the `Runner` interface (`run()`, `messages`, `run_streamed()`), so they compose freely:

```python
# Workflow dispatching to an autonomous agent
router = Router.of(
    client,
    classifier_prompt="...",
    handlers={
        "research": Agent(client, "Research thoroughly.", tools=builtin.web),  # autonomous
        "math":     Chain.of(client, ["...", "..."]),                          # workflow
    },
)
```

The `messages` property merges sub-agent message dicts recursively, so you can inspect the full nested execution regardless of which branch a turn went through.

## The orchestrator pattern

`OrchestratorAgent` is a third pattern that sits between agents and workflows: an *autonomous* agent whose tools are other agents. The orchestrator LLM decides which worker agents to call (and in what order), but each worker runs its own agentic loop.

This is most useful when the workers themselves benefit from tool use. A research orchestrator might have three workers (overview, examples, counterpoints), each of which can call web-search tools. The orchestrator decides which workers to dispatch — the workers decide how to do their work.

Use `OrchestratorAgent.assemble(client, system_message, workers=[...])` for the simple case (each worker is auto-wrapped as a `@tool`), or subclass and call `self._init_orchestrator(...)` when you need custom tool signatures.

## Further reading

- Anthropic's [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — the original taxonomy.
- [Architecture](architecture.md) — how `Runner` / `BaseAgent` / `Workflow` are arranged in the codebase.
- [How-to: build an orchestrator](../how-to/build-orchestrator.md) — practical patterns.
