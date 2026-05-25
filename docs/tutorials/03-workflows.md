# Workflows

In ~15 minutes you'll build the three load-bearing code-controlled patterns: **Chain**, **Router**, and **Parallel**. Each has a one-line `.from_client(client, ...)` factory.

If you've done [First agent with tools](02-first-agent-with-tools.md), you've already used the *autonomous* path — the LLM decides what tools to call. Workflows are the opposite: the LLM is invoked at points *you* control. You pick which is right for a given problem.

## When to pick a workflow over an agent

| Use case | Pick |
|---|---|
| Open-ended task, unknown number of steps | **`Agent`** (autonomous) |
| Fixed pipeline: step 1 → step 2 → step 3 | **`Chain`** |
| Classify the input, dispatch to a specialist | **`Router`** |
| Run several independent perspectives in parallel | **`Parallel`** |
| Iterate generate → critique → revise until acceptable | **`EvaluatorOptimizer`** |

This taxonomy is from *[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)*. See [explanation: agents vs workflows](../explanation/agents-vs-workflows.md) for the underlying argument.

## Setup

```python
import aimu

client = aimu.client("ollama:qwen3.5:9b")
```

## 1. Chain — sequential pipeline

A `Chain` runs agents in order; each step's text output becomes the next step's input.

```python
from aimu.workflows import Chain

chain = Chain.from_client(client, [
    "Break the user's task into 3 concrete steps. Output a numbered list.",
    "Execute each step. Output a result for each.",
    "Polish the result into a single paragraph for a non-technical audience.",
])

result = chain.run("Research the top 3 Python web frameworks.")
print(result)
```

`Chain.from_client(client, [prompt1, prompt2, ...])` builds three `Agent` instances sharing the same client, each with `reset_messages_on_run=True` so steps don't see each other's history.

## 2. Router — classify and dispatch

A `Router` runs a classifier first, then dispatches to the matching handler.

```python
from aimu.agents import Agent
from aimu.workflows import Router

router = Router.from_client(
    client,
    classifier_prompt="Classify as code, writing, or math. Reply with only the category name.",
    handlers={
        "code":    Agent(client, "You are a coder.",         name="coder"),
        "writing": Agent(client, "You are a writer.",        name="writer"),
        "math":    Agent(client, "You are a mathematician.", name="math"),
    },
)

print(router.run("Write a haiku about recursion."))   # → routed to writer
print(router.run("Factorise 42."))                    # → routed to math
```

Route matching is case-insensitive, whitespace-stripped. Pass `fallback=Agent(...)` to handle unmatched routes.

Handlers can be any `Runner` — agents *or* nested workflows. A router can dispatch to another router, a chain, or a parallel.

## 3. Parallel — concurrent workers

A `Parallel` runs workers concurrently via `ThreadPoolExecutor` and (optionally) feeds the joined output to an aggregator.

```python
from aimu.workflows import Parallel

parallel = Parallel.from_client(
    client,
    worker_prompts=[
        "Analyse this code for security issues.",
        "Analyse this code for performance issues.",
        "Analyse this code for readability issues.",
    ],
    aggregator_prompt=(
        "Synthesise the three perspectives into one prioritised review. "
        "Order by severity."
    ),
)

print(parallel.run(your_code_snippet))
```

Without `aggregator_prompt=`, worker outputs are joined by `separator` (default `\n\n---\n\n`) and returned directly.

## 4. EvaluatorOptimizer — iterate with critique

The fourth pattern is generate → critique → revise. There's no `.from_client()` factory — build it directly:

```python
from aimu.agents import Agent
from aimu.workflows import EvaluatorOptimizer

eo = EvaluatorOptimizer(
    generator=Agent(
        client,
        "Write a clear, accurate explanation.",
        name="writer",
    ),
    evaluator=Agent(
        client,
        (
            "Review the response. If it's clear and accurate, reply with exactly: PASS\n"
            "Otherwise reply with: REVISE: <specific feedback>"
        ),
        name="critic",
    ),
    max_rounds=4,
)

print(eo.run("Explain gradient descent."))
```

The loop stops when the evaluator's response contains the `pass_keyword` (default `"PASS"`) or `max_rounds` is reached.

## Composing patterns

Every workflow accepts any `Runner` as a sub-component. Compose freely:

```python
# Route between a parallel (for opinions) and a chain (for factual research)
composed = Router.from_client(
    client,
    classifier_prompt="Classify as opinion or factual. Reply only the category.",
    handlers={
        "opinion":  Parallel.from_client(client, worker_prompts=[...], aggregator_prompt="..."),
        "factual":  Chain.from_client(client, [...]),
    },
)
```

The `messages` property merges sub-agent message dicts recursively, so you can still inspect everything.

## What's next

You've now seen all four workflow patterns plus the autonomous `Agent`. Pick the right one per task; mix freely.

- **[Vision and streaming](04-vision-and-streaming.md)** — last tutorial.
- **[How-to: build an orchestrator](../how-to/build-orchestrator.md)** — when you want an LLM to dispatch to other agents *autonomously* via tools (a different pattern from `Router`, which dispatches in code).
- **[Explanation: agents vs workflows](../explanation/agents-vs-workflows.md)** — the design argument.
- Notebook [10 - Agent Workflows](https://github.com/saxman/aimu/blob/main/notebooks/10%20-%20Agent%20Workflows.ipynb) — interactive versions of every example here.
