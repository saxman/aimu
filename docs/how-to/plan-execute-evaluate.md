# Plan, execute, evaluate, replan

`PlanExecuteEvaluator` is a code-controlled workflow that runs a **plan → execute → evaluate → replan-on-fail** loop. A planner produces a plan; an executor runs it with tools; a scorer judges the output. If the score is below `pass_threshold`, the planner is invoked again with the prior round's feedback to produce a *new* plan. The loop bounds at `max_rounds` and returns the highest-scoring attempt.

This is the right pattern when:

- The task has measurable success criteria you can express as text.
- Single-shot prompting isn't reliable enough — you want the system to recognise failure and try a different approach.
- You want to keep evaluation honest by using a separate (often stronger) judge model.

## Quick start

```python
import aimu
from aimu.agents import PlanExecuteEvaluator
from aimu.tools import builtin

wf = PlanExecuteEvaluator.from_client(
    client=aimu.client("anthropic:claude-sonnet-4-6"),
    judge_client=aimu.client("openai:gpt-4o"),               # stronger judge
    executor_tools=builtin.web + builtin.fs,
    criteria="The summary cites 3+ sources and is under 200 words.",
    max_rounds=3,
    pass_threshold=0.7,
)

result = wf.run("Summarise the latest news on quantum computing.")
print(result)
```

The factory builds a `SkillAgent` planner, an `Agent` executor with your tools, and an `LLMJudgeScorer` over your judge client.

## Two criteria modes

**User-supplied criteria** (recommended) — pass `criteria="..."` to `PlanExecuteEvaluator.from_client()` and the criteria string is included in the planner's prompt every round and used as the scorer's `reference` field on every row.

**Planner-invented criteria** — pass `criteria=None`. The planner is asked to return its response in two sections:

```
## Evaluation criteria
<one-paragraph criteria the planner thinks defines success>

## Plan
<numbered steps>
```

The workflow parses both. The criteria the planner produces is what the scorer uses for that round. This mode is more flexible for open-ended tasks but more vulnerable to drift — the planner can quietly soften its own criteria on replans. Use a separate (stronger) judge if you go this route.

## Custom scorer

The `.from_client()` factory uses `LLMJudgeScorer` by default. To plug in a different `Scorer` (DeepEval, a custom callable, etc.), construct `PlanExecuteEvaluator` directly:

```python
from aimu.agents import PlanExecuteEvaluator, Agent, SkillAgent
from aimu.evals import DeepEvalModel, DeepEvalScorer
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

geval = GEval(
    name="quality",
    criteria="Concise, accurate, properly cited.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=DeepEvalModel(aimu.client("openai:gpt-4o")),
)

wf = PlanExecuteEvaluator(
    planner=SkillAgent(client, "Plan the task carefully.", name="planner"),
    executor=Agent(client, "Execute the plan using your tools.", name="executor", tools=builtin.web),
    scorer=DeepEvalScorer([geval]),
    criteria="Concise, accurate, properly cited.",
    max_rounds=3,
)
```

## Why a SkillAgent planner?

Using `SkillAgent` for the planner lets you drop a `SKILL.md` file describing planning strategies for known task types. For example, a `code-task` skill might tell the planner *"break the task into design → implement → test → review; flag any external dependencies"*. A `research-task` skill might tell it *"identify sources first, then synthesise"*.

Skills are auto-discovered from `.agents/skills/`, `.claude/skills/`, and the `~/` variants. Pass `skill_manager=` to override:

```python
from aimu.skills import SkillManager

wf = PlanExecuteEvaluator.from_client(
    client,
    skill_manager=SkillManager(skill_dirs=["./my-planning-skills"]),
    criteria="...",
)
```

When no skills are installed, `SkillAgent` behaves like a regular `Agent`.

## Inspect what happened

After `run()` completes, the per-round attempt log is available:

```python
result = wf.run("...")

for attempt in wf.last_attempts:
    print(f"Round {attempt['round']}: score={attempt['score']:.2f}")
    print(f"  Plan: {attempt['plan'][:80]}...")
    print(f"  Output: {attempt['output'][:80]}...")
    print(f"  Feedback: {attempt['feedback']}")
```

Each attempt dict has `round`, `plan`, `criteria`, `output`, `score`, and `feedback`. Useful for debugging replans and tuning `pass_threshold`.

## Stop conditions

The loop exits when one of these is true:

- `score >= pass_threshold` for the current round (score-based pass).
- `pass_keyword` is set and the scorer's feedback contains it (e.g. `pass_keyword="APPROVED"` for scorers that emit marker tokens).
- `max_rounds` reached. On exhaustion, the workflow returns the **highest-scoring** attempt, not the most recent.

## Streaming

`run(stream=True)` yields a single `StreamChunk(GENERATING, final_output)` chunk after the loop completes. Intermediate plans and executions are not streamed — match `EvaluatorOptimizer`'s behaviour. Inspect `wf.last_attempts` for per-round visibility.

## Related

- [`EvaluatorOptimizer`](../reference/api/agents.md#aimu.agents.EvaluatorOptimizer) — simpler generate → critique → revise loop. Use when revision is enough; use `PlanExecuteEvaluator` when you want full replanning + tool execution.
- [Build an orchestrator](build-orchestrator.md) — autonomous LLM-directed dispatch to workers. Use when the *model* should pick which worker to call; use `PlanExecuteEvaluator` when the control flow is fixed (plan → execute → judge).
- [Tune prompts](tune-prompts.md) — uses the same `Scorer` protocol for prompt-level optimisation against labelled data.
