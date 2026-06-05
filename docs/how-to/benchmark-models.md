# Benchmark models

`Benchmark` runs one prompt over a dataset across multiple `BaseModelClient` instances and aggregates per-row scores into a comparison DataFrame. Rows are driven through `chat()` with `client.reset()` between rows, so the harness handles plain `ModelClient` and `Agent.as_model_client()` views uniformly.

## Basic comparison

```python
import aimu
import pandas as pd
from aimu.evals import Benchmark
from aimu.prompts.tuners.scorers import LLMJudgeScorer

df = pd.DataFrame({
    "content": [
        "Summarise the water cycle.",
        "Explain photosynthesis briefly.",
        "What is the Doppler effect?",
    ]
})
prompt = "Answer in one sentence: {content}"

judge = aimu.client("ollama:qwen3.5:9b")
scorer = LLMJudgeScorer(judge, criteria="Concise, accurate, plain English.")

bench = Benchmark(prompt=prompt, data=df, scorer=scorer, pass_threshold=0.7)
results = bench.run({
    "qwen3.5-9b": aimu.client("ollama:qwen3.5:9b"),
    "gemma4-e4b": aimu.client("ollama:gemma4:e4b"),
})

print(results.metrics)
#               score  pass_rate
# qwen3.5-9b    0.82       1.00
# gemma4-e4b    0.74       0.67
```

## Compare with an agentic client

`Agent.as_model_client()` lets an agentic loop slot into the benchmark next to plain clients:

```python
from aimu.agents import Agent

results = bench.run({
    "plain":   aimu.client("ollama:qwen3.5:9b"),
    "agentic": Agent(aimu.client("ollama:qwen3.5:9b")).as_model_client(),
})
```

## Reference column for scorers that use it

If your dataset has a `reference` column, it's forwarded to the scorer (and surfaces as `expected_output` in DeepEval metrics):

```python
df = pd.DataFrame({
    "content":   ["What is the capital of France?", "What is 2+2?"],
    "reference": ["Paris", "4"],
})
```

## Export results

```python
results.to_csv("bench.csv")
results.to_json("bench.json")

# Persist per-client metrics into a PromptCatalog (auto-versioned)
from aimu.prompts import PromptCatalog
with PromptCatalog("bench.db") as catalog:
    results.to_catalog(catalog, prompt_name="summary-bench")
```

Each client becomes one `Prompt(name=prompt_name, model_id=client_name, ...)` row. Re-running the benchmark with the same `prompt_name` appends new versions.

## Use DeepEval metrics as the scorer

Swap `LLMJudgeScorer` for `DeepEvalScorer([metric, ...])`. See [Integrate DeepEval](integrate-deepeval.md) for the full setup.

## How it works internally

For each client, the harness:

1. Snapshots `client.messages` so the run is reversible.
2. For every row: calls `client.reset()` (clears history, preserves `system_message`), then `client.chat(prompt.format(content=row.content))`.
3. Scores the output via `scorer.score(row_with_output)`.
4. Restores the original `messages` in a `try/finally` so a failure mid-client doesn't leak state.

Rows are sequential within a client; clients are sequential across runs. Concurrent execution is deferred.

## See also

- [`aimu.evals.Benchmark`](../reference/api/evals.md#aimu.evals.Benchmark)
- [Integrate DeepEval](integrate-deepeval.md)
- Notebook [13 - Benchmarking](https://github.com/saxman/aimu/blob/main/notebooks/13%20-%20Benchmarking.ipynb)
