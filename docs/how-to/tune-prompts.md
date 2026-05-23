# Tune prompts

`PromptTuner` runs a hill-climbing loop to automatically improve a prompt against labelled data. Each round: apply the current prompt to the data, evaluate, propose a mutation, keep it if it improves the score, otherwise revert. No ML machinery — just a model client and a scorer.

Four concrete tuners ship in `aimu.prompts`:

| Tuner | Task | Data columns |
|---|---|---|
| `ClassificationPromptTuner` | Binary YES / NO | `content`, `actual_class` (bool) |
| `MultiClassPromptTuner` | N-way `[ClassName]` | `content`, `actual_class` (str) |
| `ExtractionPromptTuner` | JSON field extraction | `content`, `expected` (dict) |
| `JudgedPromptTuner` | Open-ended generation, scored per-row | `content`, optionally `reference` |

## Binary classification

```python
import pandas as pd
import aimu
from aimu.prompts import ClassificationPromptTuner

client = aimu.client("ollama:qwen3.5:9b")

df = pd.DataFrame({
    "content": [
        "LLMs are transforming AI.",
        "The recipe calls for flour and butter.",
        "GPT-4 outperforms earlier models on benchmarks.",
        ...
    ],
    "actual_class": [True, False, True, ...],
})

tuner = ClassificationPromptTuner(model_client=client)
best_prompt = tuner.tune(
    df,
    initial_prompt="Is this about AI? Reply [YES] or [NO]. Content: {content}",
    max_iterations=15,
)
```

The prompt template must contain `{content}`. The model output must include `[YES]` or `[NO]`.

## Multi-class classification

```python
from aimu.prompts import MultiClassPromptTuner

classes = ["bug", "feature-request", "question"]
tuner = MultiClassPromptTuner(model_client=client, classes=classes)
df = pd.DataFrame({
    "content": ["Login fails on Safari.", "Add dark mode please.", "How do I export CSV?"],
    "actual_class": ["bug", "feature-request", "question"],
})
best = tuner.tune(df, initial_prompt="Classify: {content}. Reply with [bug], [feature-request], or [question].")
```

Output must contain a `[ClassName]` token matching one of the registered classes.

## JSON extraction

```python
from aimu.prompts import ExtractionPromptTuner

tuner = ExtractionPromptTuner(model_client=client, fields=["name", "city", "role"])
df = pd.DataFrame({
    "content": ["Alice is a developer in Berlin.", "Bob, the designer, lives in NYC."],
    "expected": [
        {"name": "Alice", "city": "Berlin", "role": "developer"},
        {"name": "Bob",   "city": "NYC",    "role": "designer"},
    ],
})
best = tuner.tune(df, initial_prompt='Extract name, city, role as JSON: {"name": "...", "city": "...", "role": "..."}. Content: {content}')
```

The model must emit parseable JSON (raw, fenced, or `{...}` substring).

## Judged generation

For open-ended tasks where there's no labelled "correct" answer, plug in a `Scorer`:

```python
from aimu.prompts import JudgedPromptTuner, LLMJudgeScorer

writer = aimu.client("ollama:qwen3.5:9b")
judge  = aimu.client("anthropic:claude-sonnet-4-6")

tuner = JudgedPromptTuner(
    model_client=writer,
    scorer=LLMJudgeScorer(judge, criteria="Concise, accurate, plain English."),
)
df = pd.DataFrame({"content": ["Summarise the water cycle.", "Explain photosynthesis briefly."]})
best = tuner.tune(df, initial_prompt="Answer in one sentence: {content}")
```

Or use a `DeepEvalScorer` to score with DeepEval metrics — see [Integrate DeepEval](integrate-deepeval.md).

## Save versions to a catalog

Pass `catalog=` and `prompt_name=` to persist every improvement:

```python
from aimu.prompts import PromptCatalog

with PromptCatalog("prompts.db") as catalog:
    best = tuner.tune(
        df,
        initial_prompt="...",
        catalog=catalog,
        prompt_name="ai-classifier",
    )

    # Later
    latest = catalog.retrieve_last("ai-classifier", catalog.retrieve_model_ids()[0])
    print(f"v{latest.version}: {latest.prompt}")
```

Each improvement appends a new version keyed by `(name, model_id)`.

## Custom tuners

Subclass `PromptTuner` and implement `apply_prompt`, `evaluate`, `mutation_prompt` for any task type. Optionally override `score(metrics)` to optimise a non-default metric.

## See also

- [`aimu.prompts` API reference](../reference/api/prompts.md)
- [Benchmark models](benchmark-models.md) — compare clients on a fixed prompt
- Notebook [05 - Prompt Tuning](https://github.com/saxman/aimu/blob/main/notebooks/05%20-%20Prompt%20Tuning.ipynb)
