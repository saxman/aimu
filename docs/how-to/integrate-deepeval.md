# Integrate DeepEval

`aimu[deepeval]` ships two adapters:

- **`DeepEvalModel`**: wraps any AIMU `BaseModelClient` as a DeepEval judge (`DeepEvalBaseLLM`). Pass it as `model=` to any DeepEval metric.
- **`DeepEvalScorer`**: wraps a list of DeepEval metrics as an AIMU `Scorer` for `JudgedPromptTuner` and `Benchmark`.

## Install

```bash
pip install aimu[deepeval]
```

## Use any AIMU client as a DeepEval judge

```python
import aimu
from aimu.evals import DeepEvalModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

judge = DeepEvalModel(aimu.client("ollama:qwen3.5:9b"))

metric = GEval(
    name="Correctness",
    criteria="Is the actual output factually correct and directly responsive to the question?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.5,
)

case = LLMTestCase(input="What is the capital of France?", actual_output="Paris.")
metric.measure(case)
print(metric.score, metric.reason)
```

Swap in a stronger cloud judge for harder evaluations:

```python
judge = DeepEvalModel(aimu.client("anthropic:claude-sonnet-4-6"))
```

Works with any metric (`GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, etc.) and any AIMU client (Ollama, HuggingFace, OpenAI, Anthropic, Gemini, OpenAI-compatible servers).

## Drive prompt tuning with DeepEval metrics

`DeepEvalScorer` adapts one or more DeepEval metrics to AIMU's `Scorer` protocol so they plug into `JudgedPromptTuner`:

```python
from aimu.prompts import JudgedPromptTuner
from aimu.evals import DeepEvalModel, DeepEvalScorer

writer = aimu.client("ollama:qwen3.5:9b")
judge  = aimu.client("anthropic:claude-sonnet-4-6")

geval = GEval(
    name="quality",
    criteria="Concise, accurate, plain English.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=DeepEvalModel(judge),
)

tuner = JudgedPromptTuner(model_client=writer, scorer=DeepEvalScorer([geval]))
best = tuner.tune(df, initial_prompt="Answer in one sentence: {content}")
```

For multiple metrics, pass them all; `DeepEvalScorer` averages the per-metric scores:

```python
scorer = DeepEvalScorer([geval, AnswerRelevancyMetric(model=DeepEvalModel(judge))])
```

## Use DeepEval metrics in benchmarks

The same `DeepEvalScorer` slots into `Benchmark`:

```python
from aimu.evals import Benchmark, DeepEvalScorer

bench = Benchmark(prompt=prompt, data=df, scorer=DeepEvalScorer([geval]))
results = bench.run({"qwen": aimu.client("ollama:qwen3.5:9b")})
```

## Structured output

`DeepEvalModel.generate(prompt, schema=PydanticModel)` parses the response as JSON and validates against the Pydantic schema. AIMU tries raw JSON first, then fenced code blocks, then `{...}` substrings:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    response: str
    confidence: float

judge = DeepEvalModel(aimu.client("anthropic:claude-sonnet-4-6"))
parsed: Answer = judge.generate("Reply as JSON with response and confidence.", schema=Answer)
```

DeepEval calls this transparently when a metric requires structured output.

## How `reference` columns are used

For benchmark data with a `reference` column, `DeepEvalScorer` populates `LLMTestCase.expected_output`. Metrics that use the expected output (e.g. `GEval` with `EXPECTED_OUTPUT` in `evaluation_params`) get it automatically.

## See also

- [`aimu.evals.DeepEvalModel`](../reference/api/evals.md#aimu.evals.DeepEvalModel) / [`DeepEvalScorer`](../reference/api/evals.md#aimu.evals.DeepEvalScorer)
- [Benchmark models](benchmark-models.md)
- [Tune prompts](tune-prompts.md): full tuning workflow
- Notebook [16 - Evaluations](https://github.com/saxman/aimu/blob/main/notebooks/16%20-%20Evaluations.ipynb)
