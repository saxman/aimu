# Compare models

Finding out what a model can actually do means running it against something and looking at what
came back, then running a different model against the same thing. This guide is the loop: what a
model claims it can do, what it produces, what it cost, and how to scale that from one prompt to a
dataset.

Everything here works because a provider swap is a string change, so the difference you observe is
the models' and not your harness's.

## 1. What does this model claim it can do?

Capabilities are declared on the model spec, not discovered at runtime, so you can read them before
spending a token. Any client exposes the enum member it was built from:

```python
import aimu

client = aimu.client("ollama:qwen3.5:9b")

client.model.supports_tools        # True
client.model.supports_thinking     # True
client.model.supports_vision       # False
client.model.generation_kwargs     # the sampling profile from the model card
```

The same flags read off an enum member without constructing a client, which is what you want when
you're deciding *which* model to try:

```python
from aimu.models import OllamaModel

[m.name for m in OllamaModel if m.supports_vision]   # VISION_MODELS, derived from the flags
```

To go the other way, from a name to a model, `resolve_model_enum` accepts an enum member, a
`"provider:model_id"` string, or a bare member name:

```python
aimu.resolve_model_enum("QWEN_3_8B")                 # searches every installed provider
aimu.resolve_model_enum("anthropic:claude-sonnet-4-6")
```

When a bare name ships under several providers, AIMU picks one that's available locally and says so
at `WARNING` level rather than choosing silently. Pin the provider with a full
`"provider:model_id"` string when the choice matters, which for a comparison it usually does.

A claim AIMU doesn't ship a spec for raises rather than running with guessed capabilities. That is
deliberate: a wrong guess about tool support or a context window shows up as a confusing failure
three layers down. See [add a new model](add-new-model.md) to register one, or the
[model matrix](../reference/model-matrix.md) for the full table.

To see what you can run right now without downloading anything:

```python
aimu.available_text_models()   # running Ollama, then the HF cache, then reachable local servers
```

## 2. The same task, several models

One prompt, three backends, one string different each time:

```python
import aimu

TASK = "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. What does the ball cost?"

for model in ["ollama:qwen3.5:9b", "anthropic:claude-sonnet-4-6", "openai:gpt-4o-mini"]:
    print(f"--- {model}")
    print(aimu.chat(TASK, model=model))
```

`aimu.chat()` builds a fresh client per call, so there's no history bleeding between models. For a
multi-turn comparison, build the clients up front and drive them in lockstep:

```python
clients = {name: aimu.client(name, system="Answer in one sentence.") for name in [
    "ollama:qwen3.5:9b",
    "anthropic:claude-sonnet-4-6",
]}

for turn in ["What is the capital of Australia?", "What's its population?"]:
    for name, client in clients.items():
        print(name, "->", client.chat(turn))
```

## 3. What did each one cost?

`client.last_usage` holds the token counts for the most recent call, or `None` when the provider
didn't report them (in-process HuggingFace and llama.cpp never do). It's a plain dict:

```python
client = aimu.client("anthropic:claude-sonnet-4-6")
client.chat("Explain gradient descent in two sentences.")

client.last_usage
# {"input_tokens": 18, "output_tokens": 61, "total_tokens": 79}
```

Streaming populates it too, but only once the stream is fully consumed; reading it mid-stream gives
`None`. AIMU reports tokens and not dollars, deliberately: a price table would be stale within
weeks, and the token counts are the part that doesn't change.

## 4. What did each one actually do?

An answer string hides the interesting part. Two ways to see the work.

**Watch it happen.** A streamed run is labelled by phase, so reasoning, tool calls, and output are
separable rather than concatenated:

```python
for chunk in client.chat("How many r's in strawberry?", stream=True):
    print(f"[{chunk.phase.value}] {chunk.content}")
```

`aimu.pretty_print(stream)` renders that to the console with the phases already formatted. Filter
with `include=["generating"]` when you only want the answer. See
[stream output](stream-output.md).

**Read it afterwards.** `extract_tool_calls` reconstructs the tool activity from any message
history, so you can compare *how* two models attacked the same task, not just what they concluded:

```python
from aimu.tools import builtin

agent = aimu.agent("anthropic:claude-sonnet-4-6", tools=builtin.web)
agent.run("What was the high temperature in Seattle yesterday?")

for call in aimu.extract_tool_calls(agent.model_client.messages):
    print(call["iteration"], call["tool"], call["arguments"])
```

A model that solves the task in one tool call and one that flails through six both return an
answer. This is where you see which is which.

## 5. Steer reasoning and compare the difference

`thinking=` is portable, so you can hold the prompt fixed and vary only the reasoning effort:

```python
for effort in [False, "low", "high"]:
    reply = client.chat(TASK, thinking=effort)
    print(effort, client.last_usage["output_tokens"], reply)
```

A model that can't honour the request warns and answers anyway rather than raising, so this loop
survives being pointed at a different model. Which models can actually be steered is the ◆ column
in the [model matrix](../reference/model-matrix.md); the mechanics are in
[control thinking effort](control-thinking.md).

## 6. Scale it to a dataset

One prompt tells you an anecdote. `Benchmark` runs the same prompt across many clients over a
labelled dataset and returns a comparison table. Requires the `tuning` extra.

```python
import pandas as pd
import aimu
from aimu.evals import Benchmark
from aimu.prompts import LLMJudgeScorer

data = pd.DataFrame({"content": [
    "The mitochondria is the powerhouse of the cell.",
    "Water boils at 100 degrees Celsius at sea level.",
]})

scorer = LLMJudgeScorer(
    aimu.client("anthropic:claude-sonnet-4-6"),
    criteria="Is the explanation accurate and understandable to a 12-year-old?",
)

benchmark = Benchmark(prompt="Explain this to a 12-year-old:\n\n{content}", data=data, scorer=scorer)

results = benchmark.run({
    "qwen3.5:9b":       aimu.client("ollama:qwen3.5:9b"),
    "claude-sonnet-4-6": aimu.client("anthropic:claude-sonnet-4-6"),
})

print(results.metrics)   # rows = client names, columns = score, pass_rate
results.to_csv("comparison.csv")
```

The prompt template must contain `{content}`, and the dataset needs a `content` column and a unique
index. Because `Benchmark` drives clients through `chat()`, an **agent** compares against a plain
model on equal footing: pass `agent.as_model_client()` as one of the clients and its whole tool loop
runs per row.

```python
agent = aimu.agent("ollama:qwen3.5:9b", tools=builtin.web)
results = benchmark.run({
    "plain": aimu.client("ollama:qwen3.5:9b"),
    "agentic": agent.as_model_client(),
})
```

That is the comparison worth running: not just which model is better, but whether the scaffolding
you built around it earned its keep. See [benchmark models](benchmark-models.md) for the harness in
detail and [integrate DeepEval](integrate-deepeval.md) for metric-based scoring instead of a
free-text judge.

## When a model isn't reachable

Comparison runs hit dead servers and missing keys. AIMU raises rather than falling back silently, so
a model that didn't answer is never mistaken for a model that answered badly:

- `ModelConnectionError` when an inference server is unreachable, with the transport error chained
  as `__cause__`.
- `ValueError` naming the valid ids when a model string doesn't match a shipped spec.

If you want a comparison run to survive one provider being down, that's an explicit choice:
`FallbackClient([primary, backup])` tries each in turn and preserves history across the switch. See
[switch providers](switch-providers.md#provider-failover).

## See also

- [Switch providers](switch-providers.md): the string format, custom endpoints, failover
- [Model matrix](../reference/model-matrix.md): every shipped model with its capability flags
- [Provider matrix](../reference/provider-matrix.md): extras, env vars, defaults, and which
  generation parameters each backend honours
- [Benchmark models](benchmark-models.md): the harness, scorers, and result export
- [Stream output](stream-output.md): phases, filtering, `pretty_print`
