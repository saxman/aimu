# Using LLMs inside tools, and checkpointing long-running experiments

## The history pollution problem

`BaseModelClient` accumulates conversation history in `self.messages`. When a `@tool`
function needs to make its own LLM call, sharing the agent's client would:

1. Give the tool call the agent's full conversation as context (usually wrong).
2. Add the tool's messages to the agent's history (pollutes the agent's state).

**The correct solution**: use `client.generate()` for stateless calls from within tools.
`generate()` does not touch `self.messages` — it builds a one-shot request and discards it.

```python
eval_client = aimu.client("ollama:qwen3:8b", system="You are a concise evaluator.")

@tool
def evaluate_result(text: str) -> str:
    """Score a result on a scale of 1-10."""
    # generate() is stateless — no history pollution, no second client needed
    return eval_client.generate(f"Score this 1-10: {text}")
```

**Warning: do not create multiple HuggingFace or LlamaCpp client instances for the same model.**

As of v0.5.3, AIMU caches weights automatically (see below), so accidental double-loading
is prevented. Before v0.5.3, each instance loaded weights independently — doubling VRAM for
every additional client.

Cloud providers (Anthropic, OpenAI, Gemini, Ollama) make stateless API calls; multiple
instances are fine for those.

## HuggingFace and LlamaCpp weight caching

All four HuggingFace modality clients share a module-level registry keyed on the model id
and construction kwargs. A second instance for the same model reuses the loaded weights:

```python
import aimu

c1 = aimu.client("hf:Qwen/Qwen3-8B")
c2 = aimu.client("hf:Qwen/Qwen3-8B")

assert c1._hf_model is c2._hf_model  # same object — no double load
```

Weights remain in the registry for the process lifetime. To free VRAM:

```python
aimu.clear_hf_cache()              # clear all HuggingFace weights
aimu.clear_hf_cache(model=HuggingFaceModel.QWEN_3_8B)  # one model only
aimu.clear_llamacpp_cache()        # same for LlamaCpp
```

Different `model_kwargs` (e.g. `device_map="cuda:0"` vs `device_map="cuda:1"`) produce
separate cache entries and load weights independently.

## Checkpointing long-running experiments

For long agent runs (many iterations, hours of processing), save the live message state
periodically so a failed run can resume rather than restart from zero.

### Saving state

The live partial state during or after a failed run is on `agent.model_client.messages`
(the live list), **not** `agent.messages` (the post-run snapshot, updated only on
successful completion).

```python
import json
import aimu

agent = aimu.agent("anthropic:claude-sonnet-4-6", tools=[...])

try:
    result = agent.run("Begin the experiment")
except Exception:
    # Save partial state on failure
    with open("checkpoint.json", "w") as f:
        json.dump(agent.model_client.messages, f)
    raise
```

For multi-iteration loops, save after each completed iteration:

```python
for i in range(max_iterations):
    result = agent.run(f"Iteration {i}: refine the result")
    with open("checkpoint.json", "w") as f:
        json.dump(agent.model_client.messages, f)  # overwrite each round
    if done(result):
        break
```

### Restoring and resuming

`agent.restore(messages)` handles the one non-obvious issue: after the first `chat()`,
the system message is prepended into `model_client.messages`. A naive restore would
prepend it a second time. `restore()` calls `reset()` and strips any leading system
message from the saved list before restoring.

```python
with open("checkpoint.json") as f:
    saved = json.load(f)

agent.restore(saved)
result = agent.run("Continue from where you left off")
```

### EvaluatorOptimizer and Chain

`EvaluatorOptimizer.restore(messages)` restores the generator; the evaluator starts
fresh on the next round.

`Chain.restore(messages, step=0)` restores the specified step's client (default step 0).
Steps after the restored one start fresh.
