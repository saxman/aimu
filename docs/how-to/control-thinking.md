# Control thinking effort

Reasoning models let you trade latency and cost against answer quality: skip reasoning for a
trivial question, or spend more of it on a hard one. AIMU exposes this as one portable
`thinking=` argument on `chat()` and `generate()` (sync and async, plus the top-level
`aimu.chat()` / `aio.chat()` helpers), so a call site does not have to know which provider's
knob it is turning, or whether the current model has one at all.

```python
import aimu

client = aimu.client("ollama:qwen3.8:27b")

client.chat("What is 2 + 2?", thinking=False)      # skip reasoning on a trivial question
client.chat("Design a cache eviction policy.", thinking="high")

# The same code against a model with no effort control logs a warning and still answers.
gemma = aimu.client("ollama:gemma4:12b")
gemma.chat("Design a cache eviction policy.", thinking="high")
```

## The four value forms

```python
thinking=None       # default: today's behavior, byte for byte
thinking=False      # off, and select the model's instruct-mode sampling profile
thinking=True       # on, at the model's own default effort
thinking="low"      # on at low effort; also "medium" and "high"
```

`thinking=None` is the default and changes nothing: every request path is byte-for-byte what it
was before this parameter existed. `thinking=False` does two things at once: it asks the model
not to reason, and (for models whose card specifies one) it switches to the corresponding
instruct-mode sampling profile, since a model's thinking-mode and non-thinking-mode sampling
defaults are usually different. `thinking=True` turns reasoning on without pinning an effort
level, so the model keeps its own default (Qwen 3.8, for example, keeps its native `xhigh`
ceiling). A level string turns reasoning on **and** requests that specific effort.

## Swapping models: validate the argument, never the model

One rule governs resolution: **validate the argument, never the model**.

- An **invalid value** raises `ValueError` immediately, before any request is built. `thinking="xhigh"` raises rather than being silently accepted, because it is a plausible typo (Qwen's own effort ceiling is spelled exactly that way), and paying for full-effort reasoning while believing you had asked for something more modest is the wrong failure mode to allow silently.
- An **unsupported request against a specific model** logs a warning (deduplicated per client instance, so an agent loop does not repeat it every round) and the call proceeds anyway. This is what the `gemma4:12b` call above does: that model has no effort-level control, so `thinking="high"` still turns reasoning **on**, with a warning that the level itself was ignored.

The asymmetry is deliberate: this rule is what lets one call site serve a mixed fleet of models
without a model-specific branch. `thinking=False` on a model that has no reasoning to disable is
the one case that is silent rather than warned, because the statement is already true.

`thinking=False` is a no-op (warned, not applied) against **every** Gemini thinking model today,
since AIMU has no wire mechanism yet for Gemini's off-request over the OpenAI-compatible endpoint
it uses (see the table below). `GEMINI_2_5_PRO` carries an additional, stronger fact on top of
that gap: Google's own API will not let this specific model disable reasoning at all
(`thinking_optional=False`), so even a future fix to AIMU's mechanism gap would not change its
behavior. Either way, the call proceeds at full reasoning effort. If you need to confirm what
actually happened, check `client.last_usage` after the call; the reasoning tokens are billed even
though the caller asked to skip them.

## Per-provider mechanism

| Provider | off (`thinking=False`) | level (`"low"`/`"medium"`/`"high"`) |
| --- | --- | --- |
| Ollama native | `think=False` | `think="low"/"medium"/"high"` (its SDK already accepts this exact vocabulary) |
| OpenAI-compat local servers (vLLM, SGLang, LM Studio, Ollama-OpenAI, oMLX, HF-Serve, llama-server) | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | `reasoning_effort`, with `"high"` sent as Qwen's own `"xhigh"` |
| HuggingFace (in-process) | `enable_thinking=False` template kwarg | `reasoning_effort` template kwarg |
| Anthropic | omits the `thinking` request parameter | `budget_tokens`: low 2048, medium 8000, high 16000 |
| llama.cpp, OpenAI cloud, Gemini | nothing emitted | nothing emitted |

**Among the providers that share Qwen's effort vocabulary** (Ollama, the OpenAI-compatible family,
HuggingFace), only Qwen 3.8 declares effort-level support today; every other model on those
providers accepts on/off (`True`/`False`) but treats a level as advisory (see above). Anthropic is
a separate case: all six `AnthropicModel` members declare effort-level support too, mapped to
`budget_tokens` rather than this shared vocabulary (see "Anthropic: exact token budgets" below).
The last table row is a scope boundary rather than an absolute limitation for two of its
three providers: OpenAI's o-series models never declare `thinking=True` in AIMU's catalog, so
there is nothing to control on that path, and Google's OpenAI-compatible endpoint for Gemini does
in fact document a `reasoning_effort` field (its vocabulary is `minimal`/`low`/`medium`/`high`,
plus `none` to disable reasoning on `gemini-2.5-flash`). AIMU does not emit it yet because that
vocabulary does not include the `"xhigh"` value the shared Qwen mapping sends for `"high"`, and
doing this correctly needs a second, Gemini-specific effort vocabulary, which is a deferred
follow-up. `llama.cpp` has no mechanism at all on either axis.

## Anthropic: exact token budgets

The portable levels above map to `budget_tokens` 2048 / 8000 / 16000. If you need an exact
number, pass `thinking_budget_tokens` directly in `generate_kwargs`; it takes precedence over any
`thinking=` level:

```python
client = aimu.client("anthropic:claude-sonnet-4-6")
client.chat(
    "Design a cache eviction policy.",
    thinking=True,
    generate_kwargs={"thinking_budget_tokens": 4000},
)
```

This stays a separate, Anthropic-specific parameter rather than folding into `thinking=`, since
no other provider has an equivalent numeric knob to translate it to.

Anthropic's `ADAPTIVE`-style models (Opus 4.7+, Fable 5) decide for themselves whether and how
much to think, and their request shape has no budget parameter at all: `thinking_budget_tokens`
is dropped on these models whether or not you also pass a `thinking=` level. Passing a level
(`thinking="low"`, etc.) additionally logs a warning naming the level that was ignored.
`thinking=True`/`thinking=False` still work on these models, since they can be told to think or
not, just not how hard.

## See also

- [Thinking content and the model context](../explanation/thinking-and-context.md): what happens
  to reasoning after a turn, why prior-turn reasoning is never re-fed to the model, and the full
  resolution-semantics table behind the "validate the argument, never the model" rule.
- [Add a new model](add-new-model.md): the general recipe for a new `ModelSpec` catalog entry.
