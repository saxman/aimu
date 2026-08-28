# Control thinking effort

Reasoning models let you trade latency and cost against answer quality: skip reasoning for a
trivial question, or spend more of it on a hard one. AIMU exposes this as one portable
`thinking=` argument on `chat()` and `generate()` (sync and async, plus the top-level
`aimu.chat()` / `aio.chat()` helpers) and on `Agent`, so a call site does not have to know which
provider's knob it is turning, or whether the current model has one at all.

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
| Anthropic | `{"type": "disabled"}` on the adaptive models (Opus 4.7+, Sonnet 5), omits the parameter on the rest | `budget_tokens`: low 2048, medium 8000, high 16000 |
| llama.cpp, OpenAI cloud, Gemini | nothing emitted | nothing emitted |

**Among the providers that share Qwen's effort vocabulary** (Ollama, the OpenAI-compatible family,
HuggingFace), only Qwen 3.8 declares effort-level support today; every other model on those
providers accepts on/off (`True`/`False`) but treats a level as advisory (see above). Anthropic is
a separate case: every `AnthropicModel` member declares effort-level support too, mapped to
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

Anthropic's `ADAPTIVE`-style models (Opus 4.7+, Sonnet 5, Fable 5) decide for themselves whether
and how much to think, and their request shape has no budget parameter at all:
`thinking_budget_tokens` is dropped on these models whether or not you also pass a `thinking=`
level. Passing a level (`thinking="low"`, etc.) additionally logs a warning naming the level that
was ignored. `thinking=True`/`thinking=False` still work on these models, since they can be told
to think or not, just not how hard.

Turning thinking off is where the two styles differ on the wire. On the `ENABLED`-style models
(Opus 4.6, Sonnet 4.6, Haiku 4.5) an absent `thinking` parameter *is* off, so `thinking=False`
sends nothing. Opus 5 and Sonnet 5 reason by default when the parameter is absent, so AIMU sends
an explicit `{"type": "disabled"}` there instead. `CLAUDE_FABLE_5` is the one member that cannot
be turned off at all (it declares `thinking_optional=False`): `thinking=False` warns and the call
proceeds with reasoning on, as it does for any model that cannot honour the request.

The adaptive models also reject `temperature`, `top_p`, and `top_k` outright, thinking or not, so
those keys are dropped from every request to them -- including the `temperature` the Anthropic
client's own defaults supply.

## On an agent

`Agent` (sync and async) takes the same argument, as a standing field and as a per-run override:

```python
from aimu.agents import Agent

# A field: every run of this agent reasons at high effort.
agent = Agent(client, "You are a careful analyst.", tools=[...], thinking="high")

agent.run("Audit this quarter's numbers.")            # high, from the field
agent.run("What is the file called again?", thinking=False)   # off, for this run only
```

`None` means "use the field", so `thinking=False` on a run is a real override rather than an
absent one. The effective value is applied to **every** model turn the run makes, not just the
first: each tool round, the continuation nudge, and the forced tools-disabled wrap-up all carry
it, so effort is uniform across a run instead of decaying after the opening turn. It reaches
the `schema=` structured-output turn too.

Because the agent forwards the *argument* rather than a pre-resolved request, validation and the
warn-once behavior stay where the model is known: the model client's own `chat()`. An agent whose
model has no effort control warns once for the whole run, not once per round.

The workflow classes (`Chain`, `Router`, `Parallel`, ...) take no `thinking=` argument, for the
same reason they take no `tools=`: they compose sub-runners and have no single model turn to
apply it to. Configure it on the `Agent`s they wrap, or on the client itself.

## On a spawned sub-agent

`make_subagent_tool()` / `make_async_subagent_tool()` in typed mode read a `"thinking"` key from
each `agent_types` spec, alongside `"system_message"`, `"tools"`, and `"model"`, and set it as the
spawned agent's field. This is the only route to a sub-agent's effort, since the spawn happens
inside the tool rather than at a call site you control:

```python
spawn = make_subagent_tool(
    "ollama:qwen3.8:27b",
    agent_types={
        "researcher": {"system_message": "Research thoroughly.", "thinking": "high"},
        "formatter":  {"system_message": "Reformat text.", "thinking": False},
        "generalist": {"system_message": "Handle the task."},   # field stays None
    },
)
```

The key is read with `.get()`, so `False` is carried rather than swallowed. Note the asymmetry
with `"model"`: an omitted `"model"` falls back to the model the factory was built with, while an
omitted `"thinking"` leaves the spawned agent at `None` — the factory has no thinking tier to fall
back to. A caller wanting one default across every spec should write the resolved value into each
spec rather than expecting inheritance.

A spec may also carry `"generate_kwargs"`, a dict assigned to the spawned client's
`default_generate_kwargs`, so a roster can pair one specialist with a cold temperature and another
with a long context window. The same asymmetry with `"model"` applies here too: an omitted
`"generate_kwargs"` leaves the spawned client's defaults empty rather than inheriting anything,
since this tier sits *above* the model card in the precedence chain and a filled-in default would
shadow a card's own tuned profile.

A spec's keys are a closed set (`system_message`, `tools`, `model`, `thinking`, `generate_kwargs` —
`aimu.tools.builtin.SUBAGENT_SPEC_KEYS`), and an unrecognized one raises when the factory is called:

```python
make_subagent_tool(model, agent_types={"r": {"system_message": "R.", "thinkng": "high"}})
# ValueError: agent_types['r'] has unknown key(s): thinkng.
#             A spec may carry: generate_kwargs, model, system_message, thinking, tools.
```

This matters most for exactly the value this page is about. A misspelled effort key would otherwise
leave the spawned agent reasoning at its default, silently, while the caller believed it had asked
for something else — the same failure mode `thinking="xhigh"` raises for rather than accepting.

One precedence note for `agent.as_model_client()`: a `thinking=` passed to that view's `chat()`
is overridden by the agent's own `thinking` field, because the field is applied on the inner
turns the view drives. Set the field to `None` if the view's callers should decide.

## See also

- [Thinking content and the model context](../explanation/thinking-and-context.md): what happens
  to reasoning after a turn, why prior-turn reasoning is never re-fed to the model, and the full
  resolution-semantics table behind the "validate the argument, never the model" rule.
- [Add a new model](add-new-model.md): the general recipe for a new `ModelSpec` catalog entry.
