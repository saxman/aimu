# Manage context

A conversation is `self.messages` — a plain `list[dict]`, OpenAI format — and it only grows.
Left alone, a long-running agent eventually produces a request that no longer fits the model's
context window. `aimu.context` is three plain functions over that list — `count_tokens`,
`trim_messages`, `summarize_messages` — plus a `compaction=` field on `Agent` that runs one of
them automatically before every model turn. Nothing here is a hidden policy applied inside a
client: it's the same "plain data, plain functions" shape as `aimu.rag`, so what gets dropped and
why is always something you can print.

## Count tokens

`count_tokens` estimates a message list's size. The default counter is `len(text) // 4` over each
message JSON-serialized (AIMU's own inert bookkeeping keys — `timestamp`, `thinking`,
`provenance` — stripped first, since a provider never sees those but the structural overhead of
`role`/`tool_calls`/JSON punctuation does ride along in the real request):

```python
import aimu

messages = [{"role": "user", "content": "hello there"}]
aimu.count_tokens(messages)
# 10
```

**This is an estimate, not a measurement** — typically wrong by 20-30% for any specific model's
real tokenizer. The only exact count AIMU can report is after the fact, via `client.last_usage`
following a real call (see [compare models](compare-models.md#3-what-did-each-one-cost)). Pass
`counter=` with a real tokenizer when accuracy matters more than a zero-dependency default:

```python
aimu.count_tokens(messages, counter=lambda text: len(enc.encode(text)))  # enc: any tiktoken-like encoder
```

## Trim to a budget

`trim_messages` drops the oldest messages until the conversation fits `max_tokens` (per
`count_tokens`), and returns a new list — the input is never mutated:

```python
messages = [
    {"role": "system", "content": "You are a assistant."},
]
for i in range(6):
    messages.append({"role": "user", "content": f"This is user message number {i}, with some extra padding."})
    messages.append({"role": "assistant", "content": f"This is assistant reply number {i}, also padded out."})

aimu.count_tokens(messages)                                  # 328 (estimate)
trimmed = aimu.trim_messages(messages, max_tokens=80, keep_last=2)
aimu.count_tokens(trimmed)                                   # 65
[m["role"] for m in trimmed]
# ['system', 'user', 'assistant']
```

System messages (`keep_system=True`, the default) are always kept regardless of budget.
`keep_last` protects the trailing **messages**, not exchanges — a plain back-and-forth is 2
messages per exchange, but an agentic turn can be 5+ (user, assistant-with-tool_calls, one or more
tool results, final assistant answer). `keep_last=2` on a tool-using conversation protects only
the last 2 messages, which is likely mid-turn; pass more if you want whole exchanges kept.

**The invariant that makes this safe on tool-using conversations**: trimming never orphans a
`tool` message from the `assistant` message carrying the `tool_calls` it's answering. Every
provider rejects that shape, and it's exactly what a naive `messages[-n:]` slice produces.
`trim_messages` treats an `assistant`-with-`tool_calls` message and every `tool` message
answering it as one indivisible unit — dropped together, kept together, never split. On a real
two-turn tool-using conversation:

```python
[m["role"] for m in client.messages]
# ['system', 'user', 'assistant', 'tool', 'assistant', 'user', 'assistant', 'tool', 'assistant']

trimmed = aimu.trim_messages(client.messages, max_tokens=150, keep_last=1)
[m["role"] for m in trimmed]
# ['system', 'assistant', 'user', 'assistant', 'tool', 'assistant']
```

The oldest user turn and the tool-call group answering *it* were dropped together; the later
tool-call group (index 6-7 in the original list) survived intact even though it wasn't part of
the protected tail — the boundary was pushed outward to the group's edge rather than splitting it.

## Summarize instead of dropping

`trim_messages` discards the oldest turns outright. `summarize_messages` replaces them with one
LLM-generated summary instead, so the conversation keeps *some* memory of what came before rather
than none:

```python
client = aimu.client("ollama:qwen3:8b")
messages = [
    {"role": "user", "content": "My favorite color is blue."},
    {"role": "assistant", "content": "Got it, blue is a great color."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
summarized = aimu.summarize_messages(client, messages, keep_last=2)
for m in summarized:
    print(m["role"], ":", m["content"])
```

```
system : Summary of earlier conversation:
The user stated their favorite color is blue, and the assistant acknowledged it as a great color. No decisions or open questions were mentioned in the conversation.
user : What is 2+2?
assistant : 4
```

`client` is anything with a `generate(prompt: str) -> str` method — a plain `BaseModelClient`, or
`agent.as_model_client()` — passed in rather than constructed internally, so this function stays
free of any provider dependency. It applies the same group-protection invariant as
`trim_messages` when carving out the tail, and the same "system messages always survive" rule.
When the whole conversation already fits in `keep_last`, no summarization call is made.

## Automate it: `Agent(compaction=...)`

Both functions above are things you call by hand. `Agent(compaction=...)` (and the per-run
`agent.run(compaction=...)` override) runs a callable of your choice — `list[dict] -> list[dict]`
— right before every model turn in the loop:

```python
import logging
from aimu.agents import Agent

client = aimu.client("ollama:qwen3:8b", system="You are terse. Reply in one short sentence.")
agent = Agent(
    client,
    compaction=lambda msgs: aimu.trim_messages(msgs, max_tokens=60, keep_last=2),
)
agent.run("Tell me a fact about the moon.")
agent.run("Now tell me a fact about the sun.")
agent.run("Now tell me a fact about Mars.")
```

The default is `None` — an agent that doesn't opt in behaves exactly as it did before this field
existed. Set it to `lambda msgs: aimu.summarize_messages(client, msgs)` for the summarizing
variant instead of trimming.

**An applied compaction is never silent.** "Applied" means it actually dropped something, judged
by content (the callable may rebuild kept messages into new dict objects, so this isn't an
identity check). When that happens, two things fire together: a
[`ContextCompacted`](observe-a-run.md) event for a sink attached via `events=`, and an
unconditional `WARNING` log — so a caller with no sink still learns their conversation was
rewritten:

```
WARNING Compacted conversation for agent 'agent-cb9d30': dropped 2 message(s) (~97 -> ~62 tokens, AIMU's own estimate).
```

A compaction call that returns the conversation unchanged (the common case early in a
conversation, before the budget is actually exceeded) is a no-op and announces nothing. The
event's `before_tokens`/`after_tokens` are AIMU's own default estimate — not a measurement of
whatever the callable itself counted to decide what to drop; a callable using a real tokenizer or
a word count will disagree with these numbers, which is stated rather than hidden.

**If the compaction callable raises, the run raises.** A compaction that can't be trusted to run
should stop the turn, not be silently skipped while the caller believes their context is being
managed — the same "failures are apparent" rule as everywhere else in AIMU.

## When it's already too late: `ContextOverflowError`

Compaction is preventive. `ContextOverflowError` (`from aimu.models import ContextOverflowError`)
is what a request that *already* doesn't fit raises — the input-side counterpart of
`TruncatedTurnError` (`aimu.agents`), which reports an *output* that ran out of room instead.

Coverage differs by backend, because the failure itself looks different per backend:

| Provider | How it's detected |
|---|---|
| `ollama` | native API rejects the request after trimming past the required user turn |
| `anthropic` | a 400 whose message names the prompt as too long, or a 413 |
| `openai`, `gemini`, local OpenAI-compat servers that set it | 400 with `code=context_length_exceeded` |
| `hf`, `llamacpp` | in-process pre-flight token count against the model's own known window, before the call is even attempted |

A local OpenAI-compat server that doesn't set `context_length_exceeded` on its 400 isn't caught
here — its own error propagates unchanged, which is honest rather than a silent miss. On a
provider where `__cause__` exists (every networked backend), the original SDK exception is
chained; the in-process pre-flight path has no such error to chain and `__cause__` is `None` by
design.

```python
from aimu.models import ContextOverflowError

try:
    agent.run(task)
except ContextOverflowError:
    agent.model_client.messages = aimu.trim_messages(agent.model_client.messages, max_tokens=8000)
    agent.run(task)
```

Raising the model's own context window (`generate_kwargs={"context_length": N}` on Ollama; see
[set the context length](set-context-length.md)) is the other lever, when the model actually
supports a larger one.

## See also

- [Observe a run](observe-a-run.md) — `ContextCompacted` and every other event `compaction=` and
  the rest of a run can emit.
- [Set the context length](set-context-length.md) — the other side of the same problem: making the
  window itself bigger instead of making the conversation smaller.
- [Cancel a run](cancel-a-run.md) — resuming from partial state, the async-only sibling concern to
  compacting state that's already there.
