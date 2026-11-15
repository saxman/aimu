# Manage context

A conversation is `self.messages` — a plain `list[dict]`, OpenAI format — and it only grows.
Left alone, a long-running agent eventually produces a request that no longer fits the model's
context window. `aimu.context` holds the tools for that: `count_tokens`, the estimate every budget
is measured in, and `aimu.context.compaction`, which bundles the strategies (mask, trim, summarize)
and the ready-made compactions an `Agent(compaction=...)` applies to every request it sends, the way
`aimu.tools.builtin` bundles tools. The conversation itself stays whole. Nothing here is a policy
AIMU picks for you: it's the same "plain data, plain functions" shape as `aimu.rag`, so what gets
left out and why is always something you can print.

```python
from aimu.context import compaction
```

`aimu.trim_messages`, `aimu.summarize_messages`, and `aimu.count_tokens` are short paths to the same
functions.

For what each strategy costs the model (trimming a tool call erases the record that it ran, and
the oldest message dropped is usually the task itself), see
[Context management](../explanation/context-management.md).

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

`compaction.trim_messages` drops the oldest messages until the conversation fits `max_tokens` (per
`count_tokens`), and returns a new list — the input is never mutated. Dropping leaves no trace, so
as a compaction it is the backstop behind masking (next section), not the first resort:

```python
messages = [
    {"role": "system", "content": "You are a assistant."},
]
for i in range(6):
    messages.append({"role": "user", "content": f"This is user message number {i}, with some extra padding."})
    messages.append({"role": "assistant", "content": f"This is assistant reply number {i}, also padded out."})

aimu.count_tokens(messages)                                  # 274 (estimate)
trimmed = aimu.trim_messages(messages, max_tokens=80, keep_last=2)
aimu.count_tokens(trimmed)                                   # 78
[m["role"] for m in trimmed]
# ['system', 'user', 'user', 'assistant']
```

Three things are protected, and everything between them is droppable:

- **System messages** (`keep_system=True`, the default) are always kept, whatever the budget.
- **The first message after them** (`keep_first=1`, the default): here `user 0`. In an agent run
  this is the task, and dropping it leaves the agent working without knowing what it was asked
  (see [Context management](../explanation/context-management.md#trimming-forgets-that-the-model-acted)).
  Pass `keep_first=0` for plain oldest-first dropping.
- **The last `keep_last` messages**: here `user 5` and `assistant 5`.

That leaves nine droppable messages, `assistant 0` through `assistant 4`. `trim_messages` drops
them oldest-first and stops the moment the conversation fits `max_tokens`, so the protected
counts are floors, not targets. Here all nine have to go to get under the 80-token budget. Pass
`keep_first=0` and `user 0` becomes droppable too, which leaves room for `assistant 4` to stay
instead: `['system', 'assistant', 'user', 'assistant']`.

`keep_first` and `keep_last` count **messages**, not exchanges. A plain back-and-forth is 2
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
# ['system', 'user', 'assistant', 'user', 'assistant', 'tool', 'assistant']
```

The first user turn is pinned by `keep_first`, so the first thing dropped is the tool-call group
answering it: the `assistant` carrying `tool_calls` and its `tool` result go together. The later
tool-call group (index 6-7 in the original list) survived intact even though only its final
`assistant` reply was inside the protected tail: the boundary was pushed outward to the group's
edge rather than splitting it. `keep_first` follows the same rule, so a first message that opens a
tool-call group keeps its whole group.

## Mask instead of dropping

`compaction.mask_messages` keeps every message where it is and replaces old content with a short
note naming the message, until the list fits. The model still sees who said what and which tools
ran, and each note names the way back:

```python
from aimu.context.compaction import mask_messages

page = "The release is planned for March. " * 60
messages = [
    {"role": "system", "content": "You are a researcher."},
    {"role": "user", "content": "When is the release?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "fetch", "arguments": {"url": "https://example.com"}}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "name": "fetch", "content": page},
    {"role": "assistant", "content": "It is planned for March."},
    {"role": "user", "content": "Which year?"},
]
aimu.count_tokens(messages)                  # 621
masked = mask_messages(messages, max_tokens=200)
masked[3]["content"]
# '[Message 3, the result of fetch, is not shown here, to save context. get_message(3) returns it in full.]'
aimu.count_tokens(masked)                    # 137
```

Only message 3 changed. The call that fetched the page is still there, with its URL, so the model
knows it read the page, and the note says how to read it again.

**What goes first.** Content is masked in `compaction.MASK_ORDER`, oldest first within each kind,
and masking stops the moment the list fits:

1. **Tool results**, usually the bulk, and the call stays beside each one.
2. **Images**, in any message: a message's image blocks become one note beside its text.
3. **Assistant text**: a reply's text becomes a note, and a tool-call argument over 200 characters
   (a whole file's content, say) becomes a short placeholder while the call stays.
4. **User text**, last, since the dialogue is what the model reasons from.

The system messages, the first `keep_first` and the last `keep_last` messages are never masked, as
for `trim_messages`, and a message shorter than its note is left alone. Because no message is removed
or added, the list keeps the history's shape: roles alternate wherever they did, and a tool call
never loses its result.

**The way back is `get_message`.** The default note (`compaction.default_note`) names
`get_message(<index>)`, a tool every agent with a compaction is given automatically (see
[Automate it](#automate-it-agentcompaction)). It reads the message from the agent's own history,
which a compaction never rewrites. Pass `note=` (a function of the index, the original message, and
the kind) for your own wording; the index is the message's position in the list you pass, which is
its position in the history when masking runs as a compaction.

A note still costs tokens, so masking slows a conversation's growth without bounding it. When every
candidate has been masked and the list still does not fit, `mask_messages` returns it as masked;
follow it with a trim.

## Summarize instead of dropping

`trim_messages` discards the oldest turns outright. `summarize_messages` replaces them with one
LLM-generated summary instead, so the conversation keeps *some* memory of what came before rather
than none:

```python
client = aimu.client("ollama:qwen3:8b")
messages = [
    {"role": "user", "content": "Help me plan a weekend in Lisbon."},
    {"role": "assistant", "content": "Happy to. What do you enjoy most when you travel?"},
    {"role": "user", "content": "Tile art and seafood, and I'd rather walk than take taxis."},
    {"role": "assistant", "content": "Noted: tiles, seafood, and a walkable route."},
    {"role": "user", "content": "Which museum should I start with?"},
    {"role": "assistant", "content": "The National Tile Museum."},
]
summarized = aimu.summarize_messages(client, messages, keep_last=2)
for m in summarized:
    print(m["role"], ":", m["content"])
```

```
system : Summary of earlier conversation:
<generated summary text — varies by run and model>
user : Help me plan a weekend in Lisbon.
user : Which museum should I start with?
assistant : The National Tile Museum.
```

The `"Summary of earlier conversation:\n"` prefix is a fixed template (see
`summarize_messages`'s source). Only the summary itself is generated text, and it will read
differently on your machine. The rest is the original messages kept verbatim: the first one
(`keep_first=1`, the default, which protects an agent's task from being paraphrased away) and
the last two (`keep_last=2`). The three messages between them are what got summarized.

Two details of that layout. The summary comes *before* the kept first message, though it
describes what came after it, because several local chat templates reject a system message
anywhere but the start of the conversation. And the summarizer is shown the kept first message
as background, labeled "kept as written; do not summarize". The instruction asks it to keep what
a continuation will need, and the task is what decides that. Before this, the summarizer saw
only the middle, and a run listed "the location" as an open question because "Lisbon" was only
in the task. Expect the summary to restate the task briefly anyway: models tend to, despite the
label. `keep_first=0` summarizes from the first message with no background block.

`client` is anything with a `generate(prompt: str) -> str` method — a plain `BaseModelClient`, or
`agent.as_model_client()` — passed in rather than constructed internally, so this function stays
free of any provider dependency. It applies the same group-protection invariant as
`trim_messages` when carving out the kept first and last messages, and the same "system messages
always survive" rule. When nothing lies between the kept first and last messages, no
summarization call is made.

## Automate it: `Agent(compaction=...)`

The functions above are things you call by hand, and assigning the result back is a permanent edit
of the history. `Agent(compaction=...)` (and the per-run `agent.run(compaction=...)` override) is
different in kind: it takes a compaction (`aimu.context.Compaction`, any `list[dict] -> list[dict]`)
and applies it to every request the agent sends, while each request is built. The model is sent what
it returns; `client.messages` keeps the whole conversation.

`aimu.context.compaction` ships them ready-made, with their settings bound, the way
`aimu.tools.builtin` ships tool groups:

```python
from aimu.agents import Agent
from aimu.context import compaction

client = aimu.client("ollama:qwen3:8b", system="You are terse. Reply in one short sentence.")
agent = Agent(client, compaction=compaction.trim(60, keep_last=2))
agent.run("Tell me a fact about the moon.")
agent.run("Now tell me a fact about the sun.")
agent.run("Now tell me a fact about Mars.")

[m["role"] for m in client.messages]
# ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant']
[m["role"] for m in client.last_request["messages"]]
# ['system', 'user', 'assistant', 'user']
```

The history holds all three exchanges. The request for the Mars question sent four messages: the
system prompt, the moon question (pinned by `keep_first=1`), the sun answer, and the Mars question
(`keep_last=2`). What it left out is still in `client.messages`, so a later request under a looser
compaction, or none, sends it again. `client.last_request` is the payload the provider actually
received.

| Ready-made | What each request gets |
|---|---|
| `compaction.mask(max_tokens, **options)` | `mask_messages` with these options: old content replaced by notes, in place |
| `compaction.trim(max_tokens, **options)` | `trim_messages` with these options: the oldest messages dropped |
| `compaction.chain(a, b, ...)` | each compaction in turn, on the previous one's result |

Options are the function's own keyword arguments, checked when the compaction is made, so a
misspelled one raises then rather than on the first request.

**Mask first, trim as the backstop.** Masking keeps the record of what happened but cannot bound
growth on its own, since the notes still cost tokens; a trim bounds it but leaves no trace. Chained,
the trim only ever drops messages masking has already reduced to notes:

```python
agent = Agent(client, compaction=compaction.chain(compaction.mask(6000), compaction.trim(6000)))
```

Put a mask first in any chain. Its notes name each message by its position in the list it is given,
so it has to see the full history, before anything drops messages.

**`get_message` comes with any compaction.** An agent run with a compaction carries one more tool,
`get_message(index, offset=1, max_chars=8000)` (`aimu.tools.builtin.make_history_tools`), which
returns a message from the agent's own history in full: what a mask's note replaced, in windows for a
long one, with its images. You do not list it; a note naming a tool the model cannot call would only
produce a refusal. It sits ahead of any tool-group tail, and an agent whose own tools already include
a `get_message` is refused while it has a compaction.

The default is `None`: an agent that doesn't opt in sends its whole history on every request, and
carries no `get_message`. There is no ready-made summarizing compaction: a compaction runs on every
request, so a summarizing one would call the summarizer before every model turn, over a longer
history each time (issue #19). Use `summarize_messages` for a deliberate edit by hand.

**What a request leaves out is never silent.** When a request leaves out a message the previous
request sent, judged by content (the callable may rebuild kept messages into new dict objects, so
this isn't an identity check), two things fire together: a [`ContextCompacted`](observe-a-run.md)
event for a sink attached via `events=`, carrying the messages left out, and an unconditional
`WARNING` log, so a caller with no sink still learns the model is working from less than the
conversation holds. A masked message counts as left out, reported with its original content, since
the model was sent a note in its place:

```
WARNING Compacted the request for agent '<agent name>': 2 more message(s) left out (~<N> -> ~<M> tokens sent, AIMU's own estimate). The history keeps them.
```

The agent name, the token counts, and how many messages go all vary by run, because they depend on
how long the model's sampled replies were. In the run shown, the moon answer and the sun question
were left out. The moon question is the first message after the system prompt, so `keep_first=1`
pinned it, and it is sent without its answer, followed directly by a later answer. That is the price
of protecting the first message in a multi-turn chat; in a single-task agent run, the pinned message
is the task and is exactly what should survive.

Only a change is announced. Once the conversation is over budget every request is compacted, and a
request that leaves out exactly what the previous one did says nothing; the agent remembers that
across runs. A request that sends something the previous one left out reports it in the event's
`restored` list, at `INFO`. A compaction that returns the conversation unchanged announces nothing.
The event's `before_tokens`/`after_tokens` are AIMU's own estimate of the history and of what was
sent, not a measurement of whatever the callable itself counted; a callable using a real tokenizer
or a word count will disagree with these numbers, which is stated rather than hidden.

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
    agent.run(task, compaction=compaction.trim(8000))
```

The retry sends a trimmed request and leaves the history whole. To shrink the stored history
itself instead, assign the trimmed list back
(`agent.model_client.messages = aimu.trim_messages(...)`), a permanent edit you made and that
nothing needs to announce.

Raising the model's own context window (`generate_kwargs={"context_length": N}` on Ollama; see
[set the context length](set-context-length.md)) is the other lever, when the model actually
supports a larger one.

## See also

- [Context management](../explanation/context-management.md): the *why*, including what each
  strategy costs the model and why compacting at the budget compacts every turn.
- [Observe a run](observe-a-run.md) — `ContextCompacted` and every other event `compaction=` and
  the rest of a run can emit.
- [Set the context length](set-context-length.md) — the other side of the same problem: making the
  window itself bigger instead of making the conversation smaller.
- [Cancel a run](cancel-a-run.md) — resuming from partial state, the async-only sibling concern to
  compacting state that's already there.
