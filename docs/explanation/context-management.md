# Context management

A model sees what is in the request and nothing else. On every turn AIMU sends the conversation, `client.messages`, a plain `list[dict]` in OpenAI format, along with the system message and the advertised tool schemas. Context management is deciding what that list holds. This page explains the choices involved, what each one costs the model, and why AIMU leaves them in your hands rather than applying a policy inside the client.

For the functions themselves (`count_tokens`, `trim_messages`, `summarize_messages`, `Agent(compaction=)`), see [Manage context](../how-to/manage-context.md). This page is the *why* behind that one.

## Manual first, automatic second

The conversation is data you own. You can print it, slice it, and assign it back:

```python
client.messages = aimu.trim_messages(client.messages, max_tokens=8000)
```

That one line is complete context management. Nothing warns you, because nothing needs to: you made the edit and you hold both lists. `reset()`, `restore()`, and assigning `system_message` are the same kind of edit (see [System message lifecycle](system-message-lifecycle.md)).

`Agent(compaction=...)` is that same edit, run for you before every model turn of an agent's run. It exists because an agent's loop makes model calls you never see between: a worker reading twenty web pages grows its conversation by twenty tool results inside a single `run()`, with no point at which your code could step in. The automatic path is a convenience over the manual one, not a different mechanism, and it takes the same kind of callable: any `list[dict] -> list[dict]`.

So a compaction is never more than a function you could have called yourself. The two things AIMU adds are timing (before each turn, including the forced wrap-up turn) and announcement (see [An applied compaction is announced](#an-applied-compaction-is-announced)).

## What fills the window

The messages are not the only thing competing for the context window. A request carries:

- the system message,
- the tool schemas, advertised on every call (a reason to [defer tools](../how-to/defer-tools.md) an agent rarely needs),
- the messages,
- and room for the reply, up to `max_tokens`.

AIMU's own bookkeeping keys on messages (`timestamp`, `thinking`, `provenance`) are stripped before every request and cost nothing. That includes earlier turns' reasoning, which is recorded on the message but never sent back (see [Thinking and the model context](thinking-and-context.md)).

This is why a compaction budget is a *share* of the window, not the window itself. A budget equal to the window leaves no room for the system message, the tools, or the answer.

## Three places to manage it

**At the source.** A tool decides how much text it returns. The built-in readers window their output (`read_file`, `read_document`, `get_web_content`, and `get_webpage_html` return a slice and tell the model which `offset=` reads the next one), and `search_documents` returns excerpts rather than whole documents. This is the only lever for a single oversized result: no compaction can remove the most recent tool result without orphaning the call it answers, so a tool that returns a 200,000-character page has to cap it itself.

**Between turns.** Edit the list, by hand or through `compaction=`. This is where growth that builds up over many turns is handled.

**At the window.** Make the window bigger where the backend allows it (`context_length`, honored per request only by Ollama's native API; see [Set the context length](../how-to/set-context-length.md)). When a request no longer fits, AIMU raises `ContextOverflowError`. That is the point where prevention has already failed.

## What each strategy costs

Every compaction removes something the model could otherwise see. The strategies differ in *what* they take away, and the difference shows up in the model's behavior, not in an error.

### Trimming forgets that the model acted

`trim_messages` drops the oldest messages until the conversation fits. It never splits an `assistant` message carrying `tool_calls` from the `tool` messages answering it, because every provider rejects a tool result with no call (or a call with no result), and that orphaned shape is exactly what a naive `messages[-n:]` slice produces. The pair goes as one unit.

The consequence: when a tool result is trimmed, the *call* goes with it. The model loses not only what it read but the fact that it read anything. A worker that fetched a page twelve turns ago, and has since had that exchange trimmed, has no record of the URL. If the page still looks relevant to the task, fetching it again is the reasonable move from where the model stands. Repeated fetches under trimming are the model behaving correctly on an incomplete record, not a model bug.

Trimming also drops oldest-first among the non-system messages, and in an agent run the oldest of those is usually the **task itself**: the user message `agent.run(task)` appended. On a spawned sub-agent that message is the whole assignment. Once it is trimmed, the worker has its system message and a recent window of tool traffic, and has to infer what it was asked to do. Nothing raises; the run still returns a plausible answer to a question it can no longer see.

That is why `trim_messages` keeps the first message after the system messages by default (`keep_first=1`). It protects the conversation's first message, which is the current task only when the conversation starts with it. That holds for a spawned sub-agent, for an `Agent` constructed with its own `system_message` or with `reset_messages_on_run=True` (both start each run from an empty conversation), and for any agent's first run. In a conversation that spans several runs, the current run's task sits in the middle and is still droppable. For those, put the standing instructions in the system message, which is always kept, or write a compaction that finds and keeps the latest user message.

In a multi-turn chat the protection has a cost of its own: the pinned first question can outlive its answer, leaving it followed directly by a later question. `keep_first=0` restores plain oldest-first dropping.

### Summarizing costs a model call and loses detail differently

`summarize_messages` replaces the older messages with one `{"role": "system"}` summary, produced by a single `client.generate()` call. The transcript it summarizes includes tool calls with their arguments, so a URL can survive into the summary if the summarizer keeps it. Whether it does depends on the summarizer and the prompt (`prompt=` replaces the default instruction).

The costs: one extra model call each time it fires, plus the summarizer's own judgment about what mattered. The summary can drop the detail a later turn needed, with nothing in the transcript showing that it did.

That judgment would apply to the task too, so `summarize_messages` takes the same `keep_first=1` default as `trim_messages` and keeps the first message verbatim instead of summarizing it. The summarizer is still shown that message, labeled as background it should not summarize. Without it, the summarizer cannot tell which tool results the task depends on: it is asked to keep what a continuation will need, and need is relative to the goal. Kept out of view, the task's context goes missing from the summary; in one run the summarizer listed "the location" as an open question because the city was named only in the task. The cost of showing it is a longer summarization prompt and a summary that tends to restate the task briefly, despite the label.

### Stubbing results keeps the record and drops the bulk

AIMU ships no function for this, because it is a few lines and the placeholder wording is yours to choose. Replace the content of older tool results with a short placeholder and leave the calls in place:

```python
ELIDED = "[Result removed to save context. Call the tool again if you need it.]"

def elide_old_results(messages, keep_last=6):
    cutoff = len(messages) - keep_last
    return [
        {**message, "content": ELIDED} if message.get("role") == "tool" and index < cutoff else message
        for index, message in enumerate(messages)
    ]
```

The call, and the URL or path in its arguments, stays in the conversation. The model knows what it already did, though not what it saw, so a re-fetch becomes a deliberate choice rather than an unknowing one. The calls themselves still pile up, so this slows growth without bounding it. Pair it with a trim when the conversation must fit a fixed budget:

```python
agent = Agent(client, compaction=lambda messages: aimu.trim_messages(elide_old_results(messages), max_tokens=budget))
```

## An applied compaction is announced

When a compaction changes the conversation, the loop emits a [`ContextCompacted`](../reference/api/events.md) event carrying copies of what was removed, and logs a `WARNING` whether or not a sink is attached. Rewriting history the caller believes is still there is exactly the kind of silent change [principle 6](design-principles.md) rules out. A compaction that returns the conversation unchanged announces nothing, since a warning on every turn of every short conversation would teach people to ignore it. A compaction that raises stops the run rather than being skipped.

How AIMU tells "changed" from "unchanged" matters for the stubbing strategy. The loop compares the messages before and after **by content**, not by object identity, so a callable that rebuilds every dict without removing anything correctly reports nothing. The flip side is that a *shortened* message counts as removed: its original content is no longer in the list. A stubbing compaction therefore reports each stubbed tool result in `ContextCompacted.dropped`, with its original body, and the warning counts it as a dropped message, even though a shortened version is still in the conversation. The placeholder itself is not reported anywhere. To see exactly what the model is now working from, read `client.messages`.

The event's `before_tokens` and `after_tokens` are AIMU's own estimate (`len(text) // 4`), not whatever your callable counted to decide. Read them as orientation, not as the number that drove the decision.

## Compacting at the budget compacts every turn

`trim_messages` stops as soon as the conversation fits. Once a conversation reaches its budget, each new turn pushes it over again and the next compaction drops one more group. From then on, every turn is a compaction, and that has two costs:

- **A warning on every turn**, which is accurate but noisy.
- **No reuse of a cached prefix.** Many inference servers, local and cloud, reuse their computation for the start of a request they have already seen. Trimming changes the conversation right after the system message, so on every compacted turn the server reprocesses everything after that point. (AIMU's Anthropic `cache_prompt=True` marks only the system prompt and tools, so that particular cache survives trimming.)

The remedy is headroom: trigger at one size and trim to a smaller one, so a compaction buys several turns before the next.

```python
def trim_with_headroom(messages, window=32768):
    if aimu.count_tokens(messages) <= int(window * 0.75):
        return messages
    return aimu.trim_messages(messages, max_tokens=int(window * 0.5))
```

## Whose conversation is being compacted

A compaction rewrites `client.messages`, which is also what `ConversationManager` and `SessionStore` persist and what a chat UI renders. Compacting the conversation a user is reading deletes history they can still scroll back to, and the saved copy loses it too, for the rest of that conversation's life. If you need both a small working context and a full record, keep the record yourself: `ContextCompacted.dropped` hands you everything removed, or you can persist a copy before compacting.

A spawned sub-agent is the opposite case. Its conversation is built for one task and discarded when the task returns, so automatically rewriting it hides nothing anyone will read later. That is why `make_subagent_tool` accepts `compaction=` (at the factory level and as a per-type spec key; see [Spawn sub-agents](../how-to/spawn-subagents.md)), and why it is the most natural place to turn compaction on.

AIMU does not make this choice for you. `compaction=` defaults to `None` on every agent, so an agent that does not opt in never has its conversation touched by the loop.

## What AIMU deliberately doesn't do

- **No context policy inside the client.** A `ContextPolicy` object applied invisibly in the request path would make the most consequential edit to what the model sees the hardest one to observe. The functions take a list and return a list, so what was dropped is always something you can print.
- **No default compaction.** Automatically choosing what a model forgets would change its behavior in ways that show up as "the model got worse" rather than as an error.
- **No exact pre-request token count.** An exact count needs the provider's own tokenizer, which is not available before the request. `count_tokens` is a stated estimate; the exact number arrives afterwards, on `client.last_usage`.

## See also

- [Manage context](../how-to/manage-context.md): the functions, with worked examples.
- [Observe a run](../how-to/observe-a-run.md): `ContextCompacted` alongside every other run event.
- [Set the context length](../how-to/set-context-length.md): the window side of the same problem.
- [Spawn sub-agents](../how-to/spawn-subagents.md): `compaction=` on spawned workers.
