# Context management

A model sees what is in the request and nothing else. On every turn AIMU builds the request from the conversation, `client.messages`, a plain `list[dict]` in OpenAI format, along with the system message and the advertised tool schemas. Context management is deciding what that request holds. This page explains the choices involved, what each one costs the model, and why AIMU leaves them in your hands rather than picking a policy for you.

For the functions themselves (`count_tokens`, and `aimu.context.compaction`'s strategies and ready-made compactions for `Agent(compaction=)`), see [Manage context](../how-to/manage-context.md). This page is the *why* behind that one.

## The history and the request are two things

The conversation is data you own. You can print it, slice it, and assign it back:

```python
client.messages = aimu.trim_messages(client.messages, max_tokens=8000)
```

That one line is a complete, permanent edit of the history. Nothing warns you, because nothing needs to: you made the edit and you hold both lists. `reset()`, `restore()`, and assigning `system_message` are the same kind of edit (see [System message lifecycle](system-message-lifecycle.md)).

`Agent(compaction=...)` is not that edit run for you. It is a different thing: a function that decides what each *request* sends, applied while the request is built, to a copy of the history. The model is sent what it returns; `client.messages` keeps everything. It exists because an agent's loop makes model calls you never see between: a worker reading twenty web pages grows its conversation by twenty tool results inside a single `run()`, with no point at which your code could step in. It takes the same kind of callable as the manual edit, any `list[dict] -> list[dict]` (named `aimu.context.Compaction`), so every strategy on this page works in either place: called by hand, or handed to an agent ready-made from `aimu.context.compaction` (`compaction.mask(...)`, `compaction.trim(...)`, and `compaction.chain(...)` to run several in order), the way `aimu.tools.builtin` hands out tool groups.

Keeping the two apart is what makes compaction safe to use anywhere:

- **Nothing a compaction leaves out is lost.** A later request under a looser compaction, a per-run override, or none at all sends it again, because the history still holds it. A strategy can be swapped mid-conversation without having destroyed what the next one would want.
- **The conversation a person reads is untouched.** `ConversationManager` and `SessionStore` persist `client.messages`, and a chat UI renders it. Compacting what the model is sent hides nothing from either.
- **What the model actually received is recorded.** `client.last_request` and the `RequestPrepared` event carry the payload as the provider got it, after compaction, so "what was the model working from?" has an exact answer you can print.

The cost is that the history only grows in memory, and every request runs the compaction over all of it. For the plain functions here that is a scan of a list; for anything that calls a model, see [summarizing](#summarizing-costs-a-model-call-on-every-request-it-changes).

## What fills the window

The messages are not the only thing competing for the context window. A request carries:

- the system message,
- the tool schemas, advertised on every call (a reason to [defer tools](../how-to/defer-tools.md) an agent rarely needs),
- the messages,
- and room for the reply, up to `max_tokens`.

AIMU's own bookkeeping keys on messages (`timestamp`, `thinking`, `provenance`) are stripped before every request and cost nothing. That includes earlier turns' reasoning, which is recorded on the message but never sent back (see [Thinking and the model context](thinking-and-context.md)).

This is why a compaction budget is a *share* of the window, not the window itself. A budget equal to the window leaves no room for the system message, the tools, or the answer.

## Three places to manage it

**At the source.** A tool decides how much text it returns. The built-in readers window their output (`read_file`, `read_document`, `get_web_content`, and `get_webpage_html` return a slice and tell the model which `offset=` reads the next one), and `search_documents` returns excerpts rather than whole documents. This is the only lever for a single oversized result: no compaction can leave out the most recent tool result without orphaning the call it answers, so a tool that returns a 200,000-character page has to cap it itself.

**In the request.** `compaction=` decides what each request sends. This is where growth that builds up over many turns is handled, without touching the history.

**At the window.** Make the window bigger where the backend allows it (`context_length`, honored per request only by Ollama's native API; see [Set the context length](../how-to/set-context-length.md)). When a request no longer fits, AIMU raises `ContextOverflowError`. That is the point where prevention has already failed.

## What each strategy costs

Every compaction leaves out something the model could otherwise see. The strategies differ in *what* they leave out, and the difference shows up in the model's behavior, not in an error. That the history keeps it does not help the model: what is not in the request does not exist for it.

### Masking keeps the record and drops the bulk

`mask_messages` keeps every message where it is and replaces old content with a note naming the message: "[Message 12, the result of fetch, is not shown here, to save context. get_message(12) returns it in full.]". The call that produced a masked result is still there, with its URL or path, so the model knows what it already did, though not what it saw, and a re-fetch becomes a deliberate choice rather than an unknowing one. That is the best-evidenced technique for agents that read a lot: replacing old tool output with a placeholder while keeping the calls matched or beat summarizing it in most of the configurations a SWE-bench study tried, at about half the cost, and it is what Anthropic's server-side context editing, Claude Code, and OpenHands do in one form or another.

It masks the content cheapest to lose first, in `MASK_ORDER`: tool results (usually the bulk, and the call stays beside each), then images (large, and they come back exactly), then assistant text, then user text, since the dialogue is what the model reasons from. Oldest first within each kind, and it stops as soon as the request fits. The task, the system messages, and the most recent messages are never masked.

Because no message is removed or added, a masked request has the history's own shape. Roles alternate wherever they did, a tool call never loses its result, and nothing new needs a role of its own. The note lives inside the message it replaces, so a masked image in your message becomes a line of text in your message, and a masked tool result is still a tool result.

The cost is that a note still costs tokens, twenty or so with its message's structure, so masking slows growth without bounding it: a thousand masked messages are still a thousand notes. That is what trimming is for, behind it.

### Trimming forgets that the model acted

`trim_messages` leaves out the oldest messages until the request fits. It never splits an `assistant` message carrying `tool_calls` from the `tool` messages answering it, because every provider rejects a tool result with no call (or a call with no result), and that orphaned shape is exactly what a naive `messages[-n:]` slice produces. The pair goes as one unit.

The consequence: when a tool result is trimmed, the *call* goes with it. The model loses not only what it read but the fact that it read anything. A worker that fetched a page twelve turns ago, and whose requests have since stopped carrying that exchange, has no record of the URL. If the page still looks relevant to the task, fetching it again is the reasonable move from where the model stands. Repeated fetches under trimming are the model behaving correctly on an incomplete record, not a model bug.

Trimming also leaves out oldest-first among the non-system messages, and in an agent run the oldest of those is usually the **task itself**: the user message `agent.run(task)` appended. On a spawned sub-agent that message is the whole assignment. Once it is left out, the worker has its system message and a recent window of tool traffic, and has to infer what it was asked to do. Nothing raises; the run still returns a plausible answer to a question it can no longer see.

That is why `trim_messages` keeps the first message after the system messages by default (`keep_first=1`). It protects the conversation's first message, which is the current task only when the conversation starts with it. That holds for a spawned sub-agent, for an `Agent` constructed with its own `system_message` or with `reset_messages_on_run=True` (both start each run from an empty conversation), and for any agent's first run. In a conversation that spans several runs, the current run's task sits in the middle and is still droppable. For those, put the standing instructions in the system message, which is always kept, or write a compaction that finds and keeps the latest user message.

In a multi-turn chat the protection has a cost of its own: the pinned first question can outlive its answer, leaving it followed directly by a later question. `keep_first=0` restores plain oldest-first dropping.

### Summarizing costs a model call on every request it changes

`summarize_messages` replaces the older messages with one `{"role": "system"}` summary, produced by a single `client.generate()` call. The transcript it summarizes includes tool calls with their arguments, so a URL can survive into the summary if the summarizer keeps it. Whether it does depends on the summarizer and the prompt (`prompt=` replaces the default instruction).

The costs: one extra model call each time it runs, plus the summarizer's own judgment about what mattered. The summary can drop the detail a later turn needed, with nothing in the transcript showing that it did.

As a `compaction=` the first cost dominates. A compaction runs on every request, over the whole history, so `lambda msgs: summarize_messages(client, msgs)` calls the summarizer before every model turn, and each call summarizes a longer stretch than the last. Summarizing suits a deliberate, permanent edit you make by hand, `client.messages = summarize_messages(client, client.messages)`, at a point you choose. As a per-request compaction it needs a cache of summaries keyed by the stretch they cover, which AIMU does not ship (issue #19); masking or trimming needs none.

That judgment would apply to the task too, so `summarize_messages` takes the same `keep_first=1` default as `trim_messages` and keeps the first message verbatim instead of summarizing it. The summarizer is still shown that message, labeled as background it should not summarize. Without it, the summarizer cannot tell which tool results the task depends on: it is asked to keep what a continuation will need, and need is relative to the goal. Kept out of view, the task's context goes missing from the summary; in one run the summarizer listed "the location" as an open question because the city was named only in the task. The cost of showing it is a longer summarization prompt and a summary that tends to restate the task briefly, despite the label.

## What a request leaves out is announced

When a request leaves out a message the previous request of the same agent sent, the loop emits a [`ContextCompacted`](../reference/api/events.md) event carrying copies of those messages (`dropped`), and logs a `WARNING` whether or not a sink is attached. The history still has them, but the model is working from less than the conversation holds, which is exactly the kind of silent change [principle 6](design-principles.md) rules out. A request that sends one the previous request left out reports it as `restored`, logged at `INFO`, since nothing was lost. A compaction that raises stops the run rather than being skipped.

Only a change is announced. Once a conversation is over its budget, every request is compacted, and each one leaves out the same old messages the last one did, plus perhaps one more. Announcing every request would put a warning on every turn of every long conversation and teach people to ignore it, so a request that leaves out exactly what the previous one did says nothing, and one that leaves out one more message reports that one. The agent remembers what its last request left out across runs, so a conversation that has been over budget for an hour does not re-announce its first omissions on each new run.

How AIMU tells "left out" from "sent" matters for masking. The loop compares the history and the request **by content**, not by object identity, so a callable that rebuilds every dict without removing anything correctly reports nothing. The flip side is that a *shortened* message counts as left out: its original content is not in the request. A mask therefore reports each masked message in `ContextCompacted.dropped`, with its original content, even though a note for it was sent. The note itself is in `client.last_request`, which is where to read exactly what the model was working from.

The event's `before_tokens` and `after_tokens` are AIMU's own estimate (`len(text) // 4`) of the history and of what was sent, not whatever your callable counted to decide. Read them as orientation, not as the number that drove the decision.

## Telling the model what a request leaves out

`ContextCompacted` tells *you* what a request left out. The model needs telling too, or a compacted conversation reads to it as one that never contained what is missing, and it will answer "we never discussed that" about something it was shown an hour ago. Nothing raises; the answer is just wrong.

What it needs telling is a fact about the request it is reading, not about an event. A compaction runs again on every request, over the whole history, so the model never experiences a compaction happening: each request is a fresh view, and nothing from an earlier request's view survives into the next. So the telling is built by the compaction, into the request, every time, and never stored. A note written into the history would describe the request it was made for and go on describing it after the strategy or the conversation had moved on.

**The note goes where the content was.** A mask replaces a message's content with a note naming that message, so each note sits exactly where the missing content sat, in the message it replaced. That placement is what makes every other question disappear. No message is added, so a note needs no role of its own, the request keeps the history's shape, and no chat template sees two user turns in a row that the history did not already have. The strategy that left the content out is the one that writes the note, so nothing downstream (the client, the loop, the provider adapter) has to know a note exists.

A single note at the point where a trim cut was considered and set aside for that reason: a trim removes whole messages, so a note there would have to be a new message, with a role to choose and a neighbour it might collide with, and it could only say "something is missing here" rather than which message and how to get it. Trimming stays silent, and runs behind a mask, where it only drops messages the mask has already reduced to notes.

**Each note names the way back.** A note that only says something is missing tells the model to be careful; a note that says how to get it back lets the model act. The default note names `get_message(<index>)`, which returns the message from the agent's own history in full, in windows for a long one, with its images. It reads the history the compaction never touched, so recovery is exact: unlike repeating a fetch, it cannot return a page that has changed since.

**`get_message` comes with any compaction.** An agent run with a compaction carries the tool without listing it. The alternative, notes that name a tool only some agents hold, would send any agent without it into a refusal at the moment it most needs the content back, and the tool reads nothing the agent was not already shown, so carrying it widens nothing the agent could reach. A compaction and its way back are one decision: choosing to send less is choosing to give the model a route to the rest.

Two cautions hold whatever the wording. Keep notes factual: an instruction strong enough to make the model recall everything it is told about refills the window it was compacted to fit. And expect smaller models to use the way back less readily than larger ones; a model that answers from the note instead of reading the message is the failure to look for when choosing a budget.

## Compacting at the budget changes every request

`trim_messages` stops as soon as the request fits. Once the history reaches the budget, each new message pushes it over again and the next request leaves out one more group. From then on every request starts at a different message, and that has a cost: **no reuse of a cached prefix.** Many inference servers, local and cloud, reuse their computation for the start of a request they have already seen. Trimming changes the request right after the system message, so on every request the server reprocesses everything after that point. (AIMU's Anthropic `cache_prompt=True` marks only the system prompt and tools, so that particular cache survives trimming.)

The remedy is to move the cut rarely: leave out more than the budget strictly requires, in whole steps, so the request starts at the same message for many turns before the cut moves again. Because a compaction is a function of the history, "rarely" has to be computed from the history rather than remembered, and rounding the amount left out up to a whole step does exactly that:

```python
def trim_in_steps(messages, window=32768):
    high, step = int(window * 0.75), int(window * 0.25)
    total = aimu.count_tokens(messages)
    if total <= high:
        return messages
    over = -(-(total - high) // step) * step  # what must go, rounded up to whole steps
    return aimu.trim_messages(messages, max_tokens=total - over)
```

While the history grows within one step, `over` stays the same, so the same oldest messages are left out and the request's start does not move. In a check of 120 growing requests against a 2,000-token window, the start moved 9 times rather than on nearly every request, and no request exceeded the budget.

Masking is gentler on a cache than trimming. A request changes from the first message that is masked differently from the last request, and since masking goes oldest first within each kind, a newly masked message usually sits later in the conversation than the ones already masked, so more of the start is reused. It still changes the request once per newly masked message.

## Where to turn it on

Because a compaction never rewrites the history, the question is no longer whose conversation it is safe to rewrite. It is only whose requests need it. An agent whose conversation stays small needs none. A long-running worker reading many sources is the case it was built for, which is why `make_subagent_tool` accepts `compaction=` (at the factory level and as a per-type spec key; see [Spawn sub-agents](../how-to/spawn-subagents.md)). A long user-facing conversation can use one too, since compacting what the model is sent hides nothing from the person reading it. Either way, `compaction.chain(compaction.mask(n), compaction.trim(n))` is the shape to start from: masking keeps the record, the trim bounds what masking cannot, and the agent carries `get_message` to read back what a note names. (A trimmed message leaves no note, so the model has nothing to ask for; that is why the trim goes last.)

AIMU does not make this choice for you. `compaction=` defaults to `None` on every agent, so an agent that does not opt in sends its whole history on every request.

## What AIMU deliberately doesn't do

- **No hidden context policy.** A compaction does run inside the request path, and that would make the most consequential edit to what the model sees the hardest one to observe if it were invisible. It is not: it is a function you wrote and passed, what it left out is announced when that changes, and the request it produced is on `client.last_request`. AIMU adds the timing (every request of the run, including the forced wrap-up), the announcement, and the `get_message` tool that makes a note's way back real, and nothing else.
- **No default compaction.** Automatically choosing what a model forgets would change its behavior in ways that show up as "the model got worse" rather than as an error.
- **No exact pre-request token count.** An exact count needs the provider's own tokenizer, which is not available before the request. `count_tokens` is a stated estimate; the exact number arrives afterwards, on `client.last_usage`.

## See also

- [Manage context](../how-to/manage-context.md): the functions, with worked examples.
- [Observe a run](../how-to/observe-a-run.md): `ContextCompacted` alongside every other run event.
- [Set the context length](../how-to/set-context-length.md): the window side of the same problem.
- [Spawn sub-agents](../how-to/spawn-subagents.md): `compaction=` on spawned workers.
