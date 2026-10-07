# Give a running agent an inbox

A run that is already in progress is unreachable. The user types "actually, check the staging log
instead", and the agent is three tool rounds into the wrong file. Without somewhere to put that
message, a host has two options: drop it, or queue it behind the whole run and deliver it as the
next turn, by which point the run has finished doing the wrong thing.

`Inbox` is the third option. The agent loop asks for pending messages at each round boundary and
sends whatever it gets as that round's user message, on a model call it was going to make anyway.
The run is redirected rather than restarted.

This is the complement to [cancelling a run](cancel-a-run.md), not a replacement for it:
cancelling throws the turn away, an inbox keeps the context and changes where it is pointed.

## The protocol

`aimu.agents.inbox.Inbox` is a `Protocol` with one method:

```python
from typing import Callable, Optional, Protocol

class Inbox(Protocol):
    def reader(self, agent: Optional[str] = None) -> Callable[[], list[str]]: ...
```

`reader()` is called **once per run**, at its start, and returns that run's own drain. The drain
returns every message that arrived since it was last called, oldest first, and an empty list when
there is nothing. It is called from inside the loop, so it must not block.

A minimal host-side implementation, thread-safe because the feeding side is usually another thread:

```python
import threading
from collections import deque

class QueueInbox:
    """A mailbox a host writes to and a run reads from."""

    def __init__(self):
        self._pending = deque()
        self._lock = threading.Lock()

    def send(self, text: str) -> None:        # the host's side
        with self._lock:
            self._pending.append(text)

    def reader(self, agent=None):             # the run's side
        def drain():
            with self._lock:
                pending = list(self._pending)
                self._pending.clear()         # advance the cursor, or see "The round budget"
                return pending

        return drain
```

`Inbox` is `runtime_checkable`, so any object with a `reader` method satisfies it. There is no base
class to inherit.

## Wiring it in

`inbox` is a standing field on `Agent` with a per-run override, the same shape as `compaction`:

```python
import aimu
from aimu.agents import Agent

mailbox = QueueInbox()

agent = Agent(aimu.client("ollama:qwen3:8b"), tools=[...], inbox=mailbox)   # every run
agent.run("summarize the staging logs")

agent = Agent(aimu.client("ollama:qwen3:8b"), tools=[...])
agent.run("summarize the staging logs", inbox=mailbox)                     # this run only
```

Identical on the async surface (`aio.Agent(inbox=...)` / `await agent.run(..., inbox=...)`), and
`SkillAgent` takes it on both.

One case it does **not** cover: `run(schema=...)` makes a single structured-output turn instead of
running the tool loop, so there is no round boundary to drain at and the inbox is unused.

## Where a message lands

The loop checks for a pending message at each of the three round-boundary branches, and what the
message does differs by branch:

| Branch | What a pending message does |
| --- | --- |
| After a tool round | Sent as the next round's user message, before the model sees the tool results |
| After a degenerate turn (no content, no tool calls) | **Replaces** the built-in continuation nudge, rather than running alongside it |
| After what would have been the final turn | **Extends** the run by one more round, so a message landing as the model composes its answer is not dropped |

That last branch is what makes the timing forgiving: you do not have to land a message inside the
run's working rounds for it to be read. A healthy turn with nothing pending ends exactly as it did
before the run had an inbox.

A drain returning only whitespace delivers nothing, so a host that polls with an empty heartbeat
does not manufacture a round for it.

## Seeing it happen

A streamed run yields a `StreamingContentType.INBOX` chunk immediately before the round that reads
the message:

```python
for chunk in agent.run("summarize the staging logs", stream=True, inbox=mailbox):
    if chunk.is_inbox():
        print(f"[delivered] {chunk.content['text']}")
```

The content is `dict {"text": str}`. Every shipped consumer renders it:

- `aimu.pretty_print` and `CLIChannel` print a `[message] <text>` line, ungated by the
  `show_thinking` / `show_tools` flags (it is one line per delivered round, and it is what explains
  a run changing direction).
- `WebChannel` emits a frame of its own, `{"type": "inbox", "text": str}`, rather than a third
  `reason` on its `loop` frame. `loop` means the loop injected the round, and conflating the two
  would attribute the user's words to the assistant.

In the stored transcript, a delivered message is an ordinary `{"role": "user"}` entry carrying **no
provenance tag**, unlike the loop's own `PROVENANCE_CONTINUATION` / `PROVENANCE_FINAL_ANSWER`
injections (see [stream phases](../reference/stream-phases.md)). The words are the user's, arriving
through a different channel; they are not a prompt the loop composed for itself.

## The round budget

`max_iterations` bounds *autonomous* iteration: how far the loop may run with no human in it. A
delivered message puts a human back in the loop, so it resets the budget. The cap is measured from
the round the last message landed in, not from the raw round count.

Without the reset, a long-running delegate would spend its whole messageable window on the rounds
before you said anything, and a late message could still be cut off by the forced wrap-up moments
later. A run that is never messaged has its cap unaffected.

The reset is capped at one extension per permitted round. That is not tidiness: the reset is the
only thing standing between a run and an unbounded bill, and a drain that returns its whole list
instead of the unread slice is a one-character mistake that makes every round look like a fresh
human message. Past the cap the loop logs once:

```
The inbox has extended this run's round budget 10 times; refusing further extensions.
A drain returning already-delivered messages instead of the unread slice does this.
```

The count is the run's `max_iterations` (10 by default), since the allowance is one extension per
permitted round. Past that point the loop stops extending while still *delivering* the message, so
a host that genuinely messages this often gets a turn that ends sooner rather than one that drops
input. If you see that warning, your drain is not advancing its cursor.

## One reader per run, and the name it is opened under

`reader()` is called per run rather than once per `Inbox`, and the reason is worth knowing because
it constrains what a host can key on. Under sequential tool dispatch, a spawned sub-agent's loop
executes in the *same* asyncio task as the agent that spawned it, so a reader keyed by task would
let two runs consume each other's messages. Two runs of the same agent type would collide on a name
for the same reason.

The run's name is passed to `reader(agent)` so a host can route a message to one run instead of
broadcasting to every drain:

```python
class RoutedInbox:
    def __init__(self):
        self._per_agent = {}      # agent name -> list of pending messages

    def send(self, agent: str, text: str) -> None:
        self._per_agent.setdefault(agent, []).append(text)

    def reader(self, agent=None):
        def drain():
            pending = self._per_agent.pop(agent, [])
            return pending

        return drain
```

The name is passed **positionally**, so the parameter may be called anything, and a reader that
ignores the label may keep the `agent=None` default. What it may not do is refuse the argument: a
zero-argument `reader()` cannot be called at all, and that is checked at run start (see below).

**Name your agents if you intend to route.** `Agent` generates `agent-{id:06x}` when you pass no
`name=`, and a generated name is unguessable from outside the run that got it, so a host that never
names its agents ends up with runs it has a label for but cannot address.

## When an inbox misbehaves

Four failures are logged and the run continues with no reader, because an inbox that cannot deliver
should not end a run that is otherwise working:

- `reader()` raises.
- The drain raises.
- The drain returns the wrong shape. A bare `str` is **refused rather than iterated**: `str` is
  iterable, so an unchecked drain would accept `"stop"` and deliver it as four one-character
  messages, which is worse than a raise for looking like it worked.
- The drain never advances its cursor (the budget cap above).

One shape is refused instead, and refused *before* the run starts rather than ending one in
progress: a `reader` the loop cannot call. `isinstance` cannot catch it first, because a
`runtime_checkable` Protocol only checks that the method exists. So the loop's constructor rehearses
the real call and raises `TypeError` naming the class, the signature it found, and the fix:

```
...QueueInbox.reader() cannot be called the way an agent's loop calls it: reader(agent),
passing the name of the run opening the reader as one positional argument. Widen the signature
in the class body, `def reader(self, agent=None)`, ...
```

This one is an error rather than a log line because an inbox that never delivers is
indistinguishable from a user who never typed: it was the one failure here a host could not
discover. It fails open on a reader whose signature cannot be read (a `functools.partial` over a
C-implemented function), since an unreadable signature is no evidence of a wrong one.

## One mailbox for a roster of sub-agents

An `Inbox` reaches spawned sub-agents through the same two seams `compaction` uses. The factory
takes one, and a typed `agent_types` spec may carry its own:

```python
from aimu.tools import builtin
from aimu.tools.builtin import make_subagent_tool

spawn = make_subagent_tool(
    "ollama:qwen3:8b",
    inbox=mailbox,                                     # every spawned agent reads this
    agent_types={
        "researcher": {"system_message": "You research.", "tools": builtin.web},
        "writer": {"system_message": "You write.", "inbox": None},   # this one reads nothing
    },
)
```

Both tiers read the key by **membership**, not by `.get()`: an absent key inherits the factory's
source, and `"inbox": None` turns it off for one specialist. A `.get()` cannot tell those apart.

Each spawned run opens **its own** reader, so a caller's run and a delegate's run never share a
cursor even when both read the same `Inbox` object. The label a spawned run opens under is its own:
`subagent-<agent_type>` on the typed shape (which is what lets `RoutedInbox` above address one
specialist), and the fixed `subagent` on the generic shape, where there is no type to name. A nested
spawn (`max_depth > 1`) carries the same source to a grandchild, so one host mailbox reaches however
deep a roster's delegation goes.

A value that does not implement the protocol raises `ValueError` at factory-call time at both tiers.
Deferred, it would arrive from inside the child's loop as
`AttributeError: 'str' object has no attribute 'reader'`, which the parent turns into a *tool*
failure the parent model is asked to recover from, and a programmer error reported to a model has
the wrong reader.

## What deliberately does not cross this seam

The drain returns `list[str]`, and an `INBOX` chunk carries `{"text": str}`. No sender, no address,
no selector, no group, no roster, no receipt. Who a message came from, who may send one, and whether
delivery is acknowledged are your application's model, and a protocol carrying envelopes would make
every host inherit one host's addressing scheme. The agent name is the one thing that crosses, and
it is thin on purpose: AIMU attaches no meaning to the string beyond handing it back.

If you need routing richer than a name, put it on your side of `reader(agent)`, where you already
have the whole language of your own application available.

## See also

- Notebook [29 - An Agent's Inbox](https://github.com/saxman/aimu/blob/main/notebooks/29-agent-inbox.qmd):
  the runnable version of this page, including a threaded host and the async twin
- [Cancel a run](cancel-a-run.md): stop a run instead of redirecting it
- [Spawn sub-agents](spawn-subagents.md): the `"inbox"` spec key among the rest of a spawn roster
- [Stream phases](../reference/stream-phases.md): `INBOX` next to the other phases, and how it
  differs from `CONTINUING`
- [Observe a run](observe-a-run.md): the telemetry channel, for what the library did with a message
- [Build a personal assistant](build-personal-assistant.md): a host that already reads input
  concurrently with a run, which is what an inbox needs
