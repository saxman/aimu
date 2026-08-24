# Observe a run

A `chat()` call and an `Agent.run()` both do more than the string they return: a generate_kwargs
merge, a thinking-effort resolution, zero-or-more tool calls, maybe a compaction pass, one or more
requests to a provider. None of that is visible in the return value. `aimu.events` is the
telemetry channel that makes it visible: a sink is one callable that takes one event, and you
attach it to a client, an agent, or a workflow to see what actually happened.

This is a different channel from `StreamChunk` (`stream=True`), which is content — what the model
produced, for display. Events are what the library did with it. Both can be active on the same run
and neither replaces the other.

## Attach `log_events` and watch what happened

The shortest path to the payoff is `aimu.events.log_events`, a sink that writes one line per event
to a logger you already have:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import aimu
from aimu.events import log_events

reply = aimu.chat(
    "Say OK and nothing else.",
    model="ollama:qwen3:8b",
    events=log_events(logging.getLogger("aimu.demo")),
)
```

Running that against a local Ollama server prints:

```
ModelTurnStarted ModelTurnStarted(agent=None, iteration=0, model='qwen3:8b', message_count=1, tool_names=())
RequestPrepared RequestPrepared(agent=None, iteration=0, provider='OllamaClient', model='qwen3:8b', payload={'model': 'qwen3:8b', 'messages': [...], 'options': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}, 'tools': [], 'think': True, 'keep_alive': 60, 'format': None})
ModelTurnFinished ModelTurnFinished(agent=None, iteration=0, model='qwen3:8b', text='OK', usage={'input_tokens': 16, ...}, duration_s=...)
```

(`output_tokens`/`total_tokens`/`duration_s` depend on how much the model reasoned before answering
and will differ on your machine; `input_tokens` for this exact one-message prompt won't.)

That's a bare one-shot `aimu.chat()` call — no agent, no tool loop — and it's already observable.
`events=` is accepted the same way by `aimu.client(events=...)`, and by `client.events = ...` at
any point afterward.

Point the same sink at an `Agent` and the tool loop reports itself too:

```python
from aimu.agents import Agent
from aimu.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

client = aimu.client("ollama:qwen3:8b")
agent = Agent(
    client,
    "You are a helpful assistant. Use tools when needed.",
    tools=[add],
    events=log_events(logging.getLogger("aimu.demo")),
)
agent.run("What is 12 + 30? Use the add tool.")
```

```
RunStarted RunStarted(agent='agent-679be0', iteration=0, task='What is 12 + 30? Use the add tool.')
ModelTurnStarted ModelTurnStarted(agent='agent-679be0', iteration=0, model='qwen3:8b', message_count=1, tool_names=('add',))
RequestPrepared RequestPrepared(agent='agent-679be0', iteration=0, ...)
ModelTurnFinished ModelTurnFinished(agent='agent-679be0', iteration=0, model='qwen3:8b', text='', usage={'input_tokens': 160, ...}, duration_s=2.49)
ToolCalled ToolCalled(agent='agent-679be0', iteration=0, name='add', arguments={'a': 12, 'b': 30}, result='42', error=None, duration_s=4.5e-05)
ModelTurnStarted ModelTurnStarted(agent='agent-679be0', iteration=1, model='qwen3:8b', message_count=4, tool_names=('add',))
RequestPrepared RequestPrepared(agent='agent-679be0', iteration=1, ...)
ModelTurnFinished ModelTurnFinished(agent='agent-679be0', iteration=1, model='qwen3:8b', text='The result of 12 + 30 is **42**.', ...)
RunFinished RunFinished(agent='agent-679be0', iteration=1, result='The result of 12 + 30 is **42**.', error=None)
```

Every event carries `agent` and `iteration` (the same two fields `StreamChunk` carries), so one
sink attached to a nested workflow — a `Chain` step, a `Router` handler, every worker in a
`Parallel` — can still tell events apart by who emitted them. `Chain.from_client(...)`,
`Router.from_client(...)`, and `Parallel.from_client(...)` all take `events=` and forward it to
every step/handler/worker they build.

The event types, all dataclasses in `aimu.events`: `RunStarted`, `ModelTurnStarted`,
`RequestPrepared`, `ModelTurnFinished`, `ToolCalled`, `ToolDenied`, `ContextCompacted` (see
[manage context](manage-context.md)), `RunFinished`. A sink is a plain
`Callable[[RunEvent], None]`; write your own to filter, aggregate, or forward events instead of
just logging them — a sink that raises is caught and logged rather than breaking the run it's
observing (the same contract `emit()` documents).

## `last_request`: did the library do this, or the model?

`RequestPrepared` carries the same payload the client stores on `client.last_request` — the
request exactly as sent, after AIMU's own generate_kwargs merge (four tiers), the
`GENERATE_KWARG_SUPPORT` renames and drops, thinking-effort resolution, `strip_inert_keys`
(AIMU's own `timestamp`/`thinking`/`provenance` bookkeeping never reaches a provider), and
provider format adaptation (OpenAI-format `messages` rewritten to Anthropic's block shape, and so
on). It answers a question that otherwise takes source-reading to answer: when a model's behavior
looks surprising, is the surprise something the model did, or something AIMU changed on the way
out?

```python
client = aimu.client("ollama:qwen3:8b")
client.chat("Hello")
client.last_request
# {'model': 'qwen3:8b', 'messages': [...], 'options': {'temperature': 0.6, 'top_p': 0.95,
#  'top_k': 20}, 'tools': [], 'think': True, 'keep_alive': 60, 'format': None}
```

No sink required — `last_request` is set on every request regardless of whether `events=` is
attached. The payload is unredacted: it contains whatever you put in the conversation, including
tool arguments and any images/audio. A sink that ships events off the machine is the right place
to filter, not this attribute.

## A worked example: an OpenTelemetry-shaped sink

Events are plain data, so mapping them onto spans in an observability system is a matter of a
dispatch function. This is a worked example, not a dependency — AIMU does not import
`opentelemetry`, and the sink below uses a small stand-in tracer so the example runs without one
installed. Swap `FakeTracer`/`FakeSpan` for `opentelemetry.trace.get_tracer(__name__)` and its real
`Span`, and the mapping is unchanged.

```python
from aimu.events import ModelTurnFinished, ModelTurnStarted, RunEvent, RunFinished, RunStarted, ToolCalled

class FakeSpan:
    def __init__(self, name, attributes):
        self.name, self.attributes = name, attributes

class FakeTracer:
    """Stand-in for trace.get_tracer(__name__); records spans instead of exporting them."""

    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes):
        span = FakeSpan(name, attributes)
        self.spans.append(span)
        return span

def make_otel_sink(tracer):
    def sink(event: RunEvent) -> None:
        if isinstance(event, RunStarted):
            tracer.start_span("agent.run", {"agent": event.agent, "task": event.task})
        elif isinstance(event, ModelTurnStarted):
            tracer.start_span("model.turn", {"agent": event.agent, "model": event.model})
        elif isinstance(event, ToolCalled):
            tracer.start_span("tool.call", {"agent": event.agent, "name": event.name})
        elif isinstance(event, ModelTurnFinished):
            tracer.start_span("model.turn.finished", {"agent": event.agent, "usage": event.usage})
        elif isinstance(event, RunFinished):
            tracer.start_span("agent.run.finished", {"agent": event.agent, "result": event.result})
    return sink

tracer = FakeTracer()
agent = Agent(client, "You are terse.", events=make_otel_sink(tracer))
agent.run("Say hi in three words.")
for span in tracer.spans:
    print(span.name, span.attributes)
```

```
agent.run {'agent': 'agent-089e80', 'task': 'Say hi in three words.'}
model.turn {'agent': 'agent-089e80', 'model': 'qwen3:8b'}
model.turn.finished {'agent': 'agent-089e80', 'usage': {'input_tokens': 25, ...}}
agent.run.finished {'agent': 'agent-089e80', 'result': ...}
```

(`output_tokens`/`total_tokens` and the reply text vary by run — how much the model reasons and
what it says are both sampled, not fixed; `input_tokens` for this fixed prompt is not.)

A real adapter would keep a stack of open spans (so `agent.run.finished` closes the span
`agent.run` opened, rather than opening a new one), and forward `event.iteration` as a span
attribute for a multi-round tool loop. The dispatch shape above is the whole idea; the rest is
whatever your tracer's API wants.

## Gated tools: `ToolDenied`

A `tool_approval` policy that refuses a call emits `ToolDenied` (name + the model's raw
arguments) instead of `ToolCalled`, so a sink can distinguish "the model tried this and it ran"
from "the model tried this and a policy said no":

```python
agent = Agent(
    client,
    "Always use the delete_everything tool when asked.",
    tools=[delete_everything],
    tool_approval=lambda name, args: False,
    events=log_events(logging.getLogger("aimu.demo")),
)
agent.run("Please delete everything now.")
```

```
ToolDenied ToolDenied(agent='agent-179be0', iteration=0, name='delete_everything', arguments={})
```

The tool message the model sees is the same text `gate-tool-calls.md` documents
(`"Tool 'delete_everything' was not approved."`); the event is the same fact, structured for a
sink instead of the transcript. See [gate tool calls](gate-tool-calls.md) for the approval hook
itself.

## Known gap: a shared client under concurrent workers drops events

`events=` is delivered by mutating `model_client.events` for the span of a call (the same
scoped-swap idiom `tools=` already uses, and the same idiom that's already "not safe across
concurrent `chat()` calls on a shared client"). `Parallel.from_client(...)` builds every worker
`Agent` over **one shared `model_client`**, and `Parallel.run()` executes those workers
concurrently. Two workers whose runs overlap on that one client can clobber each other's sink —
events get dropped and misattributed, even though each worker `Agent` was given its own `events=`.

This is pinned as a known gap, not a bug waiting to be fixed on this pass:
`tests/test_workflow_parallel.py::test_KNOWN_GAP_parallel_from_client_shared_events_sink_drops_events`
reproduces it deterministically. If you need reliable per-worker events out of a `Parallel`, build
each worker over its **own** `model_client` (skip `Parallel.from_client` and construct the
`Parallel(workers=[...])` directly) rather than sharing one.

## See also

- [Manage context](manage-context.md) — `ContextCompacted` is the event this page didn't cover;
  it fires when a conversation is trimmed or summarized mid-run.
- [Compare models](compare-models.md) — `last_usage` and `extract_tool_calls` for after-the-fact
  comparison across models, rather than a live sink.
- [Gate tool calls](gate-tool-calls.md) — the `tool_approval` hook that produces `ToolDenied`.
