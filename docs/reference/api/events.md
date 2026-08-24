# `aimu.events`

Run events: a frozen-dataclass event union plus one `Callable[[RunEvent], None]` sink. Attach a
sink to a client, an `Agent`, or a workflow's `from_client(...)` to see what a run actually did.
See [how-to: observe a run](../../how-to/observe-a-run.md).

## Sink and dispatch

::: aimu.events.EventSink

::: aimu.events.emit

::: aimu.events.log_events

## Events

::: aimu.events.RunEvent

::: aimu.events.RunStarted

::: aimu.events.ModelTurnStarted

::: aimu.events.RequestPrepared

::: aimu.events.ModelTurnFinished

::: aimu.events.ToolCalled

::: aimu.events.ToolDenied

::: aimu.events.ContextCompacted

::: aimu.events.RunFinished
