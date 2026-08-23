"""
Tests for aimu.agents.Parallel: the Parallelization workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

from aimu.agents import Agent, Parallel, Runner
from aimu.models import StreamChunk, StreamingContentType
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Parallel tests
# ---------------------------------------------------------------------------


def test_parallel_all_workers_receive_same_task():
    """All workers receive the same original task string."""
    client_a = MockModelClient(["perspective A"])
    client_b = MockModelClient(["perspective B"])

    parallel = Parallel(
        workers=[
            Agent(client_a, name="worker-a"),
            Agent(client_b, name="worker-b"),
        ]
    )
    parallel.run("What is the meaning of life?")

    user_msgs_a = [m["content"] for m in client_a.messages if m["role"] == "user"]
    user_msgs_b = [m["content"] for m in client_b.messages if m["role"] == "user"]
    assert user_msgs_a == ["What is the meaning of life?"]
    assert user_msgs_b == ["What is the meaning of life?"]


def test_parallel_no_aggregator_joins_results():
    """Without an aggregator, results are joined with the separator."""
    client_a = MockModelClient(["part A"])
    client_b = MockModelClient(["part B"])

    parallel = Parallel(
        workers=[
            Agent(client_a, name="worker-a"),
            Agent(client_b, name="worker-b"),
        ],
        separator=" | ",
    )
    result = parallel.run("task")
    assert result == "part A | part B"


def test_parallel_aggregator_receives_combined_results():
    """When an aggregator is set, it receives all worker outputs joined."""
    client_a = MockModelClient(["view A"])
    client_b = MockModelClient(["view B"])
    agg_client = MockModelClient(["synthesized"])

    parallel = Parallel(
        workers=[
            Agent(client_a, name="worker-a"),
            Agent(client_b, name="worker-b"),
        ],
        aggregator=Agent(agg_client, name="synthesizer"),
        separator="\n---\n",
    )
    result = parallel.run("task")

    assert result == "synthesized"
    # Aggregator received the combined worker output
    agg_user_msgs = [m["content"] for m in agg_client.messages if m["role"] == "user"]
    assert agg_user_msgs[0] == "view A\n---\nview B"


def test_parallel_single_worker():
    """Works correctly with a single worker and no aggregator."""
    client = MockModelClient(["solo result"])
    parallel = Parallel(workers=[Agent(client, name="solo")])
    assert parallel.run("task") == "solo result"


def test_parallel_streamed_no_aggregator_yields_combined():
    """run(stream=True) without aggregator yields a single GENERATING chunk."""
    client_a = MockModelClient(["A"])
    client_b = MockModelClient(["B"])

    parallel = Parallel(
        workers=[
            Agent(client_a, name="worker-a"),
            Agent(client_b, name="worker-b"),
        ],
        separator=" | ",
    )
    chunks = list(parallel.run("task", stream=True))

    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamChunk)
    assert chunks[0].phase == StreamingContentType.GENERATING
    assert chunks[0].content == "A | B"


def test_parallel_streamed_with_aggregator_delegates_stream():
    """run(stream=True) with aggregator yields the aggregator's chunks."""
    client_a = MockModelClient(["A"])
    client_b = MockModelClient(["B"])
    agg_client = MockModelClient(["synthesized"])

    parallel = Parallel(
        workers=[
            Agent(client_a, name="worker-a"),
            Agent(client_b, name="worker-b"),
        ],
        aggregator=Agent(agg_client, name="synthesizer"),
    )
    chunks = list(parallel.run("task", stream=True))

    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert any(c.content == "synthesized" for c in generating)


def test_parallel_is_runner_subclass():
    """Parallel implements the single Runner interface."""
    client = MockModelClient(["hi"])
    parallel = Parallel(workers=[Agent(client, name="w")])
    assert isinstance(parallel, Runner)


def test_parallel_from_client_builds_workers_with_prompts():
    """Parallel.from_client(client, worker_prompts) wires each prompt to its own worker."""
    client = MockModelClient(["a-out", "b-out"])
    parallel = Parallel.from_client(client, worker_prompts=["Prompt A.", "Prompt B."])
    assert len(parallel.workers) == 2
    assert parallel.aggregator is None
    result = parallel.run("topic")
    assert "a-out" in result
    assert "b-out" in result


def test_parallel_from_client_with_aggregator_prompt():
    """Aggregator prompt builds an aggregator agent that consumes worker outputs."""
    client = MockModelClient(["a-out", "b-out", "synthesized"])
    parallel = Parallel.from_client(
        client,
        worker_prompts=["A.", "B."],
        aggregator_prompt="Synthesize.",
    )
    assert parallel.aggregator is not None
    result = parallel.run("topic")
    assert result == "synthesized"


def test_KNOWN_GAP_parallel_from_client_shared_events_sink_drops_events():
    """PINS A KNOWN GAP -- documents current (broken) behaviour, not a desired contract.
    See the concurrency caveat on ``Agent.events`` / ``BaseModelClient.events``.

    ``Parallel.from_client`` builds every worker ``Agent`` over one shared ``model_client``
    (see ``from_client`` above) and, in real use, ``Parallel.run()`` executes them
    concurrently via a ``ThreadPoolExecutor``. ``_events_override`` delivers a sink by
    mutating ``client.events`` -- the identical shared-mutable-state idiom
    ``_tools_override`` already carries for ``tools=`` and already documents as "not safe
    across concurrent chat() calls on a shared client". Two workers whose runs overlap on
    that one client can clobber each other's sink, so a caller who attaches ``events=`` to
    each worker Agent still loses and misorders events.

    Reproduced deterministically here by manually interleaving two ``_events_override``
    scopes on one client in the exact order concurrent workers can land in -- not via real
    threads, so this is not a timing-dependent/flaky test, just a pin of the mechanism.
    Delete this test (don't "fix" it in place) the day sink delivery stops being a mutation
    of shared client state -- see the coordinator's design note on why that fix is deferred.
    """
    from aimu.events import RunFinished, emit

    client = MockModelClient([])
    seen_a: list = []
    seen_b: list = []
    sink_a = seen_a.append
    sink_b = seen_b.append

    # worker-a's Agent.run() enters its _events_override scope first...
    scope_a = client._events_override(sink_a)
    scope_b = client._events_override(sink_b)
    scope_a.__enter__()
    assert client.events is sink_a

    # ...but before worker-a's run finishes, worker-b's Agent.run() (a different thread, same
    # shared client) enters its own scope and clobbers the live sink.
    scope_b.__enter__()
    assert client.events is sink_b

    emit(client.events, RunFinished(result="worker-b's own turn"))
    assert seen_b == [RunFinished(result="worker-b's own turn")]

    # worker-a finishes first and restores *its own* saved value (None, what it saw on
    # entry) -- not worker-b's, because _events_override has no idea another scope is
    # still open on the same client.
    scope_a.__exit__(None, None, None)
    assert client.events is None  # <- the gap: worker-b's scope is still open, sink is gone

    # Any event worker-b's still-running turn tries to report now silently drops (this is
    # the coordinator's reproduction: 11 of an expected 12 events observed).
    emit(client.events, RunFinished(result="dropped"))
    assert seen_b == [RunFinished(result="worker-b's own turn")]  # the second event never arrived

    # worker-b finishes and restores *its* saved value -- worker-a's sink, not the true
    # original -- leaving the client's events attribute wrong even after both runs "finished".
    scope_b.__exit__(None, None, None)
    assert client.events is sink_a  # <- also wrong: not the pre-run None
