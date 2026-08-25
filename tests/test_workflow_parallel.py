"""
Tests for aimu.agents.Parallel: the Parallelization workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

import threading
import time

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


class _RacingMockModelClient(MockModelClient):
    """A ``MockModelClient`` whose ``_chat()`` forces genuine cross-thread overlap between
    concurrent ``_events_override`` scopes on one shared client, instead of relying on
    incidental OS thread-scheduling timing (which would make a real-concurrency test flaky).

    A lock-protected counter assigns each call an arrival index the moment it reaches
    ``_chat()`` (before any sleep, so this reflects the order calls actually entered their
    ``_events_override`` scope). A ``threading.Barrier`` then holds every caller until *all*
    of them have arrived -- guaranteeing every worker's scope is open before any of them can
    close -- and each caller then sleeps for a duration keyed by its arrival index, chosen so
    the scopes close in a *different* order than they opened: a "crossing" (non-nested)
    overlap, the exact pattern that corrupts a plain shared-attribute swap (nested
    enter/exit restores correctly; crossing enter/exit does not).
    """

    def __init__(self, responses: list, *, barrier: threading.Barrier, delays: dict[int, float]):
        super().__init__(responses)
        self._barrier = barrier
        self._delays = delays
        self._arrival_lock = threading.Lock()
        self._next_arrival = 0

    def _chat(self, *args, **kwargs):
        with self._arrival_lock:
            arrival = self._next_arrival
            self._next_arrival += 1
        self._barrier.wait(timeout=5)
        time.sleep(self._delays[arrival])
        return super()._chat(*args, **kwargs)


def test_parallel_from_client_shared_events_sink_survives_concurrent_workers():
    """Isolation now holds for a sink shared across concurrent workers on one client.

    Replaces ``test_KNOWN_GAP_parallel_from_client_shared_events_sink_drops_events``
    (removed), which pinned the bug this test now proves fixed: delivering a scoped event
    sink via a ``contextvars.ContextVar`` (``aimu.models._internal.chat_state
    ._ACTIVE_EVENT_SINK`` / ``_effective_sink``) rather than a mutation of the client's own
    ``self.events`` attribute. See the (corrected) concurrency notes on ``Agent.events`` /
    ``BaseModelClient.events``.

    ``Parallel.from_client`` builds every worker ``Agent`` over one shared ``model_client``,
    and ``Parallel.run()`` really does execute them concurrently via ``ThreadPoolExecutor``
    (``Parallel._run_workers``) -- this test exercises that real path, not a hand-simulated
    interleaving, using ``_RacingMockModelClient`` above to force the three workers'
    ``_events_override`` scopes to overlap in a crossing (non-nested) pattern on separate OS
    threads. Every worker's ``RunStarted`` / ``ModelTurnStarted`` / ``ModelTurnFinished`` /
    ``RunFinished`` must still arrive, attributed to the right worker, in causal order --
    under the old attribute-swap mechanism this reliably reproduced misattribution and
    out-of-order delivery (a worker's ``RunFinished`` landing before its own
    ``ModelTurnFinished``, matching the bug reported during v0.21.0's review).
    """
    barrier = threading.Barrier(3)
    # Arrival order need not match worker order (whichever OS thread reaches _chat() first
    # gets arrival 0); what matters is that the scopes close out of the order they opened.
    delays = {0: 0.15, 1: 0.05, 2: 0.10}
    client = _RacingMockModelClient(["A", "B", "C"], barrier=barrier, delays=delays)

    seen: list = []
    parallel = Parallel.from_client(
        client,
        worker_prompts=["Do A.", "Do B.", "Do C."],
        events=seen.append,
    )
    parallel.run("task")

    by_agent: dict[str, list[str]] = {}
    for event in seen:
        by_agent.setdefault(event.agent, []).append(type(event).__name__)

    assert set(by_agent) == {"worker-0", "worker-1", "worker-2"}, by_agent
    expected_sequence = ["RunStarted", "ModelTurnStarted", "ModelTurnFinished", "RunFinished"]
    for name, sequence in by_agent.items():
        assert sequence == expected_sequence, f"{name}: {sequence}"
