"""
Tests for aimu.agents.Parallel — the Parallelization workflow pattern.

All tests use MockModelClient from conftest (deterministic, no backend needed).
"""

from aimu.agents import AgentChunk, Runner, SimpleAgent, Workflow
from aimu.agents.parallel import Parallel
from aimu.models.base_client import StreamingContentType
from conftest import MockModelClient


# ---------------------------------------------------------------------------
# Parallel tests
# ---------------------------------------------------------------------------


def test_parallel_all_workers_receive_same_task():
    """All workers receive the same original task string."""
    client_a = MockModelClient(["perspective A"])
    client_b = MockModelClient(["perspective B"])

    parallel = Parallel(
        workers=[
            SimpleAgent(client_a, name="worker-a"),
            SimpleAgent(client_b, name="worker-b"),
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
            SimpleAgent(client_a, name="worker-a"),
            SimpleAgent(client_b, name="worker-b"),
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
            SimpleAgent(client_a, name="worker-a"),
            SimpleAgent(client_b, name="worker-b"),
        ],
        aggregator=SimpleAgent(agg_client, name="synthesizer"),
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
    parallel = Parallel(workers=[SimpleAgent(client, name="solo")])
    assert parallel.run("task") == "solo result"


def test_parallel_streamed_no_aggregator_yields_combined():
    """run_streamed() without aggregator yields a single GENERATING chunk."""
    client_a = MockModelClient(["A"])
    client_b = MockModelClient(["B"])

    parallel = Parallel(
        workers=[
            SimpleAgent(client_a, name="worker-a"),
            SimpleAgent(client_b, name="worker-b"),
        ],
        separator=" | ",
    )
    chunks = list(parallel.run_streamed("task"))

    assert len(chunks) == 1
    assert isinstance(chunks[0], AgentChunk)
    assert chunks[0].phase == StreamingContentType.GENERATING
    assert chunks[0].content == "A | B"


def test_parallel_streamed_with_aggregator_delegates_stream():
    """run_streamed() with aggregator yields the aggregator's chunks."""
    client_a = MockModelClient(["A"])
    client_b = MockModelClient(["B"])
    agg_client = MockModelClient(["synthesized"])

    parallel = Parallel(
        workers=[
            SimpleAgent(client_a, name="worker-a"),
            SimpleAgent(client_b, name="worker-b"),
        ],
        aggregator=SimpleAgent(agg_client, name="synthesizer"),
    )
    chunks = list(parallel.run_streamed("task"))

    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert any(c.content == "synthesized" for c in generating)


def test_parallel_is_workflow_subclass():
    """Parallel must be a Workflow (not an Agent) and also a Runner."""
    client = MockModelClient(["hi"])
    parallel = Parallel(workers=[SimpleAgent(client, name="w")])

    from aimu.agents import Agent

    assert isinstance(parallel, Workflow)
    assert isinstance(parallel, Runner)
    assert not isinstance(parallel, Agent)
