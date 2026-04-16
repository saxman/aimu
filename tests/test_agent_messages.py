"""
Tests for the .messages property across the Runner hierarchy.

Covers SimpleAgent, Chain, Router, Parallel, and EvaluatorOptimizer — including the
shared-client scenario that surfaced in the notebook, where a single ModelClient is
reused across all agents in a Router and each agent's _prepare_run() wipes the client's
messages before it runs.  Without snapshotting, every key in router.messages would alias
the same live list (the last agent's messages).  The snapshot taken at the end of
SimpleAgent.run() / run_streamed() isolates each agent's history.
"""

import pytest

from aimu.agents import Chain, EvaluatorOptimizer, MessageHistory, Parallel, Router, SimpleAgent
from conftest import MockModelClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_contents(msgs: list[dict]) -> list[str]:
    return [m["content"] for m in msgs if m["role"] == "user"]


def _assistant_contents(msgs: list[dict]) -> list[str]:
    return [m["content"] for m in msgs if m["role"] == "assistant"]


# ---------------------------------------------------------------------------
# SimpleAgent.messages
# ---------------------------------------------------------------------------


def test_simple_agent_messages_empty_before_run():
    """messages is empty before any run."""
    agent = SimpleAgent(MockModelClient(["hi"]), name="a")
    assert agent.messages == {"a": []}


def test_simple_agent_messages_keyed_by_name():
    """messages dict key matches the agent name."""
    agent = SimpleAgent(MockModelClient(["x"]), name="my-agent")
    agent.run("task")
    assert "my-agent" in agent.messages


def test_simple_agent_messages_contains_exchange():
    """messages contains both the user turn and the assistant reply."""
    client = MockModelClient(["hello back"])
    agent = SimpleAgent(client, name="greeter")
    agent.run("say hi")
    msgs = agent.messages["greeter"]
    assert any(m["role"] == "user" for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)


def test_simple_agent_messages_snapshot_survives_client_clear():
    """Snapshot is preserved even after model_client.messages is cleared externally."""
    client = MockModelClient(["response"])
    agent = SimpleAgent(client, name="a")
    agent.run("task")
    client.messages = []
    assert len(agent.messages["a"]) > 0


def test_simple_agent_messages_updated_after_second_run():
    """Snapshot is replaced by the most recent run."""
    client = MockModelClient(["first", "second"])
    agent = SimpleAgent(client, name="a")
    agent.run("run 1")
    first_snapshot = list(agent.messages["a"])
    agent.run("run 2")
    second_snapshot = agent.messages["a"]
    assert second_snapshot != first_snapshot


def test_simple_agent_messages_streamed():
    """Snapshot is also taken after run_streamed() is fully consumed."""
    client = MockModelClient(["streamed response"])
    agent = SimpleAgent(client, name="streamer")
    list(agent.run_streamed("task"))  # consume the generator
    msgs = agent.messages["streamer"]
    assert any(m["role"] == "assistant" for m in msgs)


# ---------------------------------------------------------------------------
# Chain.messages
# ---------------------------------------------------------------------------


def test_chain_messages_keyed_by_step_names():
    """messages keys match each step agent's name."""
    chain = Chain(
        agents=[
            SimpleAgent(MockModelClient(["step-a"]), name="step-a"),
            SimpleAgent(MockModelClient(["step-b"]), name="step-b"),
        ]
    )
    chain.run("go")
    assert set(chain.messages.keys()) == {"step-a", "step-b"}


def test_chain_messages_step_inputs_are_correct():
    """First step receives the original task; second step receives the first step's output."""
    chain = Chain(
        agents=[
            SimpleAgent(MockModelClient(["intermediate"]), name="first"),
            SimpleAgent(MockModelClient(["final"]), name="second"),
        ]
    )
    chain.run("original task")
    assert _user_contents(chain.messages["first"]) == ["original task"]
    assert _user_contents(chain.messages["second"]) == ["intermediate"]


def test_chain_messages_steps_are_independent_lists():
    """Each step returns its own list, not a shared reference."""
    chain = Chain(
        agents=[
            SimpleAgent(MockModelClient(["a"]), name="step-a"),
            SimpleAgent(MockModelClient(["b"]), name="step-b"),
        ]
    )
    chain.run("go")
    assert chain.messages["step-a"] is not chain.messages["step-b"]


def test_chain_messages_returns_message_history_type():
    """Return type is dict[str, list[dict]] (MessageHistory)."""
    chain = Chain(agents=[SimpleAgent(MockModelClient(["x"]), name="only")])
    chain.run("go")
    m = chain.messages
    assert isinstance(m, dict)
    assert all(isinstance(v, list) for v in m.values())


# ---------------------------------------------------------------------------
# Router.messages
# ---------------------------------------------------------------------------


def test_router_messages_includes_routing_agent():
    """messages always contains the routing agent's history."""
    router = Router(
        routing_agent=SimpleAgent(MockModelClient(["code"]), name="classifier"),
        handlers={"code": SimpleAgent(MockModelClient(["done"]), name="coder")},
    )
    router.run("task")
    assert "classifier" in router.messages


def test_router_messages_includes_dispatched_handler():
    """The dispatched handler's messages are non-empty after a run."""
    router = Router(
        routing_agent=SimpleAgent(MockModelClient(["math"]), name="classifier"),
        handlers={
            "math": SimpleAgent(MockModelClient(["42"]), name="mathematician"),
            "writing": SimpleAgent(MockModelClient([]), name="writer"),
        },
    )
    router.run("What is 6 * 7?")
    assert len(router.messages["mathematician"]) > 0


def test_router_messages_includes_all_handler_keys():
    """All handler keys appear even when they were not dispatched."""
    router = Router(
        routing_agent=SimpleAgent(MockModelClient(["a"]), name="classifier"),
        handlers={
            "a": SimpleAgent(MockModelClient(["result-a"]), name="handler-a"),
            "b": SimpleAgent(MockModelClient([]), name="handler-b"),
        },
    )
    router.run("task")
    assert "handler-a" in router.messages
    assert "handler-b" in router.messages


def test_router_messages_unvisited_handler_is_empty():
    """An unvisited handler's messages list is empty."""
    router = Router(
        routing_agent=SimpleAgent(MockModelClient(["writing"]), name="classifier"),
        handlers={
            "writing": SimpleAgent(MockModelClient(["a poem"]), name="writer"),
            "code": SimpleAgent(MockModelClient([]), name="coder"),
        },
    )
    router.run("write a poem")
    assert len(router.messages["coder"]) == 0


def test_router_messages_includes_fallback():
    """messages includes the fallback runner when one is configured."""
    router = Router(
        routing_agent=SimpleAgent(MockModelClient(["unknown"]), name="classifier"),
        handlers={"code": SimpleAgent(MockModelClient([]), name="coder")},
        fallback=SimpleAgent(MockModelClient(["fallback answer"]), name="fallback"),
    )
    router.run("task")
    assert "fallback" in router.messages


def test_router_messages_shared_client_isolation():
    """
    Regression: all agents share one ModelClient (the notebook pattern).

    Before the snapshot fix every key in router.messages aliased the same list
    (whichever agent ran last). The fix snapshots model_client.messages at the
    end of SimpleAgent.run(), giving each agent its own independent copy.
    """
    shared_client = MockModelClient(["math", "42"])
    router = Router(
        routing_agent=SimpleAgent(shared_client, name="classifier", system_message="classify"),
        handlers={"math": SimpleAgent(shared_client, name="mathematician", system_message="calculate")},
    )
    router.run("What is 6 * 7?")

    classifier_msgs = router.messages["classifier"]
    mathematician_msgs = router.messages["mathematician"]

    # Each agent has its own snapshot — not the same list object
    assert classifier_msgs is not mathematician_msgs

    # Classifier recorded the routing exchange
    assert len(classifier_msgs) > 0
    assert any(m["role"] == "assistant" for m in classifier_msgs)

    # Mathematician recorded the math exchange
    assert len(mathematician_msgs) > 0
    assert any(m["role"] == "assistant" for m in mathematician_msgs)


# ---------------------------------------------------------------------------
# Parallel.messages
# ---------------------------------------------------------------------------


def test_parallel_messages_includes_all_workers():
    """messages contains every worker name."""
    parallel = Parallel(
        workers=[
            SimpleAgent(MockModelClient(["view-a"]), name="worker-a"),
            SimpleAgent(MockModelClient(["view-b"]), name="worker-b"),
        ]
    )
    parallel.run("task")
    assert "worker-a" in parallel.messages
    assert "worker-b" in parallel.messages


def test_parallel_messages_includes_aggregator():
    """messages contains the aggregator name when an aggregator is set."""
    parallel = Parallel(
        workers=[SimpleAgent(MockModelClient(["view"]), name="worker")],
        aggregator=SimpleAgent(MockModelClient(["summary"]), name="aggregator"),
    )
    parallel.run("task")
    assert "aggregator" in parallel.messages


def test_parallel_messages_without_aggregator_has_no_aggregator_key():
    """When no aggregator is set, only worker keys appear."""
    parallel = Parallel(
        workers=[
            SimpleAgent(MockModelClient(["x"]), name="solo"),
        ]
    )
    parallel.run("task")
    assert set(parallel.messages.keys()) == {"solo"}


def test_parallel_workers_have_distinct_messages():
    """Each worker has its own message list with its own response."""
    parallel = Parallel(
        workers=[
            SimpleAgent(MockModelClient(["view-a"]), name="worker-a"),
            SimpleAgent(MockModelClient(["view-b"]), name="worker-b"),
        ]
    )
    parallel.run("task")
    msgs_a = parallel.messages["worker-a"]
    msgs_b = parallel.messages["worker-b"]
    assert msgs_a is not msgs_b
    assert any(m["content"] == "view-a" for m in msgs_a if m["role"] == "assistant")
    assert any(m["content"] == "view-b" for m in msgs_b if m["role"] == "assistant")


# ---------------------------------------------------------------------------
# EvaluatorOptimizer.messages
# ---------------------------------------------------------------------------


def test_eo_messages_includes_generator_and_evaluator():
    """messages contains both generator and evaluator keys."""
    eo = EvaluatorOptimizer(
        generator=SimpleAgent(MockModelClient(["draft"]), name="writer"),
        evaluator=SimpleAgent(MockModelClient(["PASS"]), name="critic"),
    )
    eo.run("task")
    assert "writer" in eo.messages
    assert "critic" in eo.messages


def test_eo_messages_generator_has_exchange():
    """Generator messages include at least one user-assistant pair."""
    eo = EvaluatorOptimizer(
        generator=SimpleAgent(MockModelClient(["draft"]), name="writer"),
        evaluator=SimpleAgent(MockModelClient(["PASS"]), name="critic"),
    )
    eo.run("Explain X.")
    writer_msgs = eo.messages["writer"]
    assert any(m["role"] == "user" for m in writer_msgs)
    assert any(m["role"] == "assistant" for m in writer_msgs)


def test_eo_messages_after_revision_round():
    """After a revision, both generator and evaluator have non-empty histories."""
    eo = EvaluatorOptimizer(
        generator=SimpleAgent(MockModelClient(["draft-1", "draft-2"]), name="writer"),
        evaluator=SimpleAgent(MockModelClient(["REVISE: more detail", "PASS"]), name="critic"),
        max_rounds=3,
    )
    eo.run("Explain gradient descent.")
    assert len(eo.messages["writer"]) > 0
    assert len(eo.messages["critic"]) > 0


# ---------------------------------------------------------------------------
# Nested workflow: Router dispatching to a Chain
# ---------------------------------------------------------------------------


def test_nested_router_of_chain_deep_merge():
    """messages from a Router whose handler is a Chain merges all sub-agent names."""
    inner_chain = Chain(
        agents=[
            SimpleAgent(MockModelClient(["chain-step-1"]), name="chain-a"),
            SimpleAgent(MockModelClient(["chain-step-2"]), name="chain-b"),
        ]
    )
    router = Router(
        routing_agent=SimpleAgent(MockModelClient(["chain-route"]), name="classifier"),
        handlers={"chain-route": inner_chain},
    )
    router.run("task")
    keys = set(router.messages.keys())
    assert "classifier" in keys
    assert "chain-a" in keys
    assert "chain-b" in keys
