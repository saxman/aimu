"""Tests for restore() across runners: Agent, Chain, EvaluatorOptimizer, Router, Parallel, OrchestratorAgent."""

from __future__ import annotations

import pytest

from helpers import MockModelClient
from aimu.agents import Agent, OrchestratorAgent, Parallel, Router
from aimu.agents.workflows.evaluator import EvaluatorOptimizer
from aimu.agents.workflows.chain import Chain


def _agent(system="You are helpful."):
    return Agent(MockModelClient([]), system_message=system, name="test-agent")


def _messages_with_system(system="You are helpful."):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


# ---------------------------------------------------------------------------
# Agent.restore
# ---------------------------------------------------------------------------


def test_restore_strips_leading_system_message():
    agent = _agent(system="My system msg")
    agent.restore(_messages_with_system(system="My system msg"))
    roles = [m["role"] for m in agent.model_client.messages]
    assert "system" not in roles


def test_restore_preserves_conversation_content():
    agent = _agent()
    agent.restore(_messages_with_system())
    contents = [m["content"] for m in agent.model_client.messages]
    assert "Hello" in contents
    assert "Hi there" in contents


def test_restore_without_system_message_in_saved():
    agent = _agent(system="Some system")
    saved = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    agent.restore(saved)
    assert len(agent.model_client.messages) == 2


def test_restore_leaves_system_message_settable():
    client = MockModelClient([])
    agent = Agent(client, system_message="Original", name="a")
    agent.restore(_messages_with_system(system="Original"))
    # After restore, system_message can be reassigned (no immutability lock).
    client.system_message = "Updated"
    assert client.system_message == "Updated"


def test_restore_empty_messages():
    agent = _agent()
    agent.restore([])
    assert agent.model_client.messages == []


# ---------------------------------------------------------------------------
# EvaluatorOptimizer.restore
# ---------------------------------------------------------------------------


def test_evaluator_optimizer_restore_delegates_to_generator():
    gen_client = MockModelClient([])
    eval_client = MockModelClient([])
    generator = Agent(gen_client, system_message="Generate", name="gen")
    evaluator = Agent(eval_client, system_message="Evaluate", name="eval")
    eo = EvaluatorOptimizer(generator=generator, evaluator=evaluator, pass_keyword="PASS")

    eo.restore(_messages_with_system(system="Generate"))

    # generator's client should have restored messages (minus system)
    contents = [m["content"] for m in gen_client.messages]
    assert "Hello" in contents
    # evaluator's client untouched
    assert eval_client.messages == []


# ---------------------------------------------------------------------------
# Chain.restore
# ---------------------------------------------------------------------------


def test_chain_restore_default_step_zero():
    c1 = MockModelClient([])
    c2 = MockModelClient([])
    agent1 = Agent(c1, system_message="Step 1", name="s1")
    agent2 = Agent(c2, system_message="Step 2", name="s2")
    chain = Chain(agents=[agent1, agent2])

    chain.restore(_messages_with_system(system="Step 1"))

    assert "Hello" in [m["content"] for m in c1.messages]
    assert c2.messages == []  # step 2 untouched


def test_chain_restore_specific_step():
    c1 = MockModelClient([])
    c2 = MockModelClient([])
    agent1 = Agent(c1, system_message="Step 1", name="s1")
    agent2 = Agent(c2, system_message="Step 2", name="s2")
    chain = Chain(agents=[agent1, agent2])

    chain.restore(_messages_with_system(system="Step 2"), step=1)

    assert c1.messages == []  # step 0 untouched
    assert "Hello" in [m["content"] for m in c2.messages]


# ---------------------------------------------------------------------------
# Router.restore
# ---------------------------------------------------------------------------


def test_router_restore_handler_by_route():
    rc = MockModelClient([])
    hc = MockModelClient([])
    routing = Agent(rc, system_message="Classify", name="rc")
    handler = Agent(hc, system_message="Handle", name="coder")
    router = Router(routing_agent=routing, handlers={"coder": handler})

    router.restore(_messages_with_system(system="Handle"), route="coder")

    assert "Hello" in [m["content"] for m in hc.messages]
    assert rc.messages == []  # routing agent untouched


def test_router_restore_defaults_to_routing_agent():
    rc = MockModelClient([])
    hc = MockModelClient([])
    routing = Agent(rc, system_message="Classify", name="rc")
    handler = Agent(hc, system_message="Handle", name="coder")
    router = Router(routing_agent=routing, handlers={"coder": handler})

    router.restore(_messages_with_system(system="Classify"))

    assert "Hello" in [m["content"] for m in rc.messages]
    assert hc.messages == []


def test_router_restore_unknown_route_raises_keyerror():
    routing = Agent(MockModelClient([]), system_message="Classify", name="rc")
    handler = Agent(MockModelClient([]), system_message="Handle", name="coder")
    router = Router(routing_agent=routing, handlers={"coder": handler})

    with pytest.raises(KeyError):
        router.restore(_messages_with_system(), route="missing")


# ---------------------------------------------------------------------------
# Parallel.restore
# ---------------------------------------------------------------------------


def test_parallel_restore_worker_by_index():
    c0 = MockModelClient([])
    c1 = MockModelClient([])
    parallel = Parallel(workers=[Agent(c0, system_message="W0", name="w0"), Agent(c1, system_message="W1", name="w1")])

    parallel.restore(_messages_with_system(system="W1"), worker=1)

    assert "Hello" in [m["content"] for m in c1.messages]
    assert c0.messages == []  # other workers untouched


def test_parallel_restore_defaults_to_first_worker():
    c0 = MockModelClient([])
    c1 = MockModelClient([])
    parallel = Parallel(workers=[Agent(c0, system_message="W0", name="w0"), Agent(c1, system_message="W1", name="w1")])

    parallel.restore(_messages_with_system(system="W0"))

    assert "Hello" in [m["content"] for m in c0.messages]
    assert c1.messages == []


# ---------------------------------------------------------------------------
# OrchestratorAgent.restore
# ---------------------------------------------------------------------------


def test_orchestrator_restore_delegates_to_inner_orchestrator():
    orch_client = MockModelClient([])
    worker = Agent(MockModelClient([]), system_message="Work", name="w")
    orchestrator = OrchestratorAgent.assemble(orch_client, "Orchestrate the work.", workers=[worker])

    orchestrator.restore(_messages_with_system(system="Orchestrate the work."))

    assert "Hello" in [m["content"] for m in orch_client.messages]
