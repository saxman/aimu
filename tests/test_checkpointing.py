"""Tests for Agent.restore(), EvaluatorOptimizer.restore(), Chain.restore(step=)."""

from __future__ import annotations

from helpers import MockModelClient
from aimu.agents import Agent
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
