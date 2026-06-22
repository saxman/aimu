"""Async parity tests for restore() across aio runners (P1-C).

restore() is pure (sync) state manipulation, so these are plain functions: only
construction is exercised, no event loop needed.
"""

from __future__ import annotations

import pytest

from helpers_aio import MockAsyncModelClient
from aimu.aio import Agent, Chain, EvaluatorOptimizer, OrchestratorAgent, Parallel, Router


def _messages_with_system(system="You are helpful."):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


# --- Agent ---


def test_async_agent_restore_strips_system_and_keeps_content():
    client = MockAsyncModelClient([])
    agent = Agent(client, system_message="Sys", name="a")
    agent.restore(_messages_with_system(system="Sys"))
    roles = [m["role"] for m in client.messages]
    assert "system" not in roles
    assert "Hello" in [m["content"] for m in client.messages]


# --- Chain ---


def test_async_chain_restore_step():
    c1 = MockAsyncModelClient([])
    c2 = MockAsyncModelClient([])
    chain = Chain(agents=[Agent(c1, system_message="S1", name="s1"), Agent(c2, system_message="S2", name="s2")])
    chain.restore(_messages_with_system(system="S2"), step=1)
    assert c1.messages == []
    assert "Hello" in [m["content"] for m in c2.messages]


# --- EvaluatorOptimizer ---


def test_async_evaluator_optimizer_restore_delegates_to_generator():
    gen = MockAsyncModelClient([])
    ev = MockAsyncModelClient([])
    eo = EvaluatorOptimizer(
        generator=Agent(gen, system_message="Gen", name="gen"),
        evaluator=Agent(ev, system_message="Eval", name="eval"),
        pass_keyword="PASS",
    )
    eo.restore(_messages_with_system(system="Gen"))
    assert "Hello" in [m["content"] for m in gen.messages]
    assert ev.messages == []


# --- Router ---


def test_async_router_restore_handler_by_route():
    rc = MockAsyncModelClient([])
    hc = MockAsyncModelClient([])
    router = Router(
        routing_agent=Agent(rc, system_message="Classify", name="rc"),
        handlers={"coder": Agent(hc, system_message="Handle", name="coder")},
    )
    router.restore(_messages_with_system(system="Handle"), route="coder")
    assert "Hello" in [m["content"] for m in hc.messages]
    assert rc.messages == []


def test_async_router_restore_defaults_to_routing_agent():
    rc = MockAsyncModelClient([])
    hc = MockAsyncModelClient([])
    router = Router(
        routing_agent=Agent(rc, system_message="Classify", name="rc"),
        handlers={"coder": Agent(hc, system_message="Handle", name="coder")},
    )
    router.restore(_messages_with_system(system="Classify"))
    assert "Hello" in [m["content"] for m in rc.messages]
    assert hc.messages == []


def test_async_router_restore_unknown_route_raises():
    router = Router(
        routing_agent=Agent(MockAsyncModelClient([]), system_message="c", name="rc"),
        handlers={"coder": Agent(MockAsyncModelClient([]), name="coder")},
    )
    with pytest.raises(KeyError):
        router.restore(_messages_with_system(), route="missing")


# --- Parallel ---


def test_async_parallel_restore_worker_by_index():
    c0 = MockAsyncModelClient([])
    c1 = MockAsyncModelClient([])
    parallel = Parallel(workers=[Agent(c0, system_message="W0", name="w0"), Agent(c1, system_message="W1", name="w1")])
    parallel.restore(_messages_with_system(system="W1"), worker=1)
    assert "Hello" in [m["content"] for m in c1.messages]
    assert c0.messages == []


# --- OrchestratorAgent ---


def test_async_orchestrator_restore_delegates_to_inner():
    orch_client = MockAsyncModelClient([])
    worker = Agent(MockAsyncModelClient([]), system_message="Work", name="w")
    orchestrator = OrchestratorAgent.assemble(orch_client, "Orchestrate.", workers=[worker])
    orchestrator.restore(_messages_with_system(system="Orchestrate."))
    assert "Hello" in [m["content"] for m in orch_client.messages]
