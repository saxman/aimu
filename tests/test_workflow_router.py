"""
Tests for aimu.agents.Router — the Routing workflow pattern.

All tests use MockModelClient from conftest (deterministic, no backend needed).
"""

import pytest

from aimu.agents import AgentChunk, Router, Runner, SimpleAgent, Workflow
from aimu.models.base import StreamingContentType
from conftest import MockModelClient


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def test_router_dispatches_to_correct_handler():
    """Router classifies the task and dispatches to the matching handler."""
    routing_client = MockModelClient(["code"])
    code_client = MockModelClient(["here is some code"])
    writing_client = MockModelClient(["here is some writing"])

    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={
            "code": SimpleAgent(code_client, name="coder"),
            "writing": SimpleAgent(writing_client, name="writer"),
        },
    )
    result = router.run("Write a Python function.")

    assert result == "here is some code"
    assert code_client._call_count == 1
    assert writing_client._call_count == 0


def test_router_route_is_case_insensitive():
    """Route names are normalised to lowercase before lookup."""
    routing_client = MockModelClient(["Code"])  # uppercase
    handler_client = MockModelClient(["handled"])

    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={"code": SimpleAgent(handler_client, name="coder")},
    )
    result = router.run("task")
    assert result == "handled"


def test_router_uses_fallback_on_unknown_route():
    """When the classified route has no handler, the fallback is used."""
    routing_client = MockModelClient(["unknown"])
    fallback_client = MockModelClient(["fallback response"])

    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={"code": SimpleAgent(MockModelClient(["should not be called"]), name="coder")},
        fallback=SimpleAgent(fallback_client, name="fallback"),
    )
    result = router.run("task")
    assert result == "fallback response"
    assert fallback_client._call_count == 1


def test_router_raises_on_unknown_route_without_fallback():
    """ValueError is raised when route is unknown and no fallback is set."""
    routing_client = MockModelClient(["unknown"])

    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={"code": SimpleAgent(MockModelClient(["x"]), name="coder")},
    )
    with pytest.raises(ValueError, match="No handler for route 'unknown'"):
        router.run("task")


def test_router_streamed_yields_agent_chunks():
    """run_streamed() yields AgentChunk from classifier then from handler."""
    routing_client = MockModelClient(["math"])
    math_client = MockModelClient(["42"])

    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={"math": SimpleAgent(math_client, name="mathematician")},
    )
    chunks = list(router.run_streamed("What is 6 times 7?"))

    assert all(isinstance(c, AgentChunk) for c in chunks)
    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    # First GENERATING chunk is the route name; last is the handler's answer
    contents = [c.content for c in generating]
    assert "math" in contents
    assert "42" in contents


def test_router_streamed_raises_on_unknown_route():
    """run_streamed() raises ValueError when route is unknown and no fallback set."""
    routing_client = MockModelClient(["unknown"])

    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={"code": SimpleAgent(MockModelClient(["x"]), name="coder")},
    )
    with pytest.raises(ValueError, match="No handler for route 'unknown'"):
        list(router.run_streamed("task"))


def test_router_is_workflow_subclass():
    """Router must be a Workflow (not an Agent) and also a Runner."""
    routing_client = MockModelClient(["code"])
    router = Router(
        routing_agent=SimpleAgent(routing_client, name="classifier"),
        handlers={"code": SimpleAgent(MockModelClient(["done"]), name="coder")},
    )
    from aimu.agents import Agent

    assert isinstance(router, Workflow)
    assert isinstance(router, Runner)
    assert not isinstance(router, Agent)


def test_router_from_config():
    """from_config builds a Router with routing agent and handlers from config dicts."""
    client = MockModelClient(["writing", "a poem"])
    router = Router.from_config(
        routing_config={"name": "classifier"},
        handler_configs={"writing": {"name": "writer"}},
        client=client,
    )
    assert router.routing_agent.name == "classifier"
    assert "writing" in router.handlers
    assert router.handlers["writing"].name == "writer"
    result = router.run("Write a poem.")
    assert result == "a poem"
