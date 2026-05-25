"""
Tests for aimu.agents.Router: the Routing workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

import pytest

from aimu.agents import Agent, AgentChunk, Router, Runner
from aimu.models import StreamingContentType
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def test_router_dispatches_to_correct_handler():
    """Router classifies the task and dispatches to the matching handler."""
    routing_client = MockModelClient(["code"])
    code_client = MockModelClient(["here is some code"])
    writing_client = MockModelClient(["here is some writing"])

    router = Router(
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={
            "code": Agent(code_client, name="coder"),
            "writing": Agent(writing_client, name="writer"),
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
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={"code": Agent(handler_client, name="coder")},
    )
    result = router.run("task")
    assert result == "handled"


def test_router_uses_fallback_on_unknown_route():
    """When the classified route has no handler, the fallback is used."""
    routing_client = MockModelClient(["unknown"])
    fallback_client = MockModelClient(["fallback response"])

    router = Router(
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={"code": Agent(MockModelClient(["should not be called"]), name="coder")},
        fallback=Agent(fallback_client, name="fallback"),
    )
    result = router.run("task")
    assert result == "fallback response"
    assert fallback_client._call_count == 1


def test_router_raises_on_unknown_route_without_fallback():
    """ValueError is raised when route is unknown and no fallback is set."""
    routing_client = MockModelClient(["unknown"])

    router = Router(
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={"code": Agent(MockModelClient(["x"]), name="coder")},
    )
    with pytest.raises(ValueError, match="No handler for route 'unknown'"):
        router.run("task")


def test_router_streamed_yields_agent_chunks():
    """run_streamed() yields AgentChunk from classifier then from handler."""
    routing_client = MockModelClient(["math"])
    math_client = MockModelClient(["42"])

    router = Router(
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={"math": Agent(math_client, name="mathematician")},
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
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={"code": Agent(MockModelClient(["x"]), name="coder")},
    )
    with pytest.raises(ValueError, match="No handler for route 'unknown'"):
        list(router.run_streamed("task"))


def test_router_is_runner_subclass():
    """Router implements the single Runner interface."""
    routing_client = MockModelClient(["code"])
    router = Router(
        routing_agent=Agent(routing_client, name="classifier"),
        handlers={"code": Agent(MockModelClient(["done"]), name="coder")},
    )

    assert isinstance(router, Runner)


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
