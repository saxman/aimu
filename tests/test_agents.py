"""
Tests for aimu.agents — SimpleAgent, Chain, and AgenticModelClient.

Unit tests use MockModelClient from conftest (deterministic, no backend needed).
The model_client fixture is available for integration tests:
  - Default (no --client): MockModelClient
  - pytest tests/test_agents.py --client=ollama --model=LLAMA_3_2_3B
"""

from typing import Iterable
from unittest.mock import patch

import pytest

from aimu.agents import Agent, Chain, AgentChunk, AgenticModelClient, Runner, SimpleAgent, Workflow
from aimu.models import ModelClient
from aimu.models.base_client import StreamChunk, StreamingContentType
from conftest import MockModelClient, create_real_model_client, resolve_model_params

_MOCK = "mock"


def pytest_generate_tests(metafunc):
    if "model_client" not in metafunc.fixturenames:
        return
    params = resolve_model_params(metafunc.config, default_params=[_MOCK])
    metafunc.parametrize("model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[ModelClient]:
    if request.param == _MOCK:
        yield MockModelClient(["Hello, I am ready to help."])
        return
    yield from create_real_model_client(request)


# ---------------------------------------------------------------------------
# SimpleAgent tests
# ---------------------------------------------------------------------------


def test_agent_no_tools_calls_chat_once():
    """If the model never uses tools, SimpleAgent.run() calls chat() exactly once."""
    client = MockModelClient(["final answer"])
    agent = SimpleAgent(client, name="test")
    result = agent.run("do something")

    assert result == "final answer"
    assert client._call_count == 1


def test_agent_one_tool_round_then_done():
    """
    After a tool-using turn the agent sends one continuation to confirm the model
    is done. Responses consumed: tool(2 slots) + plain confirmation(1 slot) = 3.
    """
    client = MockModelClient(["tool", "after tool answer", "all done"])
    agent = SimpleAgent(client, name="test")
    result = agent.run("do something with tools")

    assert result == "all done"
    assert client._call_count == 3


def test_agent_two_tool_rounds():
    """
    Two consecutive tool-using turns, then a plain confirmation.
    Loop: chat(task)→tool → continuation→tool → continuation→plain → stop.
    """
    client = MockModelClient(["tool", "after first tool", "tool", "after second tool", "final answer"])
    agent = SimpleAgent(client, name="test", max_iterations=10)
    result = agent.run("multi-round task")

    assert result == "final answer"
    assert client._call_count == 5  # tool(2) + tool(2) + final(1)


def test_agent_max_iterations_stops_loop():
    """SimpleAgent stops after max_iterations even if tools keep being called."""
    # All responses are tool calls — would loop forever without a limit
    client = MockModelClient(["tool", "still going"] * 10)
    agent = SimpleAgent(client, name="test", max_iterations=3)
    agent.run("never-ending task")

    assert client._call_count <= 6  # 3 iterations × 2 reads each at most


def test_agent_uses_continuation_prompt():
    """After a tool-calling turn, SimpleAgent sends the continuation_prompt."""
    client = MockModelClient(["tool", "done", "confirmed"])
    agent = SimpleAgent(client, name="test", continuation_prompt="KEEP GOING")
    agent.run("start")

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages[0] == "start"
    assert user_messages[1] == "KEEP GOING"


def test_agent_from_config_sets_system_message():
    client = MockModelClient(["hello"])
    SimpleAgent.from_config({"name": "cfg_agent", "system_message": "Be helpful.", "max_iterations": 5}, client)

    assert client.system_message == "Be helpful."


def test_agent_from_config_defaults():
    client = MockModelClient(["hello"])
    agent = SimpleAgent.from_config({}, client)

    assert agent.name == "agent"
    assert agent.max_iterations == 10


def test_agent_streamed_yields_agent_chunks():
    client = MockModelClient(["streamed answer"])
    agent = SimpleAgent(client, name="streamer")
    chunks = list(agent.run_streamed("task"))

    assert all(isinstance(c, AgentChunk) for c in chunks)
    assert all(c.agent_name == "streamer" for c in chunks)
    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert len(generating) == 1
    assert generating[0].content == "streamed answer"


def test_agent_streamed_iteration_increments_on_tool_use():
    client = MockModelClient(["tool", "done", "confirmed"])
    agent = SimpleAgent(client, name="it_agent")
    chunks = list(agent.run_streamed("task"))

    iterations = {c.iteration for c in chunks}
    assert 0 in iterations  # first call
    assert 1 in iterations  # continuation after tool use


def test_simple_agent_is_agent_subclass():
    """SimpleAgent must be a concrete subclass of the Agent ABC."""
    client = MockModelClient(["hi"])
    agent = SimpleAgent(client)
    assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# AgenticModelClient tests
# ---------------------------------------------------------------------------


def test_agentic_client_chat_runs_agent_loop():
    """chat() runs the full SimpleAgent loop, not a single turn."""
    client = MockModelClient(["tool", "after tool", "done"])
    ac = AgenticModelClient(SimpleAgent(client, max_iterations=5))
    result = ac.chat("do something with tools")
    assert result == "done"
    assert client._call_count == 3  # tool(2) + confirmation(1)


def test_agentic_client_chat_no_tools_single_call():
    """chat() without tool use resolves in one model call."""
    client = MockModelClient(["plain answer"])
    ac = AgenticModelClient(SimpleAgent(client))
    assert ac.chat("hello") == "plain answer"
    assert client._call_count == 1


def test_agentic_client_chat_streamed_yields_stream_chunks():
    """chat_streamed() yields StreamChunk, not AgentChunk."""
    client = MockModelClient(["stream result"])
    ac = AgenticModelClient(SimpleAgent(client))
    chunks = list(ac.chat_streamed("task"))
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert not any(isinstance(c, AgentChunk) for c in chunks)


def test_agentic_client_messages_delegated():
    """messages property delegates to inner_client — both refer to the same list."""
    client = MockModelClient(["reply"])
    ac = AgenticModelClient(SimpleAgent(client))
    ac.chat("hi")
    assert ac.messages is client.messages


def test_agentic_client_mcp_client_delegated():
    """Setting mcp_client on AgenticModelClient propagates to inner_client."""
    client = MockModelClient(["hi"])
    ac = AgenticModelClient(SimpleAgent(client))
    mock_mcp = object()
    ac.mcp_client = mock_mcp
    assert client.mcp_client is mock_mcp


def test_agentic_client_system_message_delegated():
    """system_message property delegates to inner_client."""
    client = MockModelClient(["hi"])
    ac = AgenticModelClient(SimpleAgent(client))
    ac.system_message = "Be helpful."
    assert client.system_message == "Be helpful."


def test_agentic_client_generate_delegates_to_inner():
    """generate() bypasses the Agent loop and calls inner_client directly."""
    client = MockModelClient(["generated"])
    ac = AgenticModelClient(SimpleAgent(client))
    result = ac.generate("prompt")
    assert result == "generated"
    assert client._call_count == 1


def test_agentic_client_rejects_workflow():
    """AgenticModelClient raises TypeError when given a Workflow (Chain)."""
    client = MockModelClient(["hi"])
    chain = Chain(agents=[SimpleAgent(client)])
    with pytest.raises(TypeError, match="AgenticModelClient only accepts SimpleAgent"):
        AgenticModelClient(chain)
