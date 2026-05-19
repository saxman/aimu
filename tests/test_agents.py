"""
Tests for aimu.agents: SimpleAgent, Chain, AgenticModelClient, and example agents.

Unit tests use MockModelClient from helpers (deterministic, no backend needed).
The model_client fixture is available for integration tests:
  - Default (no --client): MockModelClient
  - pytest tests/test_agents.py --client=ollama --model=LLAMA_3_2_3B
"""

import threading
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

from aimu.agents import Agent, Chain, AgentChunk, AgenticModelClient, OrchestratorAgent, SimpleAgent
from aimu.agents.examples import ResearchReportAgent
from aimu.models import BaseModelClient, StreamChunk, StreamingContentType
from helpers import MockModelClient, create_real_model_client, resolve_model_params

_MOCK = "mock"


def pytest_generate_tests(metafunc):
    if "model_client" not in metafunc.fixturenames:
        return
    params = resolve_model_params(metafunc.config, default_params=[_MOCK])
    metafunc.parametrize("model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[BaseModelClient]:
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
    # All responses are tool calls; would loop forever without a limit
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
    """chat(..., stream=True) yields StreamChunk, not AgentChunk."""
    client = MockModelClient(["stream result"])
    ac = AgenticModelClient(SimpleAgent(client))
    chunks = list(ac.chat("task", stream=True))
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert not any(isinstance(c, AgentChunk) for c in chunks)


def test_agentic_client_messages_delegated():
    """messages property delegates to inner_client; both refer to the same list."""
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


# ---------------------------------------------------------------------------
# concurrent_tool_calls tests
# ---------------------------------------------------------------------------


def _make_mock_mcp(response_text: str = "result"):
    """Return a mock MCPClient whose call_tool returns a text response."""
    mock_mcp = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(type="text", text=response_text)]
    mock_mcp.call_tool.return_value = response
    return mock_mcp


def test_handle_tool_calls_sequential_by_default():
    """Tool calls execute sequentially by default (concurrent_tool_calls=False)."""
    client = MockModelClient([])
    client.mcp_client = _make_mock_mcp()

    call_order = []

    def record_call(name, args):
        call_order.append(name)
        response = MagicMock()
        response.content = [MagicMock(type="text", text=name)]
        return response

    client.mcp_client.call_tool.side_effect = record_call

    tools = [
        {"type": "function", "function": {"name": "tool_a"}},
        {"type": "function", "function": {"name": "tool_b"}},
    ]
    tool_calls = [{"name": "tool_a", "arguments": {}}, {"name": "tool_b", "arguments": {}}]
    client._handle_tool_calls(tool_calls, tools)

    assert call_order == ["tool_a", "tool_b"]


def test_handle_tool_calls_concurrent_runs_in_parallel():
    """With concurrent_tool_calls=True, multiple tool calls run at the same time.

    A threading.Barrier(2) forces both tools to reach a synchronization point
    simultaneously. If execution were sequential, the second call could never
    start while the first is blocked, causing the barrier to time out.
    """
    client = MockModelClient([])
    client.concurrent_tool_calls = True
    client.mcp_client = _make_mock_mcp()

    barrier = threading.Barrier(2, timeout=2.0)

    def gated_call(name, args):
        barrier.wait()  # both threads must arrive before either proceeds
        response = MagicMock()
        response.content = [MagicMock(type="text", text=name)]
        return response

    client.mcp_client.call_tool.side_effect = gated_call

    tools = [
        {"type": "function", "function": {"name": "tool_a"}},
        {"type": "function", "function": {"name": "tool_b"}},
    ]
    tool_calls = [{"name": "tool_a", "arguments": {}}, {"name": "tool_b", "arguments": {}}]
    client._handle_tool_calls(tool_calls, tools)

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_messages) == 2


def test_handle_tool_calls_concurrent_results_in_original_order():
    """Concurrent results are appended in input order, not completion order."""
    client = MockModelClient([])
    client.concurrent_tool_calls = True
    client.mcp_client = _make_mock_mcp()

    # tool_a takes longer than tool_b — tool_b would finish first if truly parallel
    def slow_first(name, args):
        import time

        if name == "tool_a":
            time.sleep(0.05)
        response = MagicMock()
        response.content = [MagicMock(type="text", text=name)]
        return response

    client.mcp_client.call_tool.side_effect = slow_first

    tools = [
        {"type": "function", "function": {"name": "tool_a"}},
        {"type": "function", "function": {"name": "tool_b"}},
    ]
    tool_calls = [{"name": "tool_a", "arguments": {}}, {"name": "tool_b", "arguments": {}}]
    client._handle_tool_calls(tool_calls, tools)

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert tool_messages[0]["name"] == "tool_a"
    assert tool_messages[1]["name"] == "tool_b"


# ---------------------------------------------------------------------------
# ResearchReportAgent tests
# ---------------------------------------------------------------------------


def _make_research_agent(tools=None):
    """Build a ResearchReportAgent with a MockModelClient, patching ModelClient
    so worker construction doesn't try to dispatch a MagicMock model enum."""
    client = MockModelClient(["report"])
    with patch("aimu.agents.examples.research_report.ModelClient") as MockMC:
        MockMC.return_value = MockModelClient(["worker response"])
        agent = ResearchReportAgent(client, tools=tools)
    return agent, client


def test_research_report_agent_without_tools_no_concurrent():
    """Without a tools server, the orchestrator's concurrent_tool_calls is False."""
    _, client = _make_research_agent()
    assert not client.concurrent_tool_calls


def test_research_report_agent_with_tools_enables_concurrent():
    """When a tools server is provided, the orchestrator enables concurrent tool calls."""
    mcp = FastMCP("test-tools")

    @mcp.tool()
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    _, client = _make_research_agent(tools=mcp)
    assert client.concurrent_tool_calls


def test_research_report_agent_is_agent_subclass():
    agent, _ = _make_research_agent()
    assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# OrchestratorAgent tests
# ---------------------------------------------------------------------------


def _make_concrete_orchestrator(client: MockModelClient) -> OrchestratorAgent:
    """Build a minimal OrchestratorAgent subclass for testing the base class."""

    class _TestOrchestrator(OrchestratorAgent):
        def __init__(self, model_client):
            worker = SimpleAgent(model_client, name="worker", system_message="Work.")

            mcp = FastMCP("Test Workers")

            @mcp.tool()
            def do_work(task: str) -> str:
                """Do the work."""
                return worker.run(task)

            self._setup_orchestrator(model_client, mcp, name="test-orchestrator", system_message="Use do_work.")

    return _TestOrchestrator(client)


def test_orchestrator_agent_is_agent_subclass():
    client = MockModelClient(["done"])
    agent = _make_concrete_orchestrator(client)
    assert isinstance(agent, Agent)
    assert isinstance(agent, OrchestratorAgent)


def test_orchestrator_agent_run_delegates():
    """run() returns the inner orchestrator's response."""
    client = MockModelClient(["orchestrator result"])
    agent = _make_concrete_orchestrator(client)
    result = agent.run("do something")
    assert result == "orchestrator result"


def test_orchestrator_agent_messages_delegates():
    """messages property returns a MessageHistory keyed by the orchestrator name."""
    client = MockModelClient(["hi"])
    agent = _make_concrete_orchestrator(client)
    agent.run("hello")
    history = agent.messages
    assert "test-orchestrator" in history
    assert any(m["content"] == "hello" for m in history["test-orchestrator"])


def test_orchestrator_agent_setup_wires_mcp_client():
    """_setup_orchestrator sets mcp_client on the model_client."""
    client = MockModelClient(["done"])
    _make_concrete_orchestrator(client)
    assert client.mcp_client is not None


def test_orchestrator_agent_setup_concurrent_tool_calls_default_false():
    """_setup_orchestrator leaves concurrent_tool_calls False by default."""
    client = MockModelClient(["done"])
    _make_concrete_orchestrator(client)
    assert not client.concurrent_tool_calls


def test_orchestrator_agent_setup_concurrent_tool_calls_can_be_enabled():
    """_setup_orchestrator respects concurrent_tool_calls=True."""

    class _ConcurrentOrchestrator(OrchestratorAgent):
        def __init__(self, model_client):
            mcp = FastMCP("Workers")

            @mcp.tool()
            def work(task: str) -> str:
                """Work."""
                return "done"

            self._setup_orchestrator(
                model_client, mcp, name="orch", system_message=".", concurrent_tool_calls=True
            )

    client = MockModelClient(["done"])
    _ConcurrentOrchestrator(client)
    assert client.concurrent_tool_calls
