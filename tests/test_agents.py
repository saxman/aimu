"""
Tests for aimu.agents: Agent, Chain, agent.as_model_client(), example agents.

Unit tests use MockModelClient from helpers (deterministic, no backend needed).
The model_client fixture is available for integration tests:
  - Default (no --client): MockModelClient
  - pytest tests/test_agents.py --client=ollama --model=LLAMA_3_2_3B
"""

import threading
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from aimu.agents import Agent, Chain, OrchestratorAgent, Runner
from aimu.agents.prebuilt import ResearchReportAgent
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
# Agent tests
# ---------------------------------------------------------------------------


def test_agent_no_tools_calls_chat_once():
    """If the model never uses tools, Agent.run() calls chat() exactly once."""
    client = MockModelClient(["final answer"])
    agent = Agent(client, name="test")
    result = agent.run("do something")

    assert result == "final answer"
    assert client._call_count == 1


def test_agent_one_tool_round_then_done():
    client = MockModelClient(["tool", "after tool answer", "all done"])
    agent = Agent(client, name="test")
    result = agent.run("do something with tools")

    assert result == "all done"
    assert client._call_count == 3


def test_agent_two_tool_rounds():
    client = MockModelClient(["tool", "after first tool", "tool", "after second tool", "final answer"])
    agent = Agent(client, name="test", max_iterations=10)
    result = agent.run("multi-round task")

    assert result == "final answer"
    assert client._call_count == 5


def test_agent_max_iterations_stops_loop():
    client = MockModelClient(["tool", "still going"] * 10)
    agent = Agent(client, name="test", max_iterations=3)
    agent.run("never-ending task")

    assert client._call_count <= 6


def test_agent_uses_continuation_prompt():
    client = MockModelClient(["tool", "done", "confirmed"])
    agent = Agent(client, name="test", continuation_prompt="KEEP GOING")
    agent.run("start")

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages[0] == "start"
    assert user_messages[1] == "KEEP GOING"


def test_agent_positional_system_message():
    """Agent accepts system_message as the second positional argument."""
    client = MockModelClient(["hi"])
    agent = Agent(client, "You are concise.", name="positional-sm")
    assert agent.system_message == "You are concise."


def test_agent_from_config_sets_system_message():
    client = MockModelClient(["hello"])
    agent = Agent.from_config({"name": "cfg_agent", "system_message": "Be helpful.", "max_iterations": 5}, client)
    assert agent.system_message == "Be helpful."


def test_agent_from_config_defaults():
    client = MockModelClient(["hello"])
    agent = Agent.from_config({}, client)

    assert agent.name is not None and agent.name.startswith("agent-")
    assert agent.max_iterations == 10


def test_agent_auto_derives_name_when_unset():
    """Agent generates a unique name when none is passed."""
    a = Agent(MockModelClient(["hi"]))
    b = Agent(MockModelClient(["hi"]))
    assert a.name != b.name
    assert a.name.startswith("agent-")


def test_agent_streamed_yields_stream_chunks_tagged_with_agent_name():
    client = MockModelClient(["streamed answer"])
    agent = Agent(client, name="streamer")
    chunks = list(agent.run("task", stream=True))

    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert all(c.agent == "streamer" for c in chunks)
    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert len(generating) == 1
    assert generating[0].content == "streamed answer"


def test_agent_streamed_iteration_increments_on_tool_use():
    client = MockModelClient(["tool", "done", "confirmed"])
    agent = Agent(client, name="it_agent")
    chunks = list(agent.run("task", stream=True))

    iterations = {c.iteration for c in chunks}
    assert 0 in iterations
    assert 1 in iterations


def test_simple_agent_is_runner_subclass():
    client = MockModelClient(["hi"])
    agent = Agent(client)
    assert isinstance(agent, Runner)


# ---------------------------------------------------------------------------
# agent.as_model_client() tests (replaces the legacy AgenticModelClient public class)
# ---------------------------------------------------------------------------


def test_as_model_client_chat_runs_agent_loop():
    """chat() on the view runs the full Agent loop, not a single turn."""
    client = MockModelClient(["tool", "after tool", "done"])
    view = Agent(client, max_iterations=5).as_model_client()
    result = view.chat("do something with tools")
    assert result == "done"
    assert client._call_count == 3


def test_as_model_client_chat_no_tools_single_call():
    client = MockModelClient(["plain answer"])
    view = Agent(client).as_model_client()
    assert view.chat("hello") == "plain answer"
    assert client._call_count == 1


def test_as_model_client_chat_streamed_yields_stream_chunks():
    client = MockModelClient(["stream result"])
    view = Agent(client).as_model_client()
    chunks = list(view.chat("task", stream=True))
    assert all(isinstance(c, StreamChunk) for c in chunks)


def test_as_model_client_messages_delegated():
    client = MockModelClient(["reply"])
    view = Agent(client).as_model_client()
    view.chat("hi")
    assert view.messages is client.messages


def test_as_model_client_mcp_client_delegated():
    client = MockModelClient(["hi"])
    view = Agent(client).as_model_client()
    mock_mcp = object()
    view.mcp_client = mock_mcp
    assert client.mcp_client is mock_mcp


def test_as_model_client_system_message_delegated():
    client = MockModelClient(["hi"])
    view = Agent(client).as_model_client()
    view.system_message = "Be helpful."
    assert client.system_message == "Be helpful."


def test_as_model_client_generate_delegates_to_inner():
    client = MockModelClient(["generated"])
    view = Agent(client).as_model_client()
    result = view.generate("prompt")
    assert result == "generated"
    assert client._call_count == 1


def test_internal_agentic_view_rejects_workflow():
    """The internal _AgenticView raises TypeError when given a Workflow (Chain)."""
    from aimu.agents.agentic_client import _AgenticView

    client = MockModelClient(["hi"])
    chain = Chain(agents=[Agent(client)])
    with pytest.raises(TypeError, match="_AgenticView only accepts Agent"):
        _AgenticView(chain)


# ---------------------------------------------------------------------------
# concurrent_tool_calls tests
# ---------------------------------------------------------------------------


def _make_mock_mcp(response_text: str = "result"):
    mock_mcp = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(type="text", text=response_text)]
    mock_mcp.call_tool.return_value = response
    return mock_mcp


def test_handle_tool_calls_sequential_by_default():
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
    client = MockModelClient([])
    client.concurrent_tool_calls = True
    client.mcp_client = _make_mock_mcp()

    barrier = threading.Barrier(2, timeout=2.0)

    def gated_call(name, args):
        barrier.wait()
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
    client = MockModelClient([])
    client.concurrent_tool_calls = True
    client.mcp_client = _make_mock_mcp()

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


def _make_research_agent(worker_tools=None):
    client = MockModelClient(["report"])
    with patch("aimu.agents.prebuilt.research_report.ModelClient") as MockMC:
        MockMC.return_value = MockModelClient(["worker response"])
        agent = ResearchReportAgent(client, worker_tools=worker_tools)
    return agent, client


def test_research_report_agent_without_worker_tools_no_concurrent():
    _, client = _make_research_agent()
    assert not client.concurrent_tool_calls


def test_research_report_agent_with_worker_tools_enables_concurrent():
    from aimu.tools import tool

    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    _, client = _make_research_agent(worker_tools=[search])
    assert client.concurrent_tool_calls


def test_research_report_agent_is_runner_subclass():
    agent, _ = _make_research_agent()
    assert isinstance(agent, Runner)


# ---------------------------------------------------------------------------
# OrchestratorAgent tests
# ---------------------------------------------------------------------------


def _make_concrete_orchestrator(client: MockModelClient) -> OrchestratorAgent:
    from aimu.tools import tool

    class _TestOrchestrator(OrchestratorAgent):
        def __init__(self, model_client):
            worker = Agent(model_client, name="worker", system_message="Work.")

            @tool
            def do_work(task: str) -> str:
                """Do the work."""
                return worker.run(task)

            self._init_orchestrator(
                model_client,
                name="test-orchestrator",
                system_message="Use do_work.",
                tools=[do_work],
            )

    return _TestOrchestrator(client)


def test_orchestrator_agent_is_runner_subclass():
    client = MockModelClient(["done"])
    agent = _make_concrete_orchestrator(client)
    assert isinstance(agent, Runner)
    assert isinstance(agent, OrchestratorAgent)


def test_orchestrator_agent_run_delegates():
    client = MockModelClient(["orchestrator result"])
    agent = _make_concrete_orchestrator(client)
    result = agent.run("do something")
    assert result == "orchestrator result"


def test_orchestrator_agent_messages_delegates():
    client = MockModelClient(["hi"])
    agent = _make_concrete_orchestrator(client)
    agent.run("hello")
    history = agent.messages
    assert "test-orchestrator" in history
    assert any(m["content"] == "hello" for m in history["test-orchestrator"])


def test_orchestrator_agent_setup_registers_tools():
    client = MockModelClient(["done"])
    _make_concrete_orchestrator(client)
    assert len(client.tools) == 1
    assert client.tools[0].__name__ == "do_work"


def test_orchestrator_agent_setup_concurrent_tool_calls_default_false():
    client = MockModelClient(["done"])
    _make_concrete_orchestrator(client)
    assert not client.concurrent_tool_calls


def test_orchestrator_agent_setup_concurrent_tool_calls_can_be_enabled():
    from aimu.tools import tool

    class _ConcurrentOrchestrator(OrchestratorAgent):
        def __init__(self, model_client):
            @tool
            def work(task: str) -> str:
                """Work."""
                return "done"

            self._init_orchestrator(
                model_client,
                name="orch",
                system_message=".",
                tools=[work],
                concurrent_tool_calls=True,
            )

    client = MockModelClient(["done"])
    _ConcurrentOrchestrator(client)
    assert client.concurrent_tool_calls


def test_orchestrator_agent_assemble_factory():
    """OrchestratorAgent.assemble builds a runnable orchestrator without subclassing."""
    client = MockModelClient(["done"])
    worker_a = Agent(MockModelClient(["worker a"]), system_message="Worker A role.", name="alpha")
    worker_b = Agent(MockModelClient(["worker b"]), system_message="Worker B role.", name="beta")

    orch = OrchestratorAgent.assemble(client, "Use the workers.", workers=[worker_a, worker_b])
    assert isinstance(orch, OrchestratorAgent)
    # Two tools were registered, one per worker, with the worker names.
    tool_names = {t.__name__ for t in client.tools}
    assert tool_names == {"alpha", "beta"}
    result = orch.run("task")
    assert result == "done"


# ---------------------------------------------------------------------------
# Agent forwards streaming-tool chunks through run(stream=True)
# ---------------------------------------------------------------------------


def test_agent_forwards_image_generating_chunks_from_streaming_tool():
    """When the underlying client emits IMAGE_GENERATING chunks (from a streaming tool),
    Agent.run(stream=True) should re-emit them with `agent` + `iteration` metadata."""

    class _ImageStreamingClient(MockModelClient):
        """Yields IMAGE_GENERATING progress + TOOL_CALLING on the first call,
        then a plain GENERATING chunk on the agent's continuation call."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._calls = 0

        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
            if not stream:
                return super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            def _gen():
                self.messages.append({"role": "user", "content": user_message})
                self._system_message_locked = True
                self._calls += 1
                if self._calls == 1:
                    # First call: emit image-gen progress + tool call.
                    for step in range(1, 4):
                        yield StreamChunk(
                            StreamingContentType.IMAGE_GENERATING,
                            {
                                "step": step,
                                "total_steps": 3,
                                "image": None,
                                "final": (step == 3),
                                "result": "/tmp/img.png" if step == 3 else None,
                            },
                        )
                    self.messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"type": "function", "function": {"name": "generate_image", "arguments": {}}, "id": "x"}
                            ],
                        }
                    )
                    self.messages.append(
                        {"role": "tool", "name": "generate_image", "content": "/tmp/img.png", "tool_call_id": "x"}
                    )
                    yield StreamChunk(
                        StreamingContentType.TOOL_CALLING,
                        {"name": "generate_image", "arguments": {}, "response": "/tmp/img.png"},
                    )
                else:
                    # Continuation call: just emit a final GENERATING chunk + assistant message.
                    self.messages.append({"role": "assistant", "content": "Done!"})
                    yield StreamChunk(StreamingContentType.GENERATING, "Done!")

            return _gen()

    client = _ImageStreamingClient(["Done!"])
    agent = Agent(client, name="streamer")
    chunks = list(agent.run("make me an image", stream=True))

    image_chunks = [c for c in chunks if c.is_image_progress()]
    assert len(image_chunks) == 3
    # Agent metadata is set on every forwarded chunk.
    assert all(c.agent == "streamer" for c in chunks)
    # Image progress + tool call come from iteration 0; "Done!" comes from iteration 1.
    assert all(c.iteration == 0 for c in image_chunks)
    # Final image chunk carries the result.
    assert image_chunks[-1].content["final"] is True
    assert image_chunks[-1].content["result"] == "/tmp/img.png"
    # TOOL_CALLING chunk also forwarded.
    tool_chunks = [c for c in chunks if c.is_tool_call()]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].content["name"] == "generate_image"
    # Final GENERATING chunk from the continuation call.
    text_chunks = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert len(text_chunks) == 1
    assert text_chunks[0].content == "Done!"
    assert text_chunks[0].iteration == 1
