"""
Tests for aimu.agents — Agent and Workflow classes.

Unit tests use MockModelClient inline (deterministic, no backend needed).
The model_client fixture is available for integration tests:
  - Default (no --client): MockModelClient
  - pytest tests/test_agents.py --client=ollama --model=LLAMA_3_2_3B
"""

from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from aimu.agents import Agent, AgentChunk, AgenticModelClient, Workflow, WorkflowChunk
from aimu.models import ModelClient
from aimu.models.base_client import StreamChunk, StreamingContentType
from conftest import create_real_model_client, resolve_model_params

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
# Minimal mock ModelClient
# ---------------------------------------------------------------------------


class MockModelClient(ModelClient):
    """
    A ModelClient stub whose chat() responses are controlled via a response queue.
    Each entry in `responses` is either:
      - str: plain text response (no tool calls)
      - "tool": simulates one tool call by appending the expected messages and returning a follow-up text
    """

    def __init__(self, responses: list):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.mcp_client = None
        self.last_thinking = ""
        self._streaming_content_type = StreamingContentType.DONE
        self._responses = list(responses)
        self._call_count = 0

    def chat(self, user_message, generate_kwargs=None, use_tools=True):
        self.messages.append({"role": "user", "content": user_message})
        response = self._responses[self._call_count]
        self._call_count += 1

        if response == "tool":
            # Simulate one tool-call round: append assistant+tool_calls, tool result, assistant+content
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "mock_tool", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "mock_tool", "content": "tool result", "tool_call_id": "x"})
            text = self._responses[self._call_count]
            self._call_count += 1
            self.messages.append({"role": "assistant", "content": text})
            return text
        else:
            self.messages.append({"role": "assistant", "content": response})
            return response

    def chat_streamed(self, user_message, generate_kwargs=None, use_tools=True):
        response = self.chat(user_message, generate_kwargs, use_tools)
        self._streaming_content_type = StreamingContentType.GENERATING
        yield StreamChunk(StreamingContentType.GENERATING, response)
        self._streaming_content_type = StreamingContentType.DONE

    def generate(self, prompt, generate_kwargs=None):
        return self.chat(prompt, generate_kwargs)

    def generate_streamed(self, prompt, generate_kwargs=None, include_thinking=True):
        self._streaming_content_type = StreamingContentType.GENERATING
        yield StreamChunk(StreamingContentType.GENERATING, self.generate(prompt, generate_kwargs))
        self._streaming_content_type = StreamingContentType.DONE

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}


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
    """
    After a tool-using turn the agent sends one continuation to confirm the model
    is done. Responses consumed: tool(2 slots) + plain confirmation(1 slot) = 3.
    """
    client = MockModelClient(["tool", "after tool answer", "all done"])
    agent = Agent(client, name="test")
    result = agent.run("do something with tools")

    assert result == "all done"
    assert client._call_count == 3


def test_agent_two_tool_rounds():
    """
    Two consecutive tool-using turns, then a plain confirmation.
    Loop: chat(task)→tool → continuation→tool → continuation→plain → stop.
    """
    client = MockModelClient(["tool", "after first tool", "tool", "after second tool", "final answer"])
    agent = Agent(client, name="test", max_iterations=10)
    result = agent.run("multi-round task")

    assert result == "final answer"
    assert client._call_count == 5  # tool(2) + tool(2) + final(1)


def test_agent_max_iterations_stops_loop():
    """Agent stops after max_iterations even if tools keep being called."""
    # All responses are tool calls — would loop forever without a limit
    client = MockModelClient(["tool", "still going"] * 10)
    agent = Agent(client, name="test", max_iterations=3)
    agent.run("never-ending task")

    assert client._call_count <= 6  # 3 iterations × 2 reads each at most


def test_agent_uses_continuation_prompt():
    """After a tool-calling turn, Agent sends the continuation_prompt."""
    client = MockModelClient(["tool", "done", "confirmed"])
    agent = Agent(client, name="test", continuation_prompt="KEEP GOING")
    agent.run("start")

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages[0] == "start"
    assert user_messages[1] == "KEEP GOING"


def test_agent_from_config_sets_system_message():
    client = MockModelClient(["hello"])
    Agent.from_config({"name": "cfg_agent", "system_message": "Be helpful.", "max_iterations": 5}, client)

    assert client.system_message == "Be helpful."


def test_agent_from_config_defaults():
    client = MockModelClient(["hello"])
    agent = Agent.from_config({}, client)

    assert agent.name == "agent"
    assert agent.max_iterations == 10


def test_agent_streamed_yields_agent_chunks():
    client = MockModelClient(["streamed answer"])
    agent = Agent(client, name="streamer")
    chunks = list(agent.run_streamed("task"))

    assert all(isinstance(c, AgentChunk) for c in chunks)
    assert all(c.agent_name == "streamer" for c in chunks)
    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert len(generating) == 1
    assert generating[0].content == "streamed answer"


def test_agent_streamed_iteration_increments_on_tool_use():
    client = MockModelClient(["tool", "done", "confirmed"])
    agent = Agent(client, name="it_agent")
    chunks = list(agent.run_streamed("task"))

    iterations = {c.iteration for c in chunks}
    assert 0 in iterations  # first call
    assert 1 in iterations  # continuation after tool use


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


def test_workflow_chains_output_to_next_input():
    """Output of step 0 becomes the task for step 1."""
    client_a = MockModelClient(["step A output"])
    client_b = MockModelClient(["step B output"])
    wf = Workflow(agents=[Agent(client_a, name="a"), Agent(client_b, name="b")])
    result = wf.run("initial task")

    assert result == "step B output"
    # Step B received step A's output as its task
    user_msgs_b = [m["content"] for m in client_b.messages if m["role"] == "user"]
    assert user_msgs_b[0] == "step A output"


def test_workflow_run_single_agent():
    client = MockModelClient(["only answer"])
    wf = Workflow(agents=[Agent(client, name="solo")])
    assert wf.run("task") == "only answer"


def test_workflow_streamed_yields_workflow_chunks():
    client_a = MockModelClient(["part one"])
    client_b = MockModelClient(["part two"])
    wf = Workflow(agents=[Agent(client_a, name="a"), Agent(client_b, name="b")])
    chunks = list(wf.run_streamed("go"))

    assert all(isinstance(c, WorkflowChunk) for c in chunks)
    steps = {c.step for c in chunks}
    assert steps == {0, 1}


def test_workflow_streamed_step_tags():
    client_a = MockModelClient(["result a"])
    client_b = MockModelClient(["result b"])
    wf = Workflow(agents=[Agent(client_a, name="alpha"), Agent(client_b, name="beta")])
    chunks = list(wf.run_streamed("start"))

    step0 = [c for c in chunks if c.step == 0]
    step1 = [c for c in chunks if c.step == 1]
    assert all(c.agent_name == "alpha" for c in step0)
    assert all(c.agent_name == "beta" for c in step1)


def test_workflow_from_config():
    def make_client(cfg):
        responses = cfg.get("_test_responses", ["ok"])
        return MockModelClient(responses)

    configs = [
        {"name": "first", "_test_responses": ["first output"]},
        {"name": "second", "_test_responses": ["second output"]},
    ]
    wf = Workflow.from_config(configs, make_client)

    assert len(wf.agents) == 2
    assert wf.agents[0].name == "first"
    assert wf.agents[1].name == "second"
    assert wf.run("go") == "second output"


# ---------------------------------------------------------------------------
# AgenticModelClient tests
# ---------------------------------------------------------------------------


def test_agentic_client_chat_runs_agent_loop():
    """chat() runs the full Agent loop, not a single turn."""
    client = MockModelClient(["tool", "after tool", "done"])
    ac = AgenticModelClient(Agent(client, max_iterations=5))
    result = ac.chat("do something with tools")
    assert result == "done"
    assert client._call_count == 3  # tool(2) + confirmation(1)


def test_agentic_client_chat_no_tools_single_call():
    """chat() without tool use resolves in one model call."""
    client = MockModelClient(["plain answer"])
    ac = AgenticModelClient(Agent(client))
    assert ac.chat("hello") == "plain answer"
    assert client._call_count == 1


def test_agentic_client_chat_streamed_yields_stream_chunks():
    """chat_streamed() yields StreamChunk, not AgentChunk."""
    client = MockModelClient(["stream result"])
    ac = AgenticModelClient(Agent(client))
    chunks = list(ac.chat_streamed("task"))
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert not any(isinstance(c, AgentChunk) for c in chunks)


def test_agentic_client_messages_delegated():
    """messages property delegates to inner_client — both refer to the same list."""
    client = MockModelClient(["reply"])
    ac = AgenticModelClient(Agent(client))
    ac.chat("hi")
    assert ac.messages is client.messages


def test_agentic_client_mcp_client_delegated():
    """Setting mcp_client on AgenticModelClient propagates to inner_client."""
    client = MockModelClient(["hi"])
    ac = AgenticModelClient(Agent(client))
    mock_mcp = object()
    ac.mcp_client = mock_mcp
    assert client.mcp_client is mock_mcp


def test_agentic_client_system_message_delegated():
    """system_message property delegates to inner_client."""
    client = MockModelClient(["hi"])
    ac = AgenticModelClient(Agent(client))
    ac.system_message = "Be helpful."
    assert client.system_message == "Be helpful."


def test_agentic_client_generate_delegates_to_inner():
    """generate() bypasses the Agent loop and calls inner_client directly."""
    client = MockModelClient(["generated"])
    ac = AgenticModelClient(Agent(client))
    result = ac.generate("prompt")
    assert result == "generated"
    assert client._call_count == 1
