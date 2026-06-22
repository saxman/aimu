"""
Tests for aimu.agents: Agent, Chain, agent.as_model_client(), example agents.

Unit tests use MockModelClient from helpers (deterministic, no backend needed).
The model_client fixture is available for integration tests:
  - Default (no --client): MockModelClient
  - pytest tests/test_agents.py --client=ollama --model=LLAMA_3_2_3B
"""

import threading
from typing import Iterable
from unittest.mock import patch

import pytest

from aimu.agents import Agent, Chain, OrchestratorAgent, Runner
from aimu.agents.prebuilt import ResearchReportAgent
from aimu.models import BaseModelClient, StreamChunk, StreamingContentType
from aimu.tools import tool
from helpers import MockModelClient, create_real_model_client, resolve_model_params

_MOCK = "mock"


@tool
def _dummy_tool(x: str) -> str:
    """A no-op tool used only to give an agent a non-empty tool list."""
    return x


class _LoopingToolClient(BaseModelClient):
    """A client that calls a tool on every turn while tools are available, and only
    produces a final answer once tools are disabled (the ``tools=[]`` per-call override).

    Used to exercise ``Agent.final_answer_prompt``: the agent loops until it hits
    ``max_iterations`` with a tool call pending, then the forced wrap-up turn (tools
    disabled) makes this client return ``final_text``. ``tools_seen`` records the value of
    ``self.tools`` on each ``_chat`` call so a test can assert the wrap-up turn had no tools.
    """

    def __init__(self, final_text: str = "FORCED SUMMARY"):
        from unittest.mock import MagicMock

        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools = []
        self.last_thinking = ""
        self.concurrent_tool_calls = False
        self._streaming_content_type = StreamingContentType.DONE
        self.final_text = final_text
        self.tools_seen: list[list] = []

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        self.messages.append({"role": "user", "content": user_message})
        self.tools_seen.append(list(self.tools))
        if self.tools:  # tools available -> emit a tool round, never finish on its own
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "t", "content": "result", "tool_call_id": "x"})
            self.messages.append({"role": "assistant", "content": ""})
            return ""
        self.messages.append({"role": "assistant", "content": self.final_text})  # tools disabled -> wrap up
        return self.final_text

    def _chat_streamed(self, user_message, generate_kwargs=None, use_tools=True, images=None):
        text = self._chat(user_message, generate_kwargs, use_tools)
        self._streaming_content_type = StreamingContentType.GENERATING
        yield StreamChunk(StreamingContentType.GENERATING, text)
        self._streaming_content_type = StreamingContentType.DONE

    def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        return self._chat(prompt, generate_kwargs)


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


def test_as_model_client_tools_delegated():
    from aimu.tools import tool

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    client = MockModelClient(["hi"])
    view = Agent(client).as_model_client()
    view.tools = [noop]
    assert client.tools == [noop]


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


def test_handle_tool_calls_sequential_by_default():
    from aimu.tools import tool

    call_order = []

    @tool
    def tool_a() -> str:
        """Tool A."""
        call_order.append("tool_a")
        return "tool_a"

    @tool
    def tool_b() -> str:
        """Tool B."""
        call_order.append("tool_b")
        return "tool_b"

    client = MockModelClient([])
    client.tools = [tool_a, tool_b]
    tool_calls = [{"name": "tool_a", "arguments": {}}, {"name": "tool_b", "arguments": {}}]
    client._handle_tool_calls(tool_calls)

    assert call_order == ["tool_a", "tool_b"]


def test_handle_tool_calls_concurrent_runs_in_parallel():
    from aimu.tools import tool

    barrier = threading.Barrier(2, timeout=2.0)

    @tool
    def tool_a() -> str:
        """Tool A."""
        barrier.wait()
        return "tool_a"

    @tool
    def tool_b() -> str:
        """Tool B."""
        barrier.wait()
        return "tool_b"

    client = MockModelClient([])
    client.concurrent_tool_calls = True
    client.tools = [tool_a, tool_b]
    tool_calls = [{"name": "tool_a", "arguments": {}}, {"name": "tool_b", "arguments": {}}]
    client._handle_tool_calls(tool_calls)

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_messages) == 2


def test_handle_tool_calls_concurrent_results_in_original_order():
    import time

    from aimu.tools import tool

    @tool
    def tool_a() -> str:
        """Tool A (slow)."""
        time.sleep(0.05)
        return "tool_a"

    @tool
    def tool_b() -> str:
        """Tool B."""
        return "tool_b"

    client = MockModelClient([])
    client.concurrent_tool_calls = True
    client.tools = [tool_a, tool_b]
    tool_calls = [{"name": "tool_a", "arguments": {}}, {"name": "tool_b", "arguments": {}}]
    client._handle_tool_calls(tool_calls)

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
# Runner.as_tool(): any runner (agent or workflow) usable as a tool
# ---------------------------------------------------------------------------


def test_runner_as_tool_dispatches_to_run():
    """as_tool() wraps run() as a @tool callable named/described from the agent."""
    agent = Agent(MockModelClient(["agent reply"]), system_message="Research the topic.\nMore detail.", name="alpha")
    fn = agent.as_tool()
    assert fn.__name__ == "alpha"
    assert fn.__tool_spec__["function"]["name"] == "alpha"
    # Description is the first line of the system message.
    assert fn.__tool_spec__["function"]["description"] == "Research the topic."
    assert fn.__tool_spec__["function"]["parameters"]["properties"] == {"task": {"type": "string"}}
    assert fn.__tool_is_async__ is False
    assert fn("anything") == "agent reply"


def test_runner_as_tool_workflow_generic_description():
    """A workflow has no system_message, so as_tool() falls back to a generic description."""
    chain = Chain(agents=[Agent(MockModelClient(["step out"]))], name="my chain")
    fn = chain.as_tool()
    assert fn.__name__ == "my_chain"  # sanitised
    assert fn.__tool_spec__["function"]["description"] == "Delegate a task to the my_chain runner."
    assert fn("go") == "step out"


def test_runner_as_tool_custom_name_and_description():
    agent = Agent(MockModelClient(["x"]), name="alpha")
    fn = agent.as_tool(name="custom name", description="Do the thing.")
    assert fn.__name__ == "custom_name"
    assert fn.__tool_spec__["function"]["description"] == "Do the thing."


def test_orchestrator_assemble_accepts_workflow_worker():
    """assemble() accepts any Runner as a worker, including a Chain workflow."""
    client = MockModelClient(["done"])
    chain = Chain(agents=[Agent(MockModelClient(["chain result"]))], name="researcher")
    agent_worker = Agent(MockModelClient(["agent result"]), system_message="Critique.", name="critic")

    orch = OrchestratorAgent.assemble(client, "Use the workers.", workers=[chain, agent_worker])
    tool_names = {t.__name__ for t in client.tools}
    assert tool_names == {"researcher", "critic"}
    assert orch.run("task") == "done"


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

        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
            if not stream:
                return super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            def _gen():
                self.messages.append({"role": "user", "content": user_message})
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


# ---------------------------------------------------------------------------
# Per-run tools= override on Agent.run()
# ---------------------------------------------------------------------------


class _ToolsRecordingClient(MockModelClient):
    """Records the value of ``self.tools`` seen on each ``_chat`` call during a run."""

    def __init__(self, responses):
        super().__init__(responses)
        self.tools_per_call = []

    def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self.tools_per_call.append(list(self.tools))
        return super()._chat(user_message, generate_kwargs, use_tools, stream, images=images, audio=audio)


def test_agent_run_tools_override_applied_each_loop_call():
    base_tool, override_tool = object(), object()
    client = _ToolsRecordingClient(["tool", "after", "done"])  # one tool round → 2 chat() calls
    agent = Agent(client, name="t", tools=[base_tool])

    agent.run("task", tools=[override_tool])

    # Both the initial and continuation chat() calls saw the override.
    assert client.tools_per_call == [[override_tool], [override_tool]]


def test_agent_run_tools_override_restored_to_configured_after_run():
    base_tool, override_tool = object(), object()
    client = MockModelClient(["done"])
    agent = Agent(client, name="t", tools=[base_tool])

    agent.run("task", tools=[override_tool])

    # _prepare_run set client.tools to the agent's configured tools; the per-call
    # override is restored, so the agent's tools remain in place afterward.
    assert client.tools == [base_tool]


def test_agent_run_tools_none_uses_configured_tools():
    base_tool = object()
    client = _ToolsRecordingClient(["done"])
    agent = Agent(client, name="t", tools=[base_tool])

    agent.run("task")  # no override

    assert client.tools_per_call == [[base_tool]]


def test_agent_run_tools_empty_disables_for_run_only():
    base_tool = object()
    client = _ToolsRecordingClient(["done"])
    agent = Agent(client, name="t", tools=[base_tool])

    agent.run("task", tools=[])

    assert client.tools_per_call == [[]]  # disabled during the run...
    assert client.tools == [base_tool]  # ...restored to configured tools after


def test_agent_run_streamed_tools_override_applied():
    override_tool = object()
    client = _ToolsRecordingClient(["tool", "after", "done"])
    agent = Agent(client, name="t", tools=[object()])

    list(agent.run("task", stream=True, tools=[override_tool]))

    # Every recorded chat() call (the mock re-dispatches internally when streaming)
    # saw the override; the point is the override is live across the whole stream.
    assert client.tools_per_call and all(seen == [override_tool] for seen in client.tools_per_call)
    assert client.tools[0] is not override_tool  # restored to configured tools


# ---------------------------------------------------------------------------
# Agent final_answer_prompt (forced wrap-up at max_iterations)
# ---------------------------------------------------------------------------


def test_agent_final_answer_prompt_forces_wrap_up_at_cap():
    client = _LoopingToolClient(final_text="FORCED SUMMARY")
    agent = Agent(
        client,
        name="capper",
        tools=[_dummy_tool],
        max_iterations=3,
        final_answer_prompt="Stop using tools and answer now.",
    )
    result = agent.run("gather forever")

    assert result == "FORCED SUMMARY"
    # The wrap-up turn ran with tools disabled; every prior turn had tools available.
    assert client.tools_seen[-1] == []
    assert all(seen for seen in client.tools_seen[:-1])
    # Exactly one wrap-up turn (only the last call had tools disabled).
    assert sum(1 for seen in client.tools_seen if seen == []) == 1
    # The final answer is in the post-run snapshot, and the wrap-up prompt was sent.
    assert agent.messages["capper"][-1]["content"] == "FORCED SUMMARY"
    assert any(m["content"] == "Stop using tools and answer now." for m in client.messages if m["role"] == "user")


def test_agent_without_final_answer_prompt_returns_last_turn_at_cap():
    client = _LoopingToolClient(final_text="FORCED SUMMARY")
    agent = Agent(client, name="capper", tools=[_dummy_tool], max_iterations=3)
    result = agent.run("gather forever")

    # No wrap-up: returns whatever the last (tool-only) turn produced, and tools were
    # never disabled. This is the empty-output failure the feature fixes.
    assert result == ""
    assert all(seen for seen in client.tools_seen)


def test_agent_final_answer_prompt_not_triggered_on_natural_finish():
    client = MockModelClient(["done"])
    agent = Agent(client, name="natural", final_answer_prompt="WRAP UP NOW")
    result = agent.run("task")

    assert result == "done"
    assert client._call_count == 1  # no extra forced turn
    assert "WRAP UP NOW" not in [m["content"] for m in client.messages if m["role"] == "user"]


def test_agent_streamed_final_answer_prompt_forces_wrap_up():
    client = _LoopingToolClient(final_text="STREAMED SUMMARY")
    agent = Agent(
        client,
        name="scap",
        tools=[_dummy_tool],
        max_iterations=3,
        final_answer_prompt="Stop using tools and answer now.",
    )
    chunks = list(agent.run("gather", stream=True))

    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING and c.content]
    assert any(c.content == "STREAMED SUMMARY" for c in generating)
    assert client.tools_seen[-1] == []  # wrap-up turn had tools disabled


def test_orchestrator_assemble_threads_final_answer_prompt():
    worker = Agent(MockModelClient(["worker out"]), name="w", system_message="Do work.")
    orch = OrchestratorAgent.assemble(
        MockModelClient(["hi"]),
        "Coordinate.",
        workers=[worker],
        final_answer_prompt="Summarize now.",
    )
    assert orch._orchestrator.final_answer_prompt == "Summarize now."
